"""Lightweight waveform preview and two-handle selection, without loading full audio."""

import json
import math
import re
import sys
from pathlib import Path

import psutil
from PySide6.QtCore import QObject, QPointF, QProcess, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPaintEvent, QPalette, QPen
from PySide6.QtWidgets import QWidget

BINS = 1200
TIME_PATTERN = re.compile(r"pts_time:([0-9.eE+-]+)")
PEAK_KEY = "lavfi.astats.Overall.Peak_level="
SECONDS_PER_MINUTE = 60


def clock(seconds: float) -> str:
    """Format a timestamp without hiding hours on long recordings."""
    whole = max(0, int(seconds))
    minutes, remainder = divmod(whole, SECONDS_PER_MINUTE)
    if minutes >= SECONDS_PER_MINUTE:
        hours, minutes = divmod(minutes, SECONDS_PER_MINUTE)
        return f"{hours}:{minutes:02d}:{remainder:02d}"
    return f"{minutes}:{remainder:02d}"


def _probe_metadata(output: bytes) -> tuple[float, int]:
    data = json.loads(output)
    streams = data.get("streams", [])
    if not streams:
        message = "No audio stream"
        raise ValueError(message)
    duration = float(data.get("format", {}).get("duration") or streams[0]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        message = "Invalid duration"
        raise ValueError(message)
    return duration, int(streams[0].get("sample_rate", 44100))


class WaveformAnalyzer(QObject):
    """Read duration, then aggregate streaming FFmpeg peak metadata into fixed bins."""

    duration_ready = Signal(float)
    envelope_ready = Signal(object)
    message = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process: QProcess | None = None
        self.generation = 0
        self.duration = 0.0
        self.peaks = [0.0] * BINS
        self.buffer = ""
        self.timestamp = 0.0
        self.timeout_timer = QTimer(self)
        self.timeout_timer.setSingleShot(True)
        self.timeout_timer.setInterval(30000)
        self.timeout_timer.timeout.connect(self._timed_out)

    def cancel(self, *, wait: bool = False) -> None:
        """Stop only this preview process; discard callbacks from the old file."""
        self.generation += 1
        self.timeout_timer.stop()
        if self.process is not None:
            process = self.process
            self.process = None
            if wait:
                process.terminate()
                if not process.waitForFinished(1000):
                    process.kill()
                    process.waitForFinished(1000)
            else:
                process.kill()
            process.finished.connect(process.deleteLater)
            if process.state() == QProcess.ProcessState.NotRunning:
                process.deleteLater()

    def _start_tool(self, process: QProcess, program: str, arguments: list[str]) -> None:
        parent = psutil.Process()
        process.start(
            sys.executable,
            [
                "-m",
                "stem_local.tool_runner",
                "--parent-pid",
                str(parent.pid),
                "--parent-created",
                str(parent.create_time()),
                "--",
                program,
                *arguments,
            ],
        )

    def load(self, path: Path) -> None:
        """Start a local, asynchronous metadata probe for an existing audio file."""
        self.cancel()
        if not path.is_file():
            self.message.emit("Входной файл отсутствует. Выберите существующий файл.")
            return
        self.duration = 0.0
        self.peaks = [0.0] * BINS
        self.message.emit("Строим волну…")
        generation = self.generation
        process = QProcess(self)
        self.process = process
        self.timeout_timer.start()
        process.errorOccurred.connect(
            lambda _error: self._failed(generation, "Волна недоступна: FFprobe не найден или не запустился."),
        )
        process.finished.connect(lambda code, _status: self._probed(process, path, generation, code))
        self._start_tool(
            process,
            "ffprobe",
            [
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-select_streams",
                "a:0",
                "-show_entries",
                "format=duration:stream=sample_rate,duration",
                "-of",
                "json",
                str(path.resolve()),
            ],
        )

    def _failed(self, generation: int, message: str) -> None:
        if generation == self.generation:
            self.timeout_timer.stop()
            self.message.emit(message)

    def _timed_out(self) -> None:
        self.cancel()
        self.message.emit("Таймаут анализа волны. Процесс остановлен; границы можно задать числами.")

    def _probed(self, process: QProcess, path: Path, generation: int, code: int) -> None:
        if generation != self.generation:
            return
        self.timeout_timer.stop()
        self.process = None
        if code:
            process.deleteLater()
            self.message.emit("Ошибка построения волны. Границы можно задать числами.")
            return
        try:
            self.duration, rate = _probe_metadata(bytes(process.readAllStandardOutput().data()))
        except (ValueError, KeyError, TypeError):
            self.message.emit("Ошибка построения волны. Границы можно задать числами.")
            return
        finally:
            process.deleteLater()
        self.duration_ready.emit(self.duration)
        self.buffer = ""
        self.timestamp = 0.0
        preview = QProcess(self)
        self.process = preview
        self.timeout_timer.start()
        preview.readyReadStandardOutput.connect(lambda: self._read_peaks(preview, generation))
        preview.errorOccurred.connect(lambda _error: self._failed(generation, "Ошибка запуска анализа волны."))
        preview.finished.connect(lambda code, _status: self._analyzed(preview, generation, code))
        samples_per_bin = min(1048576, max(1, round(rate * self.duration / BINS)))
        filters = (
            f"asetnsamples=n={samples_per_bin}:p=0,"
            "astats=metadata=1:reset=1:measure_perchannel=none:measure_overall=Peak_level,"
            "ametadata=print:key=lavfi.astats.Overall.Peak_level:file=-"
        )
        self._start_tool(
            preview,
            "ffmpeg",
            [
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-i",
                str(path.resolve()),
                "-map",
                "0:a:0",
                "-vn",
                "-af",
                filters,
                "-f",
                "null",
                "-",
            ],
        )

    def _read_peaks(self, process: QProcess, generation: int) -> None:
        if generation != self.generation:
            return
        data = bytes(process.readAllStandardOutput().data())
        if data:
            self.timeout_timer.start()
        self.buffer += data.decode("ascii", errors="replace")
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            match = TIME_PATTERN.search(line)
            if match:
                self.timestamp = float(match[1])
            elif line.startswith(PEAK_KEY):
                decibels = float(line[len(PEAK_KEY) :])
                peak = 10 ** (decibels / 20) if math.isfinite(decibels) else 0.0
                index = min(BINS - 1, max(0, int(self.timestamp / self.duration * BINS)))
                self.peaks[index] = max(self.peaks[index], min(1.0, peak))

    def _analyzed(self, process: QProcess, generation: int, code: int) -> None:
        if generation != self.generation:
            return
        self._read_peaks(process, generation)
        self.timeout_timer.stop()
        self.process = None
        process.deleteLater()
        if code:
            self.message.emit("Волна недоступна. Числовой выбор фрагмента работает независимо.")
            return
        self.envelope_ready.emit(self.peaks.copy())
        self.message.emit("Перетащите маркеры по краям, чтобы выбрать фрагмент.")


class WaveformView(QWidget):
    """A small CPU-painted waveform with two draggable, keyboard-accessible handles."""

    selection_changed = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.duration = 0.0
        self.start = 0.0
        self.end = 0.0
        self.peaks: list[float] = []
        self.active_handle = "start"
        self.dragging = False
        self.setFixedHeight(104)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Выбор фрагмента: начало и конец")
        self.setToolTip("Перетащите начало или конец. Стрелки — 0,1 секунды, Shift+стрелки — 1 секунда.")

    def reset(self) -> None:
        """Clear the previous file immediately, even while the next probe is pending."""
        self.duration = self.start = self.end = 0.0
        self.peaks = []
        self.dragging = False
        self.update()

    def set_audio(self, duration: float) -> None:
        """Set the track extent and initially select the whole track."""
        self.duration = duration
        self.set_selection(0, duration)

    def set_peaks(self, peaks: list[float]) -> None:
        """Accept a fixed-size display envelope, never the source audio itself."""
        self.peaks = peaks
        self.update()

    def set_selection(self, start: float, end: float) -> None:
        """Update selection without recursively emitting a user edit."""
        self.start = min(self.duration, max(0.0, start))
        self.end = min(self.duration, max(self.start, end))
        self.update()

    def _x(self, seconds: float) -> float:
        return 12 + seconds / self.duration * max(1, self.width() - 24) if self.duration else 12

    def _move_handle(self, seconds: float) -> None:
        gap = min(0.001, self.duration)
        if self.active_handle == "start":
            self.start = max(0.0, min(seconds, self.end - gap))
        else:
            self.end = min(self.duration, max(seconds, self.start + gap))
        self.update()
        self.selection_changed.emit(self.start, self.end)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt virtual method name.
        """Pick the closest endpoint and start dragging with the left button."""
        if not self.duration or event.button() != Qt.MouseButton.LeftButton:
            return
        self.setFocus()
        x = event.position().x()
        self.active_handle = "start" if abs(x - self._x(self.start)) <= abs(x - self._x(self.end)) else "end"
        self.dragging = True
        self.mouseMoveEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt virtual method name.
        """Clamp endpoint movement to the track and prevent reversed selections."""
        if self.dragging and self.duration:
            seconds = (event.position().x() - 12) / max(1, self.width() - 24) * self.duration
            self._move_handle(seconds)

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:  # noqa: N802 - Qt virtual method name.
        """Finish a drag; later mouse movement must not change the range."""
        self.dragging = False

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt virtual method name.
        """Allow precise adjustment without requiring a pointing device."""
        if self.duration and event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            amount = 1.0 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 0.1
            current = self.start if self.active_handle == "start" else self.end
            self._move_handle(current + (amount if event.key() == Qt.Key.Key_Right else -amount))
        else:
            super().keyPressEvent(event)

    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802 - Qt virtual method name.
        """Draw the overview, selected range and endpoints with ordinary QPainter."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        accent = palette.color(QPalette.ColorRole.Highlight)
        muted = palette.color(QPalette.ColorRole.PlaceholderText)
        painter.setPen(QPen(palette.color(QPalette.ColorRole.Mid)))
        painter.setBrush(palette.color(QPalette.ColorRole.Base))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 8, 8)
        if not self.duration:
            painter.setPen(muted)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Здесь появится волна выбранного трека")
            return
        left, right = self._x(self.start), self._x(self.end)
        selection = QColor(accent)
        selection.setAlpha(30)
        painter.fillRect(QRectF(left, 19, right - left, 68), selection)
        count = len(self.peaks)
        middle = 53.0
        for index, peak in enumerate(self.peaks):
            x = 12 + (index + 0.5) / max(1, count) * max(1, self.width() - 24)
            height = max(0.6, math.sqrt(peak) * 29)
            painter.setPen(QPen(accent if left <= x <= right else muted, 1))
            painter.drawLine(QPointF(x, middle - height), QPointF(x, middle + height))
        for x in (left, right):
            painter.setPen(QPen(accent, 2))
            painter.drawLine(QPointF(x, 16), QPointF(x, 88))
            painter.setBrush(accent)
            painter.drawRoundedRect(QRectF(x - 4, 15, 8, 15), 3, 3)
        painter.setPen(muted)
        painter.drawText(QRectF(12, 86, 120, 16), Qt.AlignmentFlag.AlignLeft, "0:00")
        painter.drawText(QRectF(self.width() - 132, 86, 120, 16), Qt.AlignmentFlag.AlignRight, clock(self.duration))
