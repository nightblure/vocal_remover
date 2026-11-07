# Copyright (c) 2026
"""Waveform tests do not download or run neural-network models."""

import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QProcess, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from stem_local.waveform import BINS, WaveformAnalyzer, WaveformView, clock


@pytest.fixture(autouse=True)
def application() -> QApplication:
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    return app


def test_waveform_defaults_to_the_full_track() -> None:
    view = WaveformView()
    duration = 222.5

    view.set_audio(duration)

    assert view.start == 0
    assert view.end == duration
    view.close()


def test_waveform_drag_clamps_both_handles() -> None:
    view = WaveformView()
    view.resize(624, 104)
    view.set_audio(60)
    selected: list[tuple[float, float]] = []
    view.selection_changed.connect(lambda start, end: selected.append((start, end)))

    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=QPoint(112, 50))
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=QPoint(112, 50))
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=QPoint(512, 50))
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=QPoint(512, 50))

    assert selected[-1] == pytest.approx((10, 50))
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=QPoint(112, 50))
    QTest.mouseMove(view, QPoint(1000, 50))
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=QPoint(1000, 50))
    assert view.start < view.end
    view.set_selection(10, 50)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=QPoint(512, 50))
    QTest.mouseMove(view, QPoint(-1000, 50))
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=QPoint(-1000, 50))
    assert 0 <= view.start < view.end <= view.duration
    view.close()


def test_waveform_has_keyboard_adjustment() -> None:
    view = WaveformView()
    view.set_audio(60)

    QTest.keyClick(view, Qt.Key.Key_Right)

    assert view.start == pytest.approx(0.1)
    view.close()


@pytest.mark.skipif(not shutil.which("ffprobe") or not shutil.which("ffmpeg"), reason="FFmpeg is not installed")
def test_waveform_analyzer_streams_real_ffmpeg_peaks(
    application: QApplication,
    audio_factory: Callable[..., tuple[Path, np.ndarray]],
) -> None:
    path, _ = audio_factory(frames=44100, sample_rate=44100)
    analyzer = WaveformAnalyzer()
    durations: list[float] = []
    envelopes: list[list[float]] = []
    analyzer.duration_ready.connect(durations.append)
    analyzer.envelope_ready.connect(envelopes.append)

    analyzer.load(path)
    deadline = time.monotonic() + 10
    while not envelopes and time.monotonic() < deadline:
        application.processEvents()
        if analyzer.process is not None:
            analyzer.process.waitForFinished(20)

    analyzer.cancel()
    minimum_peak = 0.1
    assert durations == pytest.approx([1])
    assert len(envelopes[0]) == BINS
    assert max(envelopes[0]) > minimum_peak
    assert all(0 <= value <= 1 for value in envelopes[0])


def test_waveform_cancel_invalidates_old_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    analyzer = WaveformAnalyzer()
    path = tmp_path / "source.wav"
    path.touch()
    monkeypatch.setattr(analyzer, "_start_tool", lambda *_args: None)
    messages: list[str] = []
    analyzer.message.connect(messages.append)
    analyzer.load(path)
    process = analyzer.process
    assert process is not None
    previous_generation = analyzer.generation

    analyzer.cancel()
    process.errorOccurred.emit(QProcess.ProcessError.FailedToStart)

    assert analyzer.generation > previous_generation
    assert analyzer.process is None
    assert messages == ["Строим волну…"]


def test_waveform_timeout_kills_its_own_stuck_process(application: QApplication) -> None:
    analyzer = WaveformAnalyzer()
    process = QProcess(analyzer)
    analyzer.process = process
    messages: list[str] = []
    analyzer.message.connect(messages.append)
    process.start(sys.executable, ["-c", "import time; time.sleep(60)"])
    assert process.waitForStarted(2000)
    analyzer.timeout_timer.setInterval(30)

    analyzer.timeout_timer.start()
    deadline = time.monotonic() + 3
    while analyzer.process is not None and time.monotonic() < deadline:
        application.processEvents()
        process.waitForFinished(20)

    assert analyzer.process is None
    assert any("Таймаут" in message for message in messages)
    assert not analyzer.timeout_timer.isActive()


def test_missing_waveform_source_fails_without_starting_a_process(
    tmp_path: Path,
) -> None:
    analyzer = WaveformAnalyzer()
    messages: list[str] = []
    analyzer.message.connect(messages.append)

    analyzer.load(tmp_path / "missing.wav")

    assert analyzer.process is None
    assert not analyzer.timeout_timer.isActive()
    assert "отсутствует" in messages[-1]


def test_timestamp_format_includes_hours() -> None:
    assert clock(75.5) == "1:15"
    assert clock(3661) == "1:01:01"
