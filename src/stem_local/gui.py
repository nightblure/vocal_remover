"""Qt GUI: one active QProcess task; no torch/model imports in the UI process."""

import argparse
import codecs
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
from PySide6.QtCore import QProcess, QSignalBlocker, QTimer, QUrl
from PySide6.QtGui import QCloseEvent, QColor, QDesktopServices, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from stem_local.models import DEFAULT_CACHE, MODELS
from stem_local.waveform import WaveformAnalyzer, WaveformView, clock
from stem_local.worker import EVENT_PREFIX

RANGE_TOLERANCE_SECONDS = 0.001


@dataclass
class Job:
    arguments: list[str]
    title: str = ""
    watched_paths: tuple[Path, ...] = ()


class MainWindow(QMainWindow):
    def __init__(self, *, cache: Path, output: Path, stall_timeout_seconds: float = 900) -> None:
        super().__init__()
        self.cache = cache.resolve()
        self.output_dir = output.resolve()
        self.active: Job | None = None
        self.stdout_buffer = ""
        self.stdout_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.stderr_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.last_error = ""
        self.was_cancelled = False
        self.received_success = False
        self.current_report: dict[str, Any] | None = None
        self.current_report_path: Path | None = None
        self.completed_directory: Path | None = None
        self.audio_duration = 0.0
        self.stall_timeout_seconds = stall_timeout_seconds
        self.last_activity = time.monotonic()
        self.task_failed = False
        self.active_run_dir: Path | None = None
        self.health_timer = QTimer(self)
        self.health_timer.setInterval(1000)
        self.health_timer.timeout.connect(self._check_job_health)
        self.kill_timer = QTimer(self)
        self.kill_timer.setSingleShot(True)
        self.kill_timer.timeout.connect(self._kill_worker)
        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)

        self._apply_theme()
        self._build_ui()
        self._connect_waveform()
        self._update_ready_status()

    def _apply_theme(self) -> None:
        QApplication.setStyle("Fusion")
        self.setWindowTitle("Stem Local")
        self.resize(840, 540)
        palette = self.palette()
        for role, color in (
            (QPalette.ColorRole.Window, "#1e1f23"),
            (QPalette.ColorRole.WindowText, "#ebebef"),
            (QPalette.ColorRole.Base, "#292a30"),
            (QPalette.ColorRole.Text, "#ebebef"),
            (QPalette.ColorRole.Button, "#303138"),
            (QPalette.ColorRole.ButtonText, "#ebebef"),
            (QPalette.ColorRole.Mid, "#44454f"),
            (QPalette.ColorRole.PlaceholderText, "#92929f"),
            (QPalette.ColorRole.Highlight, "#9385ef"),
            (QPalette.ColorRole.HighlightedText, "#ffffff"),
        ):
            palette.setColor(role, QColor(color))
        self.setPalette(palette)
        QApplication.setPalette(palette)
        self.setStyleSheet("""
            QMainWindow, QDialog, QWidget#content { background: #1e1f23; }
            QWidget { color: #ebebef; font-size: 13px; }
            QLabel#subtitle { color: #9999a6; font-size: 12px; }
            QGroupBox { background: #25262b; border: 1px solid #373840;
                border-radius: 9px; margin-top: 12px; padding: 10px 9px 7px; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
                background: #292a30; border: 1px solid #44454f; border-radius: 6px; padding: 6px; }
            QLineEdit:focus, QDoubleSpinBox:focus { border-color: #9385ef; }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 0px; height: 0px; border: none; }
            QComboBox { padding-right: 24px; }
            QComboBox::drop-down { border: none; width: 22px; }
            QComboBox QAbstractItemView { background: #292a30; selection-background-color: #514b70; }
            QPushButton { background: #303138; border: 1px solid #44454f;
                border-radius: 7px; padding: 6px 11px; }
            QPushButton:hover { background: #3c3c47; border-color: #696875; }
            QPushButton#primary { background: #9385ef; border-color: #9385ef;
                color: #18171d; font-weight: 650; padding: 8px 24px; }
            QPushButton#primary:hover { background: #aa9cf8; }
            QPushButton#primary:disabled { background: #484251; border-color: #484251; color: #90899f; }
            QPushButton:disabled, QDoubleSpinBox:disabled { color: #71717f; background: #25262b; }
            QProgressBar { background: #36373f; border: none; border-radius: 4px;
                text-align: center; min-height: 12px; font-size: 11px; }
            QProgressBar::chunk { background: #635a98; border-radius: 4px; }
            QToolTip { background: #303138; color: #ebebef; border: 1px solid #44454f; }
        """)
        chevron = (Path(__file__).parent / "assets" / "chevron.svg").as_posix()
        arrow_style = f"""
            QComboBox::down-arrow {{ image: url('{chevron}'); width: 12px; height: 8px; }}
        """
        self.setStyleSheet(self.styleSheet() + arrow_style)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("content")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(root)
        scroll.setStyleSheet("QScrollArea { border: none; background: #1e1f23; }")
        self.setCentralWidget(scroll)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(8)
        header = self._build_header()
        layout.addLayout(header)
        input_group = self._build_input()
        advanced = self._build_settings()
        export_group = self._build_range()
        layout.addWidget(input_group)
        layout.addWidget(export_group)
        advanced_toggle = QPushButton("Настройки")
        advanced_toggle.setCheckable(True)
        advanced_toggle.toggled.connect(advanced.setVisible)
        header.addWidget(advanced_toggle)
        catalog = QPushButton("Модели")
        catalog.clicked.connect(self._show_models)
        header.addWidget(catalog)
        layout.addWidget(advanced)
        layout.addLayout(self._build_actions())
        self.task_controls = (input_group, export_group, advanced, self.choose_output)
        self.task_label = QLabel()
        self.task_label.setWordWrap(True)
        self.task_label.hide()
        layout.addWidget(self.task_label)
        self.status = QLabel("Сначала скачайте модель. После скачивания разделение работает офлайн.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        header.addWidget(self._build_logs())
        layout.addStretch()

    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        heading = QLabel("Stem Local")
        heading.setStyleSheet("font-size: 23px; font-weight: 650")
        header.addWidget(heading)
        header.addStretch()
        subtitle = QLabel("Локально · CPU · без облака")
        subtitle.setObjectName("subtitle")
        header.addWidget(subtitle)
        return header

    def _build_input(self) -> QGroupBox:
        input_group = QGroupBox("Трек")
        input_layout = QVBoxLayout(input_group)
        file_row = QHBoxLayout()
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("Выберите музыкальный файл")
        pick = QPushButton("Выбрать…")
        pick.clicked.connect(self._pick_input)
        file_row.addWidget(self.input_path)
        file_row.addWidget(pick)
        input_layout.addLayout(file_row)
        mode_row = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItem("Только вокал", "vocals")
        self.mode.addItem("Вокал + мелодии", "vocals-melody")
        self.mode.setToolTip("Вокал и мелодии сохраняются двумя отдельными файлами, без общего инструментала.")
        self.keep_bass = QCheckBox("С басом")  # noqa: RUF001 - Russian UI text, not Latin identifiers.
        self.keep_bass.setVisible(False)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        mode_row.addWidget(self.mode, 1)
        mode_row.addWidget(self.keep_bass)
        self.format = QComboBox()
        self.format.addItem("WAV · без потерь", "WAV")
        self.format.addItem("MP3 · 320 kbps", "MP3")
        self.format.setToolTip("WAV PCM 24-bit для обработки; MP3 320 kbps для прослушивания.")
        mode_row.addWidget(self.format)
        input_layout.addLayout(mode_row)
        return input_group

    def _build_settings(self) -> QWidget:
        advanced = QWidget()
        advanced_layout = QVBoxLayout(advanced)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.model = QComboBox()
        for key, spec in MODELS.items():
            if spec.role == "vocals":
                self.model.addItem(spec.title, key)
        form.addRow("Модель", self.model)
        self.accept_license = QCheckBox("ViperX: понимаю, что лицензия весов не подтверждена")
        self.accept_license.setVisible(False)
        self.model.currentIndexChanged.connect(self._model_changed)
        form.addRow(self.accept_license)

        settings_row = QHBoxLayout()
        self.threads = QSpinBox()
        self.threads.setRange(1, os.cpu_count() or 1)
        self.threads.setValue(min(4, os.cpu_count() or 1))
        settings_row.addWidget(QLabel("Потоки"))
        settings_row.addWidget(self.threads)
        self.overlap = QComboBox()
        self.overlap.addItems(["2", "4", "8"])
        settings_row.addWidget(QLabel("Перекрытие модели"))
        settings_row.addWidget(self.overlap)
        settings_row.addStretch()
        form.addRow("Настройки", settings_row)
        advanced_layout.addLayout(form)
        advanced.setVisible(False)
        return advanced

    def _build_range(self) -> QGroupBox:
        export_group = QGroupBox("Фрагмент")
        export_layout = QVBoxLayout(export_group)
        self.waveform = WaveformView()
        self.waveform.selection_changed.connect(self._wave_selection)
        export_layout.addWidget(self.waveform)
        export_layout.addLayout(self._build_range_fields())
        self.wave_hint = QLabel("По умолчанию обрабатывается весь трек.")
        self.wave_hint.setObjectName("subtitle")
        export_layout.addWidget(self.wave_hint)
        return export_group

    def _build_range_fields(self) -> QHBoxLayout:
        preview_row = QHBoxLayout()
        self.start = QDoubleSpinBox()
        self.start.setDecimals(3)
        self.start.setRange(0, 7 * 24 * 3600)
        self.start.setSuffix(" сек.")
        self.start.setMaximumWidth(118)
        self.end = QDoubleSpinBox()
        self.end.setDecimals(3)
        self.end.setRange(0, 7 * 24 * 3600)
        self.end.setSuffix(" сек.")
        self.end.setMaximumWidth(118)
        self.end.setEnabled(False)
        self.start.valueChanged.connect(self._range_changed)
        self.end.valueChanged.connect(self._range_changed)
        self.full = QCheckBox("До конца трека")
        self.full.setChecked(True)
        self.full.toggled.connect(self._full_changed)
        preview_row.addWidget(QLabel("Начало"))
        preview_row.addWidget(self.start)
        preview_row.addWidget(QLabel("Конец"))
        preview_row.addWidget(self.end)
        preview_row.addWidget(self.full)
        preview_row.addStretch()
        self.selection_label = QLabel("Весь трек")
        self.selection_label.setObjectName("subtitle")
        preview_row.addWidget(self.selection_label)
        return preview_row

    def _build_actions(self) -> QHBoxLayout:
        actions = QHBoxLayout()
        self.download_button = QPushButton("Подготовить модели")
        self.download_button.clicked.connect(self._download)
        self.run_button = QPushButton("Запустить")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self._run)
        self.run_button.setToolTip("Запустить обработку. Одновременно выполняется только одна задача.")
        self.cancel_button = QPushButton("Отменить")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        actions.addWidget(self.download_button)
        actions.addStretch()
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.run_button)

        output_row = QHBoxLayout()
        self.output_label = QLabel(f"Сохранить: {self.output_dir.name}")
        self.output_label.setMaximumWidth(170)
        self.output_label.setToolTip(str(self.output_dir))
        self.choose_output = QPushButton("Папка…")
        self.choose_output.clicked.connect(self._pick_output)
        output_row.addWidget(self.output_label, 1)
        output_row.addWidget(self.choose_output)
        output_row.addLayout(actions)
        self.output_label.setObjectName("subtitle")
        self.open_result_button = QPushButton("Папка результата")
        self.open_result_button.setEnabled(False)
        self.open_result_button.clicked.connect(self._open_result)
        output_row.insertWidget(2, self.open_result_button)
        return output_row

    def _build_logs(self) -> QPushButton:
        self.log_dialog = QDialog(self)
        self.log_dialog.setWindowTitle("Stem Local · логи")
        self.log_dialog.resize(760, 420)
        log_layout = QVBoxLayout(self.log_dialog)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setStyleSheet("font-family: monospace; font-size: 12px")
        log_layout.addWidget(self.log)
        self.log_button = QPushButton("Логи")
        self.log_button.clicked.connect(self._show_logs)
        return self.log_button

    def _connect_waveform(self) -> None:
        self.analyzer = WaveformAnalyzer(self)
        self.analyzer.duration_ready.connect(self._wave_duration)
        self.analyzer.envelope_ready.connect(self.waveform.set_peaks)
        self.analyzer.message.connect(self.wave_hint.setText)
        self.wave_timer = QTimer(self)
        self.wave_timer.setSingleShot(True)
        self.wave_timer.timeout.connect(self._load_waveform)
        self.input_path.textChanged.connect(self._input_changed)

    def _input_changed(self) -> None:
        self.wave_timer.stop()
        self.analyzer.cancel()
        self.audio_duration = 0.0
        self.waveform.reset()
        self.start.setMaximum(7 * 24 * 3600)
        self.end.setMaximum(7 * 24 * 3600)
        self.start.setValue(0)
        self.end.setValue(0)
        self.full.setChecked(True)
        self.selection_label.setText("Весь трек")
        self.wave_hint.setText("По умолчанию обрабатывается весь трек.")
        self.wave_timer.start(300)

    def _load_waveform(self) -> None:
        path = Path(self.input_path.text()).expanduser()
        if path.is_file():
            self.analyzer.load(path)

    def _wave_duration(self, duration: float) -> None:
        self.audio_duration = duration
        self.waveform.set_audio(duration)
        self.start.setMaximum(max(0.0, duration - 0.001))
        self.end.setMaximum(duration)
        if self.full.isChecked():
            self.end.setValue(duration)
        self._range_changed()

    def _full_changed(self, checked: bool) -> None:  # noqa: FBT001 - Qt toggled signal passes a positional bool.
        self.end.setEnabled(not checked)
        if checked and self.audio_duration:
            self.end.setValue(self.audio_duration)
        self._range_changed()

    def _range_changed(self) -> None:
        start = self.start.value()
        end = self.audio_duration if self.full.isChecked() and self.audio_duration else self.end.value()
        self.waveform.set_selection(start, end)
        if self.full.isChecked() and not self.audio_duration:
            self.selection_label.setText("До конца трека")
        elif end > start:
            self.selection_label.setText(f"{clock(start)} — {clock(end)} · {end - start:.1f} сек.")
        else:
            self.selection_label.setText("Конец должен быть после начала")

    def _wave_selection(self, start: float, end: float) -> None:
        with QSignalBlocker(self.start), QSignalBlocker(self.end), QSignalBlocker(self.full):
            self.start.setValue(start)
            self.end.setValue(end)
            self.full.setChecked(abs(end - self.audio_duration) < RANGE_TOLERANCE_SECONDS)
        self.end.setEnabled(not self.full.isChecked())
        self._range_changed()

    def _pick_input(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Музыкальный файл", "", "Аудио (*);;Любые файлы (*)")
        if filename:
            self.input_path.setText(filename)

    def _pick_output(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Папка экспорта", str(self.output_dir))
        if directory:
            self.output_dir = Path(directory)
            self.output_label.setText(f"Сохранить: {self.output_dir.name}")
            self.output_label.setToolTip(str(self.output_dir))

    def _model_changed(self) -> None:
        self.accept_license.setVisible(self.model.currentData() == "viperx")
        self._update_ready_status()

    def _mode_changed(self) -> None:
        self.keep_bass.setVisible(self.mode.currentData() == "vocals-melody")
        self._update_ready_status()

    def _update_ready_status(self) -> None:
        if not hasattr(self, "status") or self.active is not None:
            return
        required = [MODELS[self.model.currentData()]]
        if self.mode.currentData() == "vocals-melody":
            required.append(MODELS["htdemucs"])
        available = all(
            (self.cache / spec.id / artifact.filename).is_file()
            for spec in required
            for artifact in (spec.config, spec.weights)
        )
        self.download_button.setVisible(not available)
        self.status.setText(
            "Готово к запуску · обработка на вашем компьютере"
            if available
            else "Нажмите «Подготовить модели». Интернет нужен только для скачивания.",
        )

    def _show_models(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Модели, источники и обновления")
        dialog.resize(760, 520)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        descriptions = [
            "Каталог входит в приложение. Автоматического поиска и установки новых весов пока нет.",
            "Новая модель добавляется после проверки совместимости, лицензии и SHA-256.",
            "Обновление не заменяет ваши старые веса. Разделение работает без интернета.",
            f"Кэш: {self.cache}\n",
        ]
        for spec in MODELS.values():
            available = (self.cache / spec.id / spec.weights.filename).is_file()
            descriptions.extend(
                [
                    spec.title,
                    f"Лицензия: {spec.license}",
                    f"Файл {'есть (SHA-256 проверяется перед запуском)' if available else 'не скачан'}",
                    f"Источник: {spec.weights.url}",
                    f"SHA-256: {spec.weights.sha256}",
                    f"Лицензия: {spec.license_source}\n",
                ],
            )
        text.setPlainText("\n".join(descriptions))
        layout.addWidget(text)
        sources = QPushButton("Открыть каталог моделей MSST в браузере")
        sources.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md"),
            ),
        )
        layout.addWidget(sources)
        dialog.exec()

    def _license_ok(self) -> bool:
        if self.model.currentData() == "viperx" and not self.accept_license.isChecked():
            self._show_error("Подтвердите ограничение лицензии ViperX или выберите Kimberley.")
            return False
        return True

    def _show_error(self, message: str) -> None:
        self.status.setText(f"Ошибка: {message}")
        self._log(message)

    def _common_arguments(self) -> list[str]:
        arguments = [
            "--model",
            self.model.currentData(),
            "--cache",
            str(self.cache),
            "--mode",
            self.mode.currentData(),
        ]
        if self.accept_license.isChecked():
            arguments.append("--accept-unverified-license")
        return arguments

    def _download(self) -> None:
        if self.active is not None or not self._license_ok():
            return
        model = MODELS[self.model.currentData()]
        answer = QMessageBox.question(
            self,
            "Скачивание весов",
            f"Скачать {model.title}?\n"
            "Kimberley: около 913 MB. Для мелодий дополнительно HTDemucs.\n"
            "Интернет нужен только для этого шага. Файлы модели проверяются по SHA-256.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        arguments = ["download", *self._common_arguments()]
        self.start_job(Job(arguments=arguments, title=f"Скачивание · {model.title}"))

    def _run(self) -> None:
        if self.active is not None:
            return
        source = Path(self.input_path.text()).expanduser()
        if not source.is_file():
            self._show_error("Входной файл отсутствует. Выберите существующий аудиофайл.")
            return
        if not self._license_ok():
            return
        if not self.full.isChecked() and self.end.value() <= self.start.value():
            self._show_error("Конец фрагмента должен быть после начала.")
            return
        spec = MODELS[self.model.currentData()]
        required = [spec]
        if self.mode.currentData() == "vocals-melody":
            required.append(MODELS["htdemucs"])
        if not all(
            (self.cache / model.id / artifact.filename).is_file()
            for model in required
            for artifact in (model.config, model.weights)
        ):
            self._show_error("Нет файлов модели. Нажмите «Подготовить модели» и дождитесь окончания.")
            return
        arguments = [
            "separate",
            str(source.resolve()),
            *self._common_arguments(),
            "--output",
            str(self.output_dir),
            "--threads",
            str(self.threads.value()),
            "--overlap",
            self.overlap.currentText(),
            "--format",
            self.format.currentData(),
            "--start",
            str(self.start.value()),
        ]
        if self.full.isChecked():
            arguments.append("--full")
        else:
            arguments.extend(["--duration", str(self.end.value() - self.start.value())])
        if self.keep_bass.isChecked() and self.mode.currentData() == "vocals-melody":
            arguments.append("--keep-bass")
        watched = [source.resolve()]
        watched.extend(
            self.cache / model.id / artifact.filename
            for model in required
            for artifact in (model.config, model.weights)
        )
        self.start_job(
            Job(
                arguments=arguments,
                title=f"Обработка · {source.name} · {self.mode.currentText()}",
                watched_paths=tuple(watched),
            ),
        )

    def start_job(self, job: Job) -> None:
        """Start a validated task, rejecting overlap and rechecking watched files."""
        if self.active is not None:
            return
        missing = next((path for path in job.watched_paths if not path.is_file()), None)
        if missing is not None:
            self._show_error(f"Файл исчез до запуска задачи: {missing}")
            return
        self.active = job
        self.stdout_buffer = ""
        self.stdout_decoder.reset()
        self.stderr_decoder.reset()
        self.last_error = ""
        self.was_cancelled = False
        self.received_success = False
        self.completed_directory = None
        self.current_report = None
        self.current_report_path = None
        self.open_result_button.setEnabled(False)
        self.active_run_dir = None
        self.task_failed = False
        self.last_activity = time.monotonic()
        self.health_timer.start()
        self.task_label.setText(job.title)
        self.task_label.setVisible(bool(job.title))
        self.run_button.setEnabled(False)
        self.download_button.setEnabled(False)
        for control in self.task_controls:
            control.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setValue(0)
        self.progress.setRange(0, 0)
        self.status.setText("Запуск отдельного процесса…")
        self.process.setArguments(
            [
                "-u",
                "-m",
                "stem_local.worker",
                *job.arguments,
                "--parent-pid",
                str(os.getpid()),
                "--parent-created",
                str(psutil.Process().create_time()),
            ],
        )
        self.process.start()

    def _read_stdout(self) -> None:
        self.stdout_buffer += self.stdout_decoder.decode(self.process.readAllStandardOutput().data())
        while "\n" in self.stdout_buffer:
            line, self.stdout_buffer = self.stdout_buffer.split("\n", 1)
            if line.startswith(EVENT_PREFIX):
                try:
                    self.handle_worker_event(json.loads(line[len(EVENT_PREFIX) :]))
                except (ValueError, KeyError, OSError) as error:
                    self._log(f"Ошибка события: {error}")
            elif line.strip():
                self._log(line)

    def _read_stderr(self) -> None:
        self._log(self.stderr_decoder.decode(self.process.readAllStandardError().data()).replace("\r", "\n"))

    def handle_worker_event(self, event: dict[str, Any]) -> None:
        """Apply a decoded worker protocol event to the current task display."""
        kind = event["event"]
        if (self.was_cancelled or self.task_failed) and kind != "error":
            return
        if kind in {"run", "status", "progress", "download", "ready", "completed"}:
            self.last_activity = time.monotonic()
        if kind == "run":
            self.active_run_dir = Path(event["directory"])
        elif kind == "status":
            self.status.setText(event["message"])
            self.progress.setRange(0, 0)
            self._log(event["message"])
        elif kind == "progress":
            self._inference_progress(done=event["done"], total=event["total"])
        elif kind == "download":
            self._download_progress(event)
        elif kind == "error":
            self.last_error = event["message"]
            self._log(self.last_error)
        elif kind == "completed":
            self._load_report(Path(event["report"]))
        elif kind == "ready":
            self.received_success = True
            self._log("Модель скачана и проверена. Можно отключить интернет.")

    def _inference_progress(self, *, done: int, total: int) -> None:
        if total > 0:
            self.progress.setRange(0, 1000)
            self.progress.setValue(round(1000 * done / total))

    def _download_progress(self, event: dict[str, Any]) -> None:
        self.status.setText(f"Скачивание {event['file']} · {event['bytes'] / 1e6:.0f} MB")
        if event["total"]:
            self.progress.setRange(0, 1000)
            self.progress.setValue(round(1000 * event["bytes"] / event["total"]))
        else:
            self.progress.setRange(0, 0)

    def _load_report(self, report_path: Path) -> None:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.current_report_path = report_path
        self.current_report = report
        self.completed_directory = report_path.parent
        self.received_success = True
        wall = report["timing"]["wall_seconds"]
        ram = report["memory"]["sampled_process_tree_peak_rss_bytes"] / 1024**3
        self._log(f"Готово · {wall:.1f} сек. · RAM (RSS, sampled): {ram:.2f} GiB; отчёт: {report_path}")

    def _finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self.health_timer.stop()
        self.kill_timer.stop()
        self._read_stdout()
        self._read_stderr()
        success = (
            exit_code == 0
            and exit_status == QProcess.ExitStatus.NormalExit
            and self.received_success
            and not self.was_cancelled
            and not self.task_failed
        )
        text = "Отменено" if self.was_cancelled else ("Готово" if success else "Ошибка")
        self.active = None
        self.task_label.clear()
        self.task_label.hide()
        self.progress.setRange(0, 1000)
        self.progress.setValue(1000 if success else 0)
        self.download_button.setEnabled(True)
        self.run_button.setEnabled(True)
        for control in self.task_controls:
            control.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._update_ready_status()
        self.status.setText(text if not self.last_error else f"{text}: {self.last_error}")
        if success and self.completed_directory is not None:
            self.open_result_button.setEnabled(True)
            self._open_directory(self.completed_directory)
        self.completed_directory = None

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.Crashed and (self.was_cancelled or self.task_failed):
            return
        if not self.last_error:
            self.last_error = self.process.errorString()
        self._log(self.last_error)
        if error == QProcess.ProcessError.FailedToStart:
            self._finished(1, QProcess.ExitStatus.CrashExit)

    def _cancel(self) -> None:
        if self.active is None:
            return
        self.was_cancelled = True
        self.status.setText("Отмена…")
        self.cancel_button.setEnabled(False)
        self._terminate_worker()

    def _terminate_worker(self) -> None:
        self.health_timer.stop()
        self.process.terminate()
        self.kill_timer.start(5000)

    def _check_job_health(self) -> None:
        if self.active is None:
            return
        missing = next((path for path in self.active.watched_paths if not path.is_file()), None)
        reason = ""
        if missing is not None:
            reason = f"При обработке исчез файл: {missing}"
        elif self.active_run_dir is not None and not self.active_run_dir.is_dir():
            reason = f"Исчезла папка текущей задачи: {self.active_run_dir}"
        elif time.monotonic() - self.last_activity >= self.stall_timeout_seconds:
            reason = f"Таймаут: нет прогресса {self.stall_timeout_seconds:g} сек. Задача остановлена."
        if reason:
            self.task_failed = True
            self.last_error = reason
            self.cancel_button.setEnabled(False)
            self._show_error(reason)
            self._terminate_worker()

    def _kill_worker(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

    def _log(self, message: str) -> None:
        self.log.appendPlainText(message.strip())

    def _show_logs(self) -> None:
        self.log_dialog.show()
        self.log_dialog.raise_()
        self.log_dialog.activateWindow()

    def _open_result(self) -> None:
        if self.current_report_path is not None:
            self._open_directory(self.current_report_path.parent)

    def _open_directory(self, directory: Path) -> None:
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory))):
            self._log(f"Ошибка автоматического открытия папки: {directory}")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt virtual method name.
        if self.active is not None:
            self._cancel()
            if not self.process.waitForFinished(5000):
                self._kill_worker()
                self.process.waitForFinished(2000)
        self.health_timer.stop()
        self.wave_timer.stop()
        self.analyzer.cancel(wait=True)
        self.log_dialog.close()
        event.accept()


def main() -> int:
    parser = argparse.ArgumentParser(description="Stem Local GUI")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--stall-timeout", type=float, default=900, help="Maximum seconds without worker progress")
    args = parser.parse_args()
    if args.stall_timeout <= 0:
        parser.error("--stall-timeout must be positive")
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Stem Local")
    window = MainWindow(cache=args.cache, output=args.output, stall_timeout_seconds=args.stall_timeout)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
