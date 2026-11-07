# Copyright (c) 2026
"""Headless Qt behavior and real worker lifecycle regression tests."""

import contextlib
import json
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import psutil
import pytest
from PySide6.QtCore import QProcess, QProcessEnvironment, QUrl
from PySide6.QtWidgets import QApplication, QGroupBox, QListWidget, QMessageBox, QPushButton

from stem_local.guardian import same_process
from stem_local.gui import Job, MainWindow
from stem_local.models import MODELS


@pytest.fixture
def app() -> QApplication:
    application = QApplication.instance() or QApplication(sys.argv[:1])
    assert isinstance(application, QApplication)
    return application


@pytest.fixture
def window(app: QApplication, tmp_path: Path) -> Iterator[MainWindow]:
    widget = MainWindow(cache=tmp_path / "models", output=tmp_path / "output")
    yield widget
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_window_starts_without_loading_torch(window: MainWindow) -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import stem_local.gui; import sys; assert 'torch' not in sys.modules"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert window.start.value() == 0
    assert window.full.isChecked()
    assert not window.end.isEnabled()
    assert window.run_button.text() == "Запустить"
    assert not window.cancel_button.isEnabled()
    assert window.model.currentData() == "kimberley"
    assert "Готовые результаты" not in {group.title() for group in window.findChildren(QGroupBox)}
    assert "Очередь" not in {group.title() for group in window.findChildren(QGroupBox)}
    assert not window.findChildren(QListWidget)
    assert not {"Продолжить очередь", "Убрать ожидающие"}.intersection(
        button.text() for button in window.findChildren(QPushButton)
    )


def test_melody_mode_enables_bass_and_selects_the_mode(window: MainWindow) -> None:
    window.mode.setCurrentIndex(1)
    window.keep_bass.setChecked(True)

    assert window.keep_bass.isEnabled()
    assert window.mode.currentData() == "vocals-melody"
    assert window.format.count() == len(("WAV", "MP3"))


def test_waveform_selection_updates_numeric_bounds(window: MainWindow) -> None:
    start_seconds, end_seconds = 30.125, 75.5
    window.analyzer.duration_ready.emit(180)

    window.waveform.selection_changed.emit(start_seconds, end_seconds)

    assert window.start.value() == start_seconds
    assert window.end.value() == end_seconds
    assert not window.full.isChecked()
    assert window.end.isEnabled()
    assert window.waveform.start == start_seconds
    assert window.waveform.end == end_seconds


def test_whole_track_restores_end_without_resetting_start(window: MainWindow) -> None:
    duration_seconds, start_seconds = 180, 30
    window.analyzer.duration_ready.emit(duration_seconds)
    window.waveform.selection_changed.emit(start_seconds, 75)

    window.full.setChecked(True)

    assert window.start.value() == start_seconds
    assert window.end.value() == duration_seconds
    assert window.waveform.end == duration_seconds


def test_late_metadata_does_not_overwrite_a_custom_numeric_range(window: MainWindow) -> None:
    start_seconds, end_seconds = 20, 50
    window.full.setChecked(False)
    window.start.setValue(start_seconds)
    window.end.setValue(end_seconds)

    window.analyzer.duration_ready.emit(180)

    assert window.start.value() == start_seconds
    assert window.end.value() == end_seconds
    assert window.waveform.start == start_seconds
    assert window.waveform.end == end_seconds


def test_new_input_resets_selection_to_the_whole_track(window: MainWindow) -> None:
    window.analyzer.duration_ready.emit(180)
    window.waveform.selection_changed.emit(20, 50)

    window.input_path.setText("another-track.wav")

    assert window.start.value() == 0
    assert window.full.isChecked()
    assert not window.end.isEnabled()
    assert window.audio_duration == 0


def test_missing_input_reports_inline_without_a_blocking_dialog(
    window: MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window.input_path.setText(str(tmp_path / "deleted.wav"))
    monkeypatch.setattr(
        "stem_local.gui.QMessageBox.warning",
        lambda *_args: pytest.fail("Missing input must not open a blocking dialog"),
    )

    window.run_button.click()

    assert window.active is None
    assert "файл" in window.status.text().casefold()
    assert window.process.state() == QProcess.ProcessState.NotRunning


@pytest.fixture
def sleeping_job(window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> Job:
    original = window.process.setArguments
    monkeypatch.setattr(
        window.process,
        "setArguments",
        lambda _arguments: original(["-c", "import time; time.sleep(60)"]),
    )
    return Job(arguments=[], title="Обработка · sleeping worker")


@pytest.mark.parametrize("action", ["run", "download", "start_job"])
def test_an_active_job_cannot_be_replaced_or_queued(
    window: MainWindow,
    app: QApplication,
    sleeping_job: Job,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    window.start_job(sleeping_job)
    assert window.process.waitForStarted(2000)
    pid = window.process.processId()
    status = window.status.text()
    monkeypatch.setattr("stem_local.gui.QMessageBox.question", lambda *_args: pytest.fail("Task is already running"))

    if action == "run":
        window.run_button.click()
    elif action == "download":
        window.download_button.click()
    else:
        window.start_job(Job(arguments=[], title="second task"))
    app.processEvents()

    assert window.active is sleeping_job
    assert window.process.processId() == pid
    assert window.task_label.text() == sleeping_job.title
    assert window.status.text() == status
    assert not window.run_button.isEnabled()
    assert not window.download_button.isEnabled()
    assert all(not control.isEnabled() for control in window.task_controls)
    assert not hasattr(window, "pending")


@pytest.mark.usefixtures("sleeping_job")
def test_download_also_blocks_removal_until_it_finishes(
    window: MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("stem_local.gui.QMessageBox.question", lambda *_args: QMessageBox.StandardButton.Yes)

    window.download_button.click()
    assert window.process.waitForStarted(2000)
    download = window.active
    window.run_button.click()

    assert download is not None
    assert download.arguments[0] == "download"
    assert window.active is download
    assert "Подготовка моделей" in window.task_label.text()
    assert not window.run_button.isEnabled()


def test_failed_process_start_restores_the_run_button(window: MainWindow, tmp_path: Path) -> None:
    window.process.setProgram(str(tmp_path / "missing-python"))

    window.start_job(Job(arguments=[], title="failed start"))
    assert not window.process.waitForStarted(2000)

    assert window.active is None
    assert window.run_button.isEnabled()
    assert not window.cancel_button.isEnabled()
    assert window.task_label.isHidden()
    assert all(control.isEnabled() for control in window.task_controls)
    assert "Ошибка" in window.status.text()


def test_progress_belongs_only_to_the_current_removal(
    window: MainWindow,
    app: QApplication,
    sleeping_job: Job,
) -> None:
    window.start_job(sleeping_job)
    assert window.process.waitForStarted(2000)
    window.handle_worker_event({"event": "status", "message": "Выделение вокала"})
    window.handle_worker_event({"event": "progress", "done": 1, "total": 4})
    assert window.progress.value() == window.progress.maximum() // 4

    window.handle_worker_event({"event": "status", "message": "Загрузка Demucs"})
    assert window.progress.maximum() == 0
    window.handle_worker_event({"event": "progress", "done": 3, "total": 4})
    assert window.progress.value() == window.progress.maximum() * 3 // 4
    window.handle_worker_event({"event": "status", "message": "Экспорт WAV"})
    assert window.status.text() == "Экспорт WAV"
    assert window.progress.maximum() == 0
    window.cancel_button.click()
    assert window.process.waitForFinished(3000)
    app.processEvents()
    assert window.progress.value() == 0
    window.handle_worker_event({"event": "progress", "done": 4, "total": 4})
    window.handle_worker_event({"event": "status", "message": "late status"})
    assert window.progress.value() == 0
    assert window.status.text() == "Отменено"

    window.start_job(sleeping_job)
    assert window.process.waitForStarted(2000)

    assert window.progress.value() == 0
    assert window.progress.maximum() == 0
    window.handle_worker_event({"event": "progress", "done": 1, "total": 10})
    assert window.progress.value() == window.progress.maximum() // 10
    assert window.task_label.text() == sleeping_job.title


def test_stalled_worker_is_stopped_and_does_not_open_results(
    window: MainWindow,
    app: QApplication,
    sleeping_job: Job,
    opened_directories: list[str],
) -> None:
    window.stall_timeout_seconds = 0.15
    window.health_timer.setInterval(20)
    window.start_job(sleeping_job)
    assert window.process.waitForStarted(2000)

    deadline = time.monotonic() + 8
    while window.active is not None and time.monotonic() < deadline:
        app.processEvents()
        window.process.waitForFinished(20)

    assert window.active is None
    assert window.task_failed
    assert "Таймаут" in window.last_error
    assert window.run_button.isEnabled()
    assert not opened_directories


def test_disappearing_file_stops_an_already_running_job(
    window: MainWindow,
    app: QApplication,
    sleeping_job: Job,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    source.touch()
    sleeping_job.watched_paths = (source,)
    window.start_job(sleeping_job)
    assert window.process.waitForStarted(2000)

    source.unlink()
    window.health_timer.timeout.emit()
    window.process.waitForFinished(3000)
    app.processEvents()

    assert window.task_failed
    assert "исчез файл" in window.last_error
    assert window.active is None


def test_close_stops_the_worker_and_log_window(
    window: MainWindow,
    sleeping_job: Job,
    app: QApplication,
) -> None:
    window.log_button.click()
    window.start_job(sleeping_job)
    assert window.process.waitForStarted(2000)

    window.close()
    app.processEvents()

    assert window.process.state() == QProcess.ProcessState.NotRunning
    assert not window.log_dialog.isVisible()
    assert not window.health_timer.isActive()


@pytest.fixture
def guarded_job(
    window: MainWindow,
    app: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[dict[int, float]]:
    ready = tmp_path / "family.json"
    child_code = (
        "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(60)"
    )
    code = (
        "import json,os,psutil,signal,subprocess,sys,time; from pathlib import Path; "
        "from stem_local.guardian import start_parent_guard; "
        "\ndef cancel(signum, frame): raise SystemExit(130)"
        "\nsignal.signal(signal.SIGTERM,cancel)"
        "\nparent = psutil.Process(os.getppid())"
        "\nstart_parent_guard(parent_pid=parent.pid,parent_created=parent.create_time())"
        "\nguardian = psutil.Process().children()[0]"
        f"\nchild = subprocess.Popen([sys.executable,'-c',{child_code!r}],stdout=subprocess.PIPE)"
        "\nchild.stdout.readline(); child.stdout.close()"
        "\nfamily = {p.pid:p.create_time() for p in (psutil.Process(),psutil.Process(child.pid),guardian)}"
        f"\nPath({str(ready)!r}).write_text(json.dumps(family))"
        "\ntime.sleep(60)"
    )
    original = window.process.setArguments
    monkeypatch.setattr(window.process, "setArguments", lambda _arguments: original(["-u", "-c", code]))
    family: dict[int, float] = {}
    try:
        window.start_job(Job(arguments=[], title="Обработка · guarded worker"))
        assert window.process.waitForStarted(2000)
        deadline = time.monotonic() + 5
        while not ready.is_file() and time.monotonic() < deadline:
            app.processEvents()
            window.process.waitForFinished(20)
        assert ready.is_file(), window.log.toPlainText()
        family = {int(pid): created for pid, created in json.loads(ready.read_text()).items()}

        yield family
    finally:
        window.close()
        deadline = time.monotonic() + 6
        while any(same_process(pid, created) for pid, created in family.items()) and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.025)
        for pid, created in family.items():
            if same_process(pid, created):
                with contextlib.suppress(psutil.NoSuchProcess):
                    psutil.Process(pid).kill()


@pytest.mark.parametrize("action", ["cancel", "timeout", "close", "kill"])
def test_guardian_cleans_the_family_without_gui_tree_traversal(
    window: MainWindow,
    app: QApplication,
    guarded_job: dict[int, float],
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    def forbid_tree_traversal(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Only the guardian may enumerate the worker's descendants")

    monkeypatch.setattr(psutil.Process, "children", forbid_tree_traversal)

    if action == "cancel":
        window.cancel_button.click()
    elif action == "timeout":
        window.last_activity = time.monotonic() - window.stall_timeout_seconds - 1
        window.health_timer.timeout.emit()
    elif action == "close":
        window.close()
    else:
        window.kill_timer.timeout.emit()
    deadline = time.monotonic() + 6
    while any(same_process(pid, created) for pid, created in guarded_job.items()) and time.monotonic() < deadline:
        app.processEvents()
        window.process.waitForFinished(20)
        time.sleep(0.025)
    app.processEvents()

    assert window.active is None
    assert all(not same_process(pid, created) for pid, created in guarded_job.items()), guarded_job


def test_removed_playback_controls_are_absent(window: MainWindow) -> None:
    assert not hasattr(window, "player")
    assert not hasattr(window, "listen_stem")
    assert not hasattr(window, "seek")


def test_numeric_range_controls_have_a_visible_text_area(window: MainWindow, app: QApplication) -> None:
    minimum_text_width = 40
    window.show()
    app.processEvents()

    assert window.start.lineEdit().width() > minimum_text_width
    assert window.end.lineEdit().width() > minimum_text_width


def test_job_with_a_deleted_source_never_starts(window: MainWindow, tmp_path: Path) -> None:
    job = Job(arguments=[], watched_paths=(tmp_path / "gone.wav",))

    window.start_job(job)

    assert window.active is None
    assert window.process.state() == QProcess.ProcessState.NotRunning
    assert window.run_button.isEnabled()
    assert "исчез" in window.status.text()


def test_closing_only_the_log_dialog_does_not_stop_processing(
    window: MainWindow,
    sleeping_job: Job,
    app: QApplication,
) -> None:
    window.show()
    window.start_job(sleeping_job)
    assert window.process.waitForStarted(2000)
    window.log_button.click()

    window.log_dialog.close()
    app.processEvents()

    assert window.active is sleeping_job
    assert window.process.state() == QProcess.ProcessState.Running


@pytest.mark.parametrize("whole_track", [True, False])
def test_run_snapshots_whole_track_or_custom_range(
    *,
    window: MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    whole_track: bool,
) -> None:
    source = tmp_path / "track.wav"
    source.touch()
    window.input_path.setText(str(source))
    window.analyzer.duration_ready.emit(180)
    if not whole_track:
        window.waveform.selection_changed.emit(30.125, 75.5)
    spec = window.model.currentData()
    directory = window.cache / spec
    directory.mkdir(parents=True)
    for artifact in (MODELS[spec].config, MODELS[spec].weights):
        (directory / artifact.filename).touch()
    started: list[Job] = []
    monkeypatch.setattr(window, "start_job", started.append)

    window.run_button.click()

    assert len(started) == 1
    arguments = started[0].arguments
    if whole_track:
        assert "--full" in arguments
        assert "--duration" not in arguments
        assert arguments[arguments.index("--start") + 1] == "0.0"
    else:
        assert "--full" not in arguments
        expected_duration = 75.5 - 30.125
        assert float(arguments[arguments.index("--duration") + 1]) == expected_duration


@pytest.fixture
def opened_directories(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    opened: list[str] = []

    def open_url(url: QUrl) -> bool:
        opened.append(url.toLocalFile())
        return True

    monkeypatch.setattr("stem_local.gui.QDesktopServices.openUrl", open_url)
    return opened


@pytest.fixture
def completion_event(tmp_path: Path) -> dict[str, str]:
    directory = tmp_path / "Музыка 🎵" / "specific-job-result"
    directory.mkdir(parents=True)
    report_path = directory / "report.json"
    report = {
        "status": "completed",
        "input": str(tmp_path / "source.wav"),
        "outputs": {"vocals": str(directory / "vocals.wav")},
        "timing": {"wall_seconds": 1},
        "memory": {"sampled_process_tree_peak_rss_bytes": 1024},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return {"event": "completed", "report": str(report_path)}


def test_success_opens_exact_job_directory_only_after_worker_exits(
    window: MainWindow,
    completion_event: dict[str, str],
    opened_directories: list[str],
) -> None:
    window.handle_worker_event(completion_event)
    assert not opened_directories
    assert not window.open_result_button.isEnabled()

    window.process.finished.emit(0, QProcess.ExitStatus.NormalExit)

    expected = str(Path(completion_event["report"]).parent)
    assert opened_directories == [expected]
    assert window.current_report_path == Path(completion_event["report"])
    assert window.open_result_button.isEnabled()
    assert window.run_button.isEnabled()
    assert window.active is None
    assert window.task_label.isHidden()
    assert window.progress.value() == window.progress.maximum()


def test_worker_unicode_completion_survives_legacy_encoding_in_the_real_qprocess(
    *,
    window: MainWindow,
    completion_event: dict[str, str],
    opened_directories: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = QProcessEnvironment.systemEnvironment()
    environment.insert("PYTHONIOENCODING", "cp1251")
    window.process.setProcessEnvironment(environment)
    code = f"from stem_local.worker import emit; emit({completion_event!r})"
    original = window.process.setArguments
    monkeypatch.setattr(window.process, "setArguments", lambda _arguments: original(["-u", "-c", code]))

    window.start_job(Job(arguments=[], title="Песня 🎵"))
    assert window.process.waitForFinished(5000)

    assert window.last_error == ""
    assert window.current_report_path == Path(completion_event["report"])
    assert opened_directories == [str(Path(completion_event["report"]).parent)]
    assert window.open_result_button.isEnabled()


@pytest.mark.parametrize(("exit_code", "cancelled"), [(1, False), (0, True)])
def test_failed_or_cancelled_job_does_not_open_directory(
    *,
    window: MainWindow,
    completion_event: dict[str, str],
    opened_directories: list[str],
    exit_code: int,
    cancelled: bool,
) -> None:
    window.handle_worker_event(completion_event)
    window.was_cancelled = cancelled

    window.process.finished.emit(exit_code, QProcess.ExitStatus.NormalExit)

    assert not opened_directories
    assert not window.open_result_button.isEnabled()
    assert window.run_button.isEnabled()
    assert window.progress.value() == 0


def test_download_does_not_open_result_directory(window: MainWindow, opened_directories: list[str]) -> None:
    window.handle_worker_event({"event": "ready", "model": "kimberley"})

    window.process.finished.emit(0, QProcess.ExitStatus.NormalExit)

    assert not opened_directories


def test_license_warning_is_model_specific(window: MainWindow) -> None:
    window.model.setCurrentIndex(1)

    assert not window.accept_license.isHidden()

    window.model.setCurrentIndex(0)

    assert window.accept_license.isHidden()


def test_existing_model_files_keep_the_check_and_repair_action_available(window: MainWindow) -> None:
    spec = MODELS[window.model.currentData()]
    directory = window.cache / spec.id
    directory.mkdir(parents=True)
    for artifact in (spec.config, spec.weights):
        (directory / artifact.filename).write_bytes(b"corrupt")

    window.mode.setCurrentIndex(1)
    window.mode.setCurrentIndex(0)

    assert not window.download_button.isHidden()
    assert window.download_button.text() == "Проверить модели"


@pytest.mark.parametrize("confirm", [True, False])
def test_check_and_repair_requires_confirmation_before_starting_a_worker(
    *,
    window: MainWindow,
    monkeypatch: pytest.MonkeyPatch,
    confirm: bool,
) -> None:
    started: list[Job] = []
    monkeypatch.setattr(window, "start_job", started.append)
    answer = QMessageBox.StandardButton.Yes if confirm else QMessageBox.StandardButton.No
    monkeypatch.setattr("stem_local.gui.QMessageBox.question", lambda *_args: answer)

    window.download_button.click()

    assert len(started) == int(confirm)
    if confirm:
        assert started[0].arguments[0] == "download"
        assert "--repair" in started[0].arguments


def test_full_inference_progress_is_not_success_before_export(window: MainWindow) -> None:
    window.handle_worker_event({"event": "progress", "done": 1, "total": 1})

    assert 0 < window.progress.value() < window.progress.maximum()
    assert not window.open_result_button.isEnabled()


def test_inner_progress_refreshes_the_stall_deadline(
    window: MainWindow,
    sleeping_job: Job,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window.start_job(sleeping_job)
    assert window.process.waitForStarted(2000)
    now = time.monotonic()
    monkeypatch.setattr("stem_local.gui.time.monotonic", lambda: now)
    window.last_activity = now - window.stall_timeout_seconds - 1

    window.handle_worker_event({"event": "progress", "done": 1, "total": 10})
    window.health_timer.timeout.emit()

    assert not window.task_failed
    assert window.process.state() == QProcess.ProcessState.Running


def test_cancel_stops_worker_and_allows_a_new_run(
    window: MainWindow,
    app: QApplication,
    sleeping_job: Job,
) -> None:
    window.start_job(sleeping_job)
    assert window.process.waitForStarted(3000)

    window.cancel_button.click()
    deadline = time.monotonic() + 8
    while window.active is not None and time.monotonic() < deadline:
        app.processEvents()
        window.process.waitForFinished(50)

    assert window.active is None
    assert window.run_button.isEnabled()
    assert "Отменено" in window.status.text()
    assert window.task_label.isHidden()

    window.start_job(sleeping_job)
    assert window.process.waitForStarted(3000)
    assert window.active is sleeping_job
    assert not window.run_button.isEnabled()
