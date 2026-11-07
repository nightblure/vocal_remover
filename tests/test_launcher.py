"""Manual-dependency launcher checks, without downloads or model loading."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QLockFile

from stem_local import launcher

RUN_SUBPROCESS = subprocess.run


@pytest.fixture(autouse=True)
def installed_dependencies(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    monkeypatch.setattr(launcher.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(launcher.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(launcher.platform, "mac_ver", lambda: ("14.0", (), ""))
    monkeypatch.setattr(launcher.sys, "version_info", (3, 11, 15))
    monkeypatch.setattr(launcher.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(launcher.shutil, "which", lambda name: f"/tools/{name}")
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="libmp3lame", stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", run)
    return calls


@pytest.mark.parametrize(("system", "machine"), [("Darwin", "arm64"), ("Windows", "AMD64")])
def test_supported_platforms(*, system: str, machine: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher.platform, "system", lambda: system)
    monkeypatch.setattr(launcher.platform, "machine", lambda: machine)

    launcher.check_dependencies()


@pytest.mark.parametrize(("system", "machine"), [("Linux", "x86_64"), ("Darwin", "x86_64"), ("Windows", "ARM64")])
def test_unsupported_platforms(*, system: str, machine: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher.platform, "system", lambda: system)
    monkeypatch.setattr(launcher.platform, "machine", lambda: machine)

    with pytest.raises(RuntimeError, match="Supported platforms"):
        launcher.check_dependencies()


def test_old_macos_reports_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher.platform, "mac_ver", lambda: ("13.7", (), ""))

    with pytest.raises(RuntimeError, match="macOS 14"):
        launcher.check_dependencies()


def test_wrong_python_reports_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher.sys, "version_info", (3, 12, 0))

    with pytest.raises(RuntimeError, match=r"Python 3\.11"):
        launcher.check_dependencies()


def test_missing_packages_report_manual_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher.importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(RuntimeError, match=r"Missing Python packages:.*PySide6.*uv sync --frozen"):
        launcher.check_dependencies()


@pytest.mark.parametrize("missing", ["ffmpeg", "ffprobe"])
def test_missing_audio_tool_reports_path_advice(missing: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None if name == missing else f"/tools/{name}")

    with pytest.raises(RuntimeError, match=rf"add its bin folder to PATH\. Missing: {missing}"):
        launcher.check_dependencies()


def test_missing_mp3_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, stdout="no encoder"),
    )

    with pytest.raises(RuntimeError, match="libmp3lame"):
        launcher.check_dependencies()


def test_check_only_runs_audio_checks_and_does_not_create_runtime(
    *,
    tmp_path: Path,
    installed_dependencies: list[list[str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = launcher.main(["--root", str(tmp_path), "--check"])

    assert result == 0
    assert installed_dependencies == [
        ["/tools/ffmpeg", "-hide_banner", "-encoders"],
        ["/tools/ffprobe", "-hide_banner", "-version"],
    ]
    assert not (tmp_path / ".runtime").exists()
    assert "Nothing was installed or downloaded" in capsys.readouterr().out


@pytest.mark.parametrize("failure", [OSError("bad binary"), subprocess.TimeoutExpired("ffprobe", 30)])
def test_broken_tools_show_instructions(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: Exception,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(launcher.subprocess, "run", fail)

    result = launcher.main(["--root", str(tmp_path), "--check"])

    assert result == 1
    assert "docs/launch.md" in capsys.readouterr().err


def test_second_instance_does_not_open_gui(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    lock = QLockFile(str(runtime / "application.lock"))
    assert lock.tryLock(0)

    try:
        result = launcher.main(["--root", str(tmp_path)])
    finally:
        lock.unlock()

    assert result == 1
    assert "already running" in capsys.readouterr().err


def test_gui_receives_project_paths_and_user_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[list[str]] = []
    monkeypatch.setattr(sys, "argv", ["test"])
    path = os.environ.get("PATH")

    def run_gui() -> int:
        received.append(sys.argv.copy())
        return 0

    monkeypatch.setattr("stem_local.gui.main", run_gui)

    result = launcher.main(["--root", str(tmp_path), "--stall-timeout", "1800"])

    assert result == 0
    assert received == [
        [
            "stem-local",
            "--cache",
            str(tmp_path / "models"),
            "--output",
            str(tmp_path / "outputs"),
            "--stall-timeout",
            "1800",
        ]
    ]
    assert os.environ.get("PATH") == path
    lock = QLockFile(str(tmp_path / ".runtime" / "application.lock"))
    assert lock.tryLock(0)
    lock.unlock()


@pytest.mark.skipif(sys.platform == "win32", reason="macOS shell entry point")
@pytest.mark.parametrize("exit_code", [0, 1])
def test_mac_script_handles_spaces_and_forwards_exit_code(
    tmp_path: Path,
    exit_code: int,
) -> None:
    root = tmp_path / "project with spaces"
    binary_dir = root / ".venv" / "bin"
    binary_dir.mkdir(parents=True)
    project = Path(__file__).resolve().parents[1]
    shutil.copy(project / "Start.command", root)
    python = binary_dir / "python"
    python.write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > args\nexit {exit_code}\n')
    python.chmod(0o700)

    result = RUN_SUBPROCESS(
        ["/bin/bash", str(root / "Start.command"), "--check"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

    assert result.returncode == exit_code
    assert (root / "args").read_text().splitlines() == ["src/stem_local/launcher.py", "--root", str(root), "--check"]
    assert not (root / ".runtime").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="macOS shell entry point")
def test_mac_script_without_venv_only_prints_instructions(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    shutil.copy(project / "Start.command", tmp_path)

    result = RUN_SUBPROCESS(
        ["/bin/bash", str(tmp_path / "Start.command")],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

    assert result.returncode == 1
    assert "uv sync --frozen" in result.stdout
    assert not (tmp_path / ".venv").exists()
    assert not (tmp_path / ".runtime").exists()


def test_launcher_without_site_packages_prints_manual_installation_advice(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]

    result = RUN_SUBPROCESS(
        [sys.executable, "-S", str(project / "src" / "stem_local" / "launcher.py"), "--root", str(tmp_path), "--check"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

    assert result.returncode == 1
    assert "Missing Python packages:" in result.stderr
    assert "uv sync --frozen" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / ".runtime").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Requires Windows cmd.exe")
def test_windows_script_without_venv_only_prints_instructions(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    shutil.copy(project / "Start.cmd", tmp_path)

    result = RUN_SUBPROCESS(
        [os.environ["COMSPEC"], "/d", "/c", "Start.cmd"],
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

    assert result.returncode == 1
    assert "uv sync --frozen" in result.stdout
    assert not (tmp_path / ".venv").exists()
    assert not (tmp_path / ".runtime").exists()
