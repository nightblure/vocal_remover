"""Check manually installed dependencies, then run the application; never install anything."""

import argparse
import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path

MIN_MACOS_VERSION = 14


def check_dependencies() -> None:
    """Fail before importing Qt or the audio runtime, with actionable installation advice."""
    supported = {("Darwin", "arm64"), ("Windows", "amd64"), ("Windows", "x86_64")}
    if (platform.system(), platform.machine().lower()) not in supported:
        message = "Supported platforms: macOS Apple Silicon and Windows x86-64. Linux is not supported."
        raise RuntimeError(message)
    if platform.system() == "Darwin" and int(platform.mac_ver()[0].split(".")[0]) < MIN_MACOS_VERSION:
        message = "Install macOS 14 or newer for the pinned PyTorch version."
        raise RuntimeError(message)
    if sys.version_info[:2] != (3, 11):
        message = "Install Python 3.11 and recreate .venv with: uv sync --frozen --python 3.11"
        raise RuntimeError(message)

    modules = ("stem_local", "PySide6", "audio_separator", "torch", "numpy", "soundfile", "psutil")
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        message = f"Missing Python packages: {', '.join(missing)}. Run in the project folder: uv sync --frozen"
        raise RuntimeError(message)

    tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    missing = [name for name, path in tools.items() if path is None]
    if missing:
        message = f"Install FFmpeg with libmp3lame and add its bin folder to PATH. Missing: {', '.join(missing)}."
        raise RuntimeError(message)
    for name, path in tools.items():
        result = subprocess.run(  # noqa: S603 - PATH-resolved executable, fixed flags, no shell.
            [str(path), "-hide_banner", "-encoders" if name == "ffmpeg" else "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=30,
        )
        if name == "ffmpeg" and "libmp3lame" not in result.stdout:
            message = "Install an FFmpeg build with the libmp3lame MP3 encoder."
            raise RuntimeError(message)


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stem Local launcher (manual dependency installation)")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Check dependencies without opening the GUI")
    args, gui_arguments = parser.parse_known_args(arguments)
    root = args.root.resolve()
    try:
        check_dependencies()
        from stem_local.process_io import configure_utf8  # noqa: PLC0415 - project may not be installed yet.

        configure_utf8()
        if args.check:
            sys.stdout.write("Dependencies are ready. Nothing was installed or downloaded.\n")
            return 0

        from PySide6.QtCore import QLockFile  # noqa: PLC0415 - check dependencies before importing Qt.

        runtime = root / ".runtime"
        runtime.mkdir(exist_ok=True)
        lock = QLockFile(str(runtime / "application.lock"))
        lock.setStaleLockTime(0)
        if not lock.tryLock(0):
            sys.stderr.write("Stem Local is already running in this folder.\n")
            return 1
        try:
            from stem_local.gui import main as run_gui  # noqa: PLC0415 - check dependencies before importing GUI.

            sys.argv = [
                "stem-local",
                "--cache",
                str(root / "models"),
                "--output",
                str(root / "outputs"),
                *gui_arguments,
            ]
            return run_gui()
        finally:
            lock.unlock()
    except (ImportError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        sys.stderr.write(f"Cannot start Stem Local: {error}\nSee docs/launch.md for installation instructions.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
