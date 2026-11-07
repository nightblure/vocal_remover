"""One isolated process per download or separation. JSON events on stdout."""

import argparse
import contextlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from types import FrameType
from typing import NotRequired, Self, TypedDict

import psutil

if os.name != "nt":
    import resource

from stem_local.guardian import start_parent_guard
from stem_local.models import DEFAULT_CACHE, MODELS, check_license, prepare_model

EVENT_PREFIX = "STEM_LOCAL "


class CancelledError(Exception):
    pass


def emit(event: dict[str, object]) -> None:
    """Write one machine-readable event without buffering progress updates."""
    sys.stdout.write(f"{EVENT_PREFIX}{json.dumps(event, ensure_ascii=False)}\n")
    sys.stdout.flush()


def block_network() -> None:
    """Reject network operations inside the local inference worker."""

    def audit(event: str, _arguments: tuple[object, ...]) -> None:
        if event in {"socket.connect", "socket.sendto", "socket.getaddrinfo"}:
            message = "Network is disabled during local inference"
            raise PermissionError(message)

    sys.addaudithook(audit)


def cancel(_signum: int, _frame: FrameType | None) -> None:
    """Translate a termination signal into cooperative task cancellation."""
    message = "Операция отменена"
    raise CancelledError(message)


class MemoryMeasurements(TypedDict):
    sampled_process_tree_peak_rss_bytes: int
    sampling_interval_seconds: float
    scope: str
    worker_os_peak_rss_bytes: NotRequired[int]


class MemoryMonitor:
    def __init__(self) -> None:
        self.peak_bytes = 0
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        process = psutil.Process()
        while not self.stopped.is_set():
            total = 0
            for child in [process, *process.children(recursive=True)]:
                with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                    total += child.memory_info().rss
            self.peak_bytes = max(self.peak_bytes, total)
            self.stopped.wait(0.1)

    def __enter__(self) -> Self:
        """Begin sampling the owned process tree."""
        self.thread.start()
        return self

    def __exit__(self, *arguments: object) -> None:
        """Stop and join the sampler before reporting measurements."""
        self.stopped.set()
        self.thread.join(timeout=1)

    def measurements(self) -> MemoryMeasurements:
        result: MemoryMeasurements = {
            "sampled_process_tree_peak_rss_bytes": self.peak_bytes,
            "sampling_interval_seconds": 0.1,
            "scope": "worker plus children; GUI and system memory excluded",
        }
        if os.name != "nt":
            multiplier = 1 if sys.platform == "darwin" else 1024
            result["worker_os_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier
        return result


def write_json(path: Path, data: dict[str, object]) -> None:
    """Publish a JSON report atomically within its destination directory."""
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_separation(args: argparse.Namespace) -> None:
    """Validate a task, execute the disk-backed pipeline and publish its report."""
    if not args.input.is_file():
        message = f"Входной файл отсутствует: {args.input}"
        raise FileNotFoundError(message)
    if shutil.which("ffmpeg") is None:
        message = "FFmpeg отсутствует в PATH. Установите FFmpeg перед запуском."
        raise RuntimeError(message)
    spec = MODELS[args.model]
    if spec.role != "vocals":
        message = "Для вокала выберите Kimberley или ViperX"
        raise ValueError(message)
    check_license(spec, accept_unverified=args.accept_unverified_license)
    block_network()
    started = time.perf_counter()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.output.resolve() / f"{args.input.stem[:48]}-{args.model}-{stamp}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "status.json", {"status": "running"})
    emit({"event": "run", "directory": str(run_dir)})

    try:
        with (
            MemoryMonitor() as memory,
            tempfile.TemporaryDirectory(prefix=".work-", dir=run_dir) as temporary,
        ):
            from stem_local.pipeline import run_pipeline  # noqa: PLC0415 - validate before importing audio runtime.

            result = run_pipeline(args=args, work=Path(temporary), emit=emit)

        audio_seconds = result.audio_seconds
        melody_seconds = result.melody.inference_seconds if result.melody else 0.0
        load_seconds = result.vocals.load_seconds + (result.melody.load_seconds if result.melody else 0.0)
        audit = {"vocals": result.vocals.audit}
        if result.melody is not None:
            audit["melody"] = result.melody.audit

        report: dict[str, object] = {
            "status": "completed",
            "input": str(args.input.resolve()),
            "audio_seconds": audio_seconds,
            "start_seconds": args.start,
            "model": spec.provenance(),
            "melody_model": MODELS["htdemucs"].provenance() if result.melody else None,
            "settings": {
                "threads": args.threads,
                "mode": args.mode,
                "keep_bass": args.keep_bass,
                "overlap_count": args.overlap,
                "model_segment_frames": 801,
                "model_segment_seconds": 8,
                "outer_window_seconds": 20,
                "outer_crossfade_seconds": 2,
                "sample_rate": 44100,
                "channels": 2,
                "export_format": args.format,
                "export_bits": 24 if args.format == "WAV" else None,
                "mp3_bitrate_kbps": 320 if args.format == "MP3" else None,
                "demucs_shifts": 0,
                "demucs_overlap": 0.25,
                "network": "blocked in Python; FFmpeg protocols restricted to file,pipe",
            },
            "timing": {
                "load_and_verify_seconds": load_seconds,
                "inference_seconds": result.vocals.inference_seconds,
                "melody_inference_seconds": melody_seconds,
                "wall_seconds": time.perf_counter() - started,
                "inference_realtime_factor": (result.vocals.inference_seconds + melody_seconds) / audio_seconds,
                "scope": "wall includes imports, decode, hash/load, inference and export; excludes download",
            },
            "memory": memory.measurements(),
            "cpu_audit": audit,
            "outputs": result.outputs,
            "environment": {
                "python": platform.python_version(),
                "processor": platform.processor(),
                "logical_cpus": os.cpu_count(),
                "physical_cpus": psutil.cpu_count(logical=False),
                "system_ram_bytes": psutil.virtual_memory().total,
                "packages": {
                    package: importlib.metadata.version(package)
                    for package in ("stem-local", "audio-separator", "torch", "numpy", "soundfile")
                },
            },
        }
        write_json(run_dir / "report.json", report)
        write_json(run_dir / "status.json", {"status": "completed"})
        emit({"event": "completed", "report": str(run_dir / "report.json"), "outputs": result.outputs})
    except BaseException as error:
        with contextlib.suppress(OSError):
            write_json(
                run_dir / "status.json",
                {"status": "cancelled" if isinstance(error, CancelledError) else "failed", "error": str(error)},
            )
        raise


def parser() -> argparse.ArgumentParser:
    """Build the worker CLI shared by downloads and separation tasks."""
    result = argparse.ArgumentParser(description="Stem Local: CPU-only offline vocal separation")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("download", "separate"):
        command = commands.add_parser(name)
        command.add_argument("--model", choices=MODELS, default="kimberley")
        command.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
        command.add_argument("--accept-unverified-license", action="store_true")
        command.add_argument("--mode", choices=("vocals", "vocals-melody"), default="vocals")
        command.add_argument("--parent-pid", type=int, default=None, help=argparse.SUPPRESS)
        command.add_argument("--parent-created", type=float, default=None, help=argparse.SUPPRESS)
        if name == "separate":
            command.add_argument("input", type=Path)
            command.add_argument("--output", type=Path, default=Path("outputs"))
            command.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
            command.add_argument("--overlap", type=int, choices=(2, 4, 8), default=2)
            command.add_argument("--format", choices=("WAV", "MP3"), default="WAV")
            command.add_argument("--keep-bass", action="store_true")
            command.add_argument("--start", type=float, default=0)
            command.add_argument("--duration", type=float, default=None, help="Optional excerpt length in seconds")
            command.add_argument(
                "--full",
                action="store_true",
                help="Process to EOF (default when duration is omitted)",
            )
    return result


def _validate_options(args: argparse.Namespace) -> None:
    if args.threads < 1 or args.start < 0 or (args.duration is not None and args.duration <= 0):
        message = "Потоки и длительность должны быть > 0, начало — >= 0."
        raise ValueError(message)


def main() -> int:
    """Run one guarded CLI task and report failures over the event stream."""
    args = parser().parse_args()
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        start_parent_guard(parent_pid=args.parent_pid, parent_created=args.parent_created)
        if args.command == "download":
            prepare_model(
                spec=MODELS[args.model],
                cache=args.cache,
                emit=emit,
                accept_unverified=args.accept_unverified_license,
            )
            if args.mode == "vocals-melody" and args.model != "htdemucs":
                prepare_model(spec=MODELS["htdemucs"], cache=args.cache, emit=emit, accept_unverified=False)
        else:
            _validate_options(args)
            run_separation(args)
    except CancelledError:
        emit({"event": "cancelled"})
        return 130
    except Exception as error:  # noqa: BLE001 - CLI boundary reports any task failure, then exits nonzero.
        traceback.print_exc(file=sys.stderr)
        emit({"event": "error", "message": str(error)})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
