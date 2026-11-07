"""Worker CLI, reports and sequential model-lifetime checks."""

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Never

import numpy as np
import pytest

from stem_local.models import ModelSpec
from stem_local.worker import CancelledError, MemoryMonitor, parser, run_separation, write_json


def test_whole_track_is_default_and_preview_is_explicit() -> None:
    args = parser().parse_args(["separate", "track.wav"])

    assert args.duration is None
    assert args.start == 0
    assert not args.full
    assert args.model == "kimberley"
    assert args.threads >= 1


def test_cli_can_request_custom_excerpt() -> None:
    start_seconds = 30.125
    duration_seconds = 45.375
    args = parser().parse_args(
        ["separate", "track.wav", "--start", str(start_seconds), "--duration", str(duration_seconds)],
    )

    assert args.start == start_seconds
    assert args.duration == duration_seconds


def test_missing_source_fails_before_importing_torch(tmp_path: Path) -> None:
    source = tmp_path / "disappeared.wav"
    code = (
        "import sys; from stem_local.worker import parser, run_separation; "
        f"args = parser().parse_args(['separate', {str(source)!r}]); "
        "\ntry: run_separation(args)"
        "\nexcept FileNotFoundError: pass"
        "\nelse: raise AssertionError('Missing source was accepted')"
        "\nassert 'torch' not in sys.modules, 'Heavy model runtime was imported before input validation'"
    )

    result = subprocess.run(  # noqa: S603 - literal test program under the current interpreter; no shell.
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("encoding", ["cp1251", "ascii"])
def test_worker_events_preserve_unicode_with_a_legacy_stdio_encoding(tmp_path: Path, encoding: str) -> None:
    source = tmp_path / "Музыка 🎵" / "песня.wav"
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = encoding

    result = subprocess.run(  # noqa: S603 - project worker and a nonexistent test input, no shell.
        [sys.executable, "-m", "stem_local.worker", "separate", str(source)],
        env=environment,
        capture_output=True,
        check=False,
        timeout=15,
    )

    events = [json.loads(line.removeprefix("STEM_LOCAL ")) for line in result.stdout.decode("utf-8").splitlines()]
    assert result.returncode == 1
    assert events[-1]["event"] == "error"
    assert str(source) in events[-1]["message"]
    assert str(source) in result.stderr.decode("utf-8")


def test_report_write_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "report.json"

    write_json(path, {"status": "completed"})

    assert json.loads(path.read_text()) == {"status": "completed"}
    assert not path.with_suffix(".json.part").exists()


def test_inference_network_guard_rejects_connect() -> None:
    code = (
        "from stem_local.worker import block_network; import socket; block_network(); "
        "socket.create_connection(('127.0.0.1', 9))"
    )

    result = subprocess.run(  # noqa: S603 - literal test program under the current interpreter; no shell.
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Network is disabled" in result.stderr


def test_memory_monitor_records_worker_rss() -> None:
    with MemoryMonitor() as memory:
        memory.thread.join(timeout=0.15)

    assert memory.measurements()["sampled_process_tree_peak_rss_bytes"] > 0


@pytest.mark.parametrize(("mode", "expected"), [("vocals", {"vocals"}), ("vocals-melody", {"vocals", "melody"})])
def test_pipeline_releases_first_model_before_loading_second(
    mode: str,
    expected: set[str],
    audio_factory: Callable[..., tuple[Path, np.ndarray]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _ = audio_factory(frames=44100, sample_rate=44100)
    alive: list[str] = []
    events: list[dict[str, object]] = []

    class FakeVocalEngine:
        def __init__(self, **_kwargs: object) -> None:
            assert not alive
            alive.append("vocals")

        def separate(self, audio: np.ndarray, *, progress: Callable[[int, int], None]) -> np.ndarray:
            progress(1, 2)
            return audio * 0.5

        def audit(self) -> dict[str, object]:
            return {"device": "cpu", "verified_forward_calls": 1}

        def __del__(self) -> None:
            alive.remove("vocals")

    class FakeMelodyEngine:
        def __init__(self, **_kwargs: object) -> None:
            assert not alive, "Both models were alive at the same time"

        def separate(self, audio: np.ndarray, *, progress: Callable[[int, int], None]) -> np.ndarray:
            progress(1, 2)
            return audio * 0.2

        def audit(self) -> dict[str, object]:
            return {"device": "cpu", "verified_forward_calls": 1}

    def factory(**kwargs: object) -> object:
        spec = kwargs["spec"]
        assert isinstance(spec, ModelSpec)
        engine_type = FakeVocalEngine if spec.role == "vocals" else FakeMelodyEngine
        return engine_type(**kwargs)

    monkeypatch.setattr("stem_local.engine.create_engine", factory)
    monkeypatch.setattr("stem_local.worker.block_network", lambda: None)
    monkeypatch.setattr("stem_local.worker.emit", events.append)
    args = parser().parse_args(
        [
            "separate",
            str(source),
            "--mode",
            mode,
            "--output",
            str(tmp_path / "results"),
            "--duration",
            "1",
        ],
    )

    run_separation(args)

    report_path = next((tmp_path / "results").glob("*/report.json"))
    report = json.loads(report_path.read_text())
    assert set(report["outputs"]).difference({"shared_gain"}) == expected
    assert {path.stem for path in report_path.parent.glob("*.wav")} == expected
    assert not list(report_path.parent.glob(".work-*"))
    assert not alive
    progress = [event for event in events if event["event"] == "progress"]
    assert len(progress) == 2 * len(expected)
    assert progress[-1]["done"] == progress[-1]["total"]
    fractions = []
    for event in progress:
        done, total = event["done"], event["total"]
        assert isinstance(done, int)
        assert isinstance(total, int)
        fractions.append(done / total)
    assert fractions == sorted(fractions)
    assert fractions[0] == 0.5 / len(expected)
    assert set(report["cpu_audit"]) == expected


@pytest.mark.parametrize(("failure", "status"), [(RuntimeError, "failed"), (CancelledError, "cancelled")])
def test_pipeline_failure_cleans_temporary_files_and_does_not_publish_success(
    failure: type[Exception],
    status: str,
    audio_factory: Callable[..., tuple[Path, np.ndarray]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _ = audio_factory(frames=44100, sample_rate=44100)
    output = tmp_path / "results"
    events: list[dict[str, object]] = []

    def fail_loading(**_kwargs: object) -> Never:
        message = "Test model failed"
        raise failure(message)

    monkeypatch.setattr("stem_local.engine.create_engine", fail_loading)
    monkeypatch.setattr("stem_local.worker.block_network", lambda: None)
    monkeypatch.setattr("stem_local.worker.emit", events.append)
    args = parser().parse_args(["separate", str(source), "--output", str(output)])

    with pytest.raises(failure, match="Test model failed"):
        run_separation(args)

    run_dir = next(output.iterdir())
    assert json.loads((run_dir / "status.json").read_text())["status"] == status
    assert not (run_dir / "report.json").exists()
    assert not list(run_dir.glob(".work-*"))
    assert not list(run_dir.glob("*.wav"))
    assert not any(event["event"] == "completed" for event in events)
