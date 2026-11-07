# Copyright (c) 2026
"""Exercise worker exits with real child processes, without loading torch."""

import contextlib
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import psutil
import pytest

from stem_local.guardian import same_process


def wait_until(condition: Callable[[], bool], timeout: float = 6) -> bool:
    """Bound every process test so a regression cannot hang the suite."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.025)
    return condition()


def stop_worker(*, worker: subprocess.Popen[bytes], finish: Path, victim: str) -> None:
    """Trigger the requested worker exit path with bounded waits."""
    if victim == "worker":
        worker.kill()
    elif victim == "cancel":
        worker.terminate()
    else:
        finish.touch()
    worker.wait(timeout=3)


@pytest.mark.parametrize("victim", ["gui", "worker", "normal", "exception", "cancel"])
def test_guardian_cleans_worker_and_its_child_after_exit(tmp_path: Path, victim: str) -> None:
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    parent_created = psutil.Process(parent.pid).create_time()
    ready = tmp_path / "ready.json"
    finish = tmp_path / "finish"
    child_code = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    code = (
        "import json,psutil,subprocess,sys,time; from pathlib import Path; "
        "from stem_local.guardian import start_parent_guard; "
        "\ndef cancel(signum, frame): raise InterruptedError('cancelled')"
        "\nimport signal; signal.signal(signal.SIGTERM, cancel)"
        f"\nstart_parent_guard(parent_pid={parent.pid}, parent_created={parent_created!r})"
        "\nguardian_pid = psutil.Process().children()[0].pid"
        f"\nchild = subprocess.Popen([sys.executable, '-c', {child_code!r}])"
        f"\nPath({str(ready)!r}).write_text(json.dumps({{'child':child.pid,'guardian':guardian_pid}}))"
        f"\nwhile not Path({str(finish)!r}).exists(): time.sleep(0.025)"
        f"\nif {victim == 'exception'!r}: raise RuntimeError('failed')"
    )
    worker = subprocess.Popen(  # noqa: S603 - literal test program under the current interpreter; no shell.
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    members: dict[int, float] = {}
    try:
        assert wait_until(ready.is_file, timeout=5), "Guardian/worker startup did not finish"
        identifiers = json.loads(ready.read_text())
        for pid in (worker.pid, identifiers["child"], identifiers["guardian"]):
            members[pid] = psutil.Process(pid).create_time()

        if victim == "gui":
            parent.kill()
            parent.wait(timeout=2)
        else:
            stop_worker(worker=worker, finish=finish, victim=victim)

        assert wait_until(lambda: all(not same_process(pid, created) for pid, created in members.items())), members
        if victim != "gui":
            assert parent.poll() is None
    finally:
        for process in (worker, parent):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)
        for pid, created in members.items():
            if same_process(pid, created):
                with contextlib.suppress(psutil.NoSuchProcess):
                    psutil.Process(pid).kill()
        if worker.stderr is not None:
            worker.stderr.close()
