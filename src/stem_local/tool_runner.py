"""Run one preview tool under the same parent-death containment as model workers."""

import argparse
import signal
import subprocess
import sys
from types import FrameType

from stem_local.guardian import start_parent_guard


def _stop(_signum: int, _frame: FrameType | None) -> None:
    message = "Preview cancelled"
    raise InterruptedError(message)


def main() -> int:
    """Forward stdout/stderr unchanged so FFprobe/FFmpeg remain streaming tools."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--parent-created", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Expected a tool command")
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        start_parent_guard(parent_pid=args.parent_pid, parent_created=args.parent_created)
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL)  # noqa: S603 - GUI builds argv; no shell.
        try:
            return child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=1)
    except InterruptedError:
        return 130


if __name__ == "__main__":
    sys.exit(main())
