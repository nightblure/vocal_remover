"""Own one worker's process family through normal exits and GUI/native-runtime crashes."""

import argparse
import contextlib
import ctypes
import os
import signal
import subprocess
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass

import psutil

from stem_local.process_io import configure_utf8


@dataclass
class _ProcessResources:
    windows_job: object | None = None
    guardian: subprocess.Popen[bytes] | None = None


# Retain OS handles for the process lifetime, not the task's try/finally block.
_resources = _ProcessResources()


def same_process(pid: int, created: float) -> bool:
    """Check process identity as well as PID, avoiding accidental PID-reuse kills."""
    try:
        process = psutil.Process(pid)
        return process.create_time() == created and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _windows_containment() -> None:
    """Keep the Windows Job handle in the worker; its death closes the entire job."""
    if sys.platform != "win32":
        message = "Windows Job Objects are only available on Windows"
        raise OSError(message)

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("process_time", ctypes.c_longlong),
            ("job_time", ctypes.c_longlong),
            ("flags", wintypes.DWORD),
            ("min_working", ctypes.c_size_t),
            ("max_working", ctypes.c_size_t),
            ("active_limit", wintypes.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority", wintypes.DWORD),
            ("scheduling", wintypes.DWORD),
        ]

    class IOCounters(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in (
                "read_count",
                "write_count",
                "other_count",
                "read_bytes",
                "write_bytes",
                "other_bytes",
            )
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("basic", BasicLimits),
            ("io", IOCounters),
            ("process_memory", ctypes.c_size_t),
            ("job_memory", ctypes.c_size_t),
            ("peak_process_memory", ctypes.c_size_t),
            ("peak_job_memory", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = ExtendedLimits()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        kernel.CloseHandle(handle)
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()):
        kernel.CloseHandle(handle)
        raise ctypes.WinError(ctypes.get_last_error())
    _resources.windows_job = handle


def isolate_worker() -> None:
    """Create containment before importing torch or launching any audio subprocess."""
    if os.name == "nt":
        _windows_containment()
    elif os.getsid(0) != os.getpid():
        os.setsid()


def _stop_family(*, worker_pid: int, worker_created: float, known: list[psutil.Process]) -> None:
    processes = list(known)
    if same_process(worker_pid, worker_created):
        worker = psutil.Process(worker_pid)
        with contextlib.suppress(psutil.NoSuchProcess):
            processes.extend(child for child in worker.children(recursive=True) if child.pid != os.getpid())
        processes.append(worker)
    if os.name != "nt":
        # This group/session was created by this exact worker, not a shell/user group.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(worker_pid, signal.SIGTERM)
    for process in processes:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            process.terminate()
    _, alive = psutil.wait_procs(processes, timeout=2)
    if os.name != "nt":
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(worker_pid, signal.SIGKILL)
    for process in alive:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            process.kill()


def watch(*, parent_pid: int, parent_created: float, worker_pid: int, worker_created: float) -> None:
    """Stop the worker family when either the GUI or the worker exits."""
    known: list[psutil.Process] = []
    while True:
        worker_alive = same_process(worker_pid, worker_created)
        if worker_alive:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                known = [
                    child for child in psutil.Process(worker_pid).children(recursive=True) if child.pid != os.getpid()
                ]
        if not worker_alive or not same_process(parent_pid, parent_created):
            _stop_family(worker_pid=worker_pid, worker_created=worker_created, known=known)
            return
        time.sleep(0.25)


def start_parent_guard(*, parent_pid: int | None, parent_created: float | None) -> None:
    """Start once before task work; the guardian cleans up after the worker exits.

    Deliberately do not stop it in finally: even a normal worker exit can leave
    children behind. The guardian exits on its own after cleaning that family.
    CLI workers without a GUI parent do not start a guardian.
    """
    if parent_pid is None and parent_created is None:
        return
    if parent_pid is None or parent_created is None:
        message = "Both parent PID and creation time are required"
        raise ValueError(message)
    if not same_process(parent_pid, parent_created):
        message = "Родительское окно уже закрыто. Worker не запущен."
        raise RuntimeError(message)
    isolate_worker()
    worker = psutil.Process()
    _resources.guardian = subprocess.Popen(  # noqa: S603 - fixed module under this interpreter; no shell.
        [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "stem_local.guardian",
            "--parent-pid",
            str(parent_pid),
            "--parent-created",
            str(parent_created),
            "--worker-pid",
            str(worker.pid),
            "--worker-created",
            str(worker.create_time()),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=os.name != "nt",
    )


def main() -> None:
    """Private entry point; never discover or terminate unrelated user processes."""
    configure_utf8()
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--parent-created", type=float, required=True)
    parser.add_argument("--worker-pid", type=int, required=True)
    parser.add_argument("--worker-created", type=float, required=True)
    args = parser.parse_args()
    watch(
        parent_pid=args.parent_pid,
        parent_created=args.parent_created,
        worker_pid=args.worker_pid,
        worker_created=args.worker_created,
    )


if __name__ == "__main__":
    main()
