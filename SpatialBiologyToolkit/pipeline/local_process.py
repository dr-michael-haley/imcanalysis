"""Linux process identity, project locking and bounded resource inspection."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from .local_models import ProcessIdentity


def require_linux() -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "Local execution requires Linux (including EC2). Planning works on any platform."
        )


def proc_stat(pid: int) -> list[str]:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()


def identity(pid: int) -> ProcessIdentity:
    stat = proc_stat(pid)
    return ProcessIdentity(
        pid=pid,
        start_ticks=int(stat[19]),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        hostname=socket.gethostname(),
    )


def same_host(process: ProcessIdentity) -> bool:
    try:
        return (
            process.hostname == socket.gethostname()
            and process.boot_id
            == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        )
    except OSError:
        return False


def alive(process: ProcessIdentity | None) -> bool:
    if process is None or not same_host(process):
        return False
    try:
        return identity(process.pid) == process and proc_stat(process.pid)[0] != "Z"
    except (OSError, ValueError, IndexError):
        return False


@contextmanager
def project_lock(state_dir: Path):
    require_linux()
    import fcntl

    # Never unlink this file: concurrent openers must lock the same inode.
    with (state_dir / "local.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "A local workflow is already running in this project."
            ) from exc
        yield lock
        # Closing this descriptor leaves the inherited worker descriptor locked.


def signal_worker(process: ProcessIdentity) -> None:
    """Signal only the recorded worker, with pidfd protection where available."""
    if not alive(process):
        raise RuntimeError(
            "The recorded worker is unavailable on this host/boot; no process was signalled."
        )
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        try:
            fd = os.pidfd_open(process.pid)
        except OSError:
            return  # The worker still polls its durable cancellation request.
        try:
            if not alive(process):
                raise RuntimeError("Worker identity changed; no process was signalled.")
            signal.pidfd_send_signal(fd, signal.SIGTERM)
        finally:
            os.close(fd)
    else:
        # Cancellation is also polled from a run-specific file. Avoid an unsafe
        # PID-only signal on older kernels/Python runtimes.
        return


def stop_child(child: subprocess.Popen, grace: float = 5.0) -> None:
    """Called only by the parent that owns this unreaped process-group leader."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace
    # Keep the group leader unreaped until after KILL to prevent PID reuse.
    while time.monotonic() < deadline:
        try:
            members = group_stats(child.pid)
        except OSError:
            break
        if not members:
            break
        time.sleep(0.1)
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def group_stats(group: int) -> dict[int, tuple[int, int]]:
    result = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = proc_stat(int(path.name))
            if int(stat[2]) == group and stat[0] != "Z":
                result[int(path.name)] = (int(stat[11]) + int(stat[12]), int(stat[21]))
        except (OSError, ValueError, IndexError):
            continue
    return result


def resource_snapshot(process: ProcessIdentity | None) -> dict:
    if not alive(process):
        return {
            "available": False,
            "reason": "No live stage process on this host/boot.",
        }
    before = group_stats(process.pid)
    started = time.monotonic()
    time.sleep(0.2)
    after = group_stats(process.pid) if alive(process) else {}
    ticks = sum(
        max(0, value[0] - before[pid][0])
        for pid, value in after.items()
        if pid in before
    )
    result = {
        "available": True,
        "pids": sorted(after),
        "cpu_percent": round(
            100 * ticks / os.sysconf("SC_CLK_TCK") / (time.monotonic() - started), 1
        ),
        "rss_mib": round(
            sum(value[1] for value in after.values())
            * os.sysconf("SC_PAGE_SIZE")
            / 2**20,
            1,
        ),
    }
    # Device utilization is global; GPU memory/process ownership is per PID.
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        compute = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        result["gpu_devices"] = (
            gpu.stdout.strip().splitlines() if gpu.returncode == 0 else []
        )
        result["gpu_processes"] = [
            line
            for line in compute.stdout.splitlines()
            if line.split(",", 1)[0].strip() in {str(pid) for pid in after}
        ]
        result["gpu_note"] = (
            "Device utilization is global. Matching PIDs show GPU allocation, not continuous compute."
        )
    except (OSError, subprocess.SubprocessError) as exc:
        result["gpu_note"] = f"nvidia-smi unavailable: {exc}"
    return result
