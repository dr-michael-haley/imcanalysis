"""Console startup for a packed Windows environment; standard library only."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def runtime_id(root: Path) -> str:
    return hashlib.sha256(str(root).upper().encode("utf-8")).hexdigest().upper()


def runtime_environment(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "QT_PLUGIN_PATH", "QML2_IMPORT_PATH",
                "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_SHLVL"):
        env.pop(key, None)
    paths = [root, root / "Library/mingw-w64/bin", root / "Library/usr/bin",
             root / "Library/bin", root / "Scripts"]
    env["PATH"] = os.pathsep.join(map(str, paths)) + os.pathsep + env.get("PATH", "")
    env.update(PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               CONDA_PREFIX=str(root))
    cache = Path(env["LOCALAPPDATA"]) / "NapariSBT/cache/numba" / runtime_id(root)[:12]
    cache.mkdir(parents=True, exist_ok=True)
    env["NUMBA_CACHE_DIR"] = str(cache)
    # These are the OpenSSL defaults normally supplied by its activation hook.
    for key, relative in (("SSL_CERT_FILE", "Library/ssl/cacert.pem"),
                          ("SSL_CERT_DIR", "Library/ssl/certs")):
        if (root / relative).exists() and not env.get(key):
            env[key] = str(root / relative)
    return env


@contextmanager
def setup_lock(root: Path):
    """Use the same Windows mutex as NapariSBT.exe during first-run setup."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateMutexW(None, False, "Local\\NapariSBT-" + runtime_id(root))
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    acquired = False
    try:
        result = kernel.WaitForSingleObject(handle, 0)
        acquired = result in (0, 0x80)
        if result == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())
        if not acquired:
            raise RuntimeError("Another launcher is preparing this folder. Wait and try again.")
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def prepare(root: Path, env: dict[str, str]) -> None:
    marker = root / ".naparisbt-location.txt"
    with setup_lock(root):
        if marker.exists():
            previous = marker.read_text(encoding="utf-8-sig").strip()
            if previous.casefold() != str(root).casefold():
                raise RuntimeError("This application folder has moved or been renamed. "
                                   "Extract the original ZIP into its final location and use that copy.")
            return
        if (root / ".naparisbt-ready").exists():
            raise RuntimeError("An older launcher prepared this folder without recording its location. "
                               "Extract the original ZIP into a fresh folder before using this launcher.")
        unpack = root / "Scripts/conda-unpack-script.py"
        if not unpack.is_file():
            raise RuntimeError("The first-run setup script is missing. Extract the entire conda-pack ZIP.")
        print("Preparing NapariSBT for this computer. This only happens on first launch.", flush=True)
        # Probe write access before relocation; publish the marker only on success.
        with tempfile.NamedTemporaryFile(dir=root, prefix=".naparisbt-write-", delete=False) as stream:
            probe = Path(stream.name)
        try:
            subprocess.run([str(root / "python.exe"), "-s", str(unpack)],
                           cwd=root, env=env, check=True)
            probe.write_text(str(root), encoding="utf-8")
            probe.replace(marker)
        finally:
            probe.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Prepare the environment, then stop (used by the batch launcher).")
    args = parser.parse_args(argv)
    root = Path(os.path.abspath(__file__)).parent
    try:
        if not (root / "python.exe").is_file() or not (root / "Lib").is_dir():
            raise RuntimeError("Put start_naparisbt.py beside python.exe in the fully extracted ZIP.")
        env = runtime_environment(root)
        prepare(root, env)
        if args.prepare_only:
            return 0
        print("Starting NapariSBT. Keep this window open while using the application.", flush=True)
        return subprocess.run([str(root / "python.exe"), "-s", "-m",
                               "SpatialBiologyToolkit.napari_sbt", "--welcome"],
                              cwd=root, env=env).returncode
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"NapariSBT could not start: {error}", file=sys.stderr)
        print("Check that the ZIP was fully extracted into a writable folder. "
              "If Windows reports a security block, contact your IT team.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
