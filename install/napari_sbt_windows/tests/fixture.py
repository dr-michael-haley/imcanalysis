"""Disposable subprocess fixture; uses a real Python executable."""
import ctypes
import os
from pathlib import Path
import sys

root = Path(sys.executable).parent
unpack = sys.argv[0].endswith("conda-unpack-script.py")
with (root / "calls.txt").open("a") as stream:
    stream.write("unpack\n" if unpack else "launch\n")
print("Unicode log output: café")
print("Diagnostic stderr captured", file=sys.stderr)
if unpack:
    sys.exit(11 if (root / "fail-unpack").exists() else 0)
assert sys.argv[1:] == ["--welcome"]
assert os.environ.get("NAPARISBT_TEST_HOOK") == "ran"
assert not os.environ.get("PYTHONPATH") and not os.environ.get("PYTHONHOME")
assert os.environ["PYTHONNOUSERSITE"] == "1"
assert Path(os.environ["NUMBA_CACHE_DIR"]).is_dir()
(root / "admin.txt").write_text(str(bool(ctypes.windll.shell32.IsUserAnAdmin())))
sys.exit(42 if (root / "fail-app").exists() else 0)
