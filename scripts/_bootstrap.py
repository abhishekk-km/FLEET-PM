import faulthandler
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

faulthandler.enable()
_wd = os.environ.get("FLEETPM_WATCHDOG")      # optional: print a stack trace every N seconds to show where a script is stuck
if _wd:
    faulthandler.dump_traceback_later(int(_wd), repeat=True)
