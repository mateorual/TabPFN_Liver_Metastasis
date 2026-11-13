"""Put the implementation root on sys.path so scripts can `import src...`."""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_quick(argv=None):
    argv = sys.argv if argv is None else argv
    return "--quick" in argv
