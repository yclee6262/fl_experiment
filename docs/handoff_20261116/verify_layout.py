"""Compatibility entry for the current layout check."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1] / "reorganization_20261116/verify_layout.py"), run_name="__main__")
