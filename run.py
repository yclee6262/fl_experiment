"""Run a classified script with the historical repository-root working directory."""
import json
import os
from pathlib import Path
import runpy
import sys
from project_support.legacy import prepare


def main():
    root = Path(__file__).resolve().parent
    flat = prepare()
    manifest = json.loads((root / "docs/reorganization_20261116/layout.json").read_text())
    entries = {row["canonical"]: str(flat / row["original"]) for row in manifest["files"]}
    entries.update({row["previous"]: str(flat / row["original"]) for row in manifest["files"]})
    aliases = {"initialization": "experiments/alpha/compare_initializations.py",
               "contribution": "experiments/contribution/replay_cost_comparison.py",
               "baseline": "experiments/baselines/compare_fedavg.py",
               "calibration": str(flat / "alpha_weight_experiments.py"),
               "stages": str(flat / "run_experiments.py")}
    entries.update(aliases)
    if len(sys.argv) == 1 or sys.argv[1] in ("--help", "-h", "list"):
        print("Usage: python run.py <classified/script.py> [script arguments]")
        print("Runs from repository root; relative output paths are relative to that root.")
        print("Some historical runners execute immediately and do not support --help.")
        print("\nClassified scripts:")
        print("\n".join(aliases))
        print("\n".join(sorted(row["canonical"] for row in manifest["files"]
                                if row["status"] != "archive")))
        return
    entry = sys.argv[1]
    if entry not in entries:
        raise SystemExit(f"Unknown entry: {entry}. Use python run.py list.")
    script = root / entries[entry]
    # Legacy scripts keep their original flat relative-source layout. New entry
    # points continue using the repository root as documented.
    if script.parent == flat:
        for i, value in enumerate(sys.argv):
            if i > 0 and sys.argv[i - 1] in ("--output-dir", "--output", "--source", "--registration") and not Path(value).is_absolute():
                sys.argv[i] = str(root / value)
        os.chdir(flat)
    else:
        os.chdir(root)
    sys.path.insert(0, str(root))
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
