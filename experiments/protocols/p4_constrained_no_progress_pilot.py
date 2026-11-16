"""Revalidate Guarded no-progress stopping after enabling hard box constraints."""

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from alpha_weight_experiments import build_parser, run, validate_args


def configurations(root, seeds):
    base = build_parser().parse_args([])
    base.constraint_mode = "box_constrained"
    base.clean_evaluator_noise = True
    base.seeds = list(seeds)
    base.initializations = ["uniform"]
    base.evaluators = ["uniform"]
    base.update_rates = [0.5]
    base.update_policies = ["guarded"]
    base.max_calibration_rounds = 20
    base.enable_coalition_stabilization = False
    base.settle_payment = False
    settings = []
    for scenario, poison_ratio, participant_ids in (
        ("clean", 0.0, [1, 2, 20]),
        ("controlled_poison", 0.05, [1, 2, 3, 4, 20]),
    ):
        for patience in (0, 2):
            current = deepcopy(base)
            current.poison_ratios = [poison_ratio]
            current.fixed_participant_ids = participant_ids
            current.guarded_no_progress_patience = patience
            current.output_dir = str(
                Path(root) / f"{scenario}_patience{patience}"
            )
            validate_args(current)
            settings.append((scenario, patience, current))
    return settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("Output directory must be empty.")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    settings = configurations(root, seeds)
    manifest = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "constraint_mode": "box_constrained",
        "comparisons": [
            {"scenario": scenario, "patience": patience,
             "arguments": vars(config)}
            for scenario, patience, config in settings
        ],
    }
    path = root / "suite_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        for _, _, config in settings:
            run(config)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
