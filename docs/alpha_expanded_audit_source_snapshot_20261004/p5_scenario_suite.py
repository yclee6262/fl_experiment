"""Run the preregistered one-factor P5 scenario expansion."""

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from alpha_weight_experiments import build_parser as build_alpha_parser
from alpha_weight_experiments import run as run_alpha
from alpha_weight_experiments import validate_args


SCENARIOS = (
    ("reference_clean", {}),
    ("heterogeneity_0p6", {"data_heterogeneity": 0.6}),
    ("sample_imbalance_4x", {"sample_imbalance_ratio": 4.0}),
    ("low_quality_25pct", {"low_quality_ratio": 0.25, "low_quality_noise_std": 0.3}),
    ("controlled_poison", {"poison_ratios": [0.05], "fixed_participant_ids": [1,2,3,4,20]}),
    ("dimension_5", {"dimensions": [5]}),
    ("controlled_coalition_5", {"fixed_participant_ids": [1,2,3,4,20]}),
    ("hard_target_2p5", {"targets": [2.5]}),
)


def build_scenario_args(root, seeds, evaluator_epochs):
    base = build_alpha_parser().parse_args([])
    base.comparison_suite = "p5"
    base.constraint_mode = "box_constrained"
    base.fixed_participant_ids = [1,2,20]
    base.clean_evaluator_noise = True
    base.seeds = list(seeds)
    base.dimensions = [2]
    base.poison_ratios = [0.0]
    base.targets = [0.0]
    base.evaluators = ["uniform"]
    base.max_calibration_rounds = 20
    base.guarded_no_progress_patience = 2
    base.external_evaluator_epochs = evaluator_epochs
    base.enable_coalition_stabilization = False
    base.settle_payment = False
    scenarios = []
    for name, overrides in SCENARIOS:
        current = deepcopy(base)
        for key, value in overrides.items():
            setattr(current, key, value)
        current.output_dir = str(Path(root) / name)
        validate_args(current)
        scenarios.append((name, current))
    return scenarios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--external-evaluator-epochs", type=int, default=100)
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("P5 suite output directory must be empty.")
    seeds = [int(value) for value in args.seeds.split(",")]
    scenarios = build_scenario_args(root, seeds, args.external_evaluator_epochs)
    manifest = {
        "schema_version": 2,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "seeds": seeds,
        "comparison": ["A_uniform_no_update", "D_uniform_guarded"],
        "expected_conditions": len(scenarios) * len(seeds) * 2,
        "settlement": "not_run_by_configuration",
        "quality_primary": "frozen pooled-data global evaluator; truth is diagnostic only",
        "confirmatory_scenarios": [name for name, _ in SCENARIOS if name != "hard_target_2p5"],
        "exploratory_scenarios": ["hard_target_2p5"],
        "statistical_rule": {
            "sampling_unit": "seed-level paired difference E_D - E_A",
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260924,
            "confirmatory_ci": "family-wise 95% Bonferroni simultaneous interval across 7 scenarios",
            "stable_advantage_requires": [
                "complete valid pairs", "Bonferroni interval entirely below zero",
                "no greater observed failure count for D than A"],
        },
        "source_sha256": {"p5_scenario_suite.py": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()},
        "pruning": "disabled_by_P3_decision",
        "scenarios": [
            {"name": name, "arguments": vars(current)}
            for name, current in scenarios
        ],
    }
    path = root / "suite_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    try:
        for name, current in scenarios:
            print(f"\n=== P5 scenario: {name} ===")
            run_alpha(current)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        raise
    finally:
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
