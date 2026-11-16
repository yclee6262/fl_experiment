"""Check saved static-alpha deliveries against frozen pooled evaluator artifacts."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from research_evaluation import load_evaluator_artifact, score_solution


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    errors = []
    expected = {(s, str(seed), m) for s in manifest["scenarios"]
                for seed in manifest["seeds"] for m in manifest["methods"]}
    rows = []
    for scenario in manifest["scenarios"]:
        directory = root / scenario
        run_manifest = directory / "run_manifest.json"
        if not run_manifest.exists():
            errors.append(f"missing run manifest: {scenario}")
            continue
        run = json.loads(run_manifest.read_text())
        for relative, digest in run.get("source_sha256", {}).items():
            source = directory / "source_snapshot" / relative
            if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                errors.append(f"source snapshot mismatch: {scenario}/{relative}")
        for relative, digest in run.get("artifact_sha256", {}).items():
            artifact = directory / relative
            if not artifact.exists() or hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
                errors.append(f"artifact mismatch: {scenario}/{relative}")
        result = directory / "summary.csv"
        if result.exists():
            with result.open(newline="") as handle:
                rows.extend(csv.DictReader(handle))
        shared_path = directory / "shared_costs.jsonl"
        if not shared_path.exists():
            errors.append(f"missing shared costs: {scenario}")
        else:
            shared = [json.loads(line) for line in shared_path.read_text().splitlines()]
            shared_by_seed = {str(item["seed"]): item for item in shared}
            if len(shared) != len(manifest["seeds"]) or len(shared_by_seed) != len(shared):
                errors.append(f"shared seed coverage mismatch: {scenario}")
            for seed in manifest["seeds"]:
                item = shared_by_seed.get(str(seed))
                if item is None:
                    errors.append(f"missing shared seed: {scenario}/{seed}")
                    continue
                if not item.get("missing_fixed_ids") and item.get("artifact_path"):
                    if not all(key in item for key in ("proposals", "coalition_ids",
                                                         "blind_test_X", "blind_test_y")):
                        errors.append(f"missing shared pairing evidence: {scenario}/{seed}")
                    elif (len(item["proposals"]) != len(item["coalition_ids"])
                          or len(item["blind_test_X"]) != len(item["blind_test_y"])):
                        errors.append(f"shared pairing mismatch: {scenario}/{seed}")
                for row in rows:
                    if row["scenario"] == scenario and row["seed"] == str(seed) and row["delivery_status"] == "valid":
                        if row["artifact_path"] != item.get("artifact_path"):
                            errors.append(f"artifact pairing mismatch: {scenario}/{seed}/{row['method']}")
    keys = Counter((r["scenario"], r["seed"], r["method"]) for r in rows)
    for key in sorted(expected - keys.keys()):
        errors.append(f"missing condition: {key}")
    for key, count in keys.items():
        if count != 1 or key not in expected:
            errors.append(f"duplicate or unexpected condition: {key}, count={count}")
    evaluator_cache = {}
    valid = 0
    for row in rows:
        if row["delivery_status"] != "valid":
            continue
        valid += 1
        key = (row["scenario"], row["seed"])
        artifact = root / row["scenario"] / row["artifact_path"]
        if not artifact.exists():
            errors.append(f"missing evaluator artifact: {key}")
            continue
        if key not in evaluator_cache:
            evaluator_cache[key] = load_evaluator_artifact(artifact)[0]
        try:
            solution = np.asarray(json.loads(row["final_solution"]), dtype=float)
            if solution.ndim != 1 or not len(solution) or not np.all(np.isfinite(solution)):
                raise ValueError("invalid X")
            violation = float(max(0., np.max(-1. - solution), np.max(solution - 1.)))
            if violation > 1e-8 or abs(violation - float(row["max_box_violation"])) > 1e-8:
                errors.append(f"box violation mismatch: {key}, {row['method']}")
            if row["final_box_feasible"] != "True":
                errors.append(f"wrong feasible flag: {key}, {row['method']}")
            target = json.loads((artifact / "metadata.json").read_text())["scenario"]["target"]
            _, score = score_solution(evaluator_cache[key], solution, target)
            if abs(score - float(row["external_global_error"])) > 1e-7:
                errors.append(f"pooled reload mismatch: {key}, {row['method']}")
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"bad valid delivery: {key}, {row['method']}, {exc}")
    report = {"suite_status": manifest["status"], "expected": len(expected),
              "observed": len(rows), "valid_deliveries": valid,
              "passed": not errors and manifest["status"] == "complete",
              "errors": errors}
    (root / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    result = verify(args.root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)
