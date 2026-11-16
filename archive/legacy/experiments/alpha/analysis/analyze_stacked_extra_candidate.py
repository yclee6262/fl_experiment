"""Development-only six-candidate selector with B-shrunk stacked initial alpha."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from agent_client import AgentNode
from research_analysis import paired_bootstrap_mean_ci
from validation_calibrated_alpha import (CANDIDATE_METHODS, fit_residual_calibrator,
                                         select_candidate)


EXTRA = "B_stacked_half"


def analyze(root, five_path, stacked_path):
    root = Path(root)
    five = json.loads(Path(five_path).read_text())
    stacked = json.loads(Path(stacked_path).read_text())
    manifest = json.loads((root / "suite_manifest.json").read_text())
    if five["seeds"] != stacked["seeds"] or not set(five["seeds"]) <= set(manifest["seeds"]):
        raise ValueError("Development cohorts are not matched")
    directory = root / manifest["scenarios"][0]
    shared = {int(r["seed"]): r for r in
              map(json.loads, (directory / "shared_costs.jsonl").read_text().splitlines())}
    five_rows = {int(r["seed"]): r for r in five["rows"]}
    stacked_rows = {int(r["seed"]): r for r in stacked["rows"]}
    output = []
    method_order = CANDIDATE_METHODS + (EXTRA,)
    for seed in five["seeds"]:
        base = five_rows[seed]
        extra = stacked_rows[seed]["methods"][EXTRA]
        item = shared[seed]
        payload = torch.load(directory / item["artifact_path"] / "local_models.pt",
                             map_location="cpu", weights_only=True)
        agents = []
        for ident in item["coalition_ids"]:
            agent = AgentNode(ident, None, n_features=payload["n_features"])
            agent.model.load_state_dict(payload["agents"][ident])
            agent.model.eval()
            agents.append(agent)
        b = base["candidates"]["B_stage0"]["alpha"]
        coefficients, _ = fit_residual_calibrator(
            agents, item["blind_test_X"], item["blind_test_y"], b)
        solutions = {m: base["candidates"][m]["solution"]
                     for m in CANDIDATE_METHODS}
        solutions[EXTRA] = extra["solution"]
        baseline_selected, baseline_scores = select_candidate(
            agents, b, coefficients, solutions, 0.0,
            method_order=CANDIDATE_METHODS)
        if baseline_selected != base["new_selected_initialization"]:
            raise ValueError(f"Five-candidate selector replay mismatch: {seed}")
        chosen, scores = select_candidate(
            agents, b, coefficients, solutions, 0.0,
            method_order=method_order)
        chosen_error = (extra["pooled_error"] if chosen == EXTRA
                        else base["candidates"][chosen]["pooled_error"])
        output.append({"seed": seed, "old_selected": baseline_selected,
                       "new_selected": chosen,
                       "old_five_pooled_error": base["new_pooled_error"],
                       "six_pooled_error": chosen_error,
                       "pooled_delta_six_minus_five": chosen_error-base["new_pooled_error"],
                       "extra_stage3_requests": extra["stage3_requests"],
                       "extra_initialization_requests": extra["initialization_requests"],
                       "extra_internal_score": scores[EXTRA]["internal_target_error"],
                       "old_best_internal_score": baseline_scores[baseline_selected][
                           "internal_target_error"]})
    delta = np.asarray([r["pooled_delta_six_minus_five"] for r in output])
    return {"study": "B_stacked_sixth_candidate_development",
            "scope": "exploratory_not_independent_quality_evidence",
            "scenario": manifest["scenarios"][0], "seeds": five["seeds"],
            "extra_selected_count": sum(r["new_selected"] == EXTRA for r in output),
            "mean_pooled_delta_six_minus_five": float(delta.mean()),
            "descriptive_95pct_ci": paired_bootstrap_mean_ci(delta),
            "wins_ties_losses": [int((delta < -1e-7).sum()),
                                   int((np.abs(delta) <= 1e-7).sum()),
                                   int((delta > 1e-7).sum())],
            "mean_extra_stage3_requests": float(np.mean([
                r["extra_stage3_requests"] for r in output])),
            "mean_extra_initialization_requests": float(np.mean([
                r["extra_initialization_requests"] for r in output])),
            "worst_pooled_regression": float(delta.max()),
            "rows": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("five_candidate_json")
    parser.add_argument("stacked_json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.five_candidate_json, args.stacked_json)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False,
                                           indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("seeds", "rows")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
