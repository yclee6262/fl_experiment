"""Independently replay the saved blind-data selector and audit its costs."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from agent_client import AgentNode
from host_server import HostServer
from initial_alpha_methods import initial_alpha
from run_validation_calibrated_alpha import (AUDIT_CANDIDATE, AUDIT_METHOD,
                                             MARGINAL_CANDIDATE, MARGINAL_METHOD)
from verify_alpha_initialization_study import verify as verify_deliveries
from validation_calibrated_alpha import diverse_four_methods


def _rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _close(actual, expected, tolerance=1e-7):
    return np.isfinite(actual).all() and np.allclose(actual, expected,
                                                     rtol=0, atol=tolerance)


def verify(root):
    root = Path(root)
    generic = verify_deliveries(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    errors = list(generic["errors"])
    audited = 0
    audited_seeds = set()
    candidate_methods = tuple(manifest["candidate_initializations"])
    variants = manifest.get("selection_variants", {
        "validation_calibrated_multistart": candidate_methods})
    selector_rules = manifest.get("selector_rules", {})
    for scenario in manifest["scenarios"]:
        directory = root / scenario
        candidate_file = directory / "candidate_rows.csv"
        if not candidate_file.exists():
            errors.append(f"missing candidate rows: {scenario}")
            continue
        candidates = _rows(candidate_file)
        summary = _rows(directory / "summary.csv")
        candidate_keys = Counter((r["seed"], r["method"]) for r in candidates)
        expected = {(str(seed), method) for seed in manifest["seeds"]
                    for method in candidate_methods}
        for key in expected:
            if candidate_keys[key] != 1:
                errors.append(f"candidate coverage mismatch: {scenario}/{key}")
        for key in candidate_keys.keys() - expected:
            errors.append(f"unexpected candidate: {scenario}/{key}")
        candidate_index = {(r["seed"], r["method"]): r for r in candidates}
        summary_index = {(r["seed"], r["method"]): r for r in summary}
        shared = {str(item["seed"]): item for item in
                  map(json.loads, (directory / "shared_costs.jsonl").read_text().splitlines())}
        for seed in map(str, manifest["seeds"]):
            item = shared.get(seed)
            if item is None or item.get("missing_fixed_ids"):
                continue
            try:
                by_method = {method: candidate_index[(seed, method)]
                             for method in candidate_methods}
                baseline = by_method["B_stage0"]
                if any(by_method[m]["delivery_status"] != "valid"
                       for m in ("A_uniform", "B_stage0")):
                    errors.append(f"baseline candidate failed: {scenario}/{seed}")
                    continue
                for method in ("A_uniform", "B_stage0"):
                    control = summary_index[(seed, method)]
                    for field in ("final_solution", "initial_alpha", "external_global_error"):
                        if control[field] != by_method[method][field]:
                            errors.append(f"baseline differs from candidate: {scenario}/{seed}/{method}/{field}")
                artifact = directory / item["artifact_path"] / "local_models.pt"
                payload = torch.load(artifact, map_location="cpu", weights_only=True)
                agents = []
                for ident in item["coalition_ids"]:
                    agent = AgentNode(ident, None, n_features=payload["n_features"])
                    agent.model.load_state_dict(payload["agents"][ident])
                    agent.model.eval()
                    agents.append(agent)
                x = np.asarray(item["blind_test_X"], float)
                y = np.asarray(item["blind_test_y"], float)
                if MARGINAL_CANDIDATE in by_method:
                    marginal_row = by_method[MARGINAL_CANDIDATE]
                    marginal_alpha, marginal_audit = initial_alpha(
                        MARGINAL_CANDIDATE, agents,
                        np.asarray(json.loads(marginal_row["stage0_mse"]), float),
                        x, y, item["proposals"], 0.0)
                    saved_alpha = np.asarray(json.loads(marginal_row["initial_alpha"]), float)
                    saved_audit = json.loads(marginal_row["alpha_audit"])
                    if (not _close(marginal_alpha, saved_alpha)
                            or marginal_row["trusted_label_count"] != str(len(y))
                            or marginal_row["extra_trusted_labels"] != "0"
                            or not _close(np.asarray(marginal_audit[
                                "blind_leave_one_out_gain"]), np.asarray(saved_audit[
                                "blind_leave_one_out_gain"]))):
                        errors.append(f"marginal alpha mismatch: {scenario}/{seed}")
                if AUDIT_CANDIDATE in by_method:
                    expanded_x = np.asarray(item["expanded_blind_test_X"], float)
                    expanded_y = np.asarray(item["expanded_blind_test_y"], float)
                    source = HostServer(0.0, payload["n_features"],
                                        test_seed=int(seed)+7919, n_test=100,
                                        constraint_mode="box_constrained")
                    if (not np.array_equal(expanded_x, source.test_X)
                            or not np.array_equal(expanded_y, source.test_y)
                            or not np.array_equal(expanded_x[:len(x)], x)
                            or not np.array_equal(expanded_y[:len(y)], y)
                            or item.get("extra_trusted_labels") != 80):
                        errors.append(f"expanded trusted labels mismatch: {scenario}/{seed}")
                    extra_row = by_method[AUDIT_CANDIDATE]
                    alpha_replayed, _ = initial_alpha(
                        "local_bias_risk", agents,
                        np.asarray(json.loads(extra_row["stage0_mse"]), float),
                        expanded_x, expanded_y, item["proposals"], 0.0)
                    if (not _close(alpha_replayed, np.asarray(json.loads(extra_row["initial_alpha"])))
                            or extra_row["trusted_label_count"] != "100"
                            or extra_row["extra_trusted_labels"] != "80"
                            or extra_row["trusted_label_source"] != "synthetic_true_function"):
                        errors.append(f"expanded alpha mismatch: {scenario}/{seed}")
                alpha = np.asarray(json.loads(baseline["initial_alpha"]), float)
                predictions = np.column_stack([np.asarray(a.api_predict(x), float)
                                               for a in agents])
                residual = predictions @ alpha - y
                design = np.column_stack((np.ones(len(x)), x))
                penalty = np.diag([0.01] + [1.0] * x.shape[1])
                coeff = np.linalg.solve(design.T @ design + penalty,
                                        design.T @ residual)
                for meta_name, method_order in variants.items():
                    method_order = tuple(method_order)
                    if meta_name == "diversified_four_portfolio":
                        alphas = {method: json.loads(by_method[method]["initial_alpha"])
                                  for method in candidate_methods}
                        method_order = diverse_four_methods(alphas)
                    selector_rule = selector_rules.get(meta_name, "global")
                    meta = summary_index[(seed, meta_name)]
                    if meta["delivery_status"] != "valid":
                        errors.append(f"meta delivery invalid: {scenario}/{seed}/{meta_name}")
                        continue
                    audit = json.loads(meta["alpha_audit"])
                    prefix = f"{scenario}/{seed}/{meta_name}"
                    if "candidate_method_order" in audit and tuple(
                            audit["candidate_method_order"]) != method_order:
                        errors.append(f"candidate method order mismatch: {prefix}")
                    if meta_name == "diversified_four_portfolio" and audit.get(
                            "portfolio_rule") != "mean_pairwise_L1_keep_A_B_four":
                        errors.append(f"portfolio rule mismatch: {prefix}")
                    if selector_rule == "global":
                        if not _close(np.asarray(audit["calibration_coefficients"]), coeff):
                            errors.append(f"calibration coefficients mismatch: {prefix}")
                    elif selector_rule == "local_bias":
                        if audit["calibration"].get("local_bandwidth") != 0.25:
                            errors.append(f"local bandwidth mismatch: {prefix}")
                    else:
                        errors.append(f"unknown selector rule: {prefix}/{selector_rule}")
                        continue
                    if audit["calibration"]["calibration_samples"] != len(y):
                        errors.append(f"calibration sample count mismatch: {prefix}")
                    if selector_rule == "global" and not _close(
                            float(audit["calibration"]["blind_residual_mse"]),
                            float(np.mean(residual**2))):
                        errors.append(f"blind residual mismatch: {prefix}")
                    scores = {}
                    for method in method_order:
                        if by_method[method]["delivery_status"] != "valid":
                            continue
                        solution = np.asarray(json.loads(by_method[method]["final_solution"]), float)
                        pred = np.asarray([float(np.asarray(a.api_predict(solution))[0])
                                           for a in agents])
                        if selector_rule == "local_bias":
                            kernel = np.exp(-np.sum((x-solution)**2, axis=1)/(2*0.25**2))
                            local_coeff = np.linalg.solve(
                                design.T @ (kernel[:, None]*design) + penalty,
                                design.T @ (kernel*residual))
                            saved_local = audit["internal_scores"][method]
                            if not _close(np.asarray(saved_local["local_coefficients"]),
                                          local_coeff):
                                errors.append(f"local coefficients mismatch: {prefix}/{method}")
                            if not _close(float(saved_local["local_kernel_mass"]),
                                          float(kernel.sum())):
                                errors.append(f"local kernel mass mismatch: {prefix}/{method}")
                            corrected = float(pred @ alpha - np.r_[1.0, solution] @ local_coeff)
                        else:
                            corrected = float(pred @ alpha - np.r_[1.0, solution] @ coeff)
                        scores[method] = abs(corrected - float(manifest["arguments"]["inverse_target"]))
                        saved = audit["internal_scores"][method]
                        if not _close(float(saved["corrected_prediction"]), corrected) or not _close(
                                float(saved["internal_target_error"]), scores[method]):
                            errors.append(f"candidate internal score mismatch: {prefix}/{method}")
                    if set(audit["internal_scores"]) != set(scores):
                        errors.append(f"candidate score coverage mismatch: {prefix}")
                    if not scores:
                        errors.append(f"no replayable candidate: {prefix}")
                        continue
                    chosen = min(scores, key=lambda method: scores[method])
                    if audit["selected_initialization"] != chosen:
                        errors.append(f"selected candidate mismatch: {prefix}")
                    selected = by_method[chosen]
                    for field in ("final_solution", "initial_alpha", "external_global_error"):
                        if meta[field] != selected[field]:
                            errors.append(f"selected delivery mismatch: {prefix}/{field}")
                    if meta_name == AUDIT_METHOD and (meta["trusted_label_count"] != "100"
                            or meta["extra_trusted_labels"] != "80"
                            or meta["trusted_label_source"] != "synthetic_true_function"):
                        errors.append(f"expanded audit accounting mismatch: {prefix}")
                    if meta_name == MARGINAL_METHOD and (meta["trusted_label_count"] != str(len(y))
                            or meta["extra_trusted_labels"] != "0"):
                        errors.append(f"marginal label accounting mismatch: {prefix}")
                    for field in ("stage3_requests", "stage3_input_samples",
                                  "initialization_requests", "initialization_input_samples"):
                        charged_methods = (candidate_methods
                                           if meta_name == "diversified_four_portfolio"
                                           and field.startswith("initialization_")
                                           else method_order)
                        expected_cost = sum(float(by_method[m][field]) for m in charged_methods)
                        if field == "initialization_requests":
                            expected_cost += len(agents) * (1 + len(method_order))
                        if field == "initialization_input_samples":
                            expected_cost += len(agents) * (len(x) + len(method_order))
                        if not _close(float(meta[field]), expected_cost):
                            errors.append(f"cost mismatch: {prefix}/{field}")
                    audited += 1
                    audited_seeds.add((scenario, seed))
            except (KeyError, ValueError, TypeError, IndexError, OSError) as exc:
                errors.append(f"selector replay failed: {scenario}/{seed}: {exc}")
    result = {"passed": generic["passed"] and not errors,
              "delivery_verification_passed": generic["passed"],
              "selector_replayed_seeds": len(audited_seeds),
              "selector_replayed_variants": audited, "errors": errors}
    (root / "selection_verification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    report = verify(args.root)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)
