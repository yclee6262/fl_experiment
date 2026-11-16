"""Experiments for separated reputation, optimization, and contribution weights.

The experiment keeps the Stage 1 coalition and Stage 2 candidates fixed across
conditions.  Stage 1 weights are retained as ex-ante reputation ``r``; Stage 3
uses an independently initialized and iteratively calibrated ``alpha``; and
Subspace Exclusion produces the ex-post contribution distribution ``q``.
"""

import argparse
import time
import copy
import csv
import json
import hashlib
import platform
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from collections import defaultdict
from itertools import product
from pathlib import Path


def _source_workspace(source_file):
    """Find the flat workspace or a standalone historical source snapshot."""
    start = Path(source_file).resolve().parent
    for candidate in (start, *start.parents):
        if all((candidate / name).is_file() for name in
               ("host_server.py", "run_experiments.py", "dataset.py")):
            return candidate
        if (candidate / "project_support/legacy.py").is_file():
            from project_support.legacy import prepare
            return prepare()
    raise FileNotFoundError(f"Cannot locate source workspace from {source_file}")


import numpy as np

from agent_client import AgentNode
from host_server import HostServer, payment_contributions
from research_evaluation import save_scenario_artifacts, score_solution
from run_experiments import build_agents, true_function


def write_run_manifest(output_dir, args, status):
    """Record provenance without collecting environment variables or private data."""
    root = _source_workspace(__file__)
    def git(*arguments):
        result = subprocess.run(
            ["git", "-C", str(root), *arguments], capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None

    packages = {}
    for name in ("numpy", "scipy", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    sources = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.glob("*.py")
    }
    snapshot = Path(output_dir) / "source_snapshot"
    if not snapshot.exists():
        snapshot.mkdir()
        for source in root.glob("*.py"):
            (snapshot / source.name).write_bytes(source.read_bytes())
    path = Path(output_dir) / "run_manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "source_sha256": sources,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "arguments": vars(args),
        "limitations": [
            "The global evaluator is an offline synthetic pooled-data study oracle, not a deployed federated model.",
            "Constraint mode is explicit; legacy runs only check box feasibility after solving.",
            "The pooled split evaluates evaluator fit; it is not an independent real-world dataset.",
        ],
    }
    manifest["status"] = status
    manifest["artifact_sha256"] = {
        str(path.relative_to(output_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(output_dir) / "artifacts").rglob("*") if path.is_file()
    } if (Path(output_dir) / "artifacts").exists() else {}
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def box_feasible(solution, config):
    """Post-hoc box check only, never an optimizer acceptance criterion."""
    if "feasible_lower" not in config or "feasible_upper" not in config:
        return None
    x = np.asarray(solution, dtype=float)
    return bool(np.all(np.isfinite(x)) and np.all(x >= config["feasible_lower"] - 1e-8)
                and np.all(x <= config["feasible_upper"] + 1e-8))


def comparison_conditions(args):
    if args.comparison_suite == "p2":
        return [
            {"method_label": "A_uniform_no_update", "initialization": "uniform",
             "initialization_replicate": 0, "evaluator": args.evaluators[0],
             "update_rate": 0.0, "update_policy": "fixed"},
            {"method_label": "B_stage0_no_update", "initialization": "stage0",
             "initialization_replicate": 0, "evaluator": args.evaluators[0],
             "update_rate": 0.0, "update_policy": "fixed"},
            {"method_label": "C_uniform_fixed", "initialization": "uniform",
             "initialization_replicate": 0, "evaluator": args.evaluators[0],
             "update_rate": 0.5, "update_policy": "fixed"},
            {"method_label": "D_uniform_guarded", "initialization": "uniform",
             "initialization_replicate": 0, "evaluator": args.evaluators[0],
             "update_rate": 0.5, "update_policy": "guarded"},
        ]
    if args.comparison_suite == "p5":
        return [
            {"method_label": "A_uniform_no_update", "initialization": "uniform",
             "initialization_replicate": 0, "evaluator": args.evaluators[0],
             "update_rate": 0.0, "update_policy": "fixed"},
            {"method_label": "D_uniform_guarded", "initialization": "uniform",
             "initialization_replicate": 0, "evaluator": args.evaluators[0],
             "update_rate": 0.5, "update_policy": "guarded"},
        ]
    conditions = []
    for initialization in args.initializations:
        replicate_count = (
            args.initialization_replicates if initialization == "dirichlet" else 1
        )
        for replicate in range(replicate_count):
            for evaluator, update_rate, update_policy in product(
                args.evaluators, args.update_rates, args.update_policies
            ):
                conditions.append({
                    "method_label": (
                        f"{initialization}_{evaluator}_{update_policy}_eta{update_rate:g}"
                    ),
                    "initialization": initialization,
                    "initialization_replicate": replicate,
                    "evaluator": evaluator,
                    "update_rate": update_rate,
                    "update_policy": update_policy,
                })
    return conditions


def parse_csv(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def normalize(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("Weights must be a non-empty one-dimensional vector.")
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("Weights must be finite and non-negative.")
    total = float(np.sum(values))
    if total <= 0:
        raise ValueError("Weights must have a positive sum.")
    return values / total


def initialize_optimization_weights(kind, reputation, rng):
    reputation = normalize(reputation)
    n_agents = len(reputation)
    if kind == "stage0":
        return reputation.copy()
    if kind == "uniform":
        return np.ones(n_agents, dtype=float) / n_agents
    if kind == "dirichlet":
        return rng.dirichlet(np.ones(n_agents, dtype=float))
    if kind == "reversed":
        return normalize(reputation[::-1])
    raise ValueError(f"Unknown initialization: {kind}")


def raw_contribution_distribution(exclusion_reports):
    positives = np.asarray(
        [row["positive_contribution"] for row in exclusion_reports],
        dtype=float,
    )
    if len(positives) == 1 and np.isposinf(positives[0]):
        return np.ones(1, dtype=float)
    if not np.all(np.isfinite(positives)):
        raise ValueError("Contribution scores must be finite.")
    total = float(np.sum(positives))
    if total <= 0:
        return None
    return normalize(positives / total)


def apply_exploration_floor(distribution, exploration_mass=0.0):
    if distribution is None:
        return None
    distribution = normalize(distribution)
    if not 0.0 <= exploration_mass < 1.0:
        raise ValueError("exploration_mass must be in [0, 1).")
    if exploration_mass:
        distribution = (
            (1.0 - exploration_mass) * distribution
            + exploration_mass / len(distribution)
        )
    return normalize(distribution)


def contribution_distribution(exclusion_reports, exploration_mass=0.0):
    """Compatibility wrapper returning the alpha-update target q_cal."""
    return apply_exploration_floor(
        raw_contribution_distribution(exclusion_reports),
        exploration_mass=exploration_mass,
    )


class StageCounter(dict):
    def __init__(self):
        super().__init__(stage="unassigned", requests=defaultdict(int),
                         input_samples=defaultdict(int), seconds=defaultdict(float))
        self.started = time.perf_counter()

    def __setitem__(self, key, value):
        if key == "stage":
            now = time.perf_counter()
            self["seconds"][self["stage"]] += now - self.started
            self.started = now
        super().__setitem__(key, value)

    def __deepcopy__(self, memo):
        clone = StageCounter()
        memo[id(self)] = clone
        for key, value in self.items():
            dict.__setitem__(clone, key, copy.deepcopy(value, memo))
        clone.started = time.perf_counter()
        return clone


def attach_query_counter(agents):
    counter = StageCounter()
    for agent in agents:
        original = getattr(agent.api_predict, "_uncounted_original", agent.api_predict)

        def counted(X_array, _original=original):
            counter["requests"][counter["stage"]] += 1
            shape = np.asarray(X_array).shape
            counter["input_samples"][counter["stage"]] += (
                int(shape[0]) if len(shape) > 1 else 1
            )
            return _original(X_array)

        counted._uncounted_original = original
        agent.api_predict = counted
    return counter


@contextmanager
def stage3_prediction_cache(agents):
    """Memoize calls to fixed AgentNode models during one Stage 3 solve.

    AgentNode converts every API input to float32; the cache uses those exact
    model-visible bytes. Never persist it across model updates or separate
    candidate solves. The wrapper is removed even if the solver fails.
    """
    if not all(isinstance(agent, AgentNode) for agent in agents):
        raise TypeError("Stage 3 prediction cache requires AgentNode models")
    originals = []
    stats = {"hits": 0, "misses": 0}
    try:
        for agent in agents:
            original = agent.api_predict
            originals.append((agent, original))
            memo = {}

            def cached(values, _original=original, _memo=memo):
                array = np.asarray(values, dtype=np.float32)
                key = (array.shape, array.tobytes())
                if key in _memo:
                    stats["hits"] += 1
                    return _memo[key].copy()
                prediction = _original(values)
                _memo[key] = np.asarray(prediction).copy()
                stats["misses"] += 1
                return prediction

            agent.api_predict = cached
        yield stats
    finally:
        for agent, original in originals:
            agent.api_predict = original


def run_stage3(server, optimizer, custom_iterations, initial_solution=None,
               cache_predictions=False):
    if cache_predictions:
        with stage3_prediction_cache(server.trusted_agents) as cache_stats:
            try:
                return _run_stage3_uncached(server, optimizer, custom_iterations,
                                            initial_solution=initial_solution)
            finally:
                server.last_stage3_cache = dict(cache_stats)
    return _run_stage3_uncached(server, optimizer, custom_iterations,
                                initial_solution=initial_solution)


def _run_stage3_uncached(server, optimizer, custom_iterations, initial_solution=None):
    if optimizer == "bfgs":
        final_S, history = server.phase3_global_optimization(
            initial_solution=initial_solution
        )
        states = ["bfgs"] * len(history)
    elif optimizer == "custom":
        final_S, history, states = server.phase3_custom_secant_optimization(
            num_iterations=custom_iterations,
            use_annealing=True,
            allow_tangent=True,
            initial_solution=initial_solution,
        )
    else:
        final_S, history, states = server.phase3_best_of_optimization(
            custom_iterations=custom_iterations,
            initial_solution=initial_solution,
        )
    return np.asarray(final_S, dtype=float), history, states


def evaluator_arguments(evaluator, reputation):
    if evaluator == "reputation":
        return "fixed", reputation
    return evaluator, None


def settle_with_separated_weights(
    server,
    reputation,
    contribution,
    positive_contribution,
    reputation_mix,
):
    """Settle the surplus from reputation and contribution, never optimization alpha."""
    if contribution is None:
        return "rejected_uninformative_contribution", [], None

    positive_indices = [
        idx for idx, value in enumerate(positive_contribution) if value > 0
    ]
    if not positive_indices:
        return "rejected_no_positive_contributors", [], None

    bids = np.asarray(
        [server._agent_bid(server.trusted_agents[idx]) for idx in positive_indices],
        dtype=float,
    )
    bid_sum = float(np.sum(bids))
    if bid_sum > server.total_budget:
        return "infeasible_minimum_bids_exceed_budget", [], None

    reputation_active = normalize(np.asarray(reputation)[positive_indices])
    contribution_active = normalize(np.asarray(contribution)[positive_indices])
    surplus = server.total_budget - bid_sum
    rows = []
    for local_idx, agent_idx in enumerate(positive_indices):
        surplus_share = (
            reputation_mix * reputation_active[local_idx]
            + (1.0 - reputation_mix) * contribution_active[local_idx]
        )
        payment = bids[local_idx] + surplus * surplus_share
        rows.append(
            {
                "agent_id": server.trusted_agents[agent_idx].agent_id,
                "reputation_share": float(reputation_active[local_idx]),
                "optimization_weight": float(server.alphas[agent_idx]),
                "contribution_share": float(contribution_active[local_idx]),
                "surplus_share": float(surplus_share),
                "bid": float(bids[local_idx]),
                "payment": float(payment),
            }
        )
    return "ok", rows, float(sum(row["payment"] for row in rows))


def settle_calibration_state(server, accepted_state, reputation, config, counter=None):
    """Settle exactly one final accepted state after coalition stabilization."""
    final_S = accepted_state["solution"]
    if counter is not None:
        counter["stage"] = "payment"
    q_pay, payment_positive, payment_source, fallback_reason = payment_contributions(
        accepted_state["payment_reports"],
        lambda: server._consensus_loss(final_S, mode="optimization"),
    )
    payment_status, payment_rows, paid_total = settle_with_separated_weights(
        server,
        reputation,
        q_pay,
        payment_positive,
        config["payment_reputation_mix"],
    )
    for row in payment_rows:
        row.update(config["identity"])
        row["payment_contribution_source"] = payment_source
        row["payment_fallback_reason"] = fallback_reason
    return {
        "payment_status": payment_status,
        "payment_contribution_source": payment_source,
        "payment_fallback_reason": fallback_reason,
        "payment_contribution_weights": json.dumps(
            [] if q_pay is None else q_pay.tolist()
        ),
        "payment_requests": (
            0 if counter is None else counter["requests"]["payment"]
        ),
        "payment_input_samples": (
            0 if counter is None else counter["input_samples"]["payment"]
        ),
        "paid_total": paid_total,
    }, payment_rows


def agent_reputation_weights(server, agents):
    """Build a deterministic reputation prior for a controlled coalition."""
    raw_scores = []
    for agent in agents:
        prediction = np.asarray(agent.api_predict(server.test_X), dtype=float)
        mse = float(np.mean((prediction - server.test_y) ** 2))
        raw_scores.append(1.0 / (mse + 1e-5))
    return normalize(raw_scores)


def configure_controlled_coalition(server, poisoned_ids, coalition_size, poison_count):
    """Construct a coalition that deliberately contains poisoned agents."""
    if coalition_size <= 0:
        raise ValueError("controlled coalition size must be positive")
    if poison_count < 0 or poison_count > coalition_size:
        raise ValueError("controlled poison count must be in [0, coalition_size]")
    if poison_count > len(poisoned_ids):
        raise ValueError("controlled poison count exceeds available poisoned agents")

    all_agents = list(getattr(server, "all_agents", []))
    by_id = {agent.agent_id: agent for agent in all_agents}
    forced_poison = [
        by_id[agent_id]
        for agent_id in sorted(poisoned_ids)[:poison_count]
    ]
    forced_ids = {agent.agent_id for agent in forced_poison}
    honest_candidates = []
    seen = set(forced_ids)
    for agent in list(server.trusted_agents) + all_agents:
        if agent.agent_id in seen or agent.agent_id in poisoned_ids:
            continue
        seen.add(agent.agent_id)
        honest_candidates.append(agent)
    honest_candidates.sort(key=lambda agent: agent.agent_id)
    selected = forced_poison + honest_candidates[: coalition_size - poison_count]
    if len(selected) != coalition_size:
        raise ValueError(
            "not enough agents to construct controlled coalition: "
            f"requested {coalition_size}, got {len(selected)}"
        )

    server.trusted_agents = selected
    server.I_list = []
    server.alphas = [
        float(value) for value in agent_reputation_weights(server, selected)
    ]
    server.phase2_collect_proposals()


def calibrate_one_condition(
    server,
    config,
    reputation,
    poisoned_ids,
    settle_payment=True,
):
    rng = np.random.default_rng(config["condition_seed"])
    alpha = initialize_optimization_weights(
        config["initialization"], reputation, rng
    )
    server.alphas = [float(value) for value in alpha]
    counter = attach_query_counter(server.trusted_agents)
    server.last_query_counter = counter
    round_rows = []
    agent_rows = []
    converged = False
    stop_reason = "max_rounds"
    accepted_state = None
    terminal_state = None
    pending_update = False
    eta_current = float(config["update_rate"])
    floor_rejection_streak = 0
    no_progress_triggered = False

    evaluation_mode, evaluation_weights = evaluator_arguments(
        config["evaluator"], reputation
    )

    for round_idx in range(config["max_rounds"]):
        alpha_before = np.asarray(server.alphas, dtype=float)
        recovery_attempts = 0
        warm_start = None
        solver_audit = {
            "status": "not_needed",
            "loo_dominates": False,
            "agent_id": None,
            "full_optimization_loss": None,
            "best_loo_optimization_loss": None,
            "improvement": None,
        }

        while True:
            counter["stage"] = "stage3"
            if warm_start is None:
                current_S, _, _ = run_stage3(
                    server, config["optimizer"], config["custom_iterations"]
                )
            else:
                current_S, _, _ = run_stage3(
                    server,
                    config["optimizer"],
                    config["custom_iterations"],
                    initial_solution=warm_start,
                )
            stage3_summary = copy.deepcopy(server.last_subspace_optimization)

            counter["stage"] = "contribution"
            current_eval_loss, exclusion_reports, exclusion_summary = server._compute_exclusion_reports(
                current_S,
                verbose=False,
                evaluation_mode=evaluation_mode,
                evaluation_weights=evaluation_weights,
                trim_fraction=config["trim_fraction"],
                optimizer="best_of",
                custom_iterations=config["custom_iterations"],
                return_summary=True,
            )
            q_raw = raw_contribution_distribution(exclusion_reports)
            q = apply_exploration_floor(
                q_raw,
                exploration_mass=config["exploration_mass"],
            )
            if q is not None:
                if recovery_attempts:
                    solver_audit["status"] = "recovered_informative"
                break

            full_opt_loss = stage3_summary.get("loss")
            if full_opt_loss is None:
                full_opt_loss = server._consensus_loss(
                    current_S, mode="optimization"
                )
            solver_audit = HostServer.audit_uninformative_exclusion(
                full_opt_loss,
                exclusion_reports,
                epsilon=config.get("solver_audit_tolerance", 1e-6),
            )
            if (
                solver_audit["loo_dominates"]
                and recovery_attempts < config.get("max_solver_recoveries", 1)
            ):
                recovery_attempts += 1
                warm_start = solver_audit["warm_start_solution"]
                continue
            if solver_audit["loo_dominates"]:
                solver_audit["status"] = "recovery_exhausted"
            break

        target_error = abs(true_function(current_S) - server.target_T)
        positive_contribution = np.asarray(
            [report["positive_contribution"] for report in exclusion_reports],
            dtype=float,
        )

        informative = q is not None
        gate_triggered = False
        reference_eval_loss = (
            None if accepted_state is None else accepted_state["evaluation_loss"]
        )
        reference_consistency = (
            None if accepted_state is None else accepted_state["consistency_l1"]
        )
        if informative:
            consistency_l1 = float(np.sum(np.abs(alpha_before - q)))
            raw_consistency_l1 = float(np.sum(np.abs(alpha_before - q_raw)))
        else:
            consistency_l1 = None
            raw_consistency_l1 = None

        if (
            config["update_policy"] == "guarded"
            and pending_update
            and reference_eval_loss is not None
            and current_eval_loss
            > reference_eval_loss + config["evaluation_tolerance"]
        ):
            gate_triggered = True

        current_state_accepted = bool(
            not gate_triggered and (informative or accepted_state is None)
        )
        update_accepted = bool(
            pending_update and current_state_accepted and informative
        )
        if current_state_accepted:
            accepted_state = {
                "round": round_idx,
                "alpha": alpha_before.copy(),
                "solution": current_S.copy(),
                "q": None if q is None else q.copy(),
                "q_raw": None if q_raw is None else q_raw.copy(),
                "positive_contribution": positive_contribution.copy(),
                "marginal_contribution": np.asarray(
                    [report["marginal_contribution"] for report in exclusion_reports],
                    dtype=float,
                ),
                "evaluation_loss": float(current_eval_loss),
                "target_error": float(target_error),
                "consistency_l1": consistency_l1,
                "raw_consistency_l1": raw_consistency_l1,
                "exclusion_summary": copy.deepcopy(exclusion_summary),
                "stage3_summary": copy.deepcopy(stage3_summary),
                "solver_audit_status": solver_audit["status"],
                "exclusion_reports": copy.deepcopy(exclusion_reports),
                "payment_reports": [
                    {key: report[key] for key in (
                        'marginal_contribution', 'restricted_optimization_loss'
                    )}
                    for report in exclusion_reports
                ],
            }

        eta_used = eta_current
        proposal_generated = False
        if informative:
            if config["update_rate"] == 0:
                alpha_after = alpha_before.copy()
                stop_reason = "no_update_baseline"
            elif gate_triggered:
                eta_current = max(
                    config["minimum_update_rate"], eta_current / 2.0
                )
                eta_used = eta_current
                if np.isclose(
                    eta_current,
                    config["minimum_update_rate"],
                    rtol=0.0,
                    atol=1e-12,
                ):
                    floor_rejection_streak += 1
                else:
                    floor_rejection_streak = 0
                accepted_alpha = accepted_state["alpha"]
                accepted_q = accepted_state["q"]
                if accepted_q is None:
                    alpha_after = accepted_alpha.copy()
                    stop_reason = "uninformative_accepted_state"
                elif (
                    config.get("guarded_no_progress_patience", 0) > 0
                    and floor_rejection_streak
                    >= config["guarded_no_progress_patience"]
                ):
                    alpha_after = accepted_alpha.copy()
                    proposal_generated = False
                    no_progress_triggered = True
                    stop_reason = "guarded_no_progress"
                else:
                    alpha_after = normalize(
                        (1.0 - eta_used) * accepted_alpha
                        + eta_used * accepted_q
                    )
                    proposal_generated = True
            elif consistency_l1 < config["tolerance"]:
                floor_rejection_streak = 0
                alpha_after = alpha_before.copy()
                converged = True
                stop_reason = "fixed_point_tolerance"
            else:
                floor_rejection_streak = 0
                if (
                    config["update_policy"] == "adaptive"
                    and reference_consistency is not None
                    and consistency_l1 > reference_consistency
                ):
                    eta_current = max(
                        config["minimum_update_rate"], eta_current / 2.0
                    )
                eta_used = eta_current
                candidate_alpha = normalize(
                    (1.0 - eta_used) * alpha_before + eta_used * q
                )
                alpha_after = candidate_alpha
                proposal_generated = True
            update_origin = (
                accepted_state["alpha"] if gate_triggered else alpha_before
            )
            update_l1 = float(np.sum(np.abs(alpha_after - update_origin)))
        else:
            q_for_update = alpha_before.copy()
            update_l1 = 0.0
            alpha_after = alpha_before.copy()
            stop_reason = solver_audit["status"]

        if informative:
            q_for_update = q

        round_rows.append(
            {
                **config["identity"],
                "round": round_idx,
                "target_error": float(target_error),
                "candidate_solution": json.dumps(current_S.tolist()),
                "candidate_box_feasible": box_feasible(current_S, config),
                "stage3_engine": stage3_summary["chosen_engine"],
                "stage3_actual_solver": stage3_summary.get("actual_solver"),
                "stage3_solver_success": stage3_summary.get("success"),
                "stage3_valid_delivery": stage3_summary.get("valid_delivery"),
                "stage3_max_box_violation": stage3_summary.get(
                    "max_box_violation"
                ),
                "stage3_engine_losses": json.dumps(
                    stage3_summary["engine_losses"], sort_keys=True
                ),
                "evaluation_loss": float(current_eval_loss),
                "delivered_solution_evaluation_loss": float(
                    exclusion_summary["delivered_solution_eval_loss"]
                ),
                "full_solution_source": exclusion_summary["full_solution_source"],
                "loo_search_objective": exclusion_summary["loo_search_objective"],
                "consistency_l1": consistency_l1,
                "raw_consistency_l1": raw_consistency_l1,
                "update_l1": update_l1,
                "eta_used": float(eta_used),
                "update_policy": config["update_policy"],
                "update_accepted": update_accepted,
                "current_state_accepted": current_state_accepted,
                "proposal_generated": proposal_generated,
                "gate_triggered": gate_triggered,
                "floor_rejection_streak": floor_rejection_streak,
                "no_progress_triggered": no_progress_triggered,
                "eval_loss_delta": (
                    None
                    if reference_eval_loss is None
                    else float(current_eval_loss - reference_eval_loss)
                ),
                "informative": informative,
                "solver_audit_status": solver_audit["status"],
                "solver_audit_agent_id": solver_audit["agent_id"],
                "solver_audit_full_loss": solver_audit[
                    "full_optimization_loss"
                ],
                "solver_audit_best_loo_loss": solver_audit[
                    "best_loo_optimization_loss"
                ],
                "solver_audit_improvement": solver_audit["improvement"],
                "solver_recovery_attempts": recovery_attempts,
                "solver_recovery_success": bool(
                    recovery_attempts > 0 and informative
                ),
                "stage3_requests_cumulative": counter["requests"]["stage3"],
                "stage3_input_samples_cumulative": counter["input_samples"]["stage3"],
                "contribution_requests_cumulative": counter["requests"]["contribution"],
                "contribution_input_samples_cumulative": counter["input_samples"]["contribution"],
            }
        )
        terminal_state = {
            "round": round_idx,
            "alpha": alpha_before.copy(),
            "solution": current_S.copy(),
            "q": None if q is None else q.copy(),
            "q_raw": None if q_raw is None else q_raw.copy(),
            "evaluation_loss": float(current_eval_loss),
            "exclusion_reports": copy.deepcopy(exclusion_reports),
            "solver_audit": copy.deepcopy(solver_audit),
            "informative": informative,
            "current_state_accepted": current_state_accepted,
        }
        for idx, report in enumerate(exclusion_reports):
            agent_rows.append(
                {
                    **config["identity"],
                    "round": round_idx,
                    "agent_id": report["agent_id"],
                    "is_poisoned": report["agent_id"] in poisoned_ids,
                    "reputation_share": float(reputation[idx]),
                    "optimization_before": float(alpha_before[idx]),
                    "marginal_contribution": float(report["marginal_contribution"]),
                    "positive_contribution": float(report["positive_contribution"]),
                    "leave_one_out_optimization_loss": float(
                        report["restricted_optimization_loss"]
                    ),
                    "leave_one_out_engine": report["restricted_engine"],
                    "leave_one_out_engine_losses": json.dumps(
                        report["restricted_engine_losses"], sort_keys=True
                    ),
                    "contribution_share": (
                        float(q_for_update[idx]) if informative else None
                    ),
                    "raw_contribution_share": (
                        float(q_raw[idx]) if informative else None
                    ),
                    "calibrated_contribution_share": (
                        float(q[idx]) if informative else None
                    ),
                    "optimization_after": float(alpha_after[idx]),
                }
            )

        server.alphas = [float(value) for value in alpha_after]
        pending_update = proposal_generated
        if (
            not informative
            or config["update_rate"] == 0
            or converged
            or stop_reason == "uninformative_accepted_state"
            or stop_reason == "guarded_no_progress"
        ):
            break

    if accepted_state is None:
        raise RuntimeError("Calibration finished without an accepted evaluated state.")

    # A proposal is only provisional until Stage 3 and L_eval evaluate it.  Always
    # settle and summarize the last accepted state, never an untested next alpha or
    # a final-round proposal rejected by the guarded evaluator.
    final_S = accepted_state["solution"]
    final_q = accepted_state["q"]
    final_q_raw = accepted_state["q_raw"]
    final_eval_loss = accepted_state["evaluation_loss"]
    exclusion_summary = accepted_state["exclusion_summary"]
    server.alphas = [float(value) for value in accepted_state["alpha"]]
    server.last_alpha_calibration_state = copy.deepcopy(accepted_state)
    server.last_stage3a_terminal_state = copy.deepcopy(terminal_state)

    if settle_payment:
        payment_summary, payment_rows = settle_calibration_state(
            server, accepted_state, reputation, config, counter=counter
        )
    else:
        payment_rows = []
        payment_summary = {
            "payment_status": "not_run_by_configuration",
            "payment_contribution_source": "",
            "payment_fallback_reason": "",
            "payment_contribution_weights": "[]",
            "payment_requests": 0,
            "payment_input_samples": 0,
            "payment_input_samples": 0,
            "paid_total": None,
        }

    final_alpha = np.asarray(server.alphas, dtype=float)
    selected_poison_weight = float(
        sum(
            final_alpha[idx]
            for idx, agent in enumerate(server.trusted_agents)
            if agent.agent_id in poisoned_ids
        )
    )
    counter["stage"] = "finished"
    summary = {
        **config["identity"],
        "selected_agent_ids": json.dumps(
            [agent.agent_id for agent in server.trusted_agents]
        ),
        "reputation_weights": json.dumps([float(x) for x in reputation]),
        "final_optimization_weights": json.dumps([float(x) for x in final_alpha]),
        "final_contribution_weights": json.dumps(
            [float(x) for x in final_q] if final_q is not None else []
        ),
        "final_raw_contribution_weights": json.dumps(
            [float(x) for x in final_q_raw] if final_q_raw is not None else []
        ),
        "rounds_executed": len(round_rows),
        "final_accepted_round": int(accepted_state["round"]),
        "converged": converged,
        "stop_reason": stop_reason,
        "final_eta_used": float(round_rows[-1]["eta_used"]),
        "updates_accepted": int(
            sum(bool(row["update_accepted"]) for row in round_rows)
        ),
        "gate_triggers": int(
            sum(bool(row["gate_triggered"]) for row in round_rows)
        ),
        "guarded_no_progress_patience": int(
            config.get("guarded_no_progress_patience", 0)
        ),
        "no_progress_triggered": no_progress_triggered,
        "stage3_seconds": counter["seconds"]["stage3"],
        "contribution_seconds": counter["seconds"]["contribution"],
        "payment_seconds": counter["seconds"]["payment"],
        "final_target_error": float(accepted_state["target_error"]),
        "final_solution": json.dumps(final_S.tolist()),
        "final_candidate_points": (
            json.dumps(np.asarray(server.I_list, dtype=float).tolist())
            if hasattr(server, "I_list") else None
        ),
        "final_box_feasible": box_feasible(final_S, config),
        "final_raw_marginal_contributions": json.dumps(
            accepted_state["marginal_contribution"].tolist()
        ),
        "final_evaluation_loss": float(final_eval_loss),
        "final_delivered_solution_evaluation_loss": float(
            exclusion_summary["delivered_solution_eval_loss"]
        ),
        "final_full_solution_source": exclusion_summary["full_solution_source"],
        "final_loo_search_objective": exclusion_summary["loo_search_objective"],
        "final_consistency_l1": accepted_state["consistency_l1"],
        "final_raw_consistency_l1": accepted_state["raw_consistency_l1"],
        "final_solver_audit_status": accepted_state["solver_audit_status"],
        "final_stage3_actual_solver": accepted_state["stage3_summary"].get(
            "actual_solver"
        ),
        "final_stage3_solver_success": accepted_state["stage3_summary"].get(
            "success"
        ),
        "final_stage3_valid_delivery": accepted_state["stage3_summary"].get(
            "valid_delivery"
        ),
        "solver_recovery_attempts": int(
            sum(row["solver_recovery_attempts"] for row in round_rows)
        ),
        "solver_recovery_successes": int(
            sum(bool(row["solver_recovery_success"]) for row in round_rows)
        ),
        "selected_poison_weight": selected_poison_weight,
        "stage3_requests": counter["requests"]["stage3"],
        "stage3_input_samples": counter["input_samples"]["stage3"],
        "contribution_requests": counter["requests"]["contribution"],
        "contribution_input_samples": counter["input_samples"]["contribution"],
        **payment_summary,
    }
    return summary, round_rows, agent_rows, payment_rows


def calibrate_stabilized_condition(server, config, reputation, poisoned_ids):
    """Run Stage 3A, prune at most one agent, then restart Stage 3A.

    The existing experiment path intentionally keeps a fixed coalition.  This
    opt-in orchestrator implements the full Stage 3A/3B nesting and settles only
    the final stable coalition.
    """
    initial_ids = [agent.agent_id for agent in server.trusted_agents]
    reputation_by_id = {
        agent.agent_id: float(reputation[index])
        for index, agent in enumerate(server.trusted_agents)
    }
    configured_max_rounds = config.get("max_coalition_rounds")
    max_rounds = (
        max(0, len(server.trusted_agents) - 1)
        if configured_max_rounds is None
        else configured_max_rounds
    )
    pruning_epsilon = config.get("pruning_tolerance", 1e-6)
    all_round_rows = []
    all_agent_rows = []
    pruning_log = []
    total_stage3_requests = 0
    total_contribution_requests = 0
    total_stage3_input_samples = 0
    total_contribution_input_samples = 0

    final_summary = None
    final_reputation = None
    final_config = None
    for coalition_round in range(max_rounds + 1):
        current_ids = [agent.agent_id for agent in server.trusted_agents]
        final_reputation = normalize(
            [reputation_by_id[agent_id] for agent_id in current_ids]
        )
        final_config = copy.deepcopy(config)
        final_config["condition_seed"] = config["condition_seed"] + coalition_round
        final_config["identity"] = {
            **config["identity"],
            "coalition_round": coalition_round,
        }

        summary, round_rows, agent_rows, _ = calibrate_one_condition(
            server,
            final_config,
            final_reputation,
            poisoned_ids,
            settle_payment=False,
        )
        all_round_rows.extend(round_rows)
        all_agent_rows.extend(agent_rows)
        total_stage3_requests += summary["stage3_requests"]
        total_contribution_requests += summary["contribution_requests"]
        total_stage3_input_samples += summary.get("stage3_input_samples", 0)
        total_contribution_input_samples += summary.get("contribution_input_samples", 0)
        final_summary = summary

        terminal = server.last_stage3a_terminal_state
        terminal_audit_status = terminal["solver_audit"]["status"]
        # An audited-uninformative terminal state has passed the L_opt solver
        # audit, so its raw C values are the evidence Stage 3B must inspect.
        # It remains rejected for alpha calibration and settlement; ordinary
        # rejected candidates still fall back to the last accepted snapshot.
        if terminal["current_state_accepted"]:
            stage3b_state = terminal
            pruning_state_source = "terminal_accepted"
        elif terminal_audit_status == "audited_uninformative":
            stage3b_state = terminal
            pruning_state_source = "terminal_audited_uninformative"
        else:
            stage3b_state = server.last_alpha_calibration_state
            pruning_state_source = "accepted_snapshot"
        reports = stage3b_state["exclusion_reports"]
        candidate = HostServer.negative_contributor_candidate(
            reports, epsilon=pruning_epsilon
        )
        log = {
            "coalition_round": coalition_round,
            "coalition_ids": current_ids,
            "stage3a_stop_reason": summary["stop_reason"],
            "solver_audit_status": terminal_audit_status,
            "pruning_state_source": pruning_state_source,
            "pruning_candidate_agent_id": (
                None if candidate is None else candidate["agent_id"]
            ),
            "pruning_candidate_contribution": (
                None if candidate is None else candidate["marginal_contribution"]
            ),
            "removed_agent_id": None,
            "status": "stable",
        }

        if candidate is None or len(server.trusted_agents) <= 1:
            if candidate is not None:
                log["status"] = "stopped_single_agent"
            pruning_log.append(log)
            break
        if coalition_round >= max_rounds:
            log["status"] = "max_coalition_rounds"
            pruning_log.append(log)
            break

        removed = server._remove_trusted_agent_at(candidate["index"])
        log["removed_agent_id"] = removed["agent_id"]
        log["status"] = "removed_negative_contributor"
        pruning_log.append(log)

    accepted_state = server.last_alpha_calibration_state
    if config.get("settle_payment", True):
        payment_summary, payment_rows = settle_calibration_state(
            server, accepted_state, final_reputation, final_config,
            counter=server.last_query_counter,
        )
    else:
        payment_summary, payment_rows = ({
            "payment_status": "not_run_by_configuration",
            "payment_contribution_source": "",
            "payment_fallback_reason": "",
            "payment_contribution_weights": "[]",
            "payment_requests": 0,
            "paid_total": None,
        }, [])
    final_summary.update(payment_summary)
    final_summary.update(
        {
            "initial_coalition_ids": json.dumps(initial_ids),
            "final_coalition_ids": json.dumps(
                [agent.agent_id for agent in server.trusted_agents]
            ),
            "coalition_rounds_executed": len(pruning_log),
            "pruning_log": json.dumps(pruning_log, sort_keys=True),
            "stage3_requests": total_stage3_requests,
            "contribution_requests": total_contribution_requests,
            "stage3_input_samples": total_stage3_input_samples,
            "contribution_input_samples": total_contribution_input_samples,
        }
    )
    return final_summary, all_round_rows, all_agent_rows, payment_rows, pruning_log


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def agent_data_controls(args):
    """Build deterministic one-factor data controls for P5 scenarios."""
    count = int(args.num_agents)
    heterogeneity = float(getattr(args, "data_heterogeneity", 0.0))
    imbalance = float(getattr(args, "sample_imbalance_ratio", 1.0))
    low_quality_ratio = float(getattr(args, "low_quality_ratio", 0.0))
    low_quality_noise = float(getattr(args, "low_quality_noise_std", 0.3))
    shifts = np.linspace(-heterogeneity, heterogeneity, count).tolist()
    if np.isclose(imbalance, 1.0):
        sample_counts = [int(args.samples_per_agent)] * count
    else:
        factors = np.geomspace(imbalance ** -0.5, imbalance ** 0.5, count)
        factors = factors / np.mean(factors)
        sample_counts = [
            max(3, int(round(args.samples_per_agent * factor)))
            for factor in factors
        ]
    noise_stds = [0.05] * count
    low_quality_count = int(round(count * low_quality_ratio))
    for index in range(count - low_quality_count, count):
        noise_stds[index] = low_quality_noise
    return sample_counts, shifts, noise_stds


def build_base_server(args, seed, dimension, poison_ratio, target):
    sample_counts, feature_shifts, noise_stds = agent_data_controls(args)
    training_started = time.perf_counter()
    agents, records = build_agents(
        num_agents=args.num_agents,
        n_features=dimension,
        poison_ratio=poison_ratio,
        samples_per_agent=args.samples_per_agent,
        epochs=args.epochs,
        seed=seed,
        return_data=True,
        sample_counts=sample_counts,
        feature_shifts=feature_shifts,
        noise_stds=noise_stds,
    )
    training_seconds = time.perf_counter() - training_started
    prep_counter = attach_query_counter(agents)
    server = HostServer(
        target_T=target,
        n_features=dimension,
        total_budget=args.total_budget,
        test_seed=seed + 7919,
        n_test=args.n_test,
        constraint_mode=getattr(args, "constraint_mode", "legacy_unconstrained"),
        feasible_lower=args.feasible_lower,
        feasible_upper=args.feasible_upper,
        feasibility_tolerance=1e-8,
    )
    prep_counter["stage"] = "qualification"
    qualification_started = time.perf_counter()
    server.phase1_filter_agents(
        agents,
        mse_threshold=args.mse_threshold,
        budget_fraction=args.budget_fraction,
        diversity_eta=args.diversity_eta,
        min_selection_score=args.min_selection_score,
        k_api=args.k_api,
        k_red=args.k_red,
        enable_inverse_check=not args.disable_inverse_check,
        inverse_target=args.inverse_target,
        inverse_loss_threshold=args.inverse_loss_threshold,
        inverse_steps=args.inverse_steps,
        feasible_lower=args.feasible_lower,
        feasible_upper=args.feasible_upper,
    )
    qualification_seconds = time.perf_counter() - qualification_started
    server.all_agents = agents
    if getattr(args, "clean_evaluator_noise", False):
        from dataset import generate_agent_datasets
        clean_records = generate_agent_datasets(num_agents=args.num_agents,
            n_samples_per_agent=args.samples_per_agent, n_features=dimension, seed=seed,
            poison_ratio=poison_ratio, sample_counts=sample_counts,
            feature_shifts=feature_shifts, noise_stds=[0.05]*args.num_agents)
        for record, clean in zip(records, clean_records):
            record["clean_y"] = clean["clean_y"]
            record["evaluator_noise_std"] = 0.05
    server.study_records = records
    prep_counter["stage"] = "proposal"
    proposal_started = time.perf_counter()
    server.phase2_collect_proposals()
    proposal_seconds = time.perf_counter() - proposal_started
    prep_counter["stage"] = "finished"
    server.preparation_costs = {
        "training_seconds": training_seconds,
        "qualification_seconds": qualification_seconds,
        "proposal_seconds": proposal_seconds,
        "qualification_requests": prep_counter["requests"]["qualification"],
        "qualification_input_samples": prep_counter["input_samples"]["qualification"],
        "proposal_requests": prep_counter["requests"]["proposal"],
        "proposal_input_samples": prep_counter["input_samples"]["proposal"],
    }
    server.preparation_counter = prep_counter
    return server


def run(args):
    args = copy.deepcopy(args)
    if getattr(args, "no_alpha_update", False):
        if args.enable_coalition_stabilization:
            raise ValueError("--no-alpha-update requires a fixed coalition; disable stabilization.")
        args.update_rates = [0.0]
        args.update_policies = ["fixed"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("Use an empty output directory to preserve previous experiments.")
    write_run_manifest(output_dir, args, "running")
    summaries = []
    rounds = []
    agents = []
    payments = []
    pruning_rows = []
    condition_index = 0

    scenarios = product(args.seeds, args.dimensions, args.poison_ratios, args.targets)
    conditions = comparison_conditions(args)
    for seed, dimension, poison_ratio, target in scenarios:
        num_poisoned = int(args.num_agents * poison_ratio)
        poisoned_ids = set(
            range(args.num_agents - num_poisoned + 1, args.num_agents + 1)
        ) if num_poisoned else set()
        base_server = build_base_server(args, seed, dimension, poison_ratio, target)
        qualification_failure = ""
        if args.fixed_participant_ids:
            qualified_ids = {
                int(row["agent_id"]) for row in base_server.phase1_report
                if row.get("reason") in {"selected", "not_selected_by_coalition_rule"}
            }
            missing_ids = sorted(set(args.fixed_participant_ids) - qualified_ids)
            if missing_ids:
                qualification_failure = (
                    "fixed participants did not pass qualification: "
                    + json.dumps(missing_ids)
                )
        if args.coalition_mode == "controlled":
            configure_controlled_coalition(
                base_server,
                poisoned_ids,
                args.controlled_coalition_size,
                args.controlled_poison_count,
            )
        if args.fixed_participant_ids and not qualification_failure:
            import torch
            by_id = {a.agent_id: a for a in base_server.all_agents}
            base_server.trusted_agents = [by_id[i] for i in args.fixed_participant_ids]
            base_server.alphas = agent_reputation_weights(base_server, base_server.trusted_agents).tolist()
            base_server.I_list = []
            base_server.preparation_counter["stage"] = "fixed_participant_proposal"
            fixed_proposal_started = time.perf_counter()
            for agent in base_server.trusted_agents:
                torch.manual_seed(seed * 100000 + agent.agent_id)
                base_server.I_list.append(agent.infer_parameters_D(target, steps=args.inverse_steps))
            base_server.preparation_costs["fixed_proposal_seconds"] = (
                time.perf_counter() - fixed_proposal_started
            )
            base_server.preparation_costs["fixed_proposal_requests"] = (
                base_server.preparation_counter["requests"]["fixed_participant_proposal"]
            )
            base_server.preparation_costs["fixed_proposal_input_samples"] = (
                base_server.preparation_counter["input_samples"]["fixed_participant_proposal"]
            )
            base_server.preparation_counter["stage"] = "finished"
        reputation = normalize(base_server.alphas)
        scenario_identity = {
            "seed": seed,
            "dimension": dimension,
            "poison_ratio": poison_ratio,
            "target": target,
        }
        evaluator_started = time.perf_counter()
        external_evaluator, artifact_dir, evaluator_metrics = save_scenario_artifacts(
            output_dir,
            scenario_identity,
            base_server.all_agents,
            base_server.study_records,
            evaluator_epochs=args.external_evaluator_epochs,
            validation_fraction=args.external_validation_fraction,
            test_fraction=args.external_test_fraction,
        )
        evaluator_seconds = time.perf_counter() - evaluator_started
        shared_preparation = getattr(base_server, "preparation_costs", {})
        del base_server.study_records

        for condition in conditions:
            initialization = condition["initialization"]
            initialization_replicate = condition["initialization_replicate"]
            evaluator = condition["evaluator"]
            update_rate = condition["update_rate"]
            update_policy = condition["update_policy"]
            identity = {
                "method_label": condition["method_label"],
                "seed": seed,
                "dimension": dimension,
                "poison_ratio": poison_ratio,
                "target": target,
                "initialization": initialization,
                "initialization_replicate": initialization_replicate,
                "evaluator": evaluator,
                "update_rate": update_rate,
                "update_policy": update_policy,
                "coalition_mode": args.coalition_mode,
                "controlled_coalition_size": (
                    args.controlled_coalition_size
                    if args.coalition_mode == "controlled"
                    else ""
                ),
                "controlled_poison_count": (
                    args.controlled_poison_count
                    if args.coalition_mode == "controlled"
                    else ""
                ),
                "data_heterogeneity": args.data_heterogeneity,
                "sample_imbalance_ratio": args.sample_imbalance_ratio,
                "low_quality_ratio": args.low_quality_ratio,
                "low_quality_noise_std": args.low_quality_noise_std,
            }
            config = {
                "identity": identity,
                "condition_seed": args.condition_seed + condition_index,
                "initialization": initialization,
                "evaluator": evaluator,
                "update_rate": update_rate,
                "update_policy": update_policy,
                "max_rounds": args.max_calibration_rounds,
                "tolerance": args.alpha_tolerance,
                "minimum_update_rate": args.minimum_update_rate,
                "evaluation_tolerance": args.evaluation_tolerance,
                "guarded_no_progress_patience": args.guarded_no_progress_patience,
                "exploration_mass": args.exploration_mass,
                "solver_audit_tolerance": args.solver_audit_tolerance,
                "max_solver_recoveries": args.max_solver_recoveries,
                "pruning_tolerance": args.pruning_tolerance,
                "max_coalition_rounds": args.max_coalition_rounds,
                "trim_fraction": args.trim_fraction,
                "optimizer": args.optimizer,
                "custom_iterations": args.custom_iterations,
                "payment_reputation_mix": args.payment_reputation_mix,
                "settle_payment": getattr(args, "settle_payment", True),
                "feasible_lower": args.feasible_lower,
                "feasible_upper": args.feasible_upper,
            }
            condition_index += 1
            condition_started = time.perf_counter()
            if qualification_failure:
                shared = getattr(base_server, "preparation_costs", {})
                summaries.append({**identity, "execution_status": "completed",
                    "failure_stage": "qualification", "failure_reason": qualification_failure,
                    "delivery_status": "qualification_failure", "payment_status": "not_run_by_configuration",
                    "selected_agent_ids": json.dumps(args.fixed_participant_ids),
                    "final_solution": "", "final_box_feasible": "", "max_box_violation": "",
                    "external_global_error": "", "final_target_error": "",
                    "stage3_requests": 0, "contribution_requests": 0, "payment_requests": 0,
                    "stage3_input_samples": 0, "contribution_input_samples": 0,
                    "payment_input_samples": 0,
                    "stage3_seconds": 0, "contribution_seconds": 0, "payment_seconds": 0,
                    "shared_training_seconds": shared.get("training_seconds"),
                    "shared_qualification_seconds": shared.get("qualification_seconds"),
                    "shared_proposal_seconds": shared.get("proposal_seconds"),
                    "shared_fixed_proposal_seconds": 0,
                    "shared_evaluator_seconds": evaluator_seconds,
                    "shared_qualification_requests": shared.get("qualification_requests"),
                    "shared_proposal_requests": shared.get("proposal_requests"),
                    "shared_fixed_proposal_requests": 0,
                    "shared_qualification_input_samples": shared.get("qualification_input_samples"),
                    "shared_proposal_input_samples": shared.get("proposal_input_samples"),
                    "shared_fixed_proposal_input_samples": 0,
                    "condition_seconds": time.perf_counter() - condition_started,
                    "scenario_artifact_dir": str(artifact_dir.relative_to(output_dir))})
                write_csv(output_dir / "summary.csv", summaries)
                continue
            server = copy.deepcopy(base_server)
            if args.enable_coalition_stabilization:
                result = calibrate_stabilized_condition(
                    server, config, reputation.copy(), poisoned_ids
                )
                summary, round_rows, agent_rows, payment_rows, condition_pruning = result
                for row in condition_pruning:
                    pruning_rows.append({**identity, **row})
            else:
                try:
                    result = calibrate_one_condition(
                        server, config, reputation.copy(), poisoned_ids,
                        settle_payment=getattr(args, "settle_payment", True),
                    )
                    summary, round_rows, agent_rows, payment_rows = result
                except ValueError as exc:
                    if args.constraint_mode != "box_constrained":
                        raise
                    failed_counter = getattr(server, "last_query_counter", None)
                    summaries.append({**identity, "execution_status": "completed",
                        "failure_stage": "stage3_or_contribution", "failure_reason": str(exc),
                        "delivery_status": "solver_failure", "payment_status": (
                            "not_run_by_configuration" if not getattr(args, "settle_payment", True)
                            else "not_reached"),
                        "stop_reason": str(exc), "final_solution": "", "external_global_error": "",
                        "final_target_error": "", "final_box_feasible": "",
                        "max_box_violation": "",
                        "stage3_requests": (failed_counter["requests"]["stage3"]
                            if failed_counter else ""),
                        "stage3_input_samples": (failed_counter["input_samples"]["stage3"]
                            if failed_counter else ""),
                        "contribution_requests": (failed_counter["requests"]["contribution"]
                            if failed_counter else ""),
                        "contribution_input_samples": (failed_counter["input_samples"]["contribution"]
                            if failed_counter else ""),
                        "payment_requests": 0 if failed_counter else "",
                        "payment_input_samples": 0 if failed_counter else "",
                        "stage3_seconds": (failed_counter["seconds"]["stage3"]
                            if failed_counter else ""),
                        "contribution_seconds": (failed_counter["seconds"]["contribution"]
                            if failed_counter else ""),
                        "payment_seconds": 0 if failed_counter else "",
                        "condition_seconds": time.perf_counter()-condition_started,
                        "scenario_artifact_dir": str(artifact_dir.relative_to(output_dir))})
                    write_csv(output_dir / "summary.csv", summaries)
                    write_csv(output_dir / "rounds.csv", rounds)
                    write_csv(output_dir / "agents.csv", agents)
                    write_csv(output_dir / "payments.csv", payments)
                    write_csv(output_dir / "pruning.csv", pruning_rows)
                    continue
                except Exception as exc:
                    failed_counter = getattr(server, "last_query_counter", None)
                    summaries.append({**identity, "execution_status": "interrupted",
                        "failure_stage": "unexpected_exception",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                        "delivery_status": "not_assessed", "payment_status": "not_reached",
                        "final_solution": "", "external_global_error": "",
                        "final_target_error": "", "final_box_feasible": "",
                        "stage3_requests": (failed_counter["requests"]["stage3"]
                            if failed_counter else ""),
                        "stage3_input_samples": (failed_counter["input_samples"]["stage3"]
                            if failed_counter else ""),
                        "contribution_requests": (failed_counter["requests"]["contribution"]
                            if failed_counter else ""),
                        "contribution_input_samples": (failed_counter["input_samples"]["contribution"]
                            if failed_counter else ""),
                        "payment_requests": "", "condition_seconds": time.perf_counter()-condition_started,
                        "scenario_artifact_dir": str(artifact_dir.relative_to(output_dir))})
                    write_csv(output_dir / "summary.csv", summaries)
                    write_csv(output_dir / "rounds.csv", rounds)
                    write_csv(output_dir / "agents.csv", agents)
                    write_csv(output_dir / "payments.csv", payments)
                    write_csv(output_dir / "pruning.csv", pruning_rows)
                    write_run_manifest(output_dir, args, "interrupted")
                    raise
            final_solution = np.asarray(json.loads(summary["final_solution"]), dtype=float)
            prediction, error = score_solution(external_evaluator, final_solution, target)
            summary.update({
                "shared_training_seconds": shared_preparation.get("training_seconds"),
                "shared_qualification_seconds": shared_preparation.get("qualification_seconds"),
                "shared_proposal_seconds": shared_preparation.get("proposal_seconds"),
                "shared_fixed_proposal_seconds": shared_preparation.get("fixed_proposal_seconds"),
                "shared_evaluator_seconds": evaluator_seconds,
                "shared_qualification_requests": shared_preparation.get("qualification_requests"),
                "shared_proposal_requests": shared_preparation.get("proposal_requests"),
                "shared_fixed_proposal_requests": shared_preparation.get("fixed_proposal_requests"),
                "shared_qualification_input_samples": shared_preparation.get("qualification_input_samples"),
                "shared_proposal_input_samples": shared_preparation.get("proposal_input_samples"),
                "shared_fixed_proposal_input_samples": shared_preparation.get("fixed_proposal_input_samples"),
                "condition_seconds": time.perf_counter() - condition_started,
                "constraint_mode": args.constraint_mode,
                "feasibility_tolerance": 1e-8,
                "max_box_violation": server.box_violation(final_solution),
                "delivery_status": "valid" if box_feasible(final_solution, config) else "box_infeasible",
                "external_global_prediction": prediction,
                "external_global_error": error,
                "external_truth_disagreement": abs(
                    prediction - true_function(final_solution)
                ),
                "external_evaluator_test_mse": evaluator_metrics["test_mse"],
                "external_evaluator_hard_test_mse": evaluator_metrics["hard_test_mse"],
                "external_evaluator_test_truth_mse": evaluator_metrics[
                    "test_truth_mse"
                ],
                "external_evaluator_hard_test_truth_mse": evaluator_metrics[
                    "hard_test_truth_mse"
                ],
                "external_evaluator_target_near_samples": evaluator_metrics[
                    "target_near_samples"
                ],
                "external_evaluator_target_near_truth_mse": evaluator_metrics[
                    "target_near_truth_mse"
                ],
                "external_evaluator_target_near_sparse": evaluator_metrics[
                    "target_near_sparse"
                ],
                "scenario_artifact_dir": str(artifact_dir.relative_to(output_dir)),
            })
            for row in round_rows:
                candidate = np.asarray(json.loads(row["candidate_solution"]), dtype=float)
                round_prediction, round_error = score_solution(
                    external_evaluator, candidate, target
                )
                row["external_global_prediction"] = round_prediction
                row["external_global_error"] = round_error
                row["external_truth_disagreement"] = abs(
                    round_prediction - true_function(candidate)
                )
            summaries.append(summary)
            rounds.extend(round_rows)
            agents.extend(agent_rows)
            payments.extend(payment_rows)
            # Persist each completed condition, including rejected candidates.
            write_csv(output_dir / "summary.csv", summaries)
            write_csv(output_dir / "rounds.csv", rounds)
            write_csv(output_dir / "agents.csv", agents)
            write_csv(output_dir / "payments.csv", payments)
            write_csv(output_dir / "pruning.csv", pruning_rows)

    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "rounds.csv", rounds)
    write_csv(output_dir / "agents.csv", agents)
    write_csv(output_dir / "payments.csv", payments)
    write_csv(output_dir / "pruning.csv", pruning_rows)
    write_run_manifest(output_dir, args, "complete")
    print(f"Alpha-weight experiments complete: {len(summaries)} conditions")
    print(f"Outputs written to: {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare ex-ante reputation, iterative Stage 3 optimization weights, "
            "and ex-post Subspace Exclusion contribution shares."
        )
    )
    parser.add_argument("--output-dir", default="alpha_weight_outputs")
    parser.add_argument("--constraint-mode", choices=["legacy_unconstrained", "box_constrained"], default="legacy_unconstrained")
    parser.add_argument("--fixed-participant-ids", type=lambda v: parse_csv(v, int), default=[])
    parser.add_argument("--clean-evaluator-noise", action="store_true")
    parser.add_argument(
        "--comparison-suite", choices=["custom", "p2", "p5"], default="custom",
        help=(
            "p2 runs A/B/C/D; p5 runs the current static A baseline and "
            "Guarded D research comparator."
        ),
    )
    parser.add_argument(
        "--no-alpha-update", action="store_true",
        help="Run one calibration round per initialization with fixed alpha and no pruning. "
             "Overrides update rates/policies; contribution audit and payment still run.",
    )
    parser.add_argument(
        "--no-settle-payment", dest="settle_payment", action="store_false",
        help="Skip settlement and record payment_status=not_run_by_configuration.",
    )
    parser.set_defaults(settle_payment=True)
    parser.add_argument("--seeds", type=lambda x: parse_csv(x, int), default=[0])
    parser.add_argument("--external-evaluator-epochs", type=int, default=100)
    parser.add_argument("--external-validation-fraction", type=float, default=0.15)
    parser.add_argument("--external-test-fraction", type=float, default=0.15)
    parser.add_argument("--dimensions", type=lambda x: parse_csv(x, int), default=[2])
    parser.add_argument(
        "--poison-ratios", type=lambda x: parse_csv(x, float), default=[0.0, 0.3]
    )
    parser.add_argument("--targets", type=lambda x: parse_csv(x, float), default=[0.0])
    parser.add_argument(
        "--initializations",
        type=lambda x: parse_csv(x, str),
        choices=None,
        default=["stage0", "uniform"],
        help="Comma-separated: stage0,uniform,dirichlet,reversed",
    )
    parser.add_argument(
        "--initialization-replicates",
        type=int,
        default=1,
        help="Number of independent Dirichlet starts per scenario.",
    )
    parser.add_argument(
        "--evaluators",
        type=lambda x: parse_csv(x, str),
        default=["trimmed"],
        help=(
            "Comma-separated: trimmed (primary), uniform/median/reputation "
            "ablations, optimization circular baseline"
        ),
    )
    parser.add_argument(
        "--update-rates", type=lambda x: parse_csv(x, float), default=[0.0, 0.3]
    )
    parser.add_argument(
        "--update-policies",
        type=lambda x: parse_csv(x, str),
        default=["guarded"],
        help="Comma-separated: fixed,adaptive,guarded",
    )
    parser.add_argument("--max-calibration-rounds", type=int, default=6)
    parser.add_argument("--alpha-tolerance", type=float, default=1e-3)
    parser.add_argument("--minimum-update-rate", type=float, default=0.05)
    parser.add_argument("--evaluation-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--guarded-no-progress-patience",
        type=int,
        default=2,
        help=(
            "Stop a guarded run after this many consecutive rejected proposals "
            "at the minimum update rate; 0 restores the previous max-round behavior."
        ),
    )
    parser.add_argument("--exploration-mass", type=float, default=0.02)
    parser.add_argument("--solver-audit-tolerance", type=float, default=1e-6)
    parser.add_argument("--max-solver-recoveries", type=int, default=1)
    parser.add_argument(
        "--enable-coalition-stabilization",
        action="store_true",
        help="Enable Stage 3B single-agent pruning and restart Stage 3A.",
    )
    parser.add_argument("--pruning-tolerance", type=float, default=1e-6)
    parser.add_argument("--max-coalition-rounds", type=int, default=None)
    parser.add_argument("--trim-fraction", type=float, default=0.2)
    parser.add_argument("--payment-reputation-mix", type=float, default=0.4)
    parser.add_argument("--condition-seed", type=int, default=20260904)
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--samples-per-agent", type=int, default=500)
    parser.add_argument("--data-heterogeneity", type=float, default=0.0)
    parser.add_argument("--sample-imbalance-ratio", type=float, default=1.0)
    parser.add_argument("--low-quality-ratio", type=float, default=0.0)
    parser.add_argument("--low-quality-noise-std", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--total-budget", type=float, default=10.0)
    parser.add_argument("--mse-threshold", type=float, default=0.1)
    parser.add_argument("--inverse-target", type=float, default=0.0)
    parser.add_argument("--inverse-loss-threshold", type=float, default=0.1)
    parser.add_argument("--inverse-steps", type=int, default=500)
    parser.add_argument("--feasible-lower", type=float, default=-1.0)
    parser.add_argument("--feasible-upper", type=float, default=1.0)
    parser.add_argument("--disable-inverse-check", action="store_true")
    parser.add_argument("--budget-fraction", type=float, default=0.8)
    parser.add_argument("--diversity-eta", type=float, default=0.5)
    parser.add_argument("--min-selection-score", type=float, default=0.0)
    parser.add_argument("--k-api", type=int, default=None)
    parser.add_argument("--k-red", type=int, default=None)
    parser.add_argument(
        "--coalition-mode",
        choices=["stage1", "controlled"],
        default="stage1",
        help="Use the normal Stage 1 coalition or force a controlled coalition.",
    )
    parser.add_argument("--controlled-coalition-size", type=int, default=5)
    parser.add_argument("--controlled-poison-count", type=int, default=1)
    parser.add_argument(
        "--optimizer",
        choices=["best_of", "custom", "bfgs"],
        default="best_of",
    )
    parser.add_argument("--custom-iterations", type=int, default=30)
    return parser


def validate_args(args):
    valid_initializations = {"stage0", "uniform", "dirichlet", "reversed"}
    valid_evaluators = {"optimization", "reputation", "uniform", "median", "trimmed"}
    valid_policies = {"fixed", "adaptive", "guarded"}
    unknown_initializations = set(args.initializations) - valid_initializations
    unknown_evaluators = set(args.evaluators) - valid_evaluators
    unknown_policies = set(args.update_policies) - valid_policies
    if unknown_initializations:
        raise ValueError(f"Unknown initializations: {sorted(unknown_initializations)}")
    if unknown_evaluators:
        raise ValueError(f"Unknown evaluators: {sorted(unknown_evaluators)}")
    if unknown_policies:
        raise ValueError(f"Unknown update policies: {sorted(unknown_policies)}")
    if any(rate < 0 or rate > 1 for rate in args.update_rates):
        raise ValueError("Every update rate must be in [0, 1].")
    if args.max_calibration_rounds < 1:
        raise ValueError("max-calibration-rounds must be positive.")
    if args.external_evaluator_epochs < 1:
        raise ValueError("external-evaluator-epochs must be positive.")
    if (
        args.external_validation_fraction <= 0
        or args.external_test_fraction <= 0
        or args.external_validation_fraction + args.external_test_fraction >= 1
    ):
        raise ValueError(
            "External validation/test fractions must be positive and sum to less than one."
        )
    if args.comparison_suite in {"p2", "p5"} and len(args.evaluators) != 1:
        raise ValueError("The p2/p5 suite requires exactly one shared evaluator.")
    if args.comparison_suite == "p2" and args.no_alpha_update:
        raise ValueError("The p2 suite already defines its static and dynamic methods.")
    if args.comparison_suite == "p2" and args.enable_coalition_stabilization:
        raise ValueError("The p2 suite requires a fixed coalition with pruning disabled.")
    if args.comparison_suite == "p5" and args.enable_coalition_stabilization:
        raise ValueError("The p5 suite follows the P3 decision to keep pruning disabled.")
    if args.initialization_replicates < 1:
        raise ValueError("initialization-replicates must be positive.")
    if not 0 <= args.payment_reputation_mix <= 1:
        raise ValueError("payment-reputation-mix must be in [0, 1].")
    if not 0 <= args.exploration_mass < 1:
        raise ValueError("exploration-mass must be in [0, 1).")
    if not 0 < args.minimum_update_rate <= 1:
        raise ValueError("minimum-update-rate must be in (0, 1].")
    if args.evaluation_tolerance < 0:
        raise ValueError("evaluation-tolerance must be non-negative.")
    if args.guarded_no_progress_patience < 0:
        raise ValueError("guarded-no-progress-patience must be non-negative.")
    if args.solver_audit_tolerance < 0:
        raise ValueError("solver-audit-tolerance must be non-negative.")
    if args.max_solver_recoveries < 0:
        raise ValueError("max-solver-recoveries must be non-negative.")
    if args.pruning_tolerance < 0:
        raise ValueError("pruning-tolerance must be non-negative.")
    if args.max_coalition_rounds is not None and args.max_coalition_rounds < 0:
        raise ValueError("max-coalition-rounds must be non-negative.")
    if args.controlled_coalition_size < 1:
        raise ValueError("controlled-coalition-size must be positive.")
    if args.controlled_poison_count < 0:
        raise ValueError("controlled-poison-count must be non-negative.")
    if args.coalition_mode == "controlled" and (
        args.controlled_poison_count > args.controlled_coalition_size
    ):
        raise ValueError("controlled-poison-count cannot exceed coalition size.")
    if not 0 <= args.data_heterogeneity < 1:
        raise ValueError("data-heterogeneity must be in [0, 1).")
    if args.sample_imbalance_ratio < 1:
        raise ValueError("sample-imbalance-ratio must be at least 1.")
    if not 0 <= args.low_quality_ratio <= 1:
        raise ValueError("low-quality-ratio must be in [0, 1].")
    if args.low_quality_noise_std < 0:
        raise ValueError("low-quality-noise-std must be non-negative.")
    if args.feasible_lower >= args.feasible_upper:
        raise ValueError("feasible-lower must be less than feasible-upper.")
    if len(set(args.fixed_participant_ids)) != len(args.fixed_participant_ids):
        raise ValueError("fixed-participant-ids must be unique.")
    if any(agent_id < 1 or agent_id > args.num_agents
           for agent_id in args.fixed_participant_ids):
        raise ValueError("fixed-participant-ids must refer to configured agents.")


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    validate_args(parsed_args)
    run(parsed_args)
