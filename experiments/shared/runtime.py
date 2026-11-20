"""Shared preparation and execution helpers used by the retained experiments."""
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
import numpy as np
from agent_client import AgentNode
from host_server import HostServer, payment_contributions
from research_evaluation import save_scenario_artifacts, score_solution
from run_experiments import build_agents, true_function


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


def parse_csv(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


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
