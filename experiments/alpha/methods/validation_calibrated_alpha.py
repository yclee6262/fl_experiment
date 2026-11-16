"""Select an ex-ante alpha candidate using only Stage 0 labels and local APIs.

The pooled evaluator is intentionally absent from every function in this file.
Each candidate must already have a feasible Stage 3 solution.
"""

import numpy as np
from itertools import combinations


CANDIDATE_METHODS = (
    "A_uniform", "B_stage0", "mse_power_1p5",
    "corr_precision_0p5", "target_local_precision",
)
LOCAL_BIAS_BANDWIDTH = 0.25
MARGINAL_SUPPORT_BOOTSTRAPS = 512
MARGINAL_SUPPORT_THRESHOLD = 0.9
CONDITIONAL_TARGET_RATIO = 0.8


def conditional_target_trigger(internal_scores, blind_residual_mse,
                               threshold=CONDITIONAL_TARGET_RATIO):
    """Add a sixth solve when five-way internal error is large vs Stage 0 noise."""
    if not internal_scores or not np.isfinite(blind_residual_mse) or blind_residual_mse < 0:
        raise ValueError("Invalid Stage 0 residual evidence")
    values = np.asarray([float(v["internal_target_error"])
                         for v in internal_scores.values()])
    if not np.isfinite(values).all() or np.any(values < 0) or not 0 < threshold:
        raise ValueError("Invalid five-way internal scores or threshold")
    best = float(values.min())
    rmse = float(np.sqrt(blind_residual_mse))
    ratio = best / max(rmse, 1e-12)
    return ratio >= threshold, {
        "ratio": ratio, "best_five_internal_error": best,
        "blind_residual_rmse": rmse, "threshold": threshold,
    }


def choose_marginal_or_target_fifth(row_gains, seed,
                                    threshold=MARGINAL_SUPPORT_THRESHOLD,
                                    bootstraps=MARGINAL_SUPPORT_BOOTSTRAPS):
    """Choose the fifth Stage 3 solve from Stage 0 per-row marginal gains.

    The statistic is the positive-gain weighted bootstrap probability that
    each agent's marginal gain remains positive. It uses no external model,
    Stage 3 solution, or pooled score. The caller must charge acquisition of
    both initial alpha vectors, even though only one is sent to Stage 3.
    """
    rows = np.asarray(row_gains, dtype=float)
    if (rows.ndim != 2 or rows.shape[0] < 2 or rows.shape[1] < 1
            or not np.isfinite(rows).all() or not 0 <= threshold <= 1
            or bootstraps < 1):
        raise ValueError("Invalid Stage 0 marginal-gain evidence")
    mean_gains = rows.mean(axis=0)
    positive = np.maximum(mean_gains, 0.0)
    if positive.sum() <= 0:
        return "target_local_precision", {
            "fallback_reason": "no_positive_marginal_gain",
            "weighted_sign_support": 0.0,
            "per_agent_sign_support": [0.0] * rows.shape[1],
            "bootstrap_resamples": bootstraps, "threshold": threshold,
        }
    rng = np.random.default_rng(20261004 + int(seed))
    sampled = rng.integers(0, rows.shape[0], size=(bootstraps, rows.shape[0]))
    support = (rows[sampled].mean(axis=1) > 0).mean(axis=0)
    weighted = float(positive @ support / positive.sum())
    choice = ("marginal_information_50" if weighted >= threshold
              else "target_local_precision")
    return choice, {
        "fallback_reason": "" if choice == "marginal_information_50"
                           else "insufficient_marginal_gain_sign_support",
        "weighted_sign_support": weighted,
        "per_agent_sign_support": support.tolist(),
        "mean_gain": mean_gains.tolist(),
        "bootstrap_resamples": bootstraps, "threshold": threshold,
    }


def diverse_four_methods(alpha_by_method):
    """Choose four ex-ante alpha options with the most L1 portfolio spread.

    A and B anchor the portfolio. All five alpha vectors must be known before
    Stage 3; the omitted initialization still has an information-acquisition
    cost even though its Stage 3 inversion is skipped.
    """
    anchors = ("A_uniform", "B_stage0")
    extras = tuple(method for method in CANDIDATE_METHODS if method not in anchors)
    values = {method: np.asarray(alpha_by_method[method], dtype=float).reshape(-1)
              for method in CANDIDATE_METHODS}
    count = len(values[anchors[0]])
    if (not count or any(len(v) != count or not np.isfinite(v).all()
                         or np.any(v < 0) or not np.isclose(v.sum(), 1.0)
                         for v in values.values())):
        raise ValueError("Candidate alpha vectors must be valid and aligned")
    def spread(methods):
        return sum(float(np.sum(np.abs(values[a]-values[b])))
                   for a,b in combinations(methods, 2))
    return max((anchors+pair for pair in combinations(extras, 2)),
               key=lambda methods: (spread(methods),
                                    -sum(extras.index(m) for m in methods if m in extras)))


def fit_residual_calibrator(agents, blind_x, blind_y, baseline_alpha):
    x = np.asarray(blind_x, dtype=float)
    y = np.asarray(blind_y, dtype=float).reshape(-1)
    alpha = np.asarray(baseline_alpha, dtype=float).reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or len(x) < x.shape[1] + 2:
        raise ValueError("Blind-test data is insufficient for calibration")
    if len(alpha) != len(agents) or not np.isfinite(alpha).all():
        raise ValueError("Baseline alpha does not match agents")
    prediction = np.column_stack([
        np.asarray(agent.api_predict(x), dtype=float).reshape(-1)
        for agent in agents
    ])
    if prediction.shape != (len(y), len(agents)) or not np.isfinite(prediction).all():
        raise ValueError("Invalid local predictions on blind-test data")
    residual = prediction @ alpha - y
    features = np.column_stack([np.ones(len(x)), x])
    penalty = np.diag([0.01] + [1.0] * x.shape[1])
    coefficients = np.linalg.solve(features.T @ features + penalty,
                                   features.T @ residual)
    return coefficients, {"calibration_samples": len(y),
                          "blind_residual_mse": float(np.mean(residual ** 2)),
                          "ridge_penalty": [0.01] + [1.0] * x.shape[1]}


def select_candidate(agents, baseline_alpha, coefficients, solutions, target,
                     lower=-1.0, upper=1.0, tolerance=1e-8,
                     method_order=CANDIDATE_METHODS):
    """Return chosen method and internal scores; never inspect external quality."""
    alpha = np.asarray(baseline_alpha, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    scores = {}
    for method in method_order:
        if method not in solutions or solutions[method] is None:
            continue
        solution = np.asarray(solutions[method], dtype=float)
        if (solution.ndim != 1 or len(solution) + 1 != len(coefficients)
                or not np.isfinite(solution).all()
                or np.any(solution < lower - tolerance)
                or np.any(solution > upper + tolerance)):
            continue
        predictions = np.asarray([
            np.asarray(agent.api_predict(solution), dtype=float).reshape(-1)[0]
            for agent in agents
        ])
        if not np.isfinite(predictions).all():
            continue
        corrected_prediction = float(predictions @ alpha -
                                     np.r_[1.0, solution] @ coefficients)
        scores[method] = {"corrected_prediction": corrected_prediction,
                          "internal_target_error": abs(corrected_prediction-target)}
    if not scores:
        raise ValueError("No feasible candidate could be internally scored")
    chosen = min(scores, key=lambda method: scores[method]["internal_target_error"])
    return chosen, scores


def select_candidate_jackknife_q90(agents, baseline_alpha, blind_x, blind_y,
                                  solutions, target, lower=-1.0, upper=1.0,
                                  tolerance=1e-8,
                                  method_order=CANDIDATE_METHODS):
    """Use the 90th percentile of leave-one-blind-label target errors.

    Each leave-one-out fit predicts at the already solved candidate X. The
    percentile is a stability score, not a calibrated coverage guarantee.
    No pooled evaluator or true-function result is accepted as an input.
    """
    x = np.asarray(blind_x, dtype=float)
    y = np.asarray(blind_y, dtype=float).reshape(-1)
    alpha = np.asarray(baseline_alpha, dtype=float).reshape(-1)
    if (x.ndim != 2 or len(x) != len(y) or len(x) < x.shape[1] + 3
            or len(alpha) != len(agents) or not np.isfinite(x).all()
            or not np.isfinite(y).all() or not np.isfinite(alpha).all()
            or np.any(alpha < 0) or not np.isclose(alpha.sum(), 1)):
        raise ValueError("Invalid Stage 0 data or baseline alpha")
    predictions = np.column_stack([
        np.asarray(agent.api_predict(x), dtype=float).reshape(-1)
        for agent in agents
    ])
    if predictions.shape != (len(y), len(agents)) or not np.isfinite(predictions).all():
        raise ValueError("Invalid local predictions on Stage 0 data")
    residual = predictions @ alpha - y
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.diag([0.01] + [1.0] * x.shape[1])
    fits = []
    for omitted in range(len(y)):
        retained = np.ones(len(y), dtype=bool)
        retained[omitted] = False
        features = design[retained]
        fits.append(np.linalg.solve(features.T @ features + penalty,
                                    features.T @ residual[retained]))
    coefficients = np.asarray(fits)
    scores = {}
    for method in method_order:
        if method not in solutions or solutions[method] is None:
            continue
        solution = np.asarray(solutions[method], dtype=float)
        if (solution.ndim != 1 or len(solution) != x.shape[1]
                or not np.isfinite(solution).all()
                or np.any(solution < lower-tolerance)
                or np.any(solution > upper+tolerance)):
            continue
        predicted = np.asarray([
            np.asarray(agent.api_predict(solution), dtype=float).reshape(-1)[0]
            for agent in agents
        ])
        if not np.isfinite(predicted).all():
            continue
        corrected = float(predicted @ alpha) - coefficients @ np.r_[1.0, solution]
        absolute = np.abs(corrected-target)
        scores[method] = {
            "jackknife_q90_target_error": float(np.quantile(absolute, .9)),
            "jackknife_mean_target_error": float(absolute.mean()),
            "jackknife_max_target_error": float(absolute.max()),
            "jackknife_refits": len(y),
        }
    if not scores:
        raise ValueError("No feasible candidate could be jackknife-scored")
    chosen = min(scores, key=lambda method: scores[method]["jackknife_q90_target_error"])
    return chosen, scores


def select_candidate_local_bias(agents, baseline_alpha, blind_x, blind_y,
                                solutions, target, lower=-1.0, upper=1.0,
                                tolerance=1e-8, method_order=CANDIDATE_METHODS,
                                bandwidth=LOCAL_BIAS_BANDWIDTH):
    """Select an initial-alpha option using nearby Stage 0 residuals only.

    For each candidate X, fit a weighted ridge bias model on the *same* blind
    labels. No frozen pooled evaluator or true function is available here.
    """
    x = np.asarray(blind_x, dtype=float)
    y = np.asarray(blind_y, dtype=float).reshape(-1)
    alpha = np.asarray(baseline_alpha, dtype=float).reshape(-1)
    if (x.ndim != 2 or len(x) != len(y) or len(x) < x.shape[1]+2
            or len(alpha) != len(agents) or not np.isfinite(x).all()
            or not np.isfinite(y).all() or not np.isfinite(alpha).all()
            or bandwidth <= 0):
        raise ValueError("Blind data, agent order, or bandwidth is invalid")
    blind_predictions = np.column_stack([
        np.asarray(agent.api_predict(x), dtype=float).reshape(-1)
        for agent in agents
    ])
    if blind_predictions.shape != (len(y), len(agents)) or not np.isfinite(blind_predictions).all():
        raise ValueError("Invalid blind predictions for local calibration")
    residual = blind_predictions @ alpha - y
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.diag([0.01] + [1.0] * x.shape[1])
    scores = {}
    for method in method_order:
        if method not in solutions or solutions[method] is None:
            continue
        solution = np.asarray(solutions[method], dtype=float)
        if (solution.ndim != 1 or len(solution) != x.shape[1]
                or not np.isfinite(solution).all()
                or np.any(solution < lower - tolerance)
                or np.any(solution > upper + tolerance)):
            continue
        kernel = np.exp(-np.sum((x-solution) ** 2, axis=1)/(2*bandwidth**2))
        coefficients = np.linalg.solve(
            design.T @ (kernel[:, None]*design) + penalty,
            design.T @ (kernel*residual))
        predictions = np.asarray([
            np.asarray(agent.api_predict(solution), dtype=float).reshape(-1)[0]
            for agent in agents
        ])
        if not np.isfinite(predictions).all():
            continue
        corrected = float(predictions @ alpha - np.r_[1.0, solution] @ coefficients)
        scores[method] = {
            "corrected_prediction": corrected,
            "internal_target_error": abs(corrected-target),
            "local_kernel_mass": float(kernel.sum()),
            "local_coefficients": coefficients.tolist(),
        }
    if not scores:
        raise ValueError("No feasible candidate could be locally scored")
    chosen = min(scores, key=lambda method: scores[method]["internal_target_error"])
    return chosen, scores
