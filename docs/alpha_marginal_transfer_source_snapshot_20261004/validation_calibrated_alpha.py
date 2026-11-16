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
