"""Ex-ante optimization weights for the static inverse-solution study.

Only Stage 0's labelled blind-test set, local engines, and Stage 2 proposals
are inputs.  The frozen pooled evaluator is deliberately unavailable here.
"""

import numpy as np
from scipy.optimize import minimize


METHODS = (
    "A_uniform",
    "B_stage0",
    "quality_shrink",
    "ensemble_shrink",
    "target_peer_shrink",
    "target_local_precision",
    "mse_power_1p5",
    "corr_precision_0p5",
    "best_agent",
    "local_bias_risk",
)
ECONOMIC_METHODS = METHODS[2:]
EPS = 1e-5
SHRINKAGE = 0.5


def _normalize(scores):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Weight scores must be a finite, nonempty vector")
    if np.any(values < 0) or not float(values.sum()) > 0:
        raise ValueError("Weight scores must be nonnegative with positive sum")
    return values / values.sum()


def _finish(weights, count):
    alpha = _normalize(weights)
    if len(alpha) != count or not np.isclose(alpha.sum(), 1.0, atol=1e-12):
        raise ValueError("Initial alpha does not match selected agents")
    return alpha


def initial_alpha(method, agents, mse, blind_x, blind_y, proposals, target,
                  lower=-1.0, upper=1.0):
    """Return (alpha, audit) in exactly the input agent order.

    API prediction calls here are charged to the caller's initialization
    counter.  Expected bad numerical inputs fall back to Uniform with a reason;
    programming errors and a mismatched coalition are not hidden.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown initial-alpha method: {method}")
    count = len(agents)
    if not count or len(mse) != count or len(proposals) != count:
        raise ValueError("Agents, Stage 0 scores, and proposals must align")
    ids = [int(agent.agent_id) for agent in agents]
    if len(set(ids)) != count:
        raise ValueError("Duplicate agent IDs")
    uniform = np.full(count, 1.0 / count)
    mse = np.asarray(mse, dtype=float)
    audit = {"agent_ids": ids, "stage0_mse": mse.tolist(),
             "fallback_reason": "", "raw_scores": [], "peer_target_errors": []}
    if not np.all(np.isfinite(mse)) or np.any(mse < 0):
        audit["fallback_reason"] = "invalid_stage0_mse"
        return uniform, audit
    quality = _normalize(1.0 / (mse + EPS))
    audit["stage0_quality_share"] = quality.tolist()
    if method == "A_uniform":
        return uniform, audit
    if method == "B_stage0":
        audit["raw_scores"] = quality.tolist()
        return _finish(quality, count), audit
    if method == "quality_shrink":
        audit["raw_scores"] = quality.tolist()
        return _finish(SHRINKAGE * uniform + SHRINKAGE * quality, count), audit
    if method == "mse_power_1p5":
        powered = _normalize((mse + EPS) ** -1.5)
        audit["raw_scores"] = powered.tolist()
        return powered, audit
    if method == "best_agent":
        winner = int(np.argmin(mse))
        concentrated = np.eye(count)[winner]
        audit["raw_scores"] = concentrated.tolist()
        audit["winner_agent_id"] = ids[winner]
        return concentrated, audit
    if method == "local_bias_risk":
        x = np.asarray(blind_x, dtype=float)
        y = np.asarray(blind_y, dtype=float).reshape(-1)
        p = np.asarray(proposals, dtype=float)
        if (x.ndim != 2 or len(x) != len(y) or len(y) < x.shape[1] + 2
                or p.shape != (count, x.shape[1]) or not np.isfinite(p).all()):
            raise ValueError("Blind data and proposals do not support local bias risk")
        predictions = np.column_stack([
            np.asarray(agent.api_predict(x), dtype=float).reshape(-1)
            for agent in agents
        ])
        if predictions.shape != (len(y), count) or not np.isfinite(predictions).all():
            audit["fallback_reason"] = "invalid_blind_predictions_use_stage0"
            return quality, audit
        residual = predictions - y[:, None]
        features = np.column_stack([np.ones(len(x)), x])
        penalty = np.diag([0.01] + [1.0] * x.shape[1])
        coefficients = np.linalg.solve(features.T @ features + penalty,
                                       features.T @ residual)
        center = p.mean(axis=0)
        local_bias = np.r_[1.0, center] @ coefficients
        variance = np.mean((residual - features @ coefficients) ** 2, axis=0)
        risk = local_bias ** 2 + variance
        scores = 1.0 / (risk + EPS)
        audit["proposal_center"] = center.tolist()
        audit["estimated_local_bias"] = local_bias.tolist()
        audit["estimated_residual_variance"] = variance.tolist()
        audit["raw_scores"] = scores.tolist()
        return _finish(scores, count), audit
    if method == "corr_precision_0p5":
        x = np.asarray(blind_x, dtype=float)
        y = np.asarray(blind_y, dtype=float).reshape(-1)
        if x.ndim != 2 or len(x) != len(y) or not len(y):
            raise ValueError("Blind-test features and labels do not align")
        predictions = np.column_stack([
            np.asarray(agent.api_predict(x), dtype=float).reshape(-1)
            for agent in agents
        ])
        if predictions.shape != (len(y), count) or not np.all(np.isfinite(predictions)):
            audit["fallback_reason"] = "invalid_blind_predictions_use_stage0"
            return quality, audit
        residual = predictions - y[:, None]
        second_moment = residual.T @ residual / len(y)
        covariance = 0.5 * np.diag(mse + EPS) + 0.5 * second_moment
        fitted = minimize(lambda w: float(w @ covariance @ w), quality,
                          method="SLSQP", bounds=[(0.0, 1.0)] * count,
                          constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
                          options={"ftol": 1e-13, "maxiter": 500})
        if not fitted.success or not np.all(np.isfinite(fitted.x)):
            audit["fallback_reason"] = "correlation_solver_failure_use_stage0"
            return quality, audit
        adjusted = _normalize(np.maximum(fitted.x, 0.0))
        audit["raw_scores"] = adjusted.tolist()
        audit["correlation_fraction"] = 0.5
        return adjusted, audit
    if method == "target_local_precision":
        x = np.asarray(blind_x, dtype=float)
        y = np.asarray(blind_y, dtype=float).reshape(-1)
        if x.ndim != 2 or len(x) != len(y) or not len(y):
            raise ValueError("Blind-test features and labels do not align")
        predictions = np.column_stack([
            np.asarray(agent.api_predict(x), dtype=float).reshape(-1)
            for agent in agents
        ])
        if predictions.shape != (len(y), count) or not np.all(np.isfinite(predictions)):
            audit["fallback_reason"] = "invalid_blind_predictions_use_stage0"
            return quality, audit
        bandwidth, prior_weight = 0.5, 5.0
        kernel = np.exp(-np.abs(y - target) / bandwidth)
        local_mse = (
            ((predictions - y[:, None]) ** 2 * kernel[:, None]).sum(axis=0)
            + prior_weight * mse
        ) / (float(kernel.sum()) + prior_weight)
        audit["target_bandwidth"] = bandwidth
        audit["global_mse_prior_weight"] = prior_weight
        audit["target_kernel_mass"] = float(kernel.sum())
        audit["target_local_mse"] = local_mse.tolist()
        adjusted = _normalize(1.0 / (local_mse + EPS))
        audit["raw_scores"] = adjusted.tolist()
        return _finish(adjusted, count), audit
    if count == 1:
        audit["fallback_reason"] = "single_participant"
        return uniform, audit
    if method == "ensemble_shrink":
        x = np.asarray(blind_x, dtype=float)
        y = np.asarray(blind_y, dtype=float).reshape(-1)
        if x.ndim != 2 or len(x) != len(y) or not len(y):
            raise ValueError("Blind-test features and labels do not align")
        predictions = np.column_stack([
            np.asarray(agent.api_predict(x), dtype=float).reshape(-1)
            for agent in agents
        ])
        if predictions.shape != (len(y), count) or not np.all(np.isfinite(predictions)):
            audit["fallback_reason"] = "invalid_blind_predictions"
            return uniform, audit
        def objective(weights):
            return float(np.mean((predictions @ weights - y) ** 2))
        result = minimize(objective, uniform, method="SLSQP",
                          bounds=[(0.0, 1.0)] * count,
                          constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
                          options={"ftol": 1e-12, "maxiter": 500})
        if not result.success or not np.all(np.isfinite(result.x)) or np.min(result.x) < -1e-8:
            audit["fallback_reason"] = "ensemble_solver_failure"
            return uniform, audit
        fitted = _normalize(np.maximum(result.x, 0.0))
        audit["raw_scores"] = fitted.tolist()
        audit["blind_ensemble_mse"] = objective(fitted)
        return _finish(SHRINKAGE * uniform + SHRINKAGE * fitted, count), audit

    proposals = np.asarray(proposals, dtype=float)
    if proposals.shape != (count, np.asarray(blind_x).shape[1]):
        raise ValueError("Proposals do not match coalition dimension")
    peer_errors = np.full(count, np.nan)
    all_errors = []
    for index, proposal in enumerate(proposals):
        if not (np.all(np.isfinite(proposal)) and
                np.all(proposal >= lower - 1e-8) and
                np.all(proposal <= upper + 1e-8)):
            continue
        errors = []
        for other, agent in enumerate(agents):
            if other == index:
                continue
            predicted = np.asarray(agent.api_predict(proposal), dtype=float).reshape(-1)
            if len(predicted) != 1 or not np.isfinite(predicted[0]):
                continue
            errors.append(abs(float(predicted[0]) - target))
        if errors:
            peer_errors[index] = float(np.median(errors))
            all_errors.extend(errors)
    audit["peer_target_errors"] = [float(x) if np.isfinite(x) else None
                                   for x in peer_errors]
    if not all_errors:
        audit["fallback_reason"] = "no_valid_peer_predictions"
        return uniform, audit
    scale = float(np.median(all_errors)) + EPS
    raw = np.where(np.isfinite(peer_errors),
                   quality * np.exp(-np.nan_to_num(peer_errors, nan=np.inf) / scale), 0.0)
    audit["raw_scores"] = raw.tolist()
    audit["peer_error_scale"] = scale
    if not np.all(np.isfinite(raw)) or raw.sum() <= 0:
        audit["fallback_reason"] = "no_positive_target_support"
        return uniform, audit
    return _finish(SHRINKAGE * uniform + SHRINKAGE * _normalize(raw), count), audit
