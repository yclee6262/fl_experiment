"""Uniform, inverse-MSE, uniform-quality mixture and constrained ensemble fitting."""
import numpy as np
from scipy.optimize import minimize
def _normalize(scores):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or (not np.all(np.isfinite(values))):
        raise ValueError('Weight scores must be a finite, nonempty vector')
    if np.any(values < 0) or not float(values.sum()) > 0:
        raise ValueError('Weight scores must be nonnegative with positive sum')
    return values / values.sum()
def _finish(weights, count):
    alpha = _normalize(weights)
    if len(alpha) != count or not np.isclose(alpha.sum(), 1.0, atol=1e-12):
        raise ValueError('Initial alpha does not match selected agents')
    return alpha

EPS = 1e-5
SHRINKAGE = 0.5
METHODS = ("A_uniform", "B_stage0", "quality_shrink", "ensemble_shrink")

def initial_alpha(method, agents, mse, blind_x, blind_y, proposals, target, lower=-1.0, upper=1.0, *, bootstrap_replicates=1000, bootstrap_seed=20261008, exponential_temperature_ratio=1.0, cached_blind_predictions=None):
    """Return (alpha, audit) in exactly the input agent order.

    API prediction calls here are charged to the caller's initialization
    counter.  Expected bad numerical inputs fall back to Uniform with a reason;
    programming errors and a mismatched coalition are not hidden.
    """
    if method not in METHODS:
        raise ValueError(f'Unknown initial-alpha method: {method}')
    count = len(agents)
    ids = [int(agent.agent_id) for agent in agents]
    if len(set(ids)) != count:
        raise ValueError('Duplicate agent IDs')
    uniform = np.full(count, 1.0 / count)
    mse = np.asarray(mse, dtype=float)
    audit = {'agent_ids': ids, 'stage0_mse': mse.tolist(), 'fallback_reason': '', 'raw_scores': [], 'peer_target_errors': []}
    if not np.all(np.isfinite(mse)) or np.any(mse < 0):
        audit['fallback_reason'] = 'invalid_stage0_mse'
        return (uniform, audit)
    quality = _normalize(1.0 / (mse + EPS))
    audit['stage0_quality_share'] = quality.tolist()
    if method == 'A_uniform':
        return (uniform, audit)
    if method == 'B_stage0':
        audit['raw_scores'] = quality.tolist()
        return (_finish(quality, count), audit)
    if method == 'quality_shrink':
        audit['raw_scores'] = quality.tolist()
        return (_finish(SHRINKAGE * uniform + SHRINKAGE * quality, count), audit)
    if count == 1:
        audit['fallback_reason'] = 'single_participant'
        return (uniform, audit)
    if method == 'ensemble_shrink':
        x = np.asarray(blind_x, dtype=float)
        y = np.asarray(blind_y, dtype=float).reshape(-1)
        if x.ndim != 2 or len(x) != len(y) or (not len(y)):
            raise ValueError('Blind-test features and labels do not align')
        predictions = np.column_stack([np.asarray(agent.api_predict(x), dtype=float).reshape(-1) for agent in agents])
        if predictions.shape != (len(y), count) or not np.all(np.isfinite(predictions)):
            audit['fallback_reason'] = 'invalid_blind_predictions'
            return (uniform, audit)

        def objective(weights):
            return float(np.mean((predictions @ weights - y) ** 2))
        result = minimize(objective, uniform, method='SLSQP', bounds=[(0.0, 1.0)] * count, constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1.0}, options={'ftol': 1e-12, 'maxiter': 500})
        if not result.success or not np.all(np.isfinite(result.x)) or np.min(result.x) < -1e-08:
            audit['fallback_reason'] = 'ensemble_solver_failure'
            return (uniform, audit)
        fitted = _normalize(np.maximum(result.x, 0.0))
        audit['raw_scores'] = fitted.tolist()
        audit['blind_ensemble_mse'] = objective(fitted)
        return (_finish(SHRINKAGE * uniform + SHRINKAGE * fitted, count), audit)
    raise ValueError(method)
