"""Qualification-only matrix shrinkage; pooled/truth never enter calibration."""
import numpy as np
from scipy.optimize import minimize
from next_alpha_initialization import initial_alpha as previous_initial_alpha

METHODS = ['A_uniform', 'B_stage0', 'bootstrap_conservative_quality',
           'corr_precision_0p5', 'adaptive_corr_precision']
PRIMARY = METHODS[-1]
GRID = (0., .25, .5, .75, 1.)
EPS = 1e-5
TIE_TOLERANCE = 1e-12

def validate(prediction, labels):
    p = np.asarray(prediction, float)
    y = np.asarray(labels, float).reshape(-1)
    if (p.ndim != 2 or len(p) != len(y) or p.shape[1] < 1 or len(y) != 20
            or not np.isfinite(p).all() or not np.isfinite(y).all()):
        raise ValueError('Expected 20 finite aligned qualification rows')
    return p, y

def fit_matrix(residual, lam):
    """Exactly the legacy fixed-0.5 objective/solver; analytic diagonal endpoint."""
    residual = np.asarray(residual, float)
    if not np.isfinite(residual).all():
        raise ValueError('Nonfinite residual')
    mse = (residual ** 2).mean(axis=0)
    score = 1 / (mse + EPS)
    quality = score / score.sum()
    moment = residual.T @ residual / len(residual)
    matrix = (1-lam)*moment + lam*np.diag(mse+EPS)
    if not np.isfinite(matrix).all():
        raise FloatingPointError('Nonfinite risk matrix')
    meta = {'lambda': float(lam), 'mse': mse.tolist(),
            'second_moment': moment.tolist(), 'risk_matrix': matrix.tolist()}
    if lam == 1:
        return quality, {**meta, 'success': True, 'solver': 'analytic_inverse_MSE',
                         'message': 'diagonal simplex optimum'}
    fit = minimize(lambda w: float(w @ matrix @ w), quality,
                   method='SLSQP', bounds=[(0., 1.)]*len(quality),
                   constraints={'type': 'eq', 'fun': lambda w: w.sum()-1},
                   options={'ftol': 1e-13, 'maxiter': 500})
    ok = bool(fit.success and np.isfinite(fit.x).all()
              and np.min(fit.x) >= -1e-8 and abs(fit.x.sum()-1) <= 1e-8)
    if not ok:
        return None, {**meta, 'success': False, 'solver': 'SLSQP', 'message': str(fit.message)}
    w = np.maximum(fit.x, 0)
    w /= w.sum()
    return w, {**meta, 'success': True, 'solver': 'SLSQP', 'message': str(fit.message)}

def choose_lambda(candidate_losses):
    valid = [(float(lam), float(loss)) for lam, loss in candidate_losses.items()
             if loss is not None and np.isfinite(loss)]
    if not valid:
        return None
    minimum = min(loss for _, loss in valid)
    return max(lam for lam, loss in valid if loss <= minimum + TIE_TOLERANCE)

def correlation_weights(prediction, labels, seed=20261011, adaptive=True):
    p, y = validate(prediction, labels)
    residual = p-y[:, None]
    base, _ = fit_matrix(residual, 1.)
    audit = {'fallback_reason': '', 'grid': list(GRID), 'fold_count': 5,
             'split_seed': int(seed), 'tie_tolerance': TIE_TOLERANCE,
             'tie_rule': 'prefer larger lambda', 'fold_audits': [],
             'matrix_semantics': 'uncentered residual second moment, retains bias',
             'tuning_loss': 'heldout mixture MSE, not inverse quality estimate'}
    lam = .5
    if adaptive:
        permutation = np.random.default_rng(np.random.SeedSequence([seed, 20261010, 5])).permutation(20)
        heldout = np.array_split(permutation, 5)
        means = {}
        for candidate in GRID:
            losses = []
            for fold, hold in enumerate(heldout):
                train = np.setdiff1d(np.arange(20), hold)
                weight, fitted = fit_matrix(residual[train], candidate)
                loss = None if weight is None else float(np.mean((p[hold] @ weight-y[hold])**2))
                if loss is not None and not np.isfinite(loss):
                    loss = None
                losses.append(loss)
                audit['fold_audits'].append({'fold': fold, 'train_indices': train.tolist(),
                    'heldout_indices': hold.tolist(), 'weight': None if weight is None else weight.tolist(),
                    'heldout_mse': loss, **fitted})
            means[str(candidate)] = float(np.mean(losses)) if all(v is not None for v in losses) else None
        audit['candidate_cv_mse'] = means
        lam = choose_lambda(means)
        if lam is None:
            audit.update(fallback_reason='all_candidates_invalid_use_B', selected_lambda=None)
            return base, audit
    fitted, final = fit_matrix(residual, lam)
    audit.update(selected_lambda=lam, final_fit=final,
                 fits=len(audit['fold_audits'])+1, definition='CV-selected matrix shrinkage' if adaptive else 'legacy fixed0.5 matrix shrinkage')
    if fitted is None:
        audit['fallback_reason'] = 'final_matrix_solver_failure_use_B'
        fitted = base
    return fitted, audit

def initial_alpha(method, agents, mse, blind_x, blind_y, proposals, target, **kwargs):
    if method not in METHODS[-2:]:
        return previous_initial_alpha(method, agents, mse, blind_x, blind_y, proposals, target, **kwargs)
    ids = [int(a.agent_id) for a in agents]
    if not ids or len(set(ids)) != len(ids) or len(ids) != len(mse) or len(ids) != len(proposals):
        raise ValueError('Coalition mismatch or duplicate IDs')
    p = kwargs.get('cached_blind_predictions')
    reused = p is not None
    if p is None:
        p = np.column_stack([a.api_predict(blind_x) for a in agents])
    if np.asarray(p).shape != (len(blind_y), len(ids)):
        raise ValueError('Qualification prediction shape mismatch')
    p, y = validate(p, blind_y)
    actual_mse = ((p-y[:, None])**2).mean(axis=0)
    if not np.allclose(actual_mse, mse, rtol=0, atol=1e-12):
        raise ValueError('Saved MSE does not match qualification residuals')
    w, audit = correlation_weights(p, y, kwargs.get('bootstrap_seed', 20261011), method == PRIMARY)
    audit.update(agent_ids=ids, blind_prediction_source='qualification_reuse' if reused else 'fresh_api', raw_scores=w.tolist())
    return w, audit
