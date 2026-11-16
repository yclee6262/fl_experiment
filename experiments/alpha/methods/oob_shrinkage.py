"""Bootstrap inverse-MSE weights with strictly out-of-bag shrinkage tuning.

This is a nonnegative quality-weight adaptation of subset shrinkage averaging,
not a reproduction of unconstrained forecast-combination optimal weights.
Only qualification predictions/labels are inputs; no evaluator or target.
"""
import numpy as np


def tune_shrinkage(prediction, labels, quality):
    """Analytic minimum of held-out mixture MSE; flat ties prefer Uniform."""
    p, y, q = np.asarray(prediction, float), np.asarray(labels, float), np.asarray(quality, float)
    if (p.ndim != 2 or y.shape != (len(p),) or q.shape != (p.shape[1],)
            or not len(y) or not np.isfinite(p).all() or not np.isfinite(y).all()
            or not np.isfinite(q).all() or np.any(q < 0)
            or not np.isclose(q.sum(), 1., atol=1e-12)):
        raise ValueError("Invalid held-out predictions, labels or quality weights")
    uniform = np.full(len(q), 1. / len(q))
    residual = p @ q - y
    direction = p @ uniform - p @ q
    denominator = float(direction @ direction)
    lam = 1. if denominator == 0. else float(np.clip(-(direction @ residual) / denominator, 0., 1.))
    weights = (1.-lam)*q + lam*uniform
    losses = {"selected": float(np.mean((p @ weights-y)**2)),
              "quality": float(np.mean(residual**2)),
              "uniform": float(np.mean((p @ uniform-y)**2))}
    return lam, weights, losses


def bootstrap_oob_quality(predictions, labels, replicates=1000, seed=20261011, eps=1e-5):
    p, y = np.asarray(predictions, float), np.asarray(labels, float).reshape(-1)
    if (p.ndim != 2 or p.shape[0] != len(y) or p.shape[1] < 1 or len(y) < 2
            or not np.isfinite(p).all() or not np.isfinite(y).all()
            or not isinstance(replicates, (int, np.integer)) or replicates < 2
            or not np.isfinite(eps) or eps <= 0):
        raise ValueError("Invalid bootstrap quality input")
    rng = np.random.default_rng(seed)
    error = (p-y[:, None])**2
    if not np.isfinite(error).all():
        raise FloatingPointError("Nonfinite squared errors")
    indices = rng.integers(0, len(y), size=(replicates, len(y)))
    weights, lambdas, sizes, losses = [], [], [], []
    skipped = 0
    for sample in indices:
        hold = np.ones(len(y), dtype=bool)
        hold[np.unique(sample)] = False
        # No in-bag row may tune lambda. Tiny OOB sets are skipped, never
        # silently replaced by in-bag observations or external test scores.
        if int(hold.sum()) < 2:
            skipped += 1
            continue
        score = 1. / (error[sample].mean(axis=0)+eps)
        q = score / score.sum()
        lam, w, loss = tune_shrinkage(p[hold], y[hold], q)
        weights.append(w)
        lambdas.append(lam)
        sizes.append(int(hold.sum()))
        losses.append(loss)
    audit = {"bootstrap_replicates": int(replicates), "bootstrap_seed": int(seed),
             "accepted_replicates": len(weights), "skipped_small_oob": skipped,
             "minimum_oob_rows": 2,
             "lambda_definition": "Uniform fraction; analytic held-out mixture-MSE minimum in [0,1]",
             "lambda_values": lambdas, "oob_sizes": sizes,
             "fallback_reason": "" if weights else "no_eligible_oob_use_uniform",
             "validation_role": "OOB tuning, not unbiased final quality estimate"}
    if not weights:
        return np.full(p.shape[1], 1./p.shape[1]), audit
    a = np.mean(weights, axis=0)
    a /= a.sum()
    audit.update(lambda_mean=float(np.mean(lambdas)),
                 lambda_quantiles=np.quantile(lambdas, [.025,.5,.975]).tolist(),
                 lambda_at_zero=int(np.sum(np.asarray(lambdas)==0)),
                 lambda_at_one=int(np.sum(np.asarray(lambdas)==1)),
                 mean_oob_tuning_losses={k:float(np.mean([x[k] for x in losses])) for k in losses[0]},
                 bootstrap_weight_std=np.std(weights, axis=0, ddof=1).tolist() if len(weights)>1 else [0.]*p.shape[1])
    return a, audit
