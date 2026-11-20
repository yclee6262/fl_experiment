"""Equal mixture of proposal support and inverse-MSE quality weights (SQ)."""
import numpy as np
from peer_support_alpha import support_signal

METHOD = "support_quality_mix50"

def mixed_weights(mse, prediction_matrix, target):
    mse = np.asarray(mse, float)
    if mse.ndim != 1 or not np.isfinite(mse).all() or np.any(mse < 0):
        raise ValueError('Invalid quality evidence')
    s, audit = support_signal(prediction_matrix, target)
    if len(s) != len(mse):
        raise ValueError('Agent order mismatch')
    b = 1 / (mse + 1e-5)
    b /= b.sum()
    alpha = .5 * s + .5 * b
    audit.update(quality=b.tolist(), support_fraction=.5, quality_fraction=.5,
                 uniform_fraction=0., formula='0.5*support + 0.5*B',
                 signal_semantics='candidate support, not participant contribution')
    return alpha, audit

def initializer(matrix):
    def initial(method, agents, mse, x, y, proposals, target, **kwargs):
        if method != METHOD:
            raise ValueError(method)
        w, a = mixed_weights(mse, matrix, target)
        a['agent_ids'] = [g.agent_id for g in agents]
        return w, a
    return initial
