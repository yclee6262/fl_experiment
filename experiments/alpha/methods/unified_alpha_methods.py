"""Simplex ensemble fit E and its equal mixture with quality weights EQ."""
import numpy as np
from initial_alpha_methods import initial_alpha as legacy

class CachedAgent:
    def __init__(self, agent, x, prediction):
        self.agent_id = agent.agent_id
        self.agent, self.x, self.prediction = agent, x, prediction

    def api_predict(self, values):
        x = np.asarray(values)
        if x.shape == self.x.shape and np.array_equal(x, self.x):
            return self.prediction.copy()
        return self.agent.api_predict(values)

METHODS = ("stacked_full", "B_stacked_half")

def initial_alpha(method, agents, mse, blind_x, blind_y, proposals, target, **kwargs):
    if method not in METHODS:
        raise ValueError("Unknown ensemble method")
    x, y = np.asarray(blind_x), np.asarray(blind_y)
    predictions = kwargs.get("cached_blind_predictions")
    if predictions is not None:
        predictions = np.asarray(predictions, float)
        if predictions.shape != (len(y), len(agents)) or not np.isfinite(predictions).all():
            raise ValueError("Qualification cache shape/values mismatch")
        agents = [CachedAgent(a, x, predictions[:, i]) for i, a in enumerate(agents)]
    quality, audit = legacy("B_stage0", agents, mse, x, y, proposals, target, **kwargs)
    _, fitted = legacy("ensemble_shrink", agents, mse, x, y, proposals, target, **kwargs)
    if fitted["fallback_reason"]:
        weights = quality
        audit["fallback_reason"] = "ensemble_fit_failed_use_B"
    else:
        ensemble = np.asarray(fitted["raw_scores"])
        weights = ensemble if method == "stacked_full" else .5 * quality + .5 * ensemble
        audit["ensemble_fit"] = fitted
    weights = np.asarray(weights, float)
    if (weights.shape != (len(agents),) or not np.isfinite(weights).all()
            or np.min(weights) < 0 or abs(weights.sum() - 1) > 1e-8):
        raise ValueError("Invalid alpha simplex")
    audit.update(unified_method=method, bootstrap_seed=int(kwargs.get("bootstrap_seed", 20261011)),
                 qualification_cache_available=predictions is not None)
    return weights, audit
