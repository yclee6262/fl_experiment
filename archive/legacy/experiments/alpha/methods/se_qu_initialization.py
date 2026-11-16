"""Fixed missing 50/50 pairs; no evaluator inputs."""
import numpy as np
from pure_es_initialization import initializer as pure_initializer
from initial_alpha_methods import initial_alpha as legacy
METHODS = ['support_ensemble_mix50', 'quality_shrink']
SCENARIOS = ['reference_clean', 'low_quality_25pct', 'sample_imbalance_4x']

def initializer(peer):
    pure = pure_initializer(peer)
    def initial(method, agents, mse, x, y, proposals, target, **kwargs):
        if method == 'quality_shrink':
            return legacy(method, agents, mse, x, y, proposals, target, **kwargs)
        if method != 'support_ensemble_mix50':
            raise ValueError(method)
        e, ea = pure('stacked_full', agents, mse, x, y, proposals, target, **kwargs)
        s, sa = pure('pure_support', agents, mse, x, y, proposals, target, **kwargs)
        alpha = .5 * np.asarray(e) + .5 * np.asarray(s)
        return alpha, dict(formula='alpha=0.5*E+0.5*S', ensemble=ea, support=sa,
                          agent_ids=[a.agent_id for a in agents], additional_labels=0)
    return initial
