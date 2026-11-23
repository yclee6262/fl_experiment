"""Hard-budget search adapters. Delivery uses only previously observed L_opt."""
import copy
import time
import numpy as np
from scipy.optimize import minimize

METHODS = ('anchor_mean', 'best_anchor', 'origin', 'coefficient', 'custom',
           'best_of', 'nelder_mead', 'direct_slsqp')
PENALTY = 1e12


class BudgetExhausted(RuntimeError):
    """Raised before an objective panel that would exceed its assigned budget."""


class PanelOracle:
    def __init__(self, server, panel_budget, engine):
        self.server = server
        self.limit = int(panel_budget)
        self.engine = engine
        self.panel_evaluations = 0
        self.model_queries = 0
        self.events = []
        self.best = None
        self.dim = int(server.n_features)
        self.weights = np.asarray(server.alphas, dtype=float)
        if len(self.weights) != len(server.trusted_agents) or len(self.weights) == 0:
            raise ValueError('A nonempty panel and matching alpha are required')
        if (self.weights < 0).any() or not np.isfinite(self.weights).all() or self.weights.sum() <= 0:
            raise ValueError('Alpha must be finite, nonnegative and have positive sum')
        self.weights = self.weights / self.weights.sum()

    def __call__(self, point):
        x = np.asarray(point, dtype=float)
        finite = x.shape == (self.dim,) and np.isfinite(x).all()
        if not finite:
            self.events.append(dict(engine=self.engine, kind='rejected_nonfinite_input', X=None,
                                    counted=False))
            return PENALTY
        if np.any(x < self.server.feasible_lower) or np.any(x > self.server.feasible_upper):
            self.events.append(dict(engine=self.engine, kind='rejected_outside_box', X=x.tolist(),
                                    counted=False))
            return PENALTY
        if self.panel_evaluations >= self.limit:
            raise BudgetExhausted('Objective panel budget exhausted before model call')
        self.panel_evaluations += 1
        event = dict(engine=self.engine, panel_index=self.panel_evaluations, X=x.tolist(),
                     wire_X=x.astype(np.float32).tolist(), counted=True, predictions=[],
                     model_queries=0, loss=None, kind='evaluated')
        self.events.append(event)
        try:
            for agent in self.server.trusted_agents:
                # A query is a model/input-sample pair. There is deliberately no cache.
                self.model_queries += 1
                event['model_queries'] += 1
                answer = agent.api_predict(x.astype(np.float32))
                prediction = float(np.asarray(answer).reshape(-1)[0])
                event['predictions'].append(prediction if np.isfinite(prediction) else None)
                if not np.isfinite(prediction):
                    event['kind'] = 'nonfinite_prediction'
                    return PENALTY
            loss = float(self.weights @ np.abs(np.asarray(event['predictions']) - self.server.target_T))
            if not np.isfinite(loss):
                event['kind'] = 'nonfinite_loss'
                return PENALTY
            event['loss'] = loss
            if self.best is None or loss < self.best['loss']:
                self.best = dict(X=x.tolist(), loss=loss)
            return loss
        except Exception as exc:
            event.update(kind='model_error', error=repr(exc))
            raise


def _run_engine(server, method, budget):
    oracle = PanelOracle(server, budget, method)
    start = time.perf_counter()
    optimizer_success = False
    reason = 'not_started'
    message = ''
    anchors = np.asarray(server.I_list, dtype=float)
    if anchors.ndim != 2 or anchors.shape[1] != server.n_features or len(anchors) == 0:
        raise ValueError('Nonempty candidate matrix with matching feature dimension required')
    x0 = anchors.mean(axis=0)
    try:
        if method == 'anchor_mean':
            oracle(x0)
            optimizer_success, reason = True, 'single_endpoint'
        elif method == 'origin':
            oracle(np.zeros(server.n_features))
            optimizer_success, reason = True, 'single_endpoint'
        elif method == 'best_anchor':
            for anchor in anchors:
                oracle(anchor)
            optimizer_success, reason = True, 'all_anchors_evaluated'
        elif method == 'coefficient':
            local = copy.copy(server)
            local.coefficient_eps = 1e-5
            result = local._solve_subspace_bfgs(anchors, oracle, history_metric=oracle)
            optimizer_success = bool(result['success'])
            reason = 'solver_completed' if optimizer_success else 'solver_unsuccessful'
            message = str(result['message'])
        elif method == 'custom':
            result = server._solve_subspace_custom(anchors, oracle, num_iterations=30,
                direction_weights=np.asarray(server.alphas), history_metric=oracle)
            optimizer_success = bool(result['success'])
            reason, message = 'solver_completed', str(result['message'])
        elif method in ('nelder_mead', 'direct_slsqp'):
            solver = 'Nelder-Mead' if method == 'nelder_mead' else 'SLSQP'
            options = ({'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-9} if solver == 'Nelder-Mead'
                       else {'maxiter': 200, 'ftol': 1e-9, 'eps': 1e-5})
            result = minimize(oracle, x0, method=solver,
                              bounds=[(server.feasible_lower, server.feasible_upper)]*server.n_features,
                              options=options)
            optimizer_success = bool(result.success)
            reason = 'solver_completed' if optimizer_success else 'solver_unsuccessful'
            message = str(result.message)
        else:
            raise ValueError('Unknown single-engine method')
    except BudgetExhausted as exc:
        reason, message = 'budget_exhausted', str(exc)
    except Exception as exc:
        reason, message = 'solver_or_model_error', repr(exc)
    best = oracle.best
    n = oracle.model_queries
    return dict(method=method, X=None if best is None else best['X'],
                loss=None if best is None else best['loss'],
                status='valid' if best is not None else 'no_valid_endpoint',
                success=optimizer_success, stop_reason=reason, message=message,
                panel_budget=budget, panel_evaluations=oracle.panel_evaluations,
                model_queries=n, input_samples=n, requests=n,
                transmitted_bytes=n*4*server.n_features + 4*sum(len(e.get('predictions',[])) for e in oracle.events),
                input_payload_bytes=n*4*server.n_features,
                output_payload_bytes=4*sum(len(e.get('predictions',[])) for e in oracle.events),
                seconds=time.perf_counter()-start, events=oracle.events,
                budget_exhausted=reason=='budget_exhausted',
                delivery_rule='best_observed_finite_box_feasible_Lopt')


def run_search(server, method, panel_budget):
    """Execute one method without exceeding the assigned model-panel budget.

    A best-of budget is divided equally, with the odd panel unassigned and no
    transfer between engines. Only observed losses may select the delivery.
    Model bytes are numerical payload only, without transport framing.
    """
    if method not in METHODS:
        raise ValueError(f'Unknown search method: {method}')
    if isinstance(panel_budget, bool) or int(panel_budget) != panel_budget or panel_budget < 0:
        raise ValueError('panel_budget must be a nonnegative integer')
    panel_budget = int(panel_budget)
    if server.constraint_mode != 'box_constrained':
        raise ValueError('Budgeted search requires explicit box constraints')
    if method != 'best_of':
        result = _run_engine(server, method, panel_budget)
        result['engine_details'] = []
        return result
    quota = panel_budget//2
    branches = [_run_engine(server, engine, quota) for engine in ('coefficient','custom')]
    eligible = [r for r in branches if r['status']=='valid']
    chosen = min(eligible, key=lambda r:r['loss']) if eligible else None
    additive = ['panel_evaluations','model_queries','input_samples','requests','transmitted_bytes','input_payload_bytes','output_payload_bytes','seconds']
    result = {f:sum(r[f] for r in branches) for f in additive}
    result.update(method=method, panel_budget=panel_budget, per_engine_budget=quota,
                  unused_split_panels=panel_budget-2*quota,
                  X=None if chosen is None else chosen['X'], loss=None if chosen is None else chosen['loss'],
                  status='no_valid_endpoint' if chosen is None else 'valid',
                  success=False if chosen is None else chosen['success'],
                  chosen_engine=None if chosen is None else chosen['method'],
                  stop_reason='no_valid_endpoint' if chosen is None else chosen['stop_reason'],
                  budget_exhausted=any(r['budget_exhausted'] for r in branches),
                  events=[e for r in branches for e in r['events']],
                  engine_details=[{k:v for k,v in r.items() if k!='events'} for r in branches],
                  delivery_rule='best_observed_finite_box_feasible_Lopt')
    return result
