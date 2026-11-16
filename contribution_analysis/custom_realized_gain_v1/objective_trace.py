"""Capture the existing API responses. No additional prediction or caching."""
import hashlib
import time
import numpy as np
from .schema import key


class ObjectiveTrace:
    def __init__(self, server, record, model_versions=None):
        self.server, self.record = server, record
        self.agents = list(server.trusted_agents)
        self.versions = model_versions or ['fixed']*len(self.agents)
        self.queries, self.evaluations = [], []
        self.query_count = self.samples = 0
        self.model_seconds = [0.]*len(self.agents)
        self.unique = set()
        self.digest = hashlib.sha256()
        self.seen = {}
        self.pending = None
        self.originals = []
        self.started = time.perf_counter()
        self.last_eval_id = None
        self.context = dict(purpose='unclassified', iteration=None, anchor_id=None, attempt=None)

    def __enter__(self):
        for j, agent in enumerate(self.agents):
            old = agent.api_predict
            self.originals.append((agent, old))
            def call(x, _j=j, _old=old):
                n = 1 if np.asarray(x).ndim == 1 else len(x)
                if self.samples+n > 20000 or time.perf_counter()-self.started > 60:
                    raise RuntimeError('per_solve_budget')
                t = time.perf_counter()
                try: y = _old(x)
                except Exception as error:
                    if self.record: self.queries.append(dict(query_id=self.query_count, model_index=_j, success=False, error=repr(error)))
                    raise
                dt = time.perf_counter()-t
                arr = np.asarray(x)
                yy = np.asarray(y)
                payload = f'{_j}:{arr.dtype}:{arr.shape}:'.encode()+arr.tobytes()+yy.tobytes()
                self.digest.update(len(payload).to_bytes(8, 'little')); self.digest.update(payload)
                self.query_count += 1; self.samples += n; self.model_seconds[_j] += dt
                self.unique.add((_j, self.versions[_j], key(x)))
                if self.record:
                    if not np.isfinite(yy).all(): raise ValueError('nonfinite_reply')
                    lookup = (_j, key(x))
                    if lookup in self.seen and self.seen[lookup] != yy.tobytes():
                        raise ValueError('unsupported_stochastic')
                    self.seen[lookup] = yy.tobytes()
                    row = dict(query_id=len(self.queries), eval_id=len(self.evaluations), model_index=_j,
                               model_id=str(self.agents[_j].agent_id), model_version=self.versions[_j],
                               x=arr.copy(), input_hex=key(x), input_dtype=str(arr.dtype), input_shape=list(arr.shape),
                               output=yy.copy(), output_dtype=str(yy.dtype), output_shape=list(yy.shape), call_seconds=dt, input_samples=n, success=True)
                    self.queries.append(row)
                    if self.pending is not None: self.pending.append(row)
                return y
            agent.api_predict = call
        return self

    def __exit__(self, *args):
        for agent, old in self.originals: agent.api_predict = old

    def __call__(self, x):
        # Preserve the original consensus implementation and operation ordering.
        if self.record: self.pending = []
        value = self.server._consensus_loss(x, agents=self.agents, mode='fixed', weights=self.server.alphas)
        if self.record:
            replies = self.pending; self.pending = None
            if len(replies) != len(self.agents): raise ValueError('missing_reply')
            losses = np.array([abs(float(r['output'][0])-self.server.target_T) for r in replies])
            eid = len(self.evaluations)
            self.evaluations.append(dict(eval_id=eid, event_index=eid, **self.context, x=np.array(x).copy(),
                                         model_query_ids=[r['query_id'] for r in replies],
                                         individual_losses=losses, objective_value=float(value), weights_version='fixed_v1'))
            self.last_eval_id = eid
        return value

    def costs(self):
        return dict(api_calls=self.query_count, input_samples=self.samples, unique_inputs=len(self.unique),
                    per_model_seconds=self.model_seconds, model_seconds=sum(self.model_seconds),
                    query_sha256=self.digest.hexdigest(), additional_prediction_calls=0)
