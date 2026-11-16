"""Raw response capture during solve; materialization after solve, before use.

Only fixed-order single-vector deterministic API calls are supported. Unsupported
replies invalidate the whole run. No cache, prediction skipping or solver edits.
"""
import hashlib
import time
import numpy as np
from contribution_analysis.custom_realized_gain_v1.schema import key,plain,validate_manifest
from contribution_analysis.custom_realized_gain_v1.observer import Observer,instrument


class CompactTrace:
    def __init__(self,server,versions):
        self.server=server;self.agents=list(server.trusted_agents);self.versions=versions
        self.context=dict(purpose='unclassified',iteration=None,anchor_id=None,attempt=None)
        self.events=[];self.pending=None;self.last_eval_id=None;self.samples=0;self.query_count=0
        self.originals=[];self.errors=[];self.started=time.perf_counter()
    def __enter__(self):
        for j,ag in enumerate(self.agents):
            old=ag.api_predict;self.originals.append((ag,old))
            def call(x,_j=j,_old=old):
                if self.pending is None or len(self.pending)!=_j:raise ValueError('unsupported_query_order')
                if x is not self.live_x:raise ValueError('unsupported_changed_query_input')
                t=time.perf_counter()
                try:y=_old(x)
                except Exception as exc:
                    self.errors.append(dict(model_index=_j,error=repr(exc),eval_id=len(self.events)));raise
                dt=time.perf_counter()-t
                self.pending.append((np.asarray(y).copy(),dt));self.samples+=1;self.query_count+=1
                return y
            ag.api_predict=call
        return self
    def __exit__(self,*args):
        for ag,old in self.originals:ag.api_predict=old
    def __call__(self,x):
        arr=np.asarray(x)
        if arr.ndim!=1 or not np.issubdtype(arr.dtype,np.floating):raise ValueError('unsupported_input')
        if self.samples+len(self.agents)>20000 or time.perf_counter()-self.started>60:raise RuntimeError('per_solve_budget')
        saved_x=arr.copy();self.pending=[];self.live_x=x
        value=self.server._consensus_loss(x,agents=self.agents,mode='fixed',weights=self.server.alphas)
        if len(self.pending)!=len(self.agents):raise ValueError('missing_reply')
        if not np.array_equal(arr,saved_x,equal_nan=True):raise ValueError('mutated_query_input')
        self.last_eval_id=len(self.events)
        self.events.append((saved_x,float(value),self.context.copy(),self.pending));self.pending=None
        return value
    def materialize(self):
        queries=[];evaluations=[];digest=hashlib.sha256();seen={};unique=set();modeltime=[0.]*len(self.agents)
        for eid,(x,value,context,replies) in enumerate(self.events):
            if not np.isfinite(x).all() or not np.isfinite(value):raise ValueError('nonfinite_reply')
            xhex=key(x);xdtype=str(x.dtype);shape=list(x.shape);qids=[];losses=[]
            for j,(y,dt) in enumerate(replies):
                if y.shape!=(1,) or not np.isfinite(y).all():raise ValueError('nonfinite_or_unsupported_reply')
                lookup=(j,xhex);yb=y.tobytes()
                if lookup in seen and seen[lookup]!=(str(y.dtype),yb):raise ValueError('unsupported_stochastic')
                seen[lookup]=(str(y.dtype),yb);unique.add((j,self.versions[j],xhex))
                payload=f'{j}:{x.dtype}:{x.shape}:'.encode()+x.tobytes()+yb
                digest.update(len(payload).to_bytes(8,'little'));digest.update(payload)
                qid=len(queries);qids.append(qid);modeltime[j]+=dt;losses.append(abs(float(y[0])-self.server.target_T))
                queries.append(dict(query_id=qid,eval_id=eid,model_index=j,model_id=str(self.agents[j].agent_id),model_version=self.versions[j],
                    x=x,input_hex=xhex,input_dtype=xdtype,input_shape=shape,output=y,output_dtype=str(y.dtype),output_shape=list(y.shape),
                    call_seconds=dt,input_samples=1,success=True))
            evaluations.append(dict(eval_id=eid,event_index=eid,**context,x=x,model_query_ids=qids,
                                    individual_losses=np.asarray(losses),objective_value=value,weights_version='fixed_v1'))
        return dict(queries=queries,evaluations=evaluations,costs=dict(api_calls=self.query_count,input_samples=self.samples,unique_inputs=len(unique),
                    per_model_seconds=modeltime,model_seconds=sum(modeltime),query_sha256=digest.hexdigest(),additional_prediction_calls=0))


def record_compact_run(server,start,manifest):
    validate_manifest(manifest)
    if manifest['history_metric_kind']!='objective':raise ValueError('unsupported_history_metric')
    from host_server import HostServer
    method=instrument(HostServer._solve_subspace_custom);trace=CompactTrace(server,manifest['model_versions']);obs=Observer(trace,manifest)
    t=time.perf_counter()
    with trace:
        result=method(server,np.asarray(manifest['anchors']),trace,direction_weights=np.asarray(manifest['direction_weights_raw']),
                      initial_solution=np.asarray(start),_observer=obs,**manifest['solver_config'])
    search=time.perf_counter()-t
    if obs.initial_eval is None or obs.final_eval is None:raise ValueError('missing_endpoint_eval')
    t=time.perf_counter();expanded=trace.materialize();materialize=time.perf_counter()-t
    return dict(result=plain(result),costs=dict(**expanded['costs'],search_seconds=search,materialize_seconds=materialize),
                queries=expanded['queries'],evaluations=expanded['evaluations'],iterations=obs.iterations,initial_eval_id=obs.initial_eval,
                final_eval_id=obs.final_eval,metric_events=obs.metric_events,execution_status='complete',instrumentation_ast_restored=True,
                capture_version='compact_v1',deferred_checks_passed=True)
