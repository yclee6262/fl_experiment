"""Minimal streaming observer, preserves the frozen solver's numeric operations."""
import run as r
import numpy as np,time,json
spec=r.importlib.util.spec_from_file_location('old_observer',r.CA/'custom_direction_v1/observer.py');obs=r.importlib.util.module_from_spec(spec);spec.loader.exec_module(obs)
compiled=obs.instrument(r.HostServer._solve_subspace_custom)
class Streaming:
 def __init__(self,n):self.total=np.zeros((n,2));self.last=np.zeros((n,2));self.h=np.zeros((n,2));self.seconds=0.;self.accepted=0
 def begin(self,k,x,mode,loss):
  t=time.perf_counter();self.x=x.copy();self.h.fill(0);self.seconds+=time.perf_counter()-t
 def direction(self,k,u,width,point,derivative,weight):
  t=time.perf_counter();self.h[k]=weight*derivative*u;self.seconds+=time.perf_counter()-t
 def combined(self,g,norm):self.norm=norm
 def trial(self,x,loss,previous):
  t=time.perf_counter()
  if loss<previous:
   self.last=-np.linalg.norm(x-self.x)*self.h/self.norm;self.total+=self.last;self.accepted+=1
  self.seconds+=time.perf_counter()-t

def main():
 records=r.read(r.OUT/'scored.json');rows=[];start=time.perf_counter();before=r.Q.total_new
 for ci,rec in enumerate(records):
  c=rec['case'];base,model,groups,rem=r.prep(c);s=r.load_server(c);n=len(c['ids']);a=np.asarray(c['alphas'])
  for rep in range(4):
   order=['CUM','LAST_ANCHOR_REPLAY'] if (ci+rep)%2 else ['LAST_ANCHOR_REPLAY','CUM']
   for name in order:
    q=r.Q(s);stream=Streaming(n);tic=time.perf_counter()
    if name=='CUM':
     br=compiled(s,s.I_list,q.objective(list(range(n))),num_iterations=30,direction_weights=a,initial_solution=np.asarray(c['fixed_start']),_observer=stream)
     raw=np.asarray(br['solution'])-stream.total
    else:
     br=r.HostServer._solve_subspace_custom(s,s.I_list,q.objective(list(range(n))),num_iterations=30,direction_weights=a,initial_solution=np.asarray(c['fixed_start']))
     # frozen engine resume state is existing input; timing excludes creating it
     k=rec['checks']['last_accepted'];raw=np.array([r.four.local(s,q,model['full'][k]['pre'],i,'anchor') for i in range(n)])
    xx=np.clip(raw,-1,1);q.stage='common';lf=q.losses(br['solution']);scores=np.array([np.dot(a,q.losses(x)-lf) for x in xx]);elapsed=time.perf_counter()-tic
    np.testing.assert_allclose(xx,rec['methods'][name]['x'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(scores,rec['methods'][name]['full'],rtol=0,atol=1e-10)
    samples=sum(not e['hit'] for e in q.events);assert samples==rec['methods'][name]['total_samples']
    if rep:rows.append(dict(key=c['key'],scenario=c['scenario'],method=name,repeat=rep,seconds=elapsed,stream_vector_seconds=stream.seconds,samples=samples))
 r.save('streaming_timing.json',rows);r.save('streaming_verification.json',dict(physical_samples=r.Q.total_new-before,wall_seconds=time.perf_counter()-start,all_endpoints_scores_counts_match=True,
 limitation='CUM includes streaming observation; LAST replay uses saved round state, so excludes online state-capture overhead. LOO timing comes from original same-session four-pass timing.'))
if __name__=='__main__':main()
