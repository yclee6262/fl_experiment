import time
import hashlib
import numpy as np
from scipy.interpolate import LinearNDInterpolator,RBFInterpolator
from scipy.spatial import cKDTree

def key(x):return np.asarray(x,dtype=np.float32).tobytes()

def record_normal(server,initial):
 totals=dict(calls=0,samples=0);points={};original=[];digest=hashlib.sha256()
 for j,agent in enumerate(server.trusted_agents):
  prior=agent.api_predict;original.append((agent,prior))
  def prediction(x,_j=j,_prior=prior):
   digest.update(_j.to_bytes(4,'little'));digest.update(np.asarray(x,dtype=np.float64).tobytes())
   y=_prior(x);totals['calls']+=1;totals['samples']+=1
   if np.asarray(x).ndim!=1:raise ValueError('scalar_only')
   k=key(x)
   if k not in points:points[k]=dict(x=np.asarray(x,dtype=np.float32).astype(float),values=np.full(len(server.trusted_agents),np.nan))
   points[k]['values'][_j]=float(y[0]);return y
  agent.api_predict=prediction
 begin=time.perf_counter()
 try:r=server.optimize_candidate_subspace(strategy='custom',mode='optimization',custom_iterations=30,initial_solution=np.array(initial))
 finally:
  for a,p in original:a.api_predict=p
 if any(not np.isfinite(p['values']).all() for p in points.values()):raise ValueError('incomplete_reply_vector')
 return dict(query_order_sha256=digest.hexdigest(),result=r,points=np.array([p['x'] for p in points.values()]),values=np.array([p['values'] for p in points.values()]),counter=totals,seconds=time.perf_counter()-begin)

def grid(points,n):
 lo=np.min(points,axis=0);hi=np.max(points,axis=0);width=np.maximum(hi-lo,.05);mid=(hi+lo)/2
 lo=np.maximum(-1,mid-.7*width);hi=np.minimum(1,mid+.7*width)
 return np.array(np.meshgrid(*[np.linspace(a,b,n) for a,b in zip(lo,hi)],indexing='ij')).reshape(len(lo),-1).T

class Proxy:
 def __init__(self,p,v,method):
  self.points=p;self.values=v;self.cache={key(x):y for x,y in zip(p,v)};self.outside_calls=0;self.calls=0
  self.tree=cKDTree(p);self.linear=LinearNDInterpolator(p,v)
  self.f=None if method=='linear' else RBFInterpolator(p,v,kernel='thin_plate_spline',degree=1,smoothing=1e-10,neighbors=min(32,len(p)))
 def __call__(self,x):
  self.calls+=1;k=key(x)
  if k in self.cache:return self.cache[k]
  z=np.asarray(x,dtype=np.float32).astype(float).reshape(1,-1);a=np.asarray(self.linear(z)).reshape(-1)
  if not np.isfinite(a).all():self.outside_calls+=1
  if self.f is not None:a=np.asarray(self.f(z)).reshape(-1)
  elif not np.isfinite(a).all():a=self.values[self.tree.query(z[0])[1]]
  if not np.isfinite(a).all():raise ValueError('nonfinite_proxy')
  return a.astype(np.float32).astype(float)

def simulate(server,proxy,case,k):
 ids=[i for j,i in enumerate(case['ids']) if j!=k];mask=np.arange(len(case['ids']))!=k;w=np.array(case['alphas'])[mask];w=w/w.sum()
 def objective(x):return float(w@abs(proxy(x)[mask]))
 before=proxy.calls;outside=proxy.outside_calls;start=time.perf_counter()
 r=server._solve_subspace_custom(np.array(case['candidates'])[mask],objective,num_iterations=30,direction_weights=w,initial_solution=np.array(case['fixed_start']))
 if np.max(abs(r['solution']))>1+1e-8:raise ValueError('infeasible_proxy_endpoint')
 return dict(result=r,remaining_ids=ids,remaining_weights=w,proxy_calls=proxy.calls-before,outside_calls=proxy.outside_calls-outside,seconds=time.perf_counter()-start)
