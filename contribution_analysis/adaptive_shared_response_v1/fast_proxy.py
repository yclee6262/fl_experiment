"""Behavior-preserving caches for the installed SciPy local RBF evaluator.

No kernel, neighborhood, point, objective, or statistical estimator changes.
"""
import numpy as np
from scipy.interpolate._rbfinterp import _build_and_solve_system
from proxy_core import Proxy

class CachedSingleRBF:
 def __init__(self,original):self.original=original;self.neighborhoods={}
 def __call__(self,x):
  r=self.original;x=np.asarray(x,dtype=float,order='C')
  if len(x)!=1:return r(x)
  _,idx=r._tree.query(x,r.neighbors);idx=np.sort(np.asarray(idx).reshape(-1));k=tuple(idx)
  if k not in self.neighborhoods:
   shift,scale,coeff=_build_and_solve_system(r.y[idx],r.d[idx],r.smoothing[idx],r.kernel,r.epsilon,r.powers);self.neighborhoods[k]=(shift,scale,coeff)
  shift,scale,coeff=self.neighborhoods[k];budget=max(x.size+r.y.size+r.d.size,1000000)
  out=r._chunk_evaluator(x,r.y[idx],shift,scale,coeff,memory_budget=budget)
  return out.view(r.d_dtype).reshape((len(x),)+r.d_shape)

class FastProxy(Proxy):
 def __init__(self,p,v,method):
  super().__init__(p,v,method);self.memo={}
  if self.f is not None:self.f=CachedSingleRBF(self.f)
 def __call__(self,x):
  # Original predictor evaluates float32 inputs; identical float32 keys are identical calls.
  k=np.asarray(x,dtype=np.float32).tobytes()
  if k in self.cache:self.calls+=1;return self.cache[k]
  if k in self.memo:
   self.calls+=1;value,outside=self.memo[k];self.outside_calls+=outside;return value
  before=self.outside_calls;value=super().__call__(x);self.memo[k]=(value,self.outside_calls-before);return value

def batch_values(proxy,xs):
 """Vectorize uncertainty-only evaluations; retain the same interpolant and float32 outputs."""
 xs=np.asarray(xs,dtype=np.float32).astype(float);linear=np.array(proxy.linear(xs),dtype=float);outside=~np.isfinite(linear).all(axis=1)
 proxy.calls+=len(xs);proxy.outside_calls+=int(outside.sum())
 if proxy.f is not None:
  f=proxy.f.original if isinstance(proxy.f,CachedSingleRBF) else proxy.f
  values=np.array(f(xs),dtype=float)
 else:
  values=linear
  if outside.any():values[outside]=proxy.values[proxy.tree.query(xs[outside])[1]]
 # This is only used for unseen uncertainty candidates, as in the original scalar loop.
 if not np.isfinite(values).all():raise ValueError('nonfinite_batch_proxy')
 return values.astype(np.float32).astype(float)
