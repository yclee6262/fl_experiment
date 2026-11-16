"""Unshrunk controls for the already completed six-method cohort."""
import torch
from unified_alpha_methods import initial_alpha as unified
from peer_support_alpha import support_signal
METHODS=['stacked_full','pure_support']
SCENARIOS=['reference_clean','low_quality_25pct','sample_imbalance_4x']
def initializer(peer):
 def initial(method,agents,mse,x,y,proposals,target,**kwargs):
  if method=='stacked_full':return unified(method,agents,mse,x,y,proposals,target,**kwargs)
  if method!='pure_support':raise ValueError(method)
  w,a=support_signal(peer,target)
  if len(w)!=len(agents):raise ValueError('Agent order/size mismatch')
  a.update(agent_ids=[g.agent_id for g in agents],uniform_fraction=0.,quality_fraction=0.,formula='alpha=s')
  return w,a
 return initial
