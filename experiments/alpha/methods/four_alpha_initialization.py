"""Frozen four-candidate 2x2 study: E/S signals mixed equally with U/B."""
import torch  # Load the existing numerical runtime before SciPy imports.
import numpy as np
from initial_alpha_methods import initial_alpha as legacy
from unified_alpha_methods import initial_alpha as unified
from support_quality_mix_experiment import mixed_weights
from peer_support_alpha import weights
METHODS=['A_uniform','B_stage0','ensemble_shrink','B_stacked_half','peer_support_shrink','support_quality_mix50']
SCENARIOS=['reference_clean','low_quality_25pct','sample_imbalance_4x']
LABELS=dict(zip(METHODS,['A：均勻','B：品質','EU：組合均勻','EB：組合品質','SU：支持均勻','SB：支持品質']))
CONTRASTS=[(m,b) for m in METHODS[2:] for b in METHODS[:2]]+[
 ('B_stacked_half','ensemble_shrink'),('support_quality_mix50','peer_support_shrink'),
 ('peer_support_shrink','ensemble_shrink'),('support_quality_mix50','B_stacked_half')]

def initializer(peer):
 def initial(method,agents,mse,x,y,proposals,target,**kwargs):
  if method=='ensemble_shrink':return legacy(method,agents,mse,x,y,proposals,target,**kwargs)
  if method=='B_stacked_half':return unified(method,agents,mse,x,y,proposals,target,**kwargs)
  if method=='support_quality_mix50':w,a=mixed_weights(mse,peer,target)
  else:w,a=weights(method,mse,peer,target)
  a['agent_ids']=[g.agent_id for g in agents];return w,a
 return initial
