"""Structural verification repairs; locked weights stay fixed."""
import json
import numpy as np
import four_way_bayes as run
import alpha_mixture_exploration as prior
from unified_alpha_methods import initial_alpha as unified
from pathlib import Path
r=run.ROOT;reg=json.loads((r/'registration.json').read_text());lock=json.loads((r/'locked_candidate.json').read_text());w=np.array(lock['w'])
# At the lower face the three outward Helmert probes had zero radius. Replace
# only those with fixed tangent transfers Q->E, Q->S, E->S (positive orientation).
for k,v in enumerate([[0,1,-1,0],[0,1,0,-1],[0,0,1,-1]]):
 v=np.array(v,float)/np.sqrt(2);run.evaluate(f'tangent_{k}',w+.05*v,reg['training_batches'])
original=prior.initializer
prior.initializer=lambda w,peer:lambda method,*args,**kwargs:unified('ensemble_shrink',*args,**kwargs)
run.evaluate('audit_EU_canonical',np.array([.5,0,.5,0]),reg['audit_batches'])
prior.initializer=original
