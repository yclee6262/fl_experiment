import json,hashlib,time,sys
from pathlib import Path
import numpy as np
WORK=Path(__file__).resolve().parents[2]
ROOT=WORK/'contribution_analysis/low_cost_local_effect_experiment_v1_20261021'
PREV=WORK/'contribution_analysis/model_step_decision_experiment_v1_20261021'
ORACLE=WORK/'contribution_analysis/combined_contribution_layer2_experiment_v1_20261021'
DESIGN=WORK/'docs/low_cost_local_effect_experiment_plan_20261021.md'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x
def save(p,v):
    with Path(p).open('x') as f:json.dump(plain(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def setup():
    sys.path.insert(0,str(ROOT/'source_snapshot'))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
def check():
    for n,h in read(ROOT/'inputs.json').items():assert sha(n)==h,('input_drift',n)
    for n,h in read(ROOT/'source_lock.json').items():assert sha(ROOT/'source_snapshot'/n)==h,('source_drift',n)
def near(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);err=float(np.max(np.abs(a-b))) if a.size else 0.
    assert err<=1e-10*max(1.,float(np.max(np.abs(b))) if b.size else 0.),('numerical_mismatch',err)
    return err

class Limit:
    def __init__(self,n,seconds):self.n=n;self.seconds=seconds;self.samples=0;self.model_seconds=0.;self.start=time.perf_counter()
    def before(self):
        if self.samples>=self.n or time.perf_counter()-self.start>self.seconds:raise RuntimeError('stage_budget_exceeded')
    def charge(self,dt):self.samples+=1;self.model_seconds+=dt
