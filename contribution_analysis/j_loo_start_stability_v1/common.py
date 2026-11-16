import hashlib,json,sys,time
from pathlib import Path
import numpy as np
WORK=Path(__file__).resolve().parents[2]; CA=WORK/'contribution_analysis'
ROOT=CA/'j_loo_start_stability_experiment_v1_20261027'
MOD=CA/'model_step_decision_experiment_v1_20261021'
L2=CA/'combined_contribution_layer2_experiment_v1_20261021'
FOUR=CA/'contribution_four_way_experiment_v2_20261022'
PLAN=WORK/'docs/j_loo_start_stability_plan_20261027.md'
EPS=1e-4
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [plain(x) for x in v]
    return v
def save(p,v):
    with Path(p).open('x') as f:json.dump(plain(v),f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
def key(i,x):return str(i)+':'+np.asarray(x,np.float64).tobytes().hex()
def near(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10)
    return float(np.max(abs(a-b))) if a.size else 0.
def setup():
    sys.path.insert(0,str(ROOT/'frozen'))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
def check():
    for p,h in read(ROOT/'inputs.json').items():assert sha(p)==h,('input_drift',p)
    for p,h in read(ROOT/'source_lock.json').items():assert sha(ROOT/p)==h,('source_drift',p)
def starts(c):
    x=np.array(c['fixed_start'],float);assert x.shape==(2,)
    values=[np.clip(x+v,-1,1) for v in [[0,0],[.25,0],[-.25,0],[0,.25],[0,-.25]]]
    assert len({v.tobytes() for v in values})==5
    assert np.isfinite(values).all()
    return values
class Budget:
    def __init__(self,cap,seconds):self.cap=cap;self.samples=0;self.start=time.perf_counter();self.seconds=seconds
    def before(self):
        if self.samples>=self.cap or time.perf_counter()-self.start>self.seconds:raise RuntimeError('budget_exceeded')
class Queries:
    def __init__(self,s,budget):self.s=s;self.budget=budget;self.raw={};self.events=[];self.stage='baseline';self.context={};self.model_seconds=0.
    def reply(self,i,x):
        k=key(i,x);hit=self.stage!='baseline' and k in self.raw
        if not hit:
            self.budget.before();tic=time.perf_counter();y=float(self.s.trusted_agents[i].api_predict(np.asarray(x))[0])
            self.model_seconds+=time.perf_counter()-tic;self.budget.samples+=1
            if k in self.raw:assert self.raw[k]['y']==y
            self.raw[k]=dict(i=i,x=np.asarray(x).tolist(),y=y)
        self.events.append(dict(key=k,stage=self.stage,hit=hit,**self.context))
        return abs(self.raw[k]['y'])
    def objective(self,panel):
        w=np.asarray(self.s.alphas)[panel];w=w/w.sum()
        return lambda x:float(np.dot(w,[self.reply(i,x) for i in panel]))
    def full(self,x):return self.objective(list(range(len(self.s.alphas))))(x)
