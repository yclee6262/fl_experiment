import copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
WORK=Path(__file__).resolve().parents[2];CA=WORK/'contribution_analysis'
ROOT=CA/'path_intervention_experiment_v1_20261024'
OLD=CA/'custom_direction_experiment_v2_20261020'
MOD=CA/'model_step_decision_experiment_v1_20261021'
L2=CA/'combined_contribution_layer2_experiment_v1_20261021'
PLAN=WORK/'docs/contribution_path_intervention_experiment_plan_20261024.md'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [plain(x) for x in v]
    return v
def save(p,v):
    with Path(p).open('x') as f:json.dump(plain(v),f,ensure_ascii=False,indent=2,allow_nan=False)
def key(i,x):return str(i)+':'+np.asarray(x,np.float64).tobytes().hex()
def near(a,b):
    aa=np.asarray(a,float);bb=np.asarray(b,float)
    np.testing.assert_allclose(aa,bb,atol=1e-10,rtol=1e-10)
    return float(np.max(abs(aa-bb))) if aa.size else 0.
def setup():
    sys.path.insert(0,str(ROOT/'source_snapshot'))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
def check():
    for p,h in read(ROOT/'inputs.json').items():assert sha(p)==h,('input_drift',p)
    for p,h in read(ROOT/'source_lock.json').items():assert sha(ROOT/'source_snapshot'/p)==h
class Budget:
    def __init__(self,cap,seconds):self.cap=cap;self.seconds=seconds;self.samples=0;self.model_seconds=0.;self.start=time.perf_counter()
    def before(self):
        if self.samples>=self.cap or time.perf_counter()-self.start>self.seconds:raise RuntimeError('resource_budget')
    def charge(self,dt):self.samples+=1;self.model_seconds+=dt
class Queries:
    def __init__(self,s,budget):self.s=s;self.budget=budget;self.raw={};self.events=[];self.stage='baseline';self.context={}
    def reply(self,i,x):
        k=key(i,x);hit=self.stage!='baseline' and k in self.raw
        if not hit:
            self.budget.before();tic=time.perf_counter();y=float(self.s.trusted_agents[i].api_predict(np.asarray(x))[0]);self.budget.charge(time.perf_counter()-tic)
            if k in self.raw:assert self.raw[k]['y']==y
            self.raw[k]=dict(i=i,x=plain(np.asarray(x)),api_input_float32=np.asarray(x,np.float32).tolist(),y=y)
        self.events.append(dict(key=k,stage=self.stage,hit=hit,**self.context));return abs(self.raw[k]['y'])
    def objective(self,panel):
        w=np.asarray(self.s.alphas)[panel];w=w/w.sum()
        return lambda x:float(np.dot(w,[self.reply(j,x) for j in panel]))
    def full(self,x):return self.objective(list(range(len(self.s.alphas))))(x)
