"""Reviewer-scoped sensitivity, duplicate identity and raw-prediction audit estimator."""
import sys,json,copy,hashlib,time,signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from contribution_analysis.model_pool_v_cm_validation_v1.run import CA,W,FROZEN,load_server,run,endpoint,construct_directions,decide,plain
from contribution_analysis.model_step_decision_v1.engine import branch_state
from contribution_analysis.senior_three_packages_v1.run import share,metrics,vector_scores
import numpy as np
OUT=CA/'senior_remaining_20261109_attempt_01'
CAP=3000000

def save(p,x):p.write_text(json.dumps(plain(x),ensure_ascii=False,indent=2,allow_nan=False))
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Q:
    total=0
    def __init__(self,s):self.s=s;self.raw={};self.count=0;self.stage='baseline';self.stages={}
    def pred(self,i,x):
        k=(i,np.asarray(x,np.float64).tobytes())
        if k not in self.raw:
            if Q.total>=CAP:raise RuntimeError('physical cap')
            v=float(self.s.trusted_agents[i].api_predict(np.asarray(x))[0]);assert np.isfinite(v)
            self.raw[k]=v;self.count+=1;Q.total+=1;self.stages[self.stage]=self.stages.get(self.stage,0)+1
        return self.raw[k]
    def loss(self,i,x):return abs(self.pred(i,x)-self.s.target_T)
    def losses(self,x):return np.array([self.loss(i,x) for i in range(len(self.s.alphas))])
    def objective(self,ids):
        ids=list(ids);a=np.array(self.s.alphas)[ids];a=a/a.sum()
        return lambda x:float(a@np.array([self.loss(i,x) for i in ids]))

def unnormalized(s,obj,x,g,before,eta,mode):
    if np.linalg.norm(g)<=1e-8:return dict(x=x,loss=before)
    for _ in range(10):
        z=x-s._bounded_step(x,-g,eta)*g;v=obj(z)
        if v<before:return dict(x=z,loss=v)
        eta/=2
    return dict(x=x,loss=before)

def scores(losses,lf,a,i):return [float(a@(losses-lf)),float(np.delete(a*(losses-lf),i).sum()/(1-a[i]))]

def execute(c,s,diagnostics=False,sparse=None,check=False):
    n=len(s.alphas);a=np.array(s.alphas);a/=a.sum();ids=list(range(n));q=Q(s)
    base,tr=run(s,q.objective(ids),30,initial=np.asarray(c['fixed_start']));baseline=q.count
    # Capture baseline cache before any attribution branch is executed.
    baseline_raw=dict(q.raw);rows=[];rng=np.random.default_rng(202611091000+int(c['seed'])+int(0 if sparse is None else sparse*100))
    for t in range(len(tr.pre)):
        f=endpoint(tr,t);p=f['pre'];x=np.asarray(p['S_current']);q.stage='features';cur,ds,H,G=construct_directions(s,q,a,f)
        assert q.count==baseline or not diagnostics or t>0
        lf=q.losses(f['post']['S_current']);g=np.array(f['g']);points=np.array([d['point'] for d in ds]);D=points-x if len(points) else np.zeros((0,len(x)))
        assert all((j,np.asarray(z,np.float64).tobytes()) in baseline_raw for z in [x,*points] for j in ids)
        rawcur=np.array([baseline_raw[(j,x.tobytes())] for j in ids]);rawpts=np.array([[baseline_raw[(j,z.tobytes())] for j in ids] for z in points]) if len(points) else np.zeros((0,n))
        B=np.linalg.lstsq(D,rawpts-rawcur,rcond=None)[0] if len(points) else np.zeros((len(x),n))
        def ploss(z):return np.abs(rawcur+(np.asarray(z)-x)@B-s.target_T)
        for i in ids:
            keep=[j for j in ids if j!=i];w=a[keep]/a[keep].sum();gm=(g-G[i])/(1-a[i]);ga=g-H[i]
            alt=[];pr=[];mask=[];diag={}
            for role,gg,panel,before in [('A',ga,ids,float(a@cur)),('M',gm,keep,float(w@cur[keep]))]:
                ww=a if role=='A' else w
                pobj=lambda z:float(ww@ploss(z)[panel])
                approx=decide(s,pobj,x,gg,before,p['eta'],p['current_method']);pv=scores(ploss(approx['x']),lf,a,i);pr.append(pv)
                sampled=sparse is None or rng.random()<sparse;mask.append(sampled)
                if sampled:
                    q.stage=role;objective=q.objective(panel);z=decide(s,objective,x,gg,before,p['eta'],p['current_method']);q.stage='common';l=q.losses(z['x']);alt.append(scores(l,lf,a,i))
                    if role=='M' and check:
                        q.stage='engine_check';state=branch_state(p,objective,s.I_list);_,bt=run(s,objective,1,resume=state)
                        assert np.allclose(z['x'],bt.final['S_current'],rtol=0,atol=1e-10)
                else:alt.append(None)
                if diagnostics:
                    q.stage='normalization';unn=unnormalized(s,q.objective(panel),x,gg,before,p['eta'],p['current_method']);diag[role+'_unnorm']=scores(q.losses(unn['x']),lf,a,i)
                    diag[role+'_norm']=float(np.linalg.norm(gg))
                    if role=='A':
                        ren=decide(s,q.objective(ids),x,gg/(1-a[i]),before,p['eta'],p['current_method']);diag['A_renorm']=scores(q.losses(ren['x']),lf,a,i)
            rows.append(dict(t=t,i=i,exact=alt,proxy=pr,sampled=mask,diag=diag,rank=int(np.linalg.matrix_rank(D)),deltas=None if sparse is not None else [ (q.losses(decide(s,q.objective(ids if r==0 else keep),x,ga if r==0 else gm,float(a@cur) if r==0 else float(w@cur[keep]),p['eta'],p['current_method'])['x'])-lf).tolist() for r in range(2)]))
    # Repeated delta read above only for endpoint scoring-weight diagnostics; all hits.
    arr=np.zeros((n,2,2))
    for r in rows:
        ex=np.array([v if v is not None else r['proxy'][k] for k,v in enumerate(r['exact'])]);proxy=np.array(r['proxy'])
        est=ex if sparse is None else proxy+(ex-proxy)*np.array(r['sampled'])[:,None]/sparse
        arr[r['i']]+=est
    return dict(key=c['key'],n=n,base=base,sums=arr,rows=rows,baseline=baseline,calls=q.count,stages=q.stages,weights=a,start=c['fixed_start'])

def alloc(S,tau=1e-10,budget=(None,500,500),fixedbasic=None):
    n=len(S);S=np.asarray(S);b=100 if fixedbasic is None else fixedbasic/n
    basic=np.where((S[:,0]<-tau)&(S[:,1]<-tau),0.,b)
    A=share(S[:,0],tau);M=share(S[:,1],tau)
    if budget[0] is not None:basic=basic/100*budget[0]/n if fixedbasic is None else basic
    amounts=basic+budget[1]*A[:-1]+budget[2]*M[:-1]
    reserve=(n*b-basic.sum())+budget[1]*A[-1]+budget[2]*M[-1]
    assert abs(amounts.sum()+reserve-(n*b+budget[1]+budget[2]))<1e-7
    return dict(basic=basic,A=A,M=M,total=amounts,reserve=reserve)

def main():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    OUT.mkdir(exist_ok=False);(OUT/'records').mkdir()
    cases=read(CA/'combined_contribution_layer2_experiment_v1_20261021/cases.json')['development'];paths=[Path(__file__),W/'docs/senior_remaining_design_20261109.md']+list(FROZEN.glob('*.py'))+list({Path(c['models']) for c in cases})
    paths+= [Path(m.__file__) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(m.__file__).startswith(str(W)) and str(m.__file__).endswith('.py')]
    locks={str(p):sha(p) for p in paths};save(OUT/'source_lock.json',locks);save(OUT/'registration.json',dict(cases=cases,rates=[.05,.1,.2,.3,.4,.5],replicates=1000,cap=CAP,wall=900,design_sha=sha(paths[1])))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall cap')));signal.alarm(900);start=time.perf_counter();summary=[]
    try:
        for ci,c in enumerate(cases):
            print(c['key'],flush=True);s=load_server(c);base=execute(c,s,True,check=True);save(OUT/'records'/f"{c['key']}_base.json",base)
            old=read(CA/f"method5_cost_experiment_20261030_attempt_04/records/{c['key']}_New.json")
            assert np.allclose(base['base']['solution'],old['base']['solution'],atol=1e-10,rtol=0)
            err=max(abs(r['exact'][k][0]-o['C_A' if k==0 else 'C_M']) for r,o in zip(base['rows'],old['rows']) for k in [0,1]);assert err<1e-8
            S=np.array(base['sums'])[:,:,0];pay=alloc(S);group='development' if int(c['seed'])%10<2 else 'validation_existing_models'
            stable=[];rng=np.random.default_rng(2026110900+ci)
            for z in range(4):
                cc=copy.deepcopy(c);cc['fixed_start']=rng.uniform(-.5,.5,2).tolist();r=execute(cc,s);save(OUT/'records'/f"{c['key']}_start{z}.json",r);stable.append(dict(S=r['sums'],allocation=alloc(np.array(r['sums'])[:,:,0])))
            weights=[]
            for owner in range(len(S)):
                ss=copy.copy(s);aa=np.ones(len(S));aa[owner]=2;ss.alphas=aa/aa.sum();r=execute(c,ss);save(OUT/'records'/f"{c['key']}_weight{owner}.json",r)
                fixed=np.zeros_like(S)
                for row in base['rows']:fixed[row['i']]+=np.array(row['deltas'])@ss.alphas
                weights.append(dict(owner=owner,alpha=ss.alphas,fixed=fixed,path=np.array(r['sums'])[:,:,0],allocation=alloc(np.array(r['sums'])[:,:,0])))
            profiles=read(Path(c['models']).parent/'metadata.json')['agent_data_profile'];poison={p['agent_id']:p['is_poisoned'] for p in profiles};owner=next(i for i,v in enumerate(c['ids']) if not poison[v]);syb=[]
            for policy in ['split_weight','equal_identity']:
                ss=copy.copy(s);ss.I_list=np.vstack([s.I_list,s.I_list[owner]]);ss.trusted_agents=list(s.trusted_agents)+[s.trusted_agents[owner]];aa=np.r_[s.alphas,0.];aa[owner]/=2;aa[-1]=aa[owner];ss.alphas=aa if policy=='split_weight' else np.ones(len(aa))/len(aa)
                r=execute(c,ss);save(OUT/'records'/f"{c['key']}_sybil_{policy}.json",r);sp=np.array(r['sums'])[:,:,0]
                syb.append(dict(policy=policy,owner=owner,original=pay['total'][owner],per_identity=alloc(sp)['total'][[owner,-1]].sum(),fixed_budget=alloc(sp,fixedbasic=100*len(S))['total'][[owner,-1]].sum(),duplicate_equal=True))
            # Actual complete LOO cost, with new empty per-arm cache and same original baseline.
            q=Q(s);ids=list(range(len(S)));q.stage='baseline';_,_=run(s,q.objective(ids),30,initial=np.asarray(c['fixed_start']));loo_base=q.count
            for i in ids:
                keep=[j for j in ids if j!=i];ss=copy.copy(s);ss.I_list=np.asarray(s.I_list)[keep];ss.alphas=np.asarray(s.alphas)[keep];ss.alphas/=ss.alphas.sum();q.stage='LOO';r,_=run(ss,q.objective(keep),30,initial=np.asarray(c['fixed_start']));q.stage='common';q.losses(r['solution']);q.losses(base['base']['solution'])
            sparse=[]
            for rate in [.05,.1,.2,.3,.4,.5]:
                r=execute(c,s,sparse=rate);save(OUT/'records'/f"{c['key']}_sparse{rate}.json",r);sparse.append(dict(rate=rate,calls=r['calls'],baseline=r['baseline'],stages=r['stages']))
            summary.append(dict(key=c['key'],group=group,scenario=c['scenario'],baseline=base['baseline'],score_error=err,stable=stable,weights=weights,sybil=syb,sparse=sparse,LOO_calls=q.count,LOO_baseline=loo_base,base_allocation=pay));save(OUT/'summary.json',summary)
        assert all(sha(Path(p))==h for p,h in locks.items());save(OUT/'completion.json',dict(complete=True,physical=Q.total,seconds=time.perf_counter()-start,source_unchanged=True))
    except Exception as e:save(OUT/'failure.json',dict(error=repr(e),physical=Q.total,teams=len(summary)));raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
