"""Offline fixed-mixture learning, bounded sequential expected improvement."""
import contextlib,json,time,warnings
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,Matern
import alpha_mixture_exploration as prior
import oob_shrinkage_research as shared
from research_evaluation import load_evaluator_artifact,score_solution,synthetic_truth
ROOT=Path('alpha_weight_outputs/four_way_bayes_v1_20261116')
def put(n,v):shared.dump(ROOT/n,v)
def read(p):return prior.read(p)
def metrics(rows,baseline):
    seeds=sorted({r['seed'] for r in rows});d=np.array([np.mean([r['pooled_error']-baseline[r['seed'],r['scenario']]['pooled_error'] for r in rows if r['seed']==s]) for s in seeds])
    return d,float(d.mean()+d.std(ddof=1)/np.sqrt(len(d)))
def propose(ws,ys,iteration):
    with warnings.catch_warnings(record=True) as ww:
        gp=GaussianProcessRegressor(kernel=ConstantKernel(1.,(.01,100))*Matern(.3,(.05,2.),nu=.5),alpha=1e-6,normalize_y=True,n_restarts_optimizer=2,random_state=20261116).fit(ws,ys)
    interior=[y for w,y in zip(ws,ys) if min(w)>=.05-1e-12];best=min(interior)
    def ei(w):
        mu,sigma=gp.predict(np.asarray(w)[None,:],return_std=True);s=max(float(sigma[0]),1e-15);imp=best-float(mu[0]);z=imp/s
        return float(imp*ndtr(z)+s*np.exp(-z*z/2)/np.sqrt(2*np.pi))
    starts=.05+.8*np.random.default_rng(20261116+iteration).dirichlet(np.ones(4),12)
    starts=np.vstack([starts,ws[min((i for i,w in enumerate(ws) if min(w)>=.05-1e-12),key=lambda i:ys[i])]])
    options=[]
    for x in starts:
        r=minimize(lambda w:-ei(w),x,method='SLSQP',bounds=[(.05,.85)]*4,constraints=[{'type':'eq','fun':lambda w:w.sum()-1},{'type':'ineq','fun':lambda w:np.min(np.linalg.norm(ws-w,axis=1))-.01}],options={'ftol':1e-12,'maxiter':150})
        if r.success and min(r.x)>=.05-1e-8 and abs(sum(r.x)-1)<1e-8 and np.min(np.linalg.norm(ws-r.x,axis=1))>=.01-1e-8:options.append(r.x)
    if not options:raise RuntimeError('No novel feasible EI proposal; preserve attempt')
    w=max(options,key=ei);mu,sigma=gp.predict(w[None,:],return_std=True)
    return w,{'kernel':str(gp.kernel_),'predicted_J':float(mu[0]),'predicted_std':float(sigma[0]),'EI':ei(w),'incumbent_J':best,'warnings':[str(x.message) for x in ww]}

def evaluate(name,w,batches):
    output=[]
    with (ROOT/'solver.log').open('a') as log:
        for b in batches:
            for seed in b['seeds']:
                for sc in prior.SCENARIOS:
                    with contextlib.redirect_stdout(log):server,item,artifact=shared.load_base(b['source'],sc,seed)
                    p=np.column_stack([a.api_predict(server.test_X) for a in server.trusted_agents]);mse=((p-server.test_y[:,None])**2).mean(0)
                    peer=np.full((3,3),np.nan);t=time.perf_counter()
                    if w[3]>0:
                        for j,a in enumerate(server.trusted_agents):
                            for i,x in enumerate(server.I_list):
                                if i!=j:peer[j,i]=float(np.asarray(a.api_predict(x)).reshape(-1)[0])
                    support=time.perf_counter()-t if w[3]>0 else 0.
                    shared.initial_alpha=prior.initializer(w,peer);np.random.seed(seed);torch.manual_seed(seed)
                    try:
                        with contextlib.redirect_stdout(log):row=shared.solve(server,name,mse,server.test_X,server.test_y,p,seed,1000)
                    except Exception as exc:
                        shared.append(ROOT/'failures.jsonl',{'method':name,'seed':seed,'scenario':sc,'error':repr(exc),**getattr(exc,'partial_row',{})});raise
                    row.update(method=name,w=np.asarray(w).tolist(),seed=seed,scenario=sc,batch=b['number'],support_seconds=support,support_requests=6 if w[3]>0 else 0,support_input_samples=6 if w[3]>0 else 0,artifact=str(artifact),additional_labels=0)
                    shared.append(ROOT/'decisions.jsonl',row.copy())
                    if row['status']!='valid':shared.append(ROOT/'results.jsonl',row);raise RuntimeError('Invalid solve; not silently excluded')
                    evaluator,_=load_evaluator_artifact(artifact);_,row['pooled_error']=score_solution(evaluator,np.array(row['X']),0);row['truth_error']=abs(float(synthetic_truth(np.array(row['X'])[None,:])[0]))
                    shared.append(ROOT/'results.jsonl',row);output.append(row)
    return output

def run():
    ROOT.mkdir(exist_ok=False);snap=ROOT/'source_snapshot';snap.mkdir()
    for p in Path('.').glob('*.py'):(snap/p.name).write_bytes(p.read_bytes())
    design=Path('docs/four_way_bayes_design_20261116.md');(ROOT/'registered_design.md').write_bytes(design.read_bytes())
    old=read(prior.ROOT/'results.jsonl');reg=json.loads((prior.ROOT/'registration.json').read_text());batches=reg['batches'];full=json.loads(Path('alpha_weight_outputs/se_qu_extension_v1_20261107/registration.json').read_text())['batches'][:3]
    audit=[{**b,'seeds':b['seeds'][10:]} for b in full]
    put('registration.json',{'status':'running','training_batches':batches,'audit_batches':audit,'max_solves':2520,'source_hashes':{p.name:shared.digest(p) for p in snap.iterdir()},'old_hashes':{str(p):shared.digest(p) for p in [prior.ROOT/'results.jsonl',prior.ROOT/'analysis.json']},'design_hash':shared.digest(design)})
    baseline={(r['seed'],r['scenario']):r for r in old if r['method']=='UE_0.500'};names=sorted({r['method'] for r in old});ws=[];ys=[];observations=[]
    for n in names:
        rr=[r for r in old if r['method']==n];d,j=metrics(rr,baseline);ws.append(rr[0]['w']);ys.append(j);observations.append({'method':n,'w':rr[0]['w'],'J':j,'mean_delta':float(d.mean()),'origin':'reused'})
    for k in range(16):
        if k<4:w=np.full(4,.2);w[k]=.4;info={'rule':'simplex vertex around centroid'}
        else:w,info=propose(np.asarray(ws),np.array(ys),k-4)
        name=f'learn_{k:02d}';shared.append(ROOT/'proposals.jsonl',{'method':name,'w':w.tolist(),**info})
        rr=evaluate(name,w,batches);d,j=metrics(rr,baseline);ws.append(w.tolist());ys.append(j);obs={'method':name,'w':w.tolist(),'J':j,'mean_delta':float(d.mean()),'origin':'learned'};observations.append(obs);shared.append(ROOT/'observations.jsonl',obs);print(name,j,float(d.mean()),flush=True)
    eligible=[o for o in observations if min(o['w'])>=.05-1e-12];chosen=min(eligible,key=lambda o:(o['J'],o['method']));put('locked_candidate.json',chosen)
    w=np.array(chosen['w']);directions=np.array([[1,-1,0,0],[1,1,-2,0],[1,1,1,-3]],float);directions/=np.linalg.norm(directions,axis=1)[:,None]
    for k,v in enumerate(directions):
        for sign in [-1,1]:
            v=sign*v;limit=min((w[i]-.05)/(-v[i]) for i in range(4) if v[i]<0);step=min(.05,.9*limit);q=w+step*v
            evaluate(f'neighbor_{k}_{sign}',q,batches)
    # Lock occurred before loading audit outcomes. Audit cannot alter candidate.
    for n,q in [('audit_candidate',w),('audit_equal',np.full(4,.25)),('audit_EU',np.array([.5,0,.5,0]))]:evaluate(n,q,audit)
    put('execution_complete.json',{'status':'complete','new_rows':len(read(ROOT/'results.jsonl'))})
if __name__=='__main__':run()
