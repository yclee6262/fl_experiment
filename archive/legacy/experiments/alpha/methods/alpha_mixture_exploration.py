"""Bounded development exploration. The initializer has no evaluator access."""
import argparse, contextlib, itertools, json, time
from pathlib import Path
import numpy as np
import torch
import oob_shrinkage_research as shared
from unified_alpha_methods import initial_alpha as unified
from peer_support_alpha import support_signal
from research_evaluation import load_evaluator_artifact, score_solution, synthetic_truth

ROOT=Path('alpha_weight_outputs/mixture_exploration_v1_20261116')
SCENARIOS=shared.SCENARIOS
BASE={'U':'A_uniform','Q':'B_stage0','E':'stacked_full','S':'pure_support',
      'UQ_0.500':'quality_shrink','UE_0.500':'ensemble_shrink','US_0.500':'peer_support_shrink',
      'QE_0.500':'B_stacked_half','QS_0.500':'support_quality_mix50','ES_0.500':'support_ensemble_mix50'}

def candidates():
    out={k:np.eye(4)[i].tolist() for i,k in enumerate('UQES')}
    for i,j in itertools.combinations(range(4),2):
        for t in [.25,.5,.75]:
            w=np.zeros(4);w[i]=t;w[j]=1-t
            out['UQES'[i]+'UQES'[j]+f'_{t:.3f}']=w.tolist()
    return out

def initializer(w,peer):
    w=np.asarray(w,float)
    if w.shape!=(4,) or not np.isfinite(w).all() or min(w)<0 or abs(sum(w)-1)>1e-12:raise ValueError('invalid mixture')
    def initial(method,agents,mse,x,y,proposals,target,**kwargs):
        components={};audit={};alpha=np.zeros(len(agents))
        for k,f in zip('UQES',w):
            if f==0:continue
            if k=='S':a,info=support_signal(peer,target)
            else:a,info=unified(BASE[k],agents,mse,x,y,proposals,target,**kwargs)
            alpha+=f*a;components[k]=a.tolist();audit[k]=info
        return alpha,{'mixture_w':w.tolist(),'components':components,'component_audits':audit}
    return initial

def read(path):return [json.loads(l) for l in Path(path).read_text().splitlines()]
def dump(name,obj):shared.dump(ROOT/name,obj)

def prepare():
    ROOT.mkdir(exist_ok=False)
    reg=json.loads(Path('alpha_weight_outputs/se_qu_extension_v1_20261107/registration.json').read_text())
    batches=[{**b,'seeds':b['seeds'][:10]} for b in reg['batches'][:3]]
    files=[Path(b[k])/'results.jsonl' for b in batches for k in ['results','six_method_results','pure_results']]
    hashes={str(p):shared.digest(p) for p in files}
    snap=ROOT/'source_snapshot';snap.mkdir()
    for p in Path('.').glob('*.py'):(snap/p.name).write_bytes(p.read_bytes())
    design=Path('docs/alpha_mixture_exploration_design_20261116.md')
    (ROOT/'registered_design.md').write_bytes(design.read_bytes())
    dump('registration.json',{'batches':batches,'coarse':candidates(),'old_result_hashes':hashes,
        'design_sha256':shared.digest(design),'source_sha256':{p.name:shared.digest(p) for p in snap.glob('*.py')},
        'kind':'development_only','seed_n':30,'max_new_solves':1890,'bootstrap_replicates':20000})
    historical=[]
    for b in batches:
        rows=[r for k in ['results','six_method_results','pure_results'] for r in read(Path(b[k])/'results.jsonl')]
        for r in rows:
            if r['seed'] in b['seeds']:
                name=next(k for k,v in BASE.items() if v==r['method'])
                historical.append({**r,'method':name,'origin':'historical','batch':b['number'],'w':candidates()[name]})
    assert len(historical)==900
    for r in historical:shared.append(ROOT/'results.jsonl',r)

def run_stage(stage,cs,replay=False):
    reg=json.loads((ROOT/'registration.json').read_text()); historical={(r['scenario'],r['seed'],r['method']):r for r in read(ROOT/'results.jsonl') if r['origin']=='historical'}
    with (ROOT/f'{stage}_solver.log').open('w') as log:
        for b in reg['batches']:
            for seed in b['seeds']:
                if replay and seed!=b['seeds'][0]:continue
                for scenario in SCENARIOS:
                    with contextlib.redirect_stdout(log):server,item,artifact=shared.load_base(b['source'],scenario,seed)
                    start=time.perf_counter();pred=np.column_stack([a.api_predict(server.test_X) for a in server.trusted_agents]);blind_s=time.perf_counter()-start
                    mse=((pred-server.test_y[:,None])**2).mean(axis=0)
                    peer=np.full((3,3),np.nan);start=time.perf_counter()
                    for j,a in enumerate(server.trusted_agents):
                        for i,x in enumerate(server.I_list):
                            if i!=j:peer[j,i]=float(np.asarray(a.api_predict(x)).reshape(-1)[0])
                    support_s=time.perf_counter()-start
                    if stage=='coarse':
                        shared.append(ROOT/'shared.jsonl',{'seed':seed,'scenario':scenario,'batch':b['number'],'preparation':item,
                            'artifact':str(artifact),'sha256':{p.name:shared.digest(p) for p in artifact.iterdir() if p.is_file()},
                            'blind_seconds':blind_s,'blind_requests':3,'blind_samples':60,'support_seconds':support_s,'support_requests':6})
                    order=list(cs);np.random.default_rng(seed+SCENARIOS.index(scenario)).shuffle(order)
                    decisions=[]
                    for name in order:
                        w=cs[name];shared.initial_alpha=initializer(w,peer)
                        np.random.seed(seed);torch.manual_seed(seed)
                        try:
                            with contextlib.redirect_stdout(log):row=shared.solve(server,name,mse,server.test_X,server.test_y,pred,seed,1000)
                        except Exception as exc:
                            shared.append(ROOT/'failures.jsonl',{'seed':seed,'scenario':scenario,'method':name,'reason':repr(exc),**getattr(exc,'partial_row',{})});raise
                        row.update(seed=seed,scenario=scenario,batch=b['number'],method=name,w=w,origin=stage,
                                   support_seconds=support_s if w[3]>0 else 0.,support_requests=6 if w[3]>0 else 0,
                                   support_input_samples=6 if w[3]>0 else 0,additional_labels=0)
                        shared.append(ROOT/f'{stage}_decisions.jsonl',row);decisions.append(row)
                    # No evaluator is supplied to initialization or search; all decisions already persisted.
                    evaluator,_=load_evaluator_artifact(artifact)
                    for row in decisions:
                        if row['status']=='valid':
                            _,row['pooled_error']=score_solution(evaluator,np.array(row['X']),0)
                            row['truth_error']=abs(float(synthetic_truth(np.array(row['X'])[None,:])[0]))
                        if replay:
                            old=historical[(scenario,seed,row['method'])]
                            row['max_replay_difference']=max(float(np.max(abs(np.asarray(row[k])-np.asarray(old[k])))) for k in ['alpha','X','pooled_error'])
                        shared.append(ROOT/('replay.jsonl' if replay else 'results.jsonl'),row)
                    print(stage,seed,scenario,flush=True)

def matrix(rows,methods):
    seeds=sorted(set(r['seed'] for r in rows));idx={(r['seed'],r['scenario'],r['method']):r for r in rows}
    a=np.array([[[idx[s,c,m]['pooled_error'] for c in SCENARIOS] for s in seeds] for m in methods])
    return seeds,a

def expansion():
    rows=read(ROOT/'results.jsonl');cs=candidates();names=list(cs);seeds,a=matrix(rows,names);d=a-a[names.index('UE_0.500')];choices=[]
    for i,j in itertools.combinations(range(4),2):
        edge='UQES'[i]+'UQES'[j];ns=[edge+f'_{t:.3f}' for t in [.25,.5,.75]]
        best=min(ns,key=lambda n:(d[names.index(n)].mean(),n));z=d[names.index(best)];t=float(best.split('_')[1]);neighbors=[]
        for q in [t-.25,t+.25]:
            nn='UQES'[j] if q==0 else 'UQES'[i] if q==1 else edge+f'_{q:.3f}'
            neighbors.append(float(d[names.index(nn)].mean()))
        gate=z.mean()<=-.001 and all(z[k:k+10].mean()<0 for k in [0,10,20]) and sum(z.mean(axis=0)<0)>=2 and min(neighbors)<0
        if gate:choices.append((float(z.mean()),best,i,j,t))
    extra={}
    for _,best,i,j,t in sorted(choices)[:2]:
        for q in [t-.125,t+.125]:
            w=np.zeros(4);w[i]=q;w[j]=1-q;extra['UQES'[i]+'UQES'[j]+f'_{q:.3f}']=w.tolist()
    for inds in itertools.combinations(range(4),3):
        w=np.zeros(4);w[list(inds)]=1/3;extra[''.join('UQES'[i] for i in inds)+'_equal']=w.tolist()
    extra['UQES_equal']=[.25]*4
    dump('expansion_decision.json',{'eligible_edges':choices,'candidates':extra,'rule':'registered_design.md; no further expansion'})
    return extra

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','coarse','replay','expand']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='coarse':run_stage('coarse',{k:v for k,v in candidates().items() if k not in BASE})
    elif a.action=='replay':run_stage('replay',{k:candidates()[k] for k in BASE},True)
    else:run_stage('expansion',expansion())
