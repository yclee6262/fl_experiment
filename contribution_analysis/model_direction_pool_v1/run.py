"""Bounded exclusive-create experiment; diagnostics follow sealed scores."""
import copy,sys,time,shutil,datetime,json
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from contribution_analysis.custom_realized_gain_v1.schema import save,sha,plain
from contribution_analysis.model_peer_pool_v1.run import cases as historical
from contribution_analysis.three_pool_t3_model_v1.fixtures import poly
from contribution_analysis.three_pool_t3_model_v1.core import evaluate
from contribution_analysis.model_consensus_pool_v1.core import settle
from .core import execute
W=Path(__file__).resolve().parents[2]

def fixtures():
    old=historical()
    for c in old:c.update(split='historical_development',iterations=1)
    new=[];rng=np.random.default_rng(20261029)
    for parameter,b in enumerate([.7,1.1]):
        noise=poly(float(rng.normal(0,2)),float(rng.uniform(-3,3)))
        templates=[('beneficial_minority',[poly(1),poly(-b),poly(-b)],['3/4','1/8','1/8'],poly(1)),('wrong_minority',[poly(1),poly(-b),poly(-b)],['3/4','1/8','1/8'],poly(-1)),('useful_restraint',[poly(1,4*b),poly(-1),poly(-1)],['1/2','1/4','1/4'],poly(0,b)),('false_restraint',[poly(1,4*b),poly(-1),poly(-1)],['1/2','1/4','1/4'],poly(-1)),('clone',[poly(1),poly(1),poly(-b)],['1/3']*3,poly(1)),('seeded_noise',[poly(1),poly(1),noise],['1/3']*3,poly(1)),('natural_tangent',[poly(.3,2*b),poly(.5,3*b),poly(.7,b)],['1/3']*3,poly(.5,2*b)),('two_dimensional',[poly(1,2*b),poly(-1,b),poly(.2,3*b)],['1/3']*3,poly(.1,2*b))]
        for cat,models,weights,ref in templates:
            anchors=[['1','0'],['-1/4','0']]
            if cat=='two_dimensional':
                anchors=[['1','1'],['-1','1'],['0','-1']]
                for j,m in enumerate(models):m['linear'][1]=str(F(j-1,2));m['quadratic'][1][1]=str(F(j+1,2))
            for n in [1,8]:
                new.append(dict(id=f'new_{cat}_{parameter}_{n}',category=cat,split='new_development' if parameter==0 else 'new_heldout',models=copy.deepcopy(models),weights=weights,reference=ref,owners=['owner_a','owner_b','owner_c'],anchors=anchors,anchor_weights=[str(F(1,len(anchors)))]*len(anchors),start=['0','0'],iterations=n))
    return old+new

def allocation(rows,beta,extra,owners=None,rounds=None):
    owners=sorted(owners if owners is not None else {r['owner'] for r in rows});round_ids=sorted(rounds if rounds is not None else {r['round'] for r in rows});nrounds=len(round_ids)
    needed=['P']+([extra] if beta>0 else [])
    valid=bool(owners and round_ids) and len(rows)==len(owners)*nrounds
    valid=valid and {(r.get('owner'),r.get('round')) for r in rows}=={(o,t) for o in owners for t in round_ids}
    valid=valid and all(isinstance(r.get(k),(int,float)) and np.isfinite(r[k]) for r in rows for k in needed)
    if not valid:return dict(status='held',amounts=None,reserve=None,reasons=['incomplete_or_nonfinite_required_matrix'])
    totals={k:{o:sum(r[k] for r in rows if r['owner']==o) for o in owners} for k in needed}
    arms=[]
    for k,B in [('P',int(round(500*(1-beta))))]+([(extra,int(round(500*beta)))] if beta>0 else []):
        arms.append(settle({o:F(str(v)) for o,v in totals[k].items()},B,nrounds))
    amounts={o:sum(a['amounts'][o] for a in arms) for o in owners};reserve=sum(a['reserve'] for a in arms)
    assert sum(amounts.values())+reserve==500
    return plain(dict(status='allocated' if sum(amounts.values()) else 'reserved',amounts=amounts,reserve=reserve,totals=totals,arms=arms))

def score(record):
    rows=copy.deepcopy(record['rows'])
    for r in rows:r['Ub']=None if r.get('U') is None else (.1*r['U'])/(1+.1*r['U'])
    args=dict(owners=record['fixture']['owners'],rounds=range(len(record['full'])))
    methods={'P_only':allocation(rows,0,'N',**args)}
    for extra in ['Ub','N']:
        for beta in [.1,.2,.3]:methods[f'P+{extra}_{beta}']=allocation(rows,beta,extra,**args)
    return dict(rows=rows,methods=methods)

def main():
    root=Path(sys.argv[1]) if len(sys.argv)>1 else W/'contribution_analysis/model_direction_pool_experiment_v1_20261029'
    root.mkdir(exist_ok=False);started=time.perf_counter();solves=0;calls=0;scale_calls=0;reference_calls=0;done=[]
    for n in ['cases','scores','diagnostics']:(root/n).mkdir()
    try:
        design=W/'docs/model_direction_pool_experiment_design_20261029.md';assert design.exists()
        deps=[W/'host_server.py',design,W/'docs/model_direction_uniqueness_design_20261029.md']
        for folder in ['model_direction_pool_v1','model_step_decision_v1','three_pool_t3_model_v1','model_peer_pool_v1','model_consensus_pool_v1','model_decision_validity_v1','custom_realized_gain_v1']:deps+=list((W/'contribution_analysis'/folder).glob('*.py'))
        locks={}
        for p in deps:
            rel=p.relative_to(W);dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);locks[str(rel)]=sha(p)
        save(root/'source_lock.json',locks)
        cases=fixtures();save(root/'fixtures.json',cases);save(root/'registration.json',dict(fixtures_sha256=sha(root/'fixtures.json'),design_sha256=sha(design),max_solves=2500,max_evaluations=250000,max_seconds=600,max_bytes=200000000,reference_in_score=False,pooled_calls=0,actual_payments=0))
        for split in ['historical_development','new_development','new_heldout']:
            for c in [c for c in cases if c['split']==split]:
                rec=execute(c,c['iterations']);solves+=rec['solves'];calls+=sum(x['evaluations'] for x in rec['cost'].values())+rec['cache_audit_evaluations'];assert solves<=2500 and calls<=250000 and time.perf_counter()-started<600
                save(root/'cases'/(c['id']+'.json'),rec);scores=score(rec);save(root/'scores'/(c['id']+'.json'),dict(score=scores,case_sha256=sha(root/'cases'/(c['id']+'.json'))))
                # Fixed-endpoint, fixed-direction scaling invariance (no solver rerun).
                errs=[]
                for factor in [.1,10,100]:
                    for i in range(len(c['models'])):
                        ft=copy.deepcopy(rec)
                        # Simultaneous rescaling numerator and denominator cancels exactly algebraically;
                        # verify actual raw slope divisions with the archived point losses.
                        scale=np.array(rec['scale']);scale[i]*=factor
                        for t,d in enumerate(rec['directions']):
                            if d['g'] is None:continue
                            base=np.array([abs(evaluate(m,rec['full'][t]['pre']['S_current'])) for m in c['models']]);scale_calls+=len(c['models']);g=[]
                            for a in d['directions']:
                                vals=np.array([abs(evaluate(m,a['point'])) for m in c['models']]);scale_calls+=len(c['models']);num=vals-base;num[i]*=factor;g.append(num/(scale*a['h']))
                            g=np.array(g).T;err=float(np.max(np.abs(g-np.array(d['g']))));assert err<1e-8;errs.append(err)
                            dist=np.sqrt(np.mean((g[:,None,:]-g[None,:,:])**2,axis=2));np.fill_diagonal(dist,np.inf);U=np.min(dist,axis=1)
                            for rr in [rr for rr in ft['rows'] if rr['round']==t]:
                                j=rr['model'];other=[z for z in range(len(c['models'])) if z!=j]
                                lf=np.array([abs(evaluate(m,rr['full'])) for m in c['models']]);lm=np.array([abs(evaluate(m,rr['minus'])) for m in c['models']]);scale_calls+=2*len(c['models']);lf[i]*=factor;lm[i]*=factor
                                w=np.array([float(F(z)) for z in c['weights']]);w/=w.sum();rr['P']=float(w[other]@((lm-lf)/scale)[other]/w[other].sum());rr['U']=float(U[j]);rr['N']=(.1*U[j])/(1+.1*U[j])*rr['I']/(1+rr['I'])
                        vs=score(ft)
                        for method in scores['methods']:
                            assert vs['methods'][method]['status']==scores['methods'][method]['status']
                            assert vs['methods'][method]['amounts']==scores['methods'][method]['amounts'],('scale_allocation',c['id'],i,factor,method)
                        for rr,orig in zip(vs['rows'],scores['rows']):
                            for key in ['P','U','I','N']:
                                if orig[key] is not None:assert abs(rr[key]-orig[key])<=1e-10+1e-9*abs(orig[key]),('scale_score',key)
                save(root/'scores'/(c['id']+'_scale.json'),dict(variants=3*len(c['models']),max_g_error=max(errs,default=0),scope='fixed endpoints and directions; raw model losses and probes rescaled; P U I N settlement checked'))
                diag=[]
                for o in c['owners']:
                    rr=[r for r in scores['rows'] if r['owner']==o];D=sum(evaluate(c['reference'],r['minus'])-evaluate(c['reference'],r['full']) for r in rr)
                    reference_calls+=2*len(rr)
                    diag.append(dict(owner=o,D=D,amounts={k:v['amounts'][o] for k,v in scores['methods'].items()}))
                save(root/'diagnostics'/(c['id']+'.json'),dict(scores_sha256=sha(root/'scores'/(c['id']+'.json')),rows=diag));done.append(dict(id=c['id'],split=split,category=c['category'],rounds=len(rec['full']),modes=[r['pre']['current_method'] for r in rec['full']],cost=rec['cost'],cache_audit_evaluations=rec['cache_audit_evaluations'],diagnostics=diag,methods=scores['methods']))
            save(root/(split+'_checkpoint.json'),dict(completed=len(done),score_hashes={p.name:sha(p) for p in (root/'scores').glob('*.json')}))
        for rel,h in locks.items():assert sha(W/rel)==h
        size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file());assert size<200000000
        save(root/'summary.json',dict(complete=True,cases=done,solves=solves,analytic_evaluations_including_cache_audit=calls,scale_control_evaluations=scale_calls,reference_diagnostic_evaluations=reference_calls,total_analytic_evaluations=calls+scale_calls+reference_calls,seconds=time.perf_counter()-started,bytes_before_receipt=size,quality_validity_established=False))
        save(root/'completion_receipt.json',dict(file_sha256={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}));print(root,'COMPLETE',solves,calls)
    except Exception as exc:save(root/'failure.json',dict(error=repr(exc),solves=solves,evaluations=calls,completed=len(done),seconds=time.perf_counter()-started));raise
if __name__=='__main__':main()
