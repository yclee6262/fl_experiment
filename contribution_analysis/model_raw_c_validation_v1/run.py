"""Offline raw full-objective C diagnostic. Never evaluates a model or reference."""
import hashlib, json, math, sys, time, shutil, subprocess
from collections import Counter
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
from contribution_analysis.model_direction_diagnostic_v1.diagnostic import dedup
from contribution_analysis.model_consensus_pool_v1.core import settle
W=Path(__file__).resolve().parents[2]
SOURCE=W/'contribution_analysis/model_direction_pool_experiment_v1_20261029_attempt_02'
HISTORY=W/'contribution_analysis/model_peer_pool_experiment_v1_20261028_attempt_02'
DESIGN=W/'docs/model_raw_c_validation_design_20261029.md'
EPS=1e-10

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def sign(x,tau=EPS):return 'positive' if x>tau else 'negative' if x< -tau else 'zero'
def index_events(events):
    out={}
    for e in events:
        key=(e['model'],tuple(e['x']));v=e['loss']
        if not math.isfinite(v):raise ValueError('nonfinite_loss')
        if key in out and out[key]!=v:raise ValueError('inconsistent_endpoint_loss')
        out[key]=v
    return out

def raw(lf,lm,w):
    weights=[Q(str(v)) for v in w];s=sum(weights)
    if s<=0 or any(v<0 for v in weights):raise ValueError('invalid_weights')
    return sum(v*(Q(str(b))-Q(str(a))) for v,a,b in zip(weights,lf,lm))/s

def allocate(rows,owners,T,beta=0,extra='N'):
    required=['C']+([extra] if beta else [])
    keys=[(r.get('owner'),r.get('round')) for r in rows]
    if len(keys)!=len(owners)*T or set(keys)!={(o,t) for o in owners for t in range(T)} or any(not isinstance(r.get(k),(int,float)) or not math.isfinite(r[k]) for r in rows for k in required):
        return dict(status='held',amounts=None,reserve=None,reasons=['incomplete_or_nonfinite_required_matrix'])
    totals={k:{o:sum((Q(str(r[k])) for r in rows if r['owner']==o),Q(0)) for o in owners} for k in required}
    arms={k:settle(totals[k],B,T) for k,B in [('C',round(500*(1-beta)))]+([(extra,round(500*beta))] if beta else [])}
    amounts={o:sum(a['amounts'][o] for a in arms.values()) for o in owners};reserve=sum(a['reserve'] for a in arms.values())
    assert sum(amounts.values())+reserve==500
    return dict(status='allocated' if sum(amounts.values()) else 'reserved',amounts=amounts,reserve=reserve,arms=arms,totals={k:{o:float(v) for o,v in z.items()} for k,z in totals.items()})

def classify(rows,T):
    tau=max(1e-8,T*EPS);cs=[r['C'] for r in rows];gp=sum(max(c,0) for c in cs);gm=sum(max(-c,0) for c in cs);net=sum(cs);sg=sign(net,tau);counts=Counter(sign(c) for c in cs)
    cancel=gp>tau and gm>tau
    if counts['zero']==T:kind='path_redundant' if all(r['distance']<=EPS for r in rows) else 'different_endpoint_equal_F'
    elif cancel:kind='cancellation_'+sg
    elif sg=='zero':kind='micro_accumulation_deadzone'
    else:kind='net_'+sg
    return dict(T=T,tau=tau,Gplus=gp,Gminus=gm,net=net,classification=sg,kind=kind,cancellation=cancel,round_sign_counts={k:counts[k] for k in ['positive','negative','zero']},trajectory=cs)

def verify_receipt(root):
    receipt=json.loads((root/'completion_receipt.json').read_text())
    for name,h in receipt['file_sha256'].items():assert sha(root/name)==h,(root,name)
    return len(receipt['file_sha256'])

def calculate(r):
    fixture=r['fixture'];w=fixture['weights'];idx=index_events(r['events']);rows=[]
    for old in r['rows']:
        lf=[idx[(j,tuple(old['full']))] for j in range(len(w))];lm=[idx[(j,tuple(old['minus']))] for j in range(len(w))]
        c=float(raw(lf,lm,w));wf=np.array([float(Q(v)) for v in w]);wf/=sum(wf)
        direct=float(wf@np.array(lm)-wf@np.array(lf));assert abs(c-direct)<=1e-10+1e-9*abs(c)
        d=r['directions'][old['round']];U=None if d['g'] is None else dedup(d['directions'],d['g'])[0][old['model']]
        dist=float(np.linalg.norm(np.array(old['full'])-np.array(old['minus'])));eta=r['full'][old['round']]['pre']['eta'];I=dist/eta
        assert abs(I-old['I'])<=1e-10+1e-9*abs(I)
        Ub=None if U is None else .1*U/(1+.1*U);Ib=I/(1+I)
        rows.append(dict(owner=old['owner'],model=old['model'],round=old['round'],full=old['full'],minus=old['minus'],loss_full=lf,loss_minus=lm,C=c,direct_F_difference=direct,distance=dist,U=U,I=I,Ub=Ub,Ib=Ib,N=None if Ub is None else Ub*Ib))
    return rows

def scale_controls(rows,w,owners,T):
    out=[];base=allocate(rows,owners,T)
    weights=[Q(v) for v in w];weights=[v/sum(weights) for v in weights]
    for kind in ['all_units','single_uncompensated','single_compensated']:
        for j in ([None] if kind=='all_units' else range(len(w))):
            for factor in [.1,10,100]:
                scales=[Q(str(factor)) if j is None or k==j else Q(1) for k in range(len(w))]
                v=[weights[k]/scales[k] if kind=='single_compensated' else weights[k] for k in range(len(w))];Z=sum(v);v=[a/Z for a in v];expected=float(Q(1)/Z) if kind=='single_compensated' else factor
                transformed=[];errors=[]
                for r in rows:
                    lf=[Q(str(a))*s for a,s in zip(r['loss_full'],scales)];lm=[Q(str(a))*s for a,s in zip(r['loss_minus'],scales)];c=float(raw(lf,lm,v));transformed.append(dict(r,C=c))
                    if kind!='single_uncompensated':errors.append(abs(c-r['C']*expected))
                a=allocate(transformed,owners,T);tau=max(1e-8,T*EPS)
                eligibility_change=[o for o in owners if (base['totals']['C'][o]>tau)!=(a['totals']['C'][o]>tau)]
                if kind!='single_uncompensated':
                    assert max(errors,default=0)<=1e-10+1e-9*max(abs(r['C']*expected) for r in rows)
                    if not eligibility_change:assert a['amounts']==base['amounts'],(kind,j,factor)
                out.append(dict(kind=kind,model=j,factor=factor,v=[float(z) for z in v],Z=float(Z),expected_multiplier=expected if kind!='single_uncompensated' else None,max_proportional_error=max(errors,default=None),eligibility_change=eligibility_change,amounts_changed=a['amounts']!=base['amounts'],sign_changed=[o for o in owners if sign(base['totals']['C'][o],tau)!=sign(a['totals']['C'][o],tau)],allocation=a,transformed_endpoints=[dict(owner=r['owner'],round=r['round'],loss_full=[float(Q(str(x))*s) for x,s in zip(r['loss_full'],scales)],loss_minus=[float(Q(str(x))*s) for x,s in zip(r['loss_minus'],scales)],C=t['C']) for r,t in zip(rows,transformed)]))
    return out

def direction_unit_controls(r):
    idx=index_events(r['events']);checks=[];n=len(r['fixture']['weights'])
    for t,d in enumerate(r['directions']):
        if d['g'] is None:continue
        start=r['full'][t]['pre']['S_current'];slopes=[]
        for a in d['directions']:
            slopes.append([(idx[(j,tuple(a['point']))]-idx[(j,tuple(start))])/(r['scale'][j]*a['h']) for j in range(n)])
        g=np.array(slopes).T;assert np.max(np.abs(g-np.array(d['g'])))<1e-8
        U=dedup(d['directions'],g.tolist())[0]
        for j in range(n):
            for factor in [.1,10,100]:
                # Numerator is taken from archived raw loss differences; fixed endpoints.
                transformed=[]
                for a in d['directions']:
                    transformed.append([((idx[(k,tuple(a['point']))]-idx[(k,tuple(start))])*(factor if k==j else 1))/(r['scale'][k]*(factor if k==j else 1)*a['h']) for k in range(n)])
                gg=np.array(transformed).T;uu=dedup(d['directions'],gg.tolist())[0];err=float(np.max(np.abs(gg-g)));uerr=max(abs(a-b) for a,b in zip(U,uu));assert err<1e-8 and uerr<1e-8
                checks.append(dict(round=t,model=j,factor=factor,max_g_error=err,max_U_error=uerr,I_unchanged_fixed_endpoints=True))
    return checks

def summarize(cases):
    out={}
    for scope in ['all','multiround','historical_development','new_development','new_heldout']:
        group=[c for c in cases if scope=='all' or (scope=='multiround' and c['T']>1) or c['split']==scope];methods={}
        for method in cases[0]['methods']:
            signs={s:dict(owners=0,paid_owners=0,units=0) for s in ['positive','negative','zero']};reserve=0;held=0;full_reserve=0;partial_reserve=0;equal_units=0;supp_neg=0
            for c in group:
                a=c['methods'][method]
                if a['status']=='held':held+=1;continue
                reserve+=a['reserve'];full_reserve+=a['reserve']==500;partial_reserve+=0<a['reserve']<500
                for o,z in c['owners'].items():
                    s=signs[z['classification']];s['owners']+=1;s['paid_owners']+=a['amounts'][o]>0;s['units']+=a['amounts'][o]
                    if z['kind']=='different_endpoint_equal_F':equal_units+=a['amounts'][o]
                    if z['classification']=='negative':supp_neg+=sum(v['amounts'][o] for k,v in a['arms'].items() if k!='C')
            methods[method]=dict(signs=signs,reserve=reserve,held_cases=held,fully_reserved_cases=full_reserve,partially_reserved_cases=partial_reserve,equal_F_different_endpoint_units=equal_units,supplement_negative_C_units=supp_neg)
        out[scope]=dict(cases=len(group),owner_cases=sum(len(c['owners']) for c in group),rounds=sum(c['T'] for c in group),classifications=dict(Counter(z['kind'] for c in group for z in c['owners'].values())),methods=methods)
    return out

def main(out):
    started=time.perf_counter();out.mkdir(exist_ok=False);(out/'cases').mkdir();(out/'scale_controls').mkdir()
    inputs=[SOURCE/'completion_receipt.json']+sorted((SOURCE/'cases').glob('*.json'))+sorted((HISTORY/'scores').glob('*.json'))
    sources=[Path(__file__),Path(__file__).with_name('test_raw_c.py'),DESIGN,W/'contribution_analysis/model_direction_diagnostic_v1/diagnostic.py',W/'contribution_analysis/model_consensus_pool_v1/core.py']
    locks={str(p.relative_to(W)):sha(p) for p in inputs+sources};save(out/'source_lock.json',locks);verified=verify_receipt(SOURCE);history_verified=verify_receipt(HISTORY);cases=[];controls=[];history=[]
    for p in sources:
        dest=out/'source_snapshot'/p.relative_to(W);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    test=subprocess.run([sys.executable,'-m','unittest','contribution_analysis.model_raw_c_validation_v1.test_raw_c'],capture_output=True,text=True,cwd=W)
    save(out/'test_report.json',dict(returncode=test.returncode,stdout=test.stdout,stderr=test.stderr));assert test.returncode==0
    for p in sorted((SOURCE/'cases').glob('*.json')):
        r=json.loads(p.read_text());f=r['fixture'];T=len(r['full']);owners=f['owners'];rows=calculate(r)
        methods={'C_only':allocate(rows,owners,T)}
        for k in ['Ub','Ib','N']:
            for beta in [.1,.2,.3]:methods[f'C+{k}_{beta}']=allocate(rows,owners,T,beta,k)
        c=dict(id=p.stem,split=f['split'],category=f['category'],T=T,rows=rows,owners={o:classify([z for z in rows if z['owner']==o],T) for o in owners},methods=methods)
        save(out/'cases'/p.name,c);cases.append(c);sc=scale_controls(rows,f['weights'],owners,T);save(out/'scale_controls'/p.name,sc);controls.extend(dict(case=p.stem,**{k:v for k,v in z.items() if k!='transformed_endpoints'}) for z in sc)
        save(out/'scale_controls'/(p.stem+'_directions.json'),direction_unit_controls(r))
        if f['split']=='historical_development':
            hist=json.loads((HISTORY/'scores'/p.name).read_text())['methods']['raw'];errors={o:abs(c['owners'][o]['net']-hist['scores'][o]) for o in owners};assert all(e<=1e-10+1e-9*abs(hist['scores'][o]) for o,e in errors.items());assert hist['amounts']==methods['C_only']['amounts'];assert hist['reserve']==methods['C_only']['reserve'];history.append(dict(case=p.stem,errors=errors,amounts_equal=True,reserve=hist['reserve']))
        assert time.perf_counter()-started<120
    summary=summarize(cases);summary.update(cases=len(cases),rounds=sum(c['T'] for c in cases),owner_rounds=sum(len(c['rows']) for c in cases),new_solver_calls=0,new_model_calls=0,new_reference_calls=0,receipt_files_verified=verified,scale_controls={k:dict(variants=sum(z['kind']==k for z in controls),amount_changes=sum(z['kind']==k and z['amounts_changed'] for z in controls),sign_changes=sum(z['kind']==k and bool(z['sign_changed']) for z in controls),threshold_exceptions=sum(z['kind']==k and bool(z['eligibility_change']) for z in controls)) for k in ['all_units','single_uncompensated','single_compensated']},historical_cases=len(history),historical_allocated=sum(z['reserve']==0 for z in history),historical_reserved=sum(z['reserve']==500 for z in history),seconds=time.perf_counter()-started)
    summary['history_receipt_files_verified']=history_verified
    for k,v in summary['scale_controls'].items():
        v['eligibility_changes']=v['threshold_exceptions']
        if k=='single_uncompensated':v['threshold_exceptions']=None
    save(out/'historical_parity.json',history);save(out/'summary.json',summary)
    threshold=[dict(scale=s,C=5e-9*s,tau=1e-8,allocation=allocate([dict(owner='a',round=0,C=5e-9*s),dict(owner='b',round=0,C=0)],['a','b'],1)) for s in [.1,1,10,100]];save(out/'threshold_examples.json',threshold)
    assert all(sha(W/p)==h for p,h in locks.items());size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file());assert size<200000000
    hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()};save(out/'completion_receipt.json',dict(status='complete',output_sha256=hashes,input_and_source_hashes_verified=True,bytes_before_receipt=size,seconds=time.perf_counter()-started));print(json.dumps({k:v for k,v in summary.items() if k not in ['all','multiround','historical_development','new_development','new_heldout']},indent=2))
if __name__=='__main__':main(Path(sys.argv[1]) if len(sys.argv)>1 else W/'contribution_analysis/model_raw_c_validation_20261029_attempt_01')
