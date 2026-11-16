"""Locked internal-F comparison; analytic solver and synthetic ledger scopes stay separate."""
import copy,json,math,shutil,subprocess,sys,time
from pathlib import Path
from fractions import Fraction as Q
from collections import Counter
from contribution_analysis.model_raw_c_validation_v1.run import calculate, classify, allocate, sha, save, verify_receipt
from contribution_analysis.model_direction_pool_v1.core import execute
from contribution_analysis.model_consensus_pool_v1.core import settle
from contribution_analysis.three_pool_t3_model_v1.fixtures import poly
W=Path(__file__).resolve().parents[2]
SOURCE=W/'contribution_analysis/model_direction_pool_experiment_v1_20261029_attempt_02'
BASE=W/'contribution_analysis/model_raw_c_validation_20261029_attempt_03'
DESIGN=W/'docs/model_cn_three_way_design_20261029.md'

def allocation(rows,owners,T,beta=0,positive=False):
    original=allocate(rows,owners,T,beta,'N')
    if not beta or not positive or original['status']=='held':return original
    tau=max(Q(1,10**8),T*Q(1,10**10))
    S={o:sum((Q(str(r['C'])) for r in rows if r['owner']==o),Q(0)) for o in owners}
    R={o:sum((Q(str(r['N'])) for r in rows if r['owner']==o),Q(0)) if S[o]>tau else Q(0) for o in owners}
    main=settle(S,round(500*(1-beta)),T);supp=settle(R,round(500*beta),T)
    amounts={o:main['amounts'][o]+supp['amounts'][o] for o in owners};reserve=main['reserve']+supp['reserve']
    return dict(status='allocated' if sum(amounts.values()) else 'reserved',amounts=amounts,reserve=reserve,arms={'C':main,'N':supp},totals=original['totals'],eligible={o:S[o]>tau for o in owners})

def methods(rows,owners,T):
    out={'C_only':allocation(rows,owners,T)}
    for b in [.1,.2,.3]:
        out[f'C+N_{b}']=allocation(rows,owners,T,b)
        out[f'C+N_positive_{b}']=allocation(rows,owners,T,b,True)
    return out

def fixtures():
    out=[]
    for parameter,b in enumerate([.9,1.3]):
        q=[16,26][parameter]
        families=[('unique_harmful',[poly(1,-2*q),poly(1)],['1/2']*2),('convex_restraint',[poly(1,4*b),poly(-1),poly(-1)],['1/2','1/4','1/4']),('mixed_tangent',[poly(.3,2*b),poly(.5,3*b),poly(.7,b)],['1/3']*3),('mixed_2d',[poly(1,2*b),poly(-1,b),poly(.2,3*b)],['1/3']*3)]
        for family,models,weights in families:
            anchors=[['1','0'],['-1/4','0']]
            if family=='mixed_2d':
                anchors=[['1','1'],['-1','1'],['0','-1']]
                for j,m in enumerate(models):m['linear'][1]=str(Q(j-1,2));m['quadratic'][1][1]=str(Q(j+1,2))
            for n in [1,8]:
                out.append(dict(id=f'{family}_{parameter}_{n}',category=family,split='new_development' if parameter==0 else 'new_parameter_holdout',models=copy.deepcopy(models),weights=weights,owners=[f'owner_{chr(97+j)}' for j in range(len(models))],anchors=anchors,anchor_weights=[str(Q(1,len(anchors)))]*len(anchors),start=['0','0'],iterations=n))
    return out

def synthetic():
    out=[]
    for k,cs in enumerate([(2,-1),(1,-2),(1,-1),(-1,2),(-2,1),(-1,1)]):
        rows=[dict(owner=o,round=t,C=c,N=n,distance=1) for o,seq,n in [('a',cs,.5),('b',[1,1],.01)] for t,c in enumerate(seq)]
        out.append(dict(id=f'ledger_cancellation_{k}',rows=rows,T=2,owners=['a','b']))
    for label,c in [('below',.99e-8),('equal',1e-8),('above',1.01e-8),('micro',2e-8)]:
        out.append(dict(id='ledger_threshold_'+label,rows=[dict(owner='a',round=0,C=c,N=.9,distance=1),dict(owner='b',round=0,C=1,N=.01,distance=1)],T=1,owners=['a','b']))
    return out

def case(name,scope,rows,owners,T):
    m=methods(rows,owners,T);cl={o:classify([r for r in rows if r['owner']==o],T) for o in owners}
    detail={}
    for key,a in m.items():
        if a['status']=='held':continue
        assert sum(a['amounts'].values())+a['reserve']==500
        if 'positive' in key:assert all(a['amounts'][o]==0 for o in owners if cl[o]['classification']!='positive')
        detail[key]=dict(max_budget_share=max(a['amounts'].values())/500,delta_units_from_C_only={o:a['amounts'][o]-m['C_only']['amounts'][o] for o in owners},fractional_entitlement={o:sum(arm['q'][o]*sum(arm['amounts'].values()) if arm['q'] is not None else 0 for arm in a['arms'].values()) for o in owners})
    return dict(id=name,scope=scope,T=T,rows=rows,owners=cl,methods=m,allocation_diagnostics=detail)

def summary(cases):
    out={}
    for scope in sorted({c['scope'] for c in cases}):
        group=[c for c in cases if c['scope']==scope];ms={}
        for key in group[0]['methods']:
            bins={s:dict(owners=0,paid_owners=0,units=0) for s in ['positive','negative','zero']};reserve=held=0
            for c in group:
                a=c['methods'][key]
                if a['status']=='held':held+=1;continue
                reserve+=a['reserve']
                for o,z in c['owners'].items():
                    v=bins[z['classification']];v['owners']+=1;v['paid_owners']+=a['amounts'][o]>0;v['units']+=a['amounts'][o]
            ms[key]=dict(signs=bins,reserve=reserve,held=held)
        out[scope]=dict(cases=len(group),rounds=sum(c['T'] for c in group),owner_rounds=sum(len(c['rows']) for c in group),classes=dict(Counter(z['kind'] for c in group for z in c['owners'].values())),methods=ms)
    return out

def main(out):
    started=time.perf_counter();out.mkdir(exist_ok=False)
    for d in ['cases','solver_records']:(out/d).mkdir()
    costs=dict(solves=0,analytic_evaluations=0,cache_audit_evaluations=0,reference_calls=0,pooled_calls=0);done=[];failures=[]
    try:
        deps=[DESIGN,W/'host_server.py']
        for folder in ['model_cn_three_way_v1','model_raw_c_validation_v1','model_direction_pool_v1','model_direction_diagnostic_v1','model_step_decision_v1','three_pool_t3_model_v1','model_consensus_pool_v1','custom_realized_gain_v1']:
            deps+=sorted((W/'contribution_analysis'/folder).glob('*.py'))
        inputs=[SOURCE/'completion_receipt.json',BASE/'completion_receipt.json']+sorted((SOURCE/'cases').glob('*.json'))+sorted((BASE/'cases').glob('*.json'))
        locks={str(p.relative_to(W)):sha(p) for p in deps+inputs};save(out/'source_lock.json',locks)
        for p in deps:
            dest=out/'source_snapshot'/p.relative_to(W);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        fs=fixtures();ss=synthetic();save(out/'fixtures.json',fs);save(out/'synthetic_fixtures.json',ss)
        save(out/'registration.json',dict(fixtures_sha256=sha(out/'fixtures.json'),synthetic_sha256=sha(out/'synthetic_fixtures.json'),max_tasks=16,max_solves=500,max_evaluations=100000,max_seconds=180,max_bytes=100000000,beta_primary=.2,beta_sensitivity=[.1,.3],r_ref_multiplier=.1,scoring='fixed_complete_internal_F',reference_in_score=False))
        verified=dict(source=verify_receipt(SOURCE),baseline=0)
        br=json.loads((BASE/'completion_receipt.json').read_text())
        for p,h in br['output_sha256'].items():assert sha(BASE/p)==h;verified['baseline']+=1
        test=subprocess.run([sys.executable,'-m','unittest','contribution_analysis.model_cn_three_way_v1.test_core'],capture_output=True,text=True,cwd=W);save(out/'test_report.json',dict(returncode=test.returncode,stdout=test.stdout,stderr=test.stderr));assert test.returncode==0
        parity=[]
        for p in sorted((SOURCE/'cases').glob('*.json')):
            r=json.loads(p.read_text());c=case(p.stem,'historical_78',calculate(r),r['fixture']['owners'],len(r['full']));old=json.loads((BASE/'cases'/p.name).read_text())
            for key in ['C_only']+[f'C+N_{b}' for b in [.1,.2,.3]]:
                for k in ['amounts','reserve','status']:assert c['methods'][key][k]==old['methods'][key][k]
            parity.append(dict(id=p.stem,matched=True));save(out/'cases'/p.name,c);done.append(c)
        save(out/'historical_parity.json',parity)
        for split in ['new_development','new_parameter_holdout']:
            for f in [z for z in fs if z['split']==split]:
                try:
                    r=execute(f,f['iterations']);costs['solves']+=r['solves'];costs['analytic_evaluations']+=sum(z['evaluations'] for z in r['cost'].values());costs['cache_audit_evaluations']+=r['cache_audit_evaluations']
                    save(out/'solver_records'/(f['id']+'.json'),r);c=case(f['id'],split,calculate(r),f['owners'],len(r['full']));save(out/'cases'/(f['id']+'.json'),c);done.append(c)
                except Exception as exc:
                    failure=dict(id=f['id'],error=repr(exc));failures.append(failure);save(out/'solver_records'/(f['id']+'_failure.json'),failure)
                assert costs['solves']<=500 and costs['analytic_evaluations']+costs['cache_audit_evaluations']<=100000 and time.perf_counter()-started<180
            save(out/(split+'_checkpoint.json'),dict(completed=[c['id'] for c in done if c['scope']==split],failures=failures,source_hashes_verified=all(sha(W/p)==h for p,h in locks.items())))
        for f in ss:
            c=case(f['id'],'synthetic_ledger',f['rows'],f['owners'],f['T']);save(out/'cases'/(f['id']+'.json'),c);done.append(c)
        s=summary(done);s.update(costs=costs,failures=failures,receipt_files_verified=verified,seconds=time.perf_counter()-started,new_solver_tasks=16,scope_warning='Synthetic ledger cases are not solver outcomes. Internal F signs are not independent quality validation.')
        save(out/'summary.json',s);assert all(sha(W/p)==h for p,h in locks.items());size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file());assert size<100000000
        save(out/'completion_receipt.json',dict(status='complete_with_case_failures' if failures else 'complete',file_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},bytes_before_receipt=size,seconds=time.perf_counter()-started));print(json.dumps(dict(output=str(out),costs=costs,seconds=time.perf_counter()-started,failures=failures)))
    except Exception as exc:
        save(out/'failure.json',dict(error=repr(exc),costs=costs,completed=[c['id'] for c in done],seconds=time.perf_counter()-started));raise
if __name__=='__main__':main(Path(sys.argv[1]) if len(sys.argv)>1 else W/'contribution_analysis/model_cn_three_way_20261029_attempt_01')
