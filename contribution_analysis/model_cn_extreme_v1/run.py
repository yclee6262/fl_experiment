"""Preregistered 64-case internal-F diagnostic, with exclusive output creation."""
import copy,json,shutil,subprocess,sys,time
from pathlib import Path
from fractions import Fraction as Q
from collections import Counter
import numpy as np
from contribution_analysis.model_cn_three_way_v1.run import allocation
from contribution_analysis.model_raw_c_validation_v1.run import calculate,sha,save,verify_receipt,classify
from contribution_analysis.model_direction_pool_v1.core import execute
W=Path(__file__).resolve().parents[2]
OLD=W/'contribution_analysis/model_cn_three_way_20261029_attempt_01'
SRC=W/'contribution_analysis/model_direction_pool_experiment_v1_20261029_attempt_02'
DESIGN=W/'docs/model_cn_extreme_design_20261030.md'
def fixtures():
    out=[]
    for family in ['exact_overlap','near_overlap','curved_overlap','opposing_unique']:
        for style in ['balanced','concentrated']:
            for delta in ['.001','.003','.01','.03']:
                d=Q(delta);weights=[(1-d)/2,(1-d)/2,d] if style=='balanced' else [Q('.95'),Q('.05')-d,d]
                for it in [1,8]:
                    models=[dict(offset='1000',linear=['1','0'],quadratic=[['0','0'],['0','0']]) for _ in range(3)]
                    models[2]['linear']=['-1','1'] if family=='opposing_unique' else ['0','1']
                    if family=='near_overlap':models[1]['linear'][1]='1/50'
                    if family=='curved_overlap':
                        for m in models:m['quadratic']=[['1/2','0'],['0','1/2']]
                    out.append(dict(id=f'{family}_{style}_{delta[1:]}_{it}',category=family,weight_style=style,delta=delta,split='new_development' if delta in ['.001','.01'] else 'new_parameter_holdout',models=models,weights=list(map(str,weights)),owners=['a','b','c'],anchors=[['1','0'],['0','1']],anchor_weights=['1/2','1/2'],start=['0','0'],iterations=it))
    return out

def ratio(x,y):return None if not y else float(x/y)
def diagnostic(record,scope,source):
    rows=calculate(record);owners=record['fixture']['owners'];T=len(record['full']);tau=max(Q('1e-8'),T*Q('1e-10'))
    S={o:sum((Q(str(r['C'])) for r in rows if r['owner']==o),Q()) for o in owners}
    R={o:sum((Q(str(r['N'])) for r in rows if r['owner']==o),Q()) for o in owners} if all(r['N'] is not None for r in rows) else None
    alloc={'C_only':allocation(rows,owners,T)}
    for b in [.1,.2,.3]:alloc[str(b)]=allocation(rows,owners,T,b,True)
    wc={o:S[o] if S[o]>tau else Q() for o in owners};cn=sum(wc.values());eligible=[o for o in owners if S[o]>tau]
    wn=None if R is None else {o:R[o] if S[o]>tau and R[o]>tau else Q() for o in owners};nn=None if wn is None else sum(wn.values());nt=None if R is None else sum(R[o] for o in eligible)
    weights=np.array([float(Q(x)) for x in record['fixture']['weights']]);weights/=weights.sum();fp=weights@np.array(record['probe']);fr=float(np.quantile(fp,.9)-np.quantile(fp,.1))
    details={}
    for o in owners:
        qc=ratio(wc[o],cn);qn=None if wn is None else ratio(wn[o],nn);qnt=None if R is None or o not in eligible else ratio(R[o],nt)
        pays={}
        for b in [.1,.2,.3]:
            a=alloc[str(b)]
            pays[str(b)]=None if a['status']=='held' else dict(C_entitlement=(qc or 0)*round(500*(1-b)),N_entitlement=(qn or 0)*round(500*b),C_units=a['arms']['C']['amounts'][o],N_units=a['arms']['N']['amounts'][o],total_units=a['amounts'][o],budget_share=a['amounts'][o]/500,fractional_budget_share=((qc or 0)*(1-b)+(qn or 0)*b))
        others={k:sum((S[z]>tau if k=='positive' else S[z]<-tau if k=='negative' else abs(S[z])<=tau) for z in owners if z!=o) for k in ['positive','negative','zero']}
        actual=pays['0.2'] is not None and pays['0.2']['N_units']>0
        details[o]=dict(**classify([r for r in rows if r['owner']==o],T),S_exact=str(S[o]),R=None if R is None else float(R[o]),R_exact=None if R is None else str(R[o]),C_over_tau=float(S[o]/tau),C_over_Fprobe_range=None if fr<=1e-12 else float(S[o])/fr,qC=qc,qC_exact=None if not cn else str(wc[o]/cn),qN=qn,qN_exact=None if not nn or wn is None else str(wn[o]/nn),qN_theoretical=qnt,other_C_signs=others,C_eligible_count=len(eligible),N_eligible_count=None if wn is None else sum(v>0 for v in wn.values()),unique_C_eligible=o in eligible and len(eligible)==1,unique_N_eligible=wn is not None and wn[o]>0 and sum(v>0 for v in wn.values())==1,multiple_positive_C_unique_N=len(eligible)>1 and wn is not None and wn[o]>0 and sum(v>0 for v in wn.values())==1,E_relative=qc is not None and qn is not None and qc<=.01 and qn>=.9 and actual,E_threshold=1<float(S[o]/tau)<=10 and qn is not None and qn>=.9 and actual,payments=pays)
    return dict(id=record['fixture']['id'],scope=scope,source=source,fixture=record['fixture'],T=T,rows=rows,owners=details,methods=alloc,Fprobe_range=fr,Fprobe_values=fp.tolist())

def summarize(cases):
    out={}
    for scope in sorted({c['scope'] for c in cases}):
        cs=[c for c in cases if c['scope']==scope];out[scope]=dict(cases=len(cs),rounds=sum(c['T'] for c in cs),owner_rounds=sum(len(c['rows']) for c in cs),owner_cases=sum(len(c['owners']) for c in cs),signs=dict(Counter(v['classification'] for c in cs for v in c['owners'].values())),E_relative=[dict(case=c['id'],owner=o) for c in cs for o,v in c['owners'].items() if v['E_relative']],E_threshold=[dict(case=c['id'],owner=o) for c in cs for o,v in c['owners'].items() if v['E_threshold']],held=sum(c['methods']['0.2']['status']=='held' for c in cs))
    return out

def main(out):
    started=time.perf_counter();out.mkdir(exist_ok=False)
    for d in ['cases','solver_records']:(out/d).mkdir()
    costs=dict(engine_calls=0,analytic_evaluations=0,cache_audit_evaluations=0,reference_calls=0,pooled_calls=0,full_round_steps=0,control_round_steps=0,minus_round_steps=0);done=[];failures=[]
    try:
        deps=[DESIGN,W/'host_server.py']
        for folder in ['model_cn_extreme_v1','model_cn_three_way_v1','model_raw_c_validation_v1','model_direction_pool_v1','model_direction_diagnostic_v1','model_step_decision_v1','three_pool_t3_model_v1','model_consensus_pool_v1','custom_realized_gain_v1']:deps+=sorted((W/'contribution_analysis'/folder).glob('*.py'))
        oldpaths=sorted((SRC/'cases').glob('*.json'))+sorted((OLD/'solver_records').glob('*.json'));assert len(oldpaths)==94
        inputs=[OLD/'completion_receipt.json',SRC/'completion_receipt.json']+oldpaths+sorted((OLD/'cases').glob('*.json'))
        locks={str(p.relative_to(W)):sha(p) for p in deps+inputs};save(out/'source_lock.json',locks)
        for p in deps:
            dest=out/'source_snapshot'/p.relative_to(W);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        fs=fixtures();save(out/'fixtures.json',fs);save(out/'registration.json',dict(fixtures_sha256=sha(out/'fixtures.json'),design_sha256=sha(DESIGN),cases=64,max_engine_calls=1200,max_analytic_evaluations_including_audit=200000,max_seconds=180,max_bytes=200000000,beta_primary=.2,beta_sensitivity=[.1,.3],no_reference_or_pooled=True))
        test=subprocess.run([sys.executable,'-m','unittest','contribution_analysis.model_cn_extreme_v1.test_core'],capture_output=True,text=True,cwd=W);save(out/'test_report.json',dict(returncode=test.returncode,stdout=test.stdout,stderr=test.stderr));assert test.returncode==0
        verified={str(p.relative_to(W)):verify_receipt(p) for p in [OLD,SRC]};parity=[];seen=set()
        for p in oldpaths:
            r=json.loads(p.read_text());id=r['fixture']['id'];assert id not in seen;seen.add(id);c=diagnostic(r,'historical_94',str(p.relative_to(W)));old=json.loads((OLD/'cases'/p.name).read_text())
            for newkey,oldkey in [('C_only','C_only')]+[(str(b),f'C+N_positive_{b}') for b in [.1,.2,.3]]:
                for k in ['amounts','reserve','status']:assert c['methods'][newkey][k]==old['methods'][oldkey][k]
            parity.append(dict(id=id,source=str(p.relative_to(W)),sha256=sha(p),matched=True));save(out/'cases'/p.name,c);done.append(c)
        save(out/'historical_parity.json',parity)
        for split in ['new_development','new_parameter_holdout']:
            for f in [f for f in fs if f['split']==split]:
                try:
                    r=execute(f,f['iterations']);costs['engine_calls']+=r['solves'];costs['analytic_evaluations']+=sum(v['evaluations'] for v in r['cost'].values());costs['cache_audit_evaluations']+=r['cache_audit_evaluations'];costs['full_round_steps']+=len(r['full']);costs['control_round_steps']+=len(r['full']);costs['minus_round_steps']+=len(r['branches'])
                    save(out/'solver_records'/(f['id']+'.json'),r);c=diagnostic(r,split,str((out/'solver_records'/(f['id']+'.json')).relative_to(W)));save(out/'cases'/(f['id']+'.json'),c);done.append(c)
                except Exception as exc:
                    fail=dict(id=f['id'],error=repr(exc));failures.append(fail);save(out/'solver_records'/(f['id']+'_failure.json'),fail)
                assert costs['engine_calls']<=1200 and costs['analytic_evaluations']+costs['cache_audit_evaluations']<=200000 and time.perf_counter()-started<180
            save(out/(split+'_checkpoint.json'),dict(completed=[c['id'] for c in done if c['scope']==split],failures=failures,source_hashes_verified=all(sha(W/p)==h for p,h in locks.items())))
        s=summarize(done);s.update(costs=costs,failures=failures,receipt_files_verified=verified,seconds=time.perf_counter()-started,solver_tasks_attempted=64,new_solver_tasks_completed=sum(c['scope']!='historical_94' for c in done),synthetic_cases_included=0);save(out/'summary.json',s)
        assert all(sha(W/p)==h for p,h in locks.items());size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file());assert size<200000000
        save(out/'completion_receipt.json',dict(status='complete_with_case_failures' if failures else 'complete',file_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},bytes_before_receipt=size,seconds=time.perf_counter()-started));print(json.dumps(s))
    except Exception as exc:save(out/'failure.json',dict(error=repr(exc),costs=costs,completed=[c['id'] for c in done]));raise
if __name__=='__main__':main(Path(sys.argv[1]) if len(sys.argv)>1 else W/'contribution_analysis/model_cn_extreme_20261030_attempt_01')
