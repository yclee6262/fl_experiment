"""Post-execution parent audit. Reads sealed attempts, never reruns a solver.

Added after attempts 01/02; its source is sealed separately in the review folder.
Uses the independent rational reference, never the production scorer/runner.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from fractions import Fraction as F
from .reference import score as reference_score, verify_branch, poly

WORK=Path(__file__).resolve().parents[2]
METHODS=('peer','raw','full_normalized','min')


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)


def near(a,b,tol=1e-10):
    if abs(F(str(a))-F(str(b)))>F(str(tol)):raise ValueError('numeric_mismatch')


def scores_check(manifest,slots,methods):
    expected=reference_score(manifest,slots)
    for method,ref in expected.items():
        got=methods[method]
        if got['status']!=ref['status']or got['amounts']!=ref['amounts']:raise ValueError('settlement_mismatch')
        if any(type(v)is not int for v in got['amounts'].values()):raise ValueError('noninteger_amount')
        if got['reserve']!=(manifest['budget']if ref['status']=='reserved'else 0):raise ValueError('reserve')
        if sum(got['amounts'].values())+got['reserve']!=manifest['budget']:raise ValueError('budget')
        if set(got['scores'])!=set(ref['scores']):raise ValueError('owner_coverage')
        for o,v in ref['scores'].items():near(v,got['scores'][o])
        if ref['q']is None:
            if got['q']is not None:raise ValueError('null_share')
        else:
            if not isinstance(got['q'],dict)or set(got['q'])!=set(ref['q']):raise ValueError('share_coverage')
            for o,v in ref['q'].items():near(v,got['q'][o])
        near(got['tau'],max(F('1e-8'),len(manifest['rounds'])*F('1e-10')))
        # These study tasks have exactly one round. Do not infer multi-round validity.
        if manifest['rounds']!=['t']:raise ValueError('unsupported_study_rounds')
        if {(r['owner'],r['round'])for r in got['rounds']}!={(o,'t')for o in manifest['owners']}:raise ValueError('rounds')
        for r in got['rounds']:near(r['P'],ref['scores'][r['owner']])
    return expected


def sgn(v):return 1 if v>1e-10 else -1 if v < -1e-10 else 0


def stats(rows,method):
    actions=len(rows);positive=[r for r in rows if r['methods'][method]['score']>1e-8]
    paid=[r for r in rows if r['methods'][method]['amount']>0]
    units=sum(r['methods'][method]['amount']for r in rows)
    harmful=sum(r['methods'][method]['amount']for r in rows if r['D'] < -1e-10)
    return dict(owner_actions=actions,sign_matches=sum(sgn(r['D'])==sgn(r['methods'][method]['score'])for r in rows),
        beneficial_negative=sum(r['D']>1e-10 and r['methods'][method]['score'] < -1e-10 for r in rows),
        beneficial_zero=sum(r['D']>1e-10 and abs(r['methods'][method]['score'])<=1e-10 for r in rows),
        harmful_positive=sum(r['D'] < -1e-10 and r['methods'][method]['score']>1e-10 for r in rows),
        positive_score_precision=None if not positive else sum(r['D']>1e-10 for r in positive)/len(positive),
        positive_amount_precision=None if not paid else sum(r['D']>1e-10 for r in paid)/len(paid),
        harmful_units=harmful,harmful_unit_fraction=None if not units else harmful/units,
        zero_D_units=sum(r['methods'][method]['amount']for r in rows if abs(r['D'])<=1e-10),
        allocated_units=units,reserve_rate=1-units/(500*len({r['id']for r in rows})))


def verify_receipt(root):
    receipt=read(root/'completion_receipt.json')
    for rel,h in receipt['file_sha256'].items():
        if sha(root/rel)!=h:raise ValueError('receipt:'+rel)
    for rel,h in read(root/'source_lock.json').items():
        if sha(root/'source_snapshot'/rel)!=h:raise ValueError('snapshot:'+rel)
    return len(receipt['file_sha256'])


def audit(root,previous,out):
    out.mkdir(exist_ok=False)
    sealed=verify_receipt(root);old_sealed=verify_receipt(previous)
    for rel,h in read(root/'source_lock.json').items():
        if sha(WORK/rel)!=h:raise ValueError('final_active_source:'+rel)
    sources=read(root/'source_lock.json');prior_sources=read(previous/'source_lock.json')
    changed=[rel for rel,h in sources.items()if prior_sources.get(rel)!=h]
    if set(changed)!={f'contribution_analysis/model_peer_pool_v1/{n}.py'for n in ['core','reference','test_core']}:raise ValueError('unexpected_attempt_source_change')
    original=WORK/'contribution_analysis/model_consensus_pool_implementation_checks_20261028/technical_receipt.json'
    oldtech=read(original)
    for rel,h in oldtech['source_sha256'].items():
        if sha(WORK/rel)!=h:raise ValueError('old_consensus_source')
    for rel,h in oldtech['artifact_sha256'].items():
        if sha(original.parent/rel)!=h:raise ValueError('old_consensus_artifact')
    for p,h in oldtech['prior_receipts'].items():
        if sha(WORK/p)!=h:raise ValueError('old_prior_receipt')
        for rel,v in read(WORK/p)['file_sha256'].items():
            if sha((WORK/p).parent/rel)!=v:raise ValueError('old_prior_artifact')
    spec=read(root/'fixtures.json');reg=read(root/'registration.json');probe=read(root/'probe_lock.json')
    if sha(root/'fixtures.json')!=sha(previous/'fixtures.json'):raise ValueError('attempt_fixtures_changed')
    if reg['fixtures_sha256']!=sha(root/'fixtures.json')or probe['fixtures_sha256']!=reg['fixtures_sha256']:raise ValueError('fixtures_hash')
    if probe['solver_calls_so_far']!=0 or len(spec['cases'])!=46:raise ValueError('registration')
    if spec['grid']!=[[a,b]for a in [-1,-.5,0,.5,1]for b in [-1,-.5,0,.5,1]]:raise ValueError('probe_grid')
    summary=read(root/'summary.json');oldsummary=read(previous/'summary.json')
    if summary['rows']!=oldsummary['rows']or summary['analysis']!=oldsummary['analysis']:raise ValueError('attempt_results_changed')
    if read(root/'technical_tests.json')['returncode']:raise ValueError('unit_tests')
    docs={};recomputed=[];raw_checks=probe_checks=mutations=branches=0;max_branch=0.;stages=dict(control=0,full=0,minus=0,common=0)
    for c in spec['cases']:
        cid=c['id'];casepath=root/'cases'/(cid+'.json');doc=read(casepath);docs[cid]=doc
        m=doc['manifest'];slots=doc['slots'];scorepath=root/'scores'/(cid+'.json');scored=read(scorepath);diag=read(root/'diagnostics'/(cid+'.json'))
        if c!=doc['fixture']or doc['probe_lock_sha256']!=sha(root/'probe_lock.json'):raise ValueError('case_link')
        if not reg['created_at']<=probe['locked_at']<=doc['started_at']<=doc['ended_at']<=scored['sealed_at']<=diag['evaluated_at']:raise ValueError('lock_order')
        if scored['case_sha256']!=sha(casepath)or diag['scores_sha256']!=sha(scorepath):raise ValueError('score_diagnostic_lock')
        if m['owners']!=c['owners']or m['rounds']!=['t']or m['budget']!=500 or len(m['models'])!=len(c['models']):raise ValueError('manifest')
        for j,model in enumerate(m['models']):
            if model['owner']!=c['owners'][j]or model['id']!='m'+str(j)or model['version']!=digest(c['models'][j]):raise ValueError('model_mapping')
            near(model['v'],F(c['weights'][j]))
            if model['probe']!=probe['probes'][cid][j]:raise ValueError('probe_lock_link')
            for point,v in zip(spec['grid'],model['probe']):near(v,abs(poly(c['models'][j],point)));probe_checks+=1
        if len(slots)!=len(c['owners'])or {s['owner']for s in slots}!=set(c['owners'])or set(doc['owner_runs'])!=set(c['owners']):raise ValueError('all_owner_coverage')
        full=None
        for slot in slots:
            o=slot['owner'];d=doc['owner_runs'][o];rc=d['fixture'];targets=[i for i,owner in enumerate(c['owners'])if owner==o]
            if rc['target']!=targets or rc['models']!=c['models']or rc['weights']!=c['weights']or rc['anchor_weights']!=c['anchor_weights']:raise ValueError('branch_inputs')
            if rc['reference']!={'offset':'1000','linear':['0','0'],'quadratic':[['0','0'],['0','0']]}:raise ValueError('true_reference_in_solver_adapter')
            if d['score']['D_analytic']!=0:raise ValueError('dummy_reference_not_dummy')
            max_branch=max(max_branch,verify_branch(rc,d));branches+=2
            if full is None:full=d['lock']['full']
            if full!=d['lock']['full']or d['lock']['full']!=d['full']['post']['S_current']or d['lock']['minus']!=d['minus']['post']['S_current']:raise ValueError('accepted_endpoints')
            if slot['manifest_hash']!=digest(m)or slot['versions']!={a['id']:a['version']for a in m['models']}:raise ValueError('version_hash')
            strip=lambda stage:[{k:v for k,v in e.items()if k!='stage'}for e in d['events']if e['stage']==stage]
            if strip('control')!=strip('full')or d['control']!=d['full_result']:raise ValueError('uninstrumented_control')
            for stage in stages:
                count=sum(e['stage']==stage for e in d['events'])
                if d['stage_calls'][stage]!=count:raise ValueError('stage_cost')
                stages[stage]+=count
            for i,e in enumerate(d['events']):
                near(e['reply'],poly(c['models'][e['model']],e['x']),1e-8);raw_checks+=1
                if e['stage']=='minus'and e['model']in targets:raise ValueError('excluded_model_called')
                if e['stage']=='common'and i<d['lock']['event_index']:raise ValueError('common_before_lock')
            for arm,label in [('full','losses_full'),('minus','losses_minus')]:
                if slot[arm]['losses']!={'m'+str(j):v for j,v in enumerate(d['score'][label])}:raise ValueError('slot_raw_loss_link')
                for j,v in enumerate(d['score'][label]):
                    candidates=[e for i,e in enumerate(d['events'])if i>=d['lock']['event_index']and e['stage']=='common'and e['model']==j and e['x']==d['lock'][arm]]
                    if not candidates or any(abs(e['reply'])!=v for e in candidates):raise ValueError('endpoint_common_reply')
            D=float(poly(c['reference'],d['lock']['minus'])-poly(c['reference'],d['lock']['full']))
            row=next(r for r in diag['rows']if r['owner']==o);near(D,row['D'])
            want=dict(id=cid,category=c['category'],split=c['split'],owner=o,D=D,
                      methods={k:dict(score=z['scores'][o],amount=z['amounts'][o],q=z['q'][o]if z['q']else None)for k,z in scored['methods'].items()})
            if row!=want:raise ValueError('diagnostic_row')
            recomputed.append(want)
        scores_check(m,slots,scored['methods'])
        # Five different corruptions per case: score, share, amount, endpoint and D.
        for field in ['scores','q','amounts']:
            bad=copy.deepcopy(scored['methods']);o=m['owners'][0]
            if bad['peer'][field]is None:bad['peer'][field]={who:0 for who in m['owners']}
            bad['peer'][field][o]+=1
            try:scores_check(m,slots,bad)
            except (AssertionError,ValueError):mutations+=1
            else:raise ValueError('undetected_settlement_mutation')
        d=copy.deepcopy(doc['owner_runs'][m['owners'][0]]);d['full']['post']['S_current'][0]+=.01
        try:verify_branch(d['fixture'],d)
        except (AssertionError,ValueError):mutations+=1
        else:raise ValueError('undetected_endpoint_mutation')
        row=diag['rows'][0];d=doc['owner_runs'][row['owner']];correct=poly(c['reference'],d['lock']['minus'])-poly(c['reference'],d['lock']['full'])
        badD=-row['D']if abs(row['D'])>1e-10 else 1
        try:near(correct,badD)
        except ValueError:mutations+=1
        else:raise ValueError('undetected_D_mutation')
    key=lambda r:(r['id'],r['owner'])
    if sorted(recomputed,key=key)!=sorted(summary['rows'],key=key):raise ValueError('summary_rows')
    for method in METHODS:
        for scope,got in summary['analysis'][method]['groups'].items():
            rr=[r for r in recomputed if scope in ['all',r['split'],r['category']]]
            if stats(rr,method)!=got:raise ValueError('summary_statistics')
        held=[r for r in recomputed if r['split']=='new_heldout']
        gate=all(sgn(r['D'])==sgn(r['methods'][method]['score'])and not(r['D']< -1e-10 and r['methods'][method]['amount']>0)for r in held)
        if gate!=summary['analysis'][method]['quality_gate_passed']:raise ValueError('quality_gate')
    variants=read(root/'scale_variants.json');expected_keys={(c['id'],j,a)for c in spec['cases']for j in range(len(c['models']))for a in [.1,10,100]}
    if len(variants)!=324 or {(r['id'],r['model'],r['factor'])for r in variants}!=expected_keys:raise ValueError('variant_coverage')
    max_scale_error=max_share_error=0.;integer_diffs=0
    for v in variants:
        doc=docs[v['id']];m=copy.deepcopy(doc['manifest']);slots=copy.deepcopy(doc['slots']);j=v['model'];factor=v['factor'];mid='m'+str(j)
        m['models'][j]['probe']=[factor*a for a in m['models'][j]['probe']]
        for s in slots:
            s['manifest_hash']=digest(m)
            for arm in ['full','minus']:s[arm]['losses'][mid]*=factor
        ref=reference_score(m,slots)['peer'];base=read(root/'scores'/(v['id']+'.json'))['methods']['peer']
        for field in ['scores','q','amounts','status']:
            if v['peer'][field]!=ref[field]:raise ValueError('variant_independent_score')
        for o,p in ref['scores'].items():
            err=abs(p-base['scores'][o]);max_scale_error=max(max_scale_error,err)
            if err!=v['errors'][o]or err>1e-10+1e-9*abs(base['scores'][o]):raise ValueError('invariance_score')
            if (p>1e-8)!=(base['scores'][o]>1e-8):raise ValueError('invariance_eligibility')
            if ref['q']is not None:
                e=abs(ref['q'][o]-base['q'][o]);max_share_error=max(max_share_error,e)
                if e>1e-8:raise ValueError('invariance_share')
            elif base['q']is not None:raise ValueError('invariance_null')
            diff=ref['amounts'][o]-base['amounts'][o]
            if diff!=v['integer_differences'][o]:raise ValueError('integer_diff_record')
            integer_diffs+=diff!=0
    controls=read(root/'controls.json');owncontrols=controls['own_endpoint']
    if len(owncontrols)!=108 or {(r['id'],r['owner'])for r in owncontrols}!={(r['id'],r['owner'])for r in recomputed}:raise ValueError('control_coverage')
    for row in owncontrols:
        d=docs[row['id']];s=copy.deepcopy(d['slots']);owner=row['owner'];j=d['fixture']['owners'].index(owner)
        next(z for z in s if z['owner']==owner)['minus']['losses']['m'+str(j)]+=17
        value=reference_score(d['manifest'],s)['peer']['scores'][owner]
        base=read(root/'scores'/(row['id']+'.json'))['methods']['peer']['scores'][owner]
        if value!=base or not row['own_endpoint_score_unchanged']:raise ValueError('own_control')
    toy=controls['indirect_allocation_toy'];original_slots=copy.deepcopy(toy['modified_slots']);original_slots[1]['minus']['losses']['a']=1.1
    for slots,label in [(original_slots,'original'),(toy['modified_slots'],'modified')]:
        r=reference_score(toy['manifest'],slots)['peer']
        for field in ['scores','q','amounts','status']:
            if toy[label][field]!=r[field]:raise ValueError('indirect_toy')
    if toy['original']['scores']['a']!=toy['modified']['scores']['a']or toy['original']['q']['a']!=.5 or toy['modified']['q']['a']!=1.:raise ValueError('indirect_toy_property')
    order=read(root/'execution_order.json');checkpoint=read(root/'development_checkpoint.json')
    if len(order)!=46 or [r['id']for r in order]!=[c['id']for split in ['old_development','new_development','new_heldout']for c in spec['cases']if c['split']==split]:raise ValueError('execution_order')
    if checkpoint['sealed_at']>min(r['started_at']for r in order if r['split']=='new_heldout'):raise ValueError('heldout_order')
    for name,h in checkpoint['scores_sha256'].items():
        if sha(root/'scores'/name)!=h:raise ValueError('checkpoint_score_changed')
    if stages!=summary['stage_api']or raw_checks!=summary['search_common_api']or probe_checks!=2700 or branches!=216:raise ValueError('cost_counts')
    supplement={}
    for method in METHODS:
        paid=[r for r in recomputed if r['methods'][method]['amount']>0];beneficial=[r for r in recomputed if r['D']>1e-10]
        states=[read(root/'scores'/(c['id']+'.json'))['methods'][method]['status']for c in spec['cases']]
        supplement[method]=dict(paid_owners=len(paid),beneficial_owners=len(beneficial),beneficial_paid=sum(r['D']>1e-10 for r in paid),
            allocated_tasks=states.count('allocated'),reserved_tasks=states.count('reserved'),held_tasks=states.count('held'))
    result=dict(passed=True,final_attempt=str(root),previous_attempt=str(previous),sealed_files=[old_sealed,sealed],
        source_changes_between_attempts=changed,attempt_scores_identical=True,owner_actions=108,branches_checked=branches,
        raw_reply_checks=raw_checks,probe_checks=probe_checks,score_method_panels=46*4,scale_variants=324,own_controls=108,
        mutations_detected=mutations,max_branch_error=max_branch,max_scale_error=max_scale_error,max_share_error=max_share_error,
        integer_share_changes=integer_diffs,counts=supplement,diagnostic_sign_counts={str(k):sum(sgn(r['D'])==k for r in recomputed)for k in [-1,0,1]},
        new_solver_calls=0,new_model_api_calls=0,post_execution_audit=True)
    save(out/'parent_audit.json',result)
    save(out/'audit_provenance.json',dict(audit_source=sha(__file__),reference_source=sha(Path(__file__).parent/'reference.py'),
         input_receipts={str(p):sha(p)for p in [root/'completion_receipt.json',previous/'completion_receipt.json',original]}))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--previous',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();audit(a.root.resolve(),a.previous.resolve(),a.output.resolve())
