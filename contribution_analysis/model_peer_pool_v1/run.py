"""Exclusive-create preregistered peer experiment; reference diagnosis after score seal."""
import copy,datetime,time,subprocess,sys,shutil,json
from pathlib import Path
from fractions import Fraction as F
from contribution_analysis.custom_realized_gain_v1.schema import save,read,sha,plain
from contribution_analysis.model_consensus_pool_v1.core import canonical_hash
from contribution_analysis.three_pool_t3_model_v1.core import execute
from contribution_analysis.three_pool_t3_model_v1.fixtures import poly
from contribution_analysis.model_decision_validity_v1.fixtures import cases as oldcases
from .core import peer_score
from .reference import score as reference_score,poly as value,verify_branch
W=Path(__file__).resolve().parents[2]
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def cases():
    out=oldcases()
    for c in out:c['split']='old_development';c['owners']=['owner_a','owner_b']
    for i,(b,q) in enumerate(zip(map(F,['3/5','17/20','11/20','9/10']),map(F,['6/5','12/5','8/5','14/5']))):
        for cat,models,weights,ref in [('truthful_minority',[poly(1),poly(-b),poly(-b)],['3/4','1/8','1/8'],poly(1)),('majority_correct',[poly(1),poly(-b),poly(-b)],['3/4','1/8','1/8'],poly(-1)),('useful_restraint',[poly(1,2*q),poly(-1),poly(-1)],['1/2','1/4','1/4'],poly(0,q)),('false_restraint',[poly(1,2*q),poly(-1),poly(-1)],['1/2','1/4','1/4'],poly(-1))]:
            out.append(dict(id=f'{cat}_{i}',category=cat,split='new_development' if i<2 else 'new_heldout',models=models,weights=weights,reference=ref,owners=['owner_a','owner_b','owner_c'],target=[0],anchor_weights=['1/2','1/2'],anchors=[['1','0'],['-1/4','0']],start=['0','0'],eta='1/10',parameter_index=i,b=str(b),q=str(q)))
    return out

def manifest(c,probes):return dict(schema_version='1',owners=c['owners'],models=[dict(id='m'+str(i),owner=o,v=float(F(c['weights'][i])),version=canonical_hash(c['models'][i]),probe=probes[i]) for i,o in enumerate(c['owners'])],rounds=['t'],budget=500,probe_id='fixed25')
def verify_scores(m,s,got):
    ref=reference_score(m,s)
    for k,r in ref.items():
        for o,v in r['scores'].items():assert abs(v-got[k]['scores'][o])<1e-10
        for key in ['q','amounts','status']:assert r[key]==got[k][key],(k,key)

def analysis(rows):
    def sign(z):return 1 if z>1e-10 else -1 if z< -1e-10 else 0
    out={}
    for method in ['peer','raw','full_normalized','min']:
        groups={}
        for scope in ['all']+sorted({r['split'] for r in rows})+sorted({r['category'] for r in rows}):
            rs=[r for r in rows if scope in ['all',r['split'],r['category']]];pos=[r for r in rs if r['methods'][method]['score']>1e-8];paid=[r for r in rs if r['methods'][method]['amount']>0];units=sum(r['methods'][method]['amount'] for r in rs)
            groups[scope]=dict(owner_actions=len(rs),sign_matches=sum(sign(r['D'])==sign(r['methods'][method]['score']) for r in rs),beneficial_negative=sum(r['D']>1e-10 and r['methods'][method]['score']< -1e-10 for r in rs),beneficial_zero=sum(r['D']>1e-10 and abs(r['methods'][method]['score'])<=1e-10 for r in rs),harmful_positive=sum(r['D']< -1e-10 and r['methods'][method]['score']>1e-10 for r in rs),positive_score_precision=sum(r['D']>1e-10 for r in pos)/len(pos) if pos else None,positive_amount_precision=sum(r['D']>1e-10 for r in paid)/len(paid) if paid else None,harmful_units=sum(r['methods'][method]['amount'] for r in rs if r['D']< -1e-10),harmful_unit_fraction=sum(r['methods'][method]['amount'] for r in rs if r['D']< -1e-10)/units if units else None,zero_D_units=sum(r['methods'][method]['amount'] for r in rs if abs(r['D'])<=1e-10),allocated_units=units,reserve_rate=1-units/(500*len({r['id'] for r in rs})))
        h=[r for r in rows if r['split']=='new_heldout'];out[method]=dict(groups=groups,quality_gate_passed=all(sign(r['D'])==sign(r['methods'][method]['score']) and not(r['D']< -1e-10 and r['methods'][method]['amount']>0) for r in h))
    return out

def main():
    root=Path(sys.argv[1]) if len(sys.argv)>1 else W/'contribution_analysis/model_peer_pool_experiment_v1_20261028';root.mkdir(exist_ok=False);started=time.perf_counter();calls=0
    for sub in ['cases','scores','diagnostics']: (root/sub).mkdir()
    try:
        deps=[W/'host_server.py',W/'docs/model_peer_pool_design_v1_20261028.md']
        for folder in ['model_peer_pool_v1','model_consensus_pool_v1','three_pool_t3_model_v1','model_step_decision_v1','model_decision_validity_v1','custom_realized_gain_v1']:deps+=list((W/'contribution_analysis'/folder).glob('*.py'))
        locked={}
        for p in deps:
            rel=p.relative_to(W);dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);locked[str(rel)]=sha(p)
        save(root/'source_lock.json',locked)
        old=W/'contribution_analysis/model_decision_validity_experiment_v1_20261028/completion_receipt.json';receipt=read(old)
        for rel,h in receipt['file_sha256'].items():assert sha(old.parent/rel)==h
        save(root/'historical_receipts.json',{str(old):dict(sha256=sha(old),verified_files=len(receipt['file_sha256']))})
        fixed=cases();grid=[[x,y] for x in [-1,-.5,0,.5,1] for y in [-1,-.5,0,.5,1]];save(root/'fixtures.json',dict(cases=fixed,grid=grid,factors=[.1,10,100]))
        save(root/'registration.json',dict(created_at=stamp(),fixtures_sha256=sha(root/'fixtures.json'),tasks=46,owner_actions=108,max_solves=324,max_api=30000,max_probe=2700,max_seconds=180,max_bytes=100000000,new_training=0,pooled_calls=0,actual_payments=0))
        t=subprocess.run([sys.executable,'-m','unittest','contribution_analysis.model_peer_pool_v1.test_core','-v'],capture_output=True,text=True,cwd=W);save(root/'technical_tests.json',dict(returncode=t.returncode,stdout=t.stdout,stderr=t.stderr));assert t.returncode==0
        probes={c['id']:[[float(abs(value(p,z))) for z in grid] for p in c['models']] for c in fixed};save(root/'probe_lock.json',dict(locked_at=stamp(),solver_calls_so_far=0,probes=probes,analytic_calls=2700,fixtures_sha256=sha(root/'fixtures.json')))
        rows=[];variants=[];controls=[];stages=dict(control=0,full=0,minus=0,common=0);maxerr=0.;execution=[]
        for split in ['old_development','new_development','new_heldout']:
            for c in [c for c in fixed if c['split']==split]:
                begin=stamp();m=manifest(c,probes[c['id']]);slots=[];runs={};full=None
                for j,o in enumerate(c['owners']):
                    run_c=copy.deepcopy(c);run_c['target']=[i for i,a in enumerate(c['owners']) if a==o];run_c['reference']=poly(0);d=plain(execute(run_c));maxerr=max(maxerr,verify_branch(run_c,d));runs[o]=d
                    if full is None:full=d['lock']['full']
                    assert full==d['lock']['full']
                    for i,e in enumerate(d['events']):
                        assert abs(float(value(c['models'][e['model']],e['x']))-e['reply'])<1e-8
                        assert not(e['stage']=='minus' and e['model'] in run_c['target'])
                        assert not(e['stage']=='common' and i<d['lock']['event_index'])
                    for stage,n in d['stage_calls'].items():stages[stage]+=n;calls+=n
                    slot=dict(owner=o,round='t',manifest_hash=canonical_hash(m),versions={a['id']:a['version'] for a in m['models']},full=dict(id=c['id']+'_full',losses={'m'+str(i):z for i,z in enumerate(d['score']['losses_full'])}),minus=dict(id=c['id']+'_'+o+'_minus',losses={'m'+str(i):z for i,z in enumerate(d['score']['losses_minus'])}));slots.append(slot)
                save(root/'cases'/(c['id']+'.json'),dict(fixture=c,manifest=m,slots=slots,owner_runs=runs,started_at=begin,ended_at=stamp(),probe_lock_sha256=sha(root/'probe_lock.json')))
                scores=peer_score(m,slots);verify_scores(m,slots,scores);save(root/'scores'/(c['id']+'.json'),dict(sealed_at=stamp(),methods=scores,case_sha256=sha(root/'cases'/(c['id']+'.json'))))
                for j in range(len(c['models'])):
                    for factor in [.1,10,100]:
                        vm=copy.deepcopy(m);vs=copy.deepcopy(slots);mid='m'+str(j);vm['models'][j]['probe']=[factor*z for z in vm['models'][j]['probe']]
                        for s in vs:
                            s['manifest_hash']=canonical_hash(vm)
                            for arm in ['full','minus']:s[arm]['losses'][mid]*=factor
                        v=peer_score(vm,vs);verify_scores(vm,vs,v);errors={o:abs(scores['peer']['scores'][o]-v['peer']['scores'][o]) for o in c['owners']};assert all(e<=1e-10+1e-9*abs(scores['peer']['scores'][o]) for o,e in errors.items())
                        assert all((scores['peer']['scores'][o]>1e-8)==(v['peer']['scores'][o]>1e-8) for o in c['owners'])
                        assert all(abs(scores['peer']['q'][o]-v['peer']['q'][o])<=1e-8 for o in c['owners']) if scores['peer']['q'] else v['peer']['q'] is None
                        variants.append(dict(id=c['id'],model=j,factor=factor,errors=errors,peer=v['peer'],integer_differences={o:v['peer']['amounts'][o]-scores['peer']['amounts'][o] for o in c['owners']}))
                for s in slots:
                    change=copy.deepcopy(slots);changed=next(z for z in change if z['owner']==s['owner']);j=c['owners'].index(s['owner']);changed['minus']['losses']['m'+str(j)]+=17
                    got=peer_score(m,change);assert got['peer']['scores'][s['owner']]==scores['peer']['scores'][s['owner']];controls.append(dict(id=c['id'],owner=s['owner'],own_endpoint_score_unchanged=True))
                diag=[]
                for o,d in runs.items():
                    D=float(value(c['reference'],d['lock']['minus'])-value(c['reference'],d['lock']['full']));r=dict(id=c['id'],category=c['category'],split=split,owner=o,D=D,methods={k:dict(score=z['scores'][o],amount=z['amounts'][o],q=z['q'][o] if z['q'] else None) for k,z in scores.items()});rows.append(r);diag.append(r)
                save(root/'diagnostics'/(c['id']+'.json'),dict(evaluated_at=stamp(),scores_sha256=sha(root/'scores'/(c['id']+'.json')),rows=diag));execution.append(dict(id=c['id'],split=split,started_at=begin,ended_at=stamp()))
                assert calls<=30000 and time.perf_counter()-started<=180
            if split=='new_development':save(root/'development_checkpoint.json',dict(sealed_at=stamp(),completed_tasks=len(execution),scores_sha256={p.name:sha(p) for p in (root/'scores').glob('*.json')},analysis=analysis(rows)))
        save(root/'heldout_checkpoint.json',dict(sealed_at=stamp(),completed_tasks=len(execution),analysis=analysis([r for r in rows if r['split']=='new_heldout'])))
        from .test_core import fixture
        tm,ts=fixture();before=peer_score(tm,ts)['peer'];ts[1]['minus']['losses']['a']=.9;after=peer_score(tm,ts)['peer'];assert before['scores']['a']==after['scores']['a'] and after['amounts']['a']==500
        save(root/'controls.json',dict(own_endpoint=controls,indirect_allocation_toy=dict(manifest=tm,original=before,modified_slots=ts,modified=after,interpretation='arbitrary reply threat only; locked-model admissibility not demonstrated')))
        save(root/'scale_variants.json',variants);save(root/'execution_order.json',execution)
        for rel,h in locked.items():assert sha(W/rel)==h and sha(root/'source_snapshot'/rel)==h
        assert sha(old)==read(root/'historical_receipts.json')[str(old)]['sha256']
        for rel,h in receipt['file_sha256'].items():assert sha(old.parent/rel)==h
        size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file());assert size<=100000000
        save(root/'independent_audit.json',dict(passed=True,independent_fraction_scoring=True,independent_fraction_branch=True,raw_reply_audits=calls,owner_actions=len(rows),max_branch_error=maxerr,variant_count=len(variants),sources_unchanged=True,historical_files_unchanged=True))
        save(root/'summary.json',dict(complete=True,rows=rows,analysis=analysis(rows),tasks=len(fixed),owner_actions=len(rows),one_round_solves=len(rows)*3,search_common_api=calls,stage_api=stages,probe_calls=2700,adapter_dummy_model_diagnostic_calls=sum(len(c['models'])**2*21 for c in fixed),adapter_dummy_reference_calls=sum(len(c['models'])* (len(c['models'])*21+2) for c in fixed),true_reference_endpoint_evaluations=216,variants=len(variants),seconds=time.perf_counter()-started,bytes_before_receipt=size,new_training=0,pooled_calls=0,actual_payments=0,quality_validity_established=False))
        save(root/'completion_receipt.json',dict(completed_at=stamp(),file_sha256={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}));print(root, 'COMPLETE',calls,'API',len(rows),'actions',flush=True)
    except Exception as exc:save(root/'failure.json',dict(error=repr(exc),api_calls=calls,seconds=time.perf_counter()-started));raise
if __name__=='__main__':main()
