"""Freeze, run account tests, old-data settlement, analytic branch diagnostics."""
import argparse
import copy
import datetime
from decimal import Decimal
from pathlib import Path
import shutil
import time
from contribution_analysis.custom_realized_gain_v1.schema import read,save,sha,plain
from .engine import settle
from .reference import verify
from .fixtures import cases,contract,events
from .study import execute,independent

WORK=Path(__file__).resolve().parents[2]
T2=WORK/'contribution_analysis/three_pool_t2_anchor_experiment_v1_20261028'
DESIGN=WORK/'docs/anchor_pool_shadow_design_v1_20261028.md'
RULES=('net','positive_rounds','equal')


def receipts():
    locks={}
    for name in ('three_pool_t0_t1_experiment_v1_20261028','three_pool_t2_anchor_experiment_v1_20261028','three_pool_t3_model_experiment_v1_20261028'):
        root=WORK/'contribution_analysis'/name;p=root/'completion_receipt.json'
        locks[str(p)]=sha(p)
        for rel,h in read(p)['file_sha256'].items():
            if sha(root/rel)!=h:raise ValueError('old_receipt:'+rel)
    return locks


def variants(c,es):
    out=[('reverse',c,list(reversed(es))),('replay',c,es+es),
         ('new_event_ids',c,[dict(e,event_id='changed:'+str(i)) for i,e in enumerate(es)])]
    if c['rounds']:
        cc=copy.deepcopy(c);t=max(cc['rounds'])+1;cc['rounds'].append(t)
        out.append(('zero_round',cc,es+[dict(owner=o,round=t,C='0',event_id=f'zero:{o}',reference_id=c['reference_id']) for o in c['owners']]))
    # Fragmentation is authorized preprocessing: all fixed halves must be present.
    # The resulting unique parent slots (not child identities) enter settlement.
    merged=[]
    for e in es:
        value=None if e['C'] is None else str(Decimal(str(e['C']))/2+Decimal(str(e['C']))/2)
        merged.append(dict(e,C=value))
    out.append(('authorized_halves_reassembled',c,merged))
    # Joint change of numeric unit, including gate and tolerance; decisions unchanged.
    cc=copy.deepcopy(c);cc['epsilon']=str(Decimal(cc['epsilon'])*7);cc['min_release']=str(Decimal(cc['min_release'])*7)
    scaled=[dict(e,C=None if e['C'] is None else str(Decimal(str(e['C']))*7)) for e in es]
    out.append(('consistent_unit_change',cc,scaled))
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=WORK/'contribution_analysis/anchor_pool_shadow_experiment_v1_20261028');args=ap.parse_args()
    root=args.output.resolve();root.mkdir(exist_ok=False);(root/'source_snapshot').mkdir();(root/'analytic').mkdir()
    started=time.perf_counter();calls=0
    paths=[DESIGN,WORK/'host_server.py',*Path(__file__).parent.glob('*.py'),
           *(WORK/'contribution_analysis/three_pool_t2_anchor_v1').glob('*.py'),
           *(WORK/'contribution_analysis/custom_realized_gain_v1').glob('*.py'),WORK/'contribution_analysis/a_e_b_validation/run.py']
    lock={}
    for p in paths:
        rel=p.relative_to(WORK);dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);lock[str(rel)]=sha(p)
    save(root/'source_lock.json',lock);fixed=cases();save(root/'fixtures.json',fixed)
    save(root/'registration.json',dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),design_sha256=sha(DESIGN),fixture_sha256=sha(root/'fixtures.json'),
        split_counts={s:sum(c['split']==s for c in fixed) for s in ('development','heldout')},max_synthetic_calls=20000,max_seconds=120,max_bytes=100000000,
        new_main_searches=0,new_formal_confirmation=False,actual_payments=0))
    try:
        previous=receipts();save(root/'historical_receipts.json',previous)
        rows=[];invariance=[]
        for split in ('development','heldout'):
            for f in [f for f in fixed if f['split']==split]:
                if f['error']:
                    try:settle(f['contract'],f['events'])
                    except ValueError as exc:
                        if str(exc)!=f['error']:raise
                        rows.append(dict(id=f['id'],split=split,rejected=str(exc)));continue
                    raise ValueError('expected_rejection_missing')
                results={r:settle(f['contract'],f['events'],r) for r in RULES}
                for r,v in results.items():verify(f['contract'],f['events'],r,v)
                expected=f['expected_net']
                if isinstance(expected,dict) and results['net']['payments']!=expected:raise ValueError('hand_golden:'+f['id'])
                if expected is None and results['net']['status']!='held':raise ValueError('held_golden')
                for name,c,es in variants(f['contract'],f['events']):
                    for rule in RULES:
                        v=settle(c,es,rule);verify(c,es,rule,v)
                        if v['payments']!=results[rule]['payments'] or v['status']!=results[rule]['status']:raise ValueError('invariance:'+name)
                        invariance.append(dict(id=f['id'],variant=name,rule=rule,contract=c,events=es,result=v,passed=True))
                rows.append(dict(id=f['id'],split=split,results=results))
            print(split,'PASSED',sum(r['split']==split for r in rows),flush=True)
        save(root/'ledger_results.json',rows);save(root/'invariance.json',invariance)
        analytic=[]
        for p in sorted((T2/'cases').glob('*.json')):
            d=execute(read(p));d=plain(d);checks=independent(d);d['independent']=checks;calls+=d['synthetic_api_calls']
            # One round diagnostic settlement, not an artificial multi-round total across fixtures.
            c=contract(owners=list(d['rows'][0]['owners']),rounds=(0,));scores={}
            for row in d['rows']:
                es=events(c,{o:[a['C']] for o,a in row['owners'].items()});v=settle(c,es);verify(c,es,'net',v);scores[str(row['factor'])]=v
            d['shadow']=scores;save(root/'analytic'/p.name,d)
            analytic.append(dict(id=d['fixture']['id'],calls=d['synthetic_api_calls'],max_error=checks['max_error'],
                cases=[dict(factor=r['factor'],sum_C=r['sum_C'],J=r['J'],interaction=r['interaction'],C={o:a['C'] for o,a in r['owners'].items()},payments=scores[str(r['factor'])]['payments'],status=scores[str(r['factor'])]['status']) for r in d['rows']]))
            if calls>20000 or time.perf_counter()-started>120:raise RuntimeError('budget_exceeded')
        historical=[];old=read(T2/'historical_reanalysis.json')
        for case in sorted({r['case'] for r in old}):
            rs=[r for r in old if r['case']==case];os=sorted({r['owner'] for r in rs});ts=sorted({r['t'] for r in rs});c=contract(os,ts)
            es=[dict(event_id=f"{r['owner']}:{r['t']}",owner=r['owner'],round=r['t'],C=str(r['C']),reference_id=c['reference_id']) for r in rs]
            result={r:settle(c,es,r) for r in RULES}
            for r,v in result.items():verify(c,es,r,v)
            historical.append(dict(case=case,contract=c,events=es,results=result,old_data_exploration=True))
        save(root/'historical_settlement.json',historical)
        if receipts()!=previous:raise ValueError('old_files_changed')
        for rel,h in lock.items():
            if sha(WORK/rel)!=h or sha(root/'source_snapshot'/rel)!=h:raise ValueError('source_drift')
        size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        if size>100000000:raise RuntimeError('size_budget')
        save(root/'summary.json',dict(complete=True,ledger_cases=len(rows),rejected=sum('rejected' in r for r in rows),invariance_checks=len(invariance),analytic=analytic,
            historical_teams=len(historical),historical_owner_rounds=len(old),new_synthetic_api_calls=calls,new_main_searches=0,
            seconds=time.perf_counter()-started,bytes_before_audit=size,trained_model_calls=0,new_training=0,pooled_calls=0,actual_payments=0,
            technical_reference_validated=True,economic_validity_established=False,low_cost_established=False,new_formal_confirmation=False))
        print('COMPLETE',len(rows),'ledger,',len(analytic),'old analytic states,',calls,'new synthetic calls')
    except Exception as exc:
        save(root/'failure.json',dict(complete=False,error=repr(exc),calls=calls));raise


if __name__=='__main__':main()
