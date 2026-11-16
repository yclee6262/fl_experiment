"""Extend saved pilot ledgers without new model calls or changes to A/E."""
import argparse
import datetime
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tracemalloc
from .schema import read,lines,save,save_lines,plain,sha
from .accounting import compute_accounts
from .ledger import build_balance_ledger
from .verify import verify_run
from .verify_balance import verify_balance

WORK=Path(__file__).resolve().parents[2];MODULE=Path(__file__).resolve().parent
OLD=WORK/'contribution_analysis/custom_realized_gain_pilot_v1_20261027'
DESIGN=WORK/'docs/custom_balance_ledger_design_v1_20261027.md'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=WORK/'contribution_analysis/custom_balance_ledger_analysis_v1_20261027');args=parser.parse_args()
    root=args.output.resolve();root.mkdir(exist_ok=False)
    try:
        prior=read(OLD/'completion_receipt.json')
        for name,h in prior['file_sha256'].items():
            if sha(OLD/name)!=h:raise ValueError('historical_artifact_changed:'+name)
        for name in ['accounting.py','schema.py','verify.py']:
            if sha(MODULE/name)!=sha(OLD/'delivery_source_snapshot'/name):raise ValueError('base_method_changed:'+name)
        snap=root/'source_snapshot';snap.mkdir()
        for p in MODULE.rglob('*'):
            if p.is_file() and p.suffix in ['.py','.md']:
                out=snap/p.relative_to(MODULE);out.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,out)
        shutil.copy2(DESIGN,root/DESIGN.name)
        inputs={str(OLD/'completion_receipt.json'):sha(OLD/'completion_receipt.json'),str(DESIGN):sha(DESIGN)}
        for name,h in prior['file_sha256'].items():inputs[str(OLD/name)]=h
        save(root/'inputs.json',inputs)
        source_lock={str(p.relative_to(root)):sha(p) for p in snap.rglob('*') if p.is_file()};save(root/'source_lock.json',source_lock)
        save(root/'registration.json',dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            complete_method_version='custom_realized_gain_with_balance_v1',component_version='custom_balance_ledger_v1',
            base_method_version='custom_realized_gain_v1',cases=[c['key'] for c in read(OLD/'cases.json')],
            source_root=str(OLD),source_receipt_sha256=sha(OLD/'completion_receipt.json'),source_receipt_files=len(prior['file_sha256']),
            old_data_exploration=True,new_confirmation=False,new_solver_runs=0,new_model_predictions=0,pooled_loaded=False,truth_loaded=False,
            sign_rule='strict p<0',display_tiny_projection=1e-10,valid_direction='original g_norm>1e-8',
            frequency_denominator='all eligible direction rounds; context-specific eligible rounds for each context',
            source_A_E_must_match_exactly=True,protection_effect_status='not_established'))
        test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(MODULE/'tests_balance'),'-v'],cwd=WORK,capture_output=True,text=True)
        save(root/'technical_tests.json',dict(passed=test.returncode==0,returncode=test.returncode,stdout=test.stdout,stderr=test.stderr))
        if test.returncode:raise RuntimeError('balance_tests_failed')
        sys.path.insert(0,str(OLD/'source_snapshot'))
        summary=[]
        for case in read(OLD/'cases.json'):
            source=OLD/case['key'];out=root/case['key'];out.mkdir();manifest=read(source/'manifest.json');run=read(source/'run_metadata.json')
            for field in ['evaluations','queries','iterations']:run[field]=lines(source/(field+'.jsonl'))
            previous=read(source/'accounts.json');previous['steps']=lines(source/'diagnostics.jsonl')
            base=compute_accounts(manifest,run['evaluations'],run['iterations'],run['initial_eval_id'],run['final_eval_id'])
            for field in base:
                if field=='verification_status':continue
                if plain(base[field])!=previous[field]:raise ValueError('base_scores_changed:'+field)
            baseverify=verify_run(manifest,run,previous)
            start=time.perf_counter();ledger=build_balance_ledger(base['anchor_owners'],run['iterations'],base['steps']);elapsed=time.perf_counter()-start
            tracemalloc.start();repeated=build_balance_ledger(base['anchor_owners'],run['iterations'],base['steps']);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            if plain(ledger)!=plain(repeated):raise ValueError('non_reproducible_ledger')
            check=verify_balance(manifest,run,plain(ledger))
            save(out/'source_reference.json',dict(source=str(source),source_manifest_sha256=sha(source/'manifest.json'),source_accounts_sha256=sha(source/'accounts.json')))
            save_lines(out/'balance_rounds.jsonl',ledger['rounds'])
            save(out/'balance_summary.json',{k:v for k,v in ledger.items() if k!='rounds'})
            save(out/'combined_accounts.json',dict(complete_method_version='custom_realized_gain_with_balance_v1',
                realized_gain={k:v for k,v in previous.items() if k!='steps'},balance=ledger['owner_summary'],
                P_source=str(source/'diagnostics.jsonl'),service_cost_source=str(source/'costs.json'),
                protection_effect_status='not_established',total_contribution_score=None,
                total_score_status='not_defined_do_not_add_A_E_b',payment_allocation=None))
            save(out/'verification.json',dict(base_A_E_exactly_unchanged=True,all_base_fields_exactly_unchanged=True,
                base=baseverify,balance=check,source_receipt_all_files_passed=True))
            save(out/'costs.json',dict(offline_ledger_seconds=elapsed,python_peak_bytes=peak,
                repeated_for_memory_measurement=True,new_model_predictions=0,new_searches=0,
                source_cost_measurement_reused_not_reestimated=True,native_memory_measured=False))
            summary.append(dict(case=case['key'],scenario=case['scenario'],G=base['G_total'],owners=base['anchor_owners'],
                A=base['A_support_owner'],A_share=base['owner_share'],E=base['E_model'],
                balance=ledger['owner_summary'],rounds=len(run['iterations']),
                base_A_E_exactly_unchanged=True,balance_verification=check,offline_seconds=elapsed,python_peak_bytes=peak))
        for name,h in inputs.items():
            if sha(name)!=h:raise ValueError('input_drift')
        for name,h in source_lock.items():
            if sha(root/name)!=h:raise ValueError('source_drift')
        save(root/'summary.json',dict(complete=True,cases=summary,tests_passed=True,new_model_predictions=0,new_searches=0,
            uninstrumented_saved_objective_replays=2,old_data_exploration=True,new_confirmation=False,protection_effect_established=False,
            economic_validity_established=False,pooled_loaded=False,truth_loaded=False))
        print('COMPLETE',root,flush=True)
    except Exception as error:
        save(root/'failure.json',dict(complete=False,error=repr(error)));raise

if __name__=='__main__':main()
