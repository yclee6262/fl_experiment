"""Verify explicit policies against task-3 records; no new scientific comparison."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import torch
import numpy as np
from .run import load_base, prepare_precision, digest, dump
from experiments.shared.solver_policy import apply_solver_policy, POLICY_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rows = [json.loads(s) for s in (args.reference/'results.jsonl').read_text().splitlines()]
    index = {(r['scenario'],r['seed'],r['variant']):r for r in rows}
    cases = sorted({(r['scenario'],r['seed']) for r in rows})
    root = Path(__file__).resolve().parents[3]
    paths = [Path(__file__), POLICY_PATH, root/'core/host_server.py', root/'experiments/shared/solver_policy.py']
    manifest = dict(status='running', git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p.resolve()):digest(p) for p in paths},
        reference_sha256=digest(args.reference/'results.jsonl'), inputs={}, cases=len(cases))
    dump(args.output_dir/'manifest.json', manifest)
    results=[]
    try:
        for scenario, seed in cases:
            base,_,artifact=load_base(args.source,scenario,seed)
            base.alphas=[1/len(base.trusted_agents)]*len(base.trusted_agents)
            for p in [artifact/'local_models.pt',args.source/scenario/'shared_costs.jsonl']:
                manifest['inputs'][str(p.resolve())]=digest(p)
            for policy, variant in [('historical','float32_default'),('float32_slsqp_v1','float32_eps_1e-5')]:
                ref=index[scenario,seed,variant]
                for engine in ['bfgs','custom','best_of']:
                    server=copy.deepcopy(base)
                    settings=apply_solver_policy(server,policy)
                    counter=prepare_precision(server,'float32')
                    actual=server.optimize_candidate_subspace(strategy=engine,custom_iterations=30,history_metric=server._true_target_error)
                    expected=ref['best_of'] if engine=='best_of' else ref['engines'][engine]
                    delta=float(np.max(np.abs(np.asarray(expected['X'])-actual['solution'])))
                    loss_delta=float(actual['loss']-expected['loss'])
                    query_delta=counter['requests']-expected['queries']['requests']
                    chosen_match=engine!='best_of' or actual['chosen_engine']==expected['chosen_engine']
                    passed=delta<=1e-12 and abs(loss_delta)<=1e-12 and query_delta==0 and chosen_match and actual['valid_delivery']
                    row=dict(scenario=scenario,seed=seed,policy=settings,engine=engine,passed=bool(passed),
                             max_X_difference=delta,loss_difference=loss_delta,query_difference=query_delta,
                             queries=counter['requests'],success=actual['success'])
                    results.append(row)
                    with (args.output_dir/'checks.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        manifest['status']='passed' if all(r['passed'] for r in results) else 'failed'
    except BaseException as exc:
        manifest.update(status='interrupted',exception=repr(exc))
        raise
    finally:
        manifest.update(checks=len(results), failures=sum(not r['passed'] for r in results),
                        added_model_queries=sum(r['queries'] for r in results))
        dump(args.output_dir/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['status','cases','checks','failures','added_model_queries']}))
    if manifest['status']!='passed': raise SystemExit(1)

if __name__=='__main__': main()
