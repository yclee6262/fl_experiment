"""Reproducible paired precision diagnostics; never changes core defaults."""
import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy
import torch
from project_support.legacy import prepare
prepare()
import host_server as host_module
from experiments.alpha.replay import load_base, SCENARIOS

VARIANTS = [('float32_default', 'float32', None),
            ('float32_eps_1e-6', 'float32', 1e-6),
            ('float32_eps_1e-5', 'float32', 1e-5),
            ('float32_eps_1e-4', 'float32', 1e-4),
            ('float64_default', 'float64', None)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple,np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


@contextmanager
def difference_step(eps):
    original = host_module.minimize
    def wrapped(*args, **kwargs):
        if eps is not None:
            kwargs['options'] = {**kwargs.get('options', {}), 'eps': eps}
        return original(*args, **kwargs)
    host_module.minimize = wrapped
    try:
        yield
    finally:
        host_module.minimize = original


def prepare_precision(server, dtype):
    counter = Counter()
    for agent in server.trusted_agents:
        agent.device = torch.device('cpu')
        agent.model = agent.model.cpu()
        if dtype == 'float64':
            agent.model.double()
        tdtype = torch.float64 if dtype == 'float64' else torch.float32
        def predict(x, model=agent.model, tdtype=tdtype):
            a = np.asarray(x)
            counter['requests'] += 1
            counter['input_samples'] += 1 if a.ndim == 1 else len(a)
            with torch.no_grad():
                t = torch.tensor(a, dtype=tdtype)
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                return model(t).cpu().numpy().flatten()
        agent.api_predict = predict
    return counter


def perturbation_check(server, eps):
    anchors = np.asarray(server.I_list)
    beta = np.ones(len(anchors))/len(anchors)
    x = beta @ anchors
    initial = server._consensus_loss(x)
    same_x, same_f = 0, 0
    for k in range(len(beta)):
        perturbed = beta.copy()
        perturbed[k] += eps
        xp = perturbed @ anchors
        same_x += int(np.array_equal(x.astype(np.float32), xp.astype(np.float32)))
        same_f += int(server._consensus_loss(xp) == initial)
    return {'dimensions':len(beta), 'same_float32_input':same_x,
            'same_objective':same_f, 'step':eps}


def run_variant(base, dtype, eps):
    server = copy.deepcopy(base)
    counter = prepare_precision(server, dtype)
    x0 = np.mean(server.I_list, axis=0)
    coefficient_x0 = np.ones(len(server.I_list)) / len(server.I_list) @ np.asarray(server.I_list)
    initial_losses = {'custom': server._consensus_loss(x0), 'bfgs': server._consensus_loss(coefficient_x0)}
    perturbation = perturbation_check(server, np.sqrt(np.finfo(float).eps) if eps is None else eps)
    diagnostic_cost = dict(counter)
    results = {}
    for engine in ('bfgs','custom'):
        counter.clear()
        start = time.perf_counter()
        initial_loss = initial_losses[engine]
        engine_x0 = coefficient_x0 if engine == 'bfgs' else x0
        try:
            with difference_step(eps):
                result = server.optimize_candidate_subspace(strategy=engine,
                    custom_iterations=30, history_metric=server._true_target_error)
            elapsed = time.perf_counter()-start
            x = result['solution']
            results[engine] = {
                'status':'valid', 'success':result['success'], 'message':result['message'],
                'X':x, 'loss':result['loss'], 'initial_loss':initial_loss,
                'improvement':initial_loss-result['loss'],
                'truth_error':server._true_target_error(x),
                'displacement_linf':float(np.max(np.abs(x-engine_x0))),
                'stagnant':bool(np.max(np.abs(x-engine_x0))<=1e-12),
                'no_improvement':bool(initial_loss-result['loss']<=1e-10),
                'box_violation':result['max_box_violation'],
                'iterations_recorded':len(result['history']),
                'seconds':elapsed, 'queries':dict(counter)}
        except Exception as exc:
            results[engine]={'status':'failure', 'exception':repr(exc),
                'seconds':time.perf_counter()-start, 'queries':dict(counter)}
    valid = [(key,value) for key,value in results.items() if value['status']=='valid']
    chosen = min(valid,key=lambda pair:pair[1]['loss'])[0] if valid else None
    best = None if chosen is None else {
        'chosen_engine':chosen, 'loss':results[chosen]['loss'],
        'truth_error':results[chosen]['truth_error'], 'X':results[chosen]['X'],
        'queries':{key:sum(r['queries'].get(key,0) for r in results.values())
                   for key in ('requests','input_samples')}}
    verification = None
    if dtype=='float32' and eps is None:
        counter.clear()
        result = server.optimize_candidate_subspace(strategy='best_of', custom_iterations=30,
                                                      history_metric=server._true_target_error)
        verification={'chosen_matches':chosen==result['chosen_engine'],
                      'max_X_difference':float(np.max(np.abs(best['X']-result['solution']))),
                      'loss_difference':float(best['loss']-result['loss']),
                      'queries':dict(counter)}
        if not verification['chosen_matches'] or verification['max_X_difference']>1e-12 or abs(verification['loss_difference'])>1e-12:
            raise AssertionError('separate-engine replay differs from best-of')
    return {'initial_X':{'custom':x0,'bfgs':coefficient_x0},'initial_loss':initial_losses,'perturbation':perturbation,
            'diagnostic_queries':diagnostic_cost,'engines':results,'best_of':best,
            'baseline_replay_verification':verification}


def summarize(rows):
    output={}
    baseline={(r['scenario'],r['seed']):r for r in rows if r['variant']=='float32_default'}
    for name,_,_ in VARIANTS:
        selected=[r for r in rows if r['variant']==name]
        engines={}
        for e in ('bfgs','custom'):
            values=[r['engines'][e] for r in selected if r.get('engines',{}).get(e,{}).get('status')=='valid']
            engines[e]={'valid':len(values),'failures':len(selected)-len(values)}
            for field in ('loss','truth_error','improvement','displacement_linf','seconds'):
                engines[e]['mean_'+field]=float(np.mean([v[field] for v in values])) if values else None
            for field in ('success','stagnant','no_improvement'):
                engines[e][field+'_count']=sum(v[field] for v in values)
            engines[e]['mean_queries']=float(np.mean([v['queries']['requests'] for v in values])) if values else None
        valid=[r for r in selected if r.get('best_of')]
        pair=[(r,baseline[(r['scenario'],r['seed'])]) for r in valid if baseline.get((r['scenario'],r['seed']),{}).get('best_of')]
        output[name]={'cases':len(selected),'engines':engines,
            'chosen_engines':dict(Counter(r['best_of']['chosen_engine'] for r in valid)),
            'mean_best_loss':float(np.mean([r['best_of']['loss'] for r in valid])) if valid else None,
            'mean_best_truth_error':float(np.mean([r['best_of']['truth_error'] for r in valid])) if valid else None,
            'mean_best_queries':float(np.mean([r['best_of']['queries']['requests'] for r in valid])) if valid else None,
            'paired_best_loss_difference':float(np.mean([r['best_of']['loss']-b['best_of']['loss'] for r,b in pair])) if pair else None,
            'paired_best_truth_difference':float(np.mean([r['best_of']['truth_error']-b['best_of']['truth_error'] for r,b in pair])) if pair else None,
            'same_float32_input_fraction':sum(r.get('perturbation',{}).get('same_float32_input',0) for r in selected)/max(1,sum(r.get('perturbation',{}).get('dimensions',0) for r in selected)),
            'same_objective_fraction':sum(r.get('perturbation',{}).get('same_objective',0) for r in selected)/max(1,sum(r.get('perturbation',{}).get('dimensions',0) for r in selected))}
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=30)
    args=parser.parse_args()
    if not 1 <= args.limit <= 30:
        parser.error('--limit must be 1..30')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[3]
    manifest={'status':'running','source':str(args.source.resolve()),'limit':args.limit,
        'design':'three historical scenarios, first sorted seeds, uniform alpha, T=0, paired numeric diagnostics',
        'python':sys.version,'platform':platform.platform(),
        'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__},
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'git_status':subprocess.check_output(['git','status','--short'],text=True),
        'source_hashes':{str(p.relative_to(root)):digest(p) for p in [root/'core/host_server.py',root/'core/agent_client.py',root/'core/models.py',root/'experiments/alpha/replay.py',Path(__file__).resolve(),Path(__file__).with_name('README.md')]},
        'input_hashes':{},'cases':[]}
    historical_path = args.source.parent / 'results/results.jsonl'
    historical = {(r['scenario'],r['seed']):r for r in (json.loads(line) for line in historical_path.read_text().splitlines()) if r['method']=='A_uniform'}
    manifest['input_hashes'][str(historical_path.resolve())]=digest(historical_path)
    manifest['snapshot_compatibility'] = {}
    for scenario in SCENARIOS:
        snapshot=args.source/scenario/'source_snapshot/host_server.py'
        match=digest(snapshot)==digest(root/'core/host_server.py')
        manifest['snapshot_compatibility'][scenario]={'path':str(snapshot.resolve()),'sha256':digest(snapshot),'matches_current':match}
        if not match:
            raise ValueError('historical host source differs')
    rows=[]
    dump(args.output_dir/'manifest.json',manifest)
    try:
        for scenario in SCENARIOS:
            shared=args.source/scenario/'shared_costs.jsonl'
            manifest['input_hashes'][str(shared.resolve())]=digest(shared)
            seeds=sorted(json.loads(line)['seed'] for line in shared.read_text().splitlines())[:args.limit]
            for seed in seeds:
                base,item,artifact=load_base(args.source,scenario,seed)
                base.alphas=[1/len(base.trusted_agents)]*len(base.trusted_agents)
                assert base.target_T==0 and len(base.I_list)==3
                model=artifact/'local_models.pt'
                manifest['input_hashes'][str(model.resolve())]=digest(model)
                manifest['cases'].append({'scenario':scenario,'seed':seed,'coalition_ids':item['coalition_ids']})
                for name,dtype,eps in VARIANTS:
                    row={'scenario':scenario,'seed':seed,'variant':name,'dtype':dtype,'eps':eps}
                    try:
                        row.update(run_variant(base,dtype,eps))
                        if name=='float32_default':
                            old=historical[(scenario,seed)]
                            b=row['best_of']
                            check={'max_X_difference':float(np.max(np.abs(np.asarray(b['X'])-old['X']))), 'loss_difference':b['loss']-old['solver']['loss'], 'chosen_matches':b['chosen_engine']==old['solver']['chosen_engine'], 'query_difference':b['queries']['requests']-old['stage3_requests']}
                            check['passed']=check['max_X_difference']<=1e-12 and abs(check['loss_difference'])<=1e-12 and check['chosen_matches'] and check['query_difference']==0
                            row['historical_U_replay']=check
                            if not check['passed']:
                                raise AssertionError('Historical U replay mismatch')
                    except Exception as exc:
                        row.update(status='failure',exception=repr(exc))
                    rows.append(row)
                    with (args.output_dir/'results.jsonl').open('a') as handle:
                        handle.write(json.dumps(clean(row),allow_nan=False)+'\n')
                print(scenario,seed,'completed',flush=True)
        manifest['status']='complete' if all(r.get('best_of') and r.get('status')!='failure' for r in rows) else 'complete_with_failures'
        dump(args.output_dir/'summary.json',{'all':summarize(rows),
            'by_scenario':{s:summarize([r for r in rows if r['scenario']==s]) for s in SCENARIOS}})
    except BaseException as exc:
        manifest.update(status='interrupted',exception=repr(exc))
        raise
    finally:
        manifest['rows']=len(rows)
        manifest['output_hashes']={p.name:digest(p) for p in args.output_dir.glob('*') if p.is_file() and p.name!='manifest.json'}
        dump(args.output_dir/'manifest.json',manifest)

if __name__=='__main__':
    main()
