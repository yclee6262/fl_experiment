"""Cross local inversion and central search starts on fixed role model teams."""
import argparse
import gzip
import inspect
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import scipy
import torch
from experiments.contribution.start_stability import engine
from experiments.contribution.nonzero_targets.run import (
    load_server, HostServer, run, endpoint, construct_directions, decide,
    invert_model, FROZEN, dump, plain, sha)

ROOT = Path(__file__).resolve().parents[3]
CA = ROOT/'contribution_analysis'
PLAN = Path(__file__).with_name('protocol.json')
SOURCE = CA/'combined_contribution_layer2_experiment_v1_20261021/cases.json'


def coefficients(case):
    n = len(case['ids'])
    rng = np.random.default_rng(case['seed']+11000000)
    return np.r_[np.full((1, n), 1/n), rng.dirichlet(np.ones(n), size=2)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cases = json.loads(SOURCE.read_text())['development']
    if a.smoke:
        cases = cases[:1]
    functions = [HostServer._solve_subspace_custom, load_server, run, endpoint,
                 construct_directions, decide, invert_model, engine.execute]
    paths = [Path(inspect.getsourcefile(f)).resolve() for f in functions]
    paths += [Path(__file__), PLAN,
        ROOT/'experiments/contribution/nonzero_targets/run.py',
        CA/'model_step_decision_v1/engine.py', CA/'three_pool_t2_anchor_v1/core.py',
        CA/'custom_realized_gain_v1/schema.py']
    paths += list(FROZEN.glob('*.py'))
    source_files = [SOURCE]+[Path(c['models']) for c in cases]
    manifest = dict(status='running', smoke=a.smoke, protocol=json.loads(PLAN.read_text()),
        runtime_sha256={str(f.resolve()):sha(f) for f in paths},
        input_sha256={str(f.resolve()):sha(f) for f in source_files},
        actual_function_sources={f.__name__:str(Path(inspect.getsourcefile(f)).resolve()) for f in functions},
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        versions=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, scipy=scipy.__version__),
        planned_jobs=4 if a.smoke else 600, completed_jobs=0)
    dump(a.output_dir/'manifest.json', manifest)
    index, preparations = [], []
    try:
        for c in cases:
            s = load_server(c)
            coeff = coefficients(c)
            for target in ([.5] if a.smoke else [-.5, .5]):
                parent = a.output_dir/c['key']/f'T{target:g}'
                parent.mkdir(parents=True)
                for r in ([0, 1] if a.smoke else range(5)):
                    inversions = [invert_model(agent.model, target,
                        c['seed']*100+int(agent.agent_id)+1000000*r, starts=1, steps=500)
                        for agent in s.trusted_agents]
                    prep = dict(key=c['key'], scenario=c['scenario'], seed=c['seed'], target=target, r=r,
                        ids=c['ids'], model_path=c['models'], inversions=inversions,
                        warm_coefficients=coeff,
                        inversion_forward_samples=sum(v['forward_samples'] for v in inversions),
                        inversion_backward_steps=sum(v['backward_steps'] for v in inversions))
                    prepfile = parent/f'r{r}_inversions.json'
                    dump(prepfile, prep)
                    preparations.append(dict(file=str(prepfile.relative_to(a.output_dir)), sha256=sha(prepfile)))
                    for w in ([0, 1] if a.smoke else range(3)):
                        if any(v['X'] is None for v in inversions):
                            result = dict(status='inversion_failure', key=c['key'], scenario=c['scenario'], target=target)
                        else:
                            anchors = np.array([v['X'] for v in inversions])
                            initial = anchors.mean(0) if w == 0 else coeff[w]@anchors
                            result = engine.execute(s, c, target, inversions, initial, include_game=r == w == 0)
                        result.update(r=r, w=w, warm_coefficients=coeff[w],
                            preparation_file=str(prepfile.relative_to(a.output_dir)), preparation_sha256=sha(prepfile))
                        filename = parent/f'r{r}_w{w}.json.gz'
                        with gzip.open(filename, 'wt') as f:
                            json.dump(plain(result), f, allow_nan=False)
                        index.append(dict(key=c['key'], scenario=c['scenario'], target=target, r=r, w=w,
                            status=result['status'], file=str(filename.relative_to(a.output_dir)), sha256=sha(filename)))
                        dump(a.output_dir/'index.json', index)
                        manifest['completed_jobs'] += 1
                    print(c['key'], target, 'inversion', r, 'complete', flush=True)
        for name, digest in {**manifest['runtime_sha256'], **manifest['input_sha256']}.items():
            assert sha(Path(name)) == digest, name
        dump(a.output_dir/'preparations.json', preparations)
        manifest.update(status='complete', preparations=len(preparations))
    except BaseException as exc:
        manifest.update(status='interrupted', exception=repr(exc))
        raise
    finally:
        dump(a.output_dir/'manifest.json', manifest)


if __name__ == '__main__':
    main()
