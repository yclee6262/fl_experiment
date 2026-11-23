"""Historical 600-game Shapley/LOO reference with five-seed inference."""
import argparse
import ast
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[3]
TOL = 1e-10
REPLICATES = 200000
RNG = 20261124
METRICS = ['spearman', 'sign_agreement', 'allocation_TV', 'mean_absolute_score_difference',
           'shapley_positive_fraction', 'loo_positive_fraction', 'shapley_reserved', 'loo_reserved']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csvrows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [plain(v) for v in value]
    return value


def dump(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')


def csvout(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(plain(value), separators=(',', ':'))
                             if isinstance(value, (list, tuple, dict, np.ndarray)) else plain(value)
                             for key, value in row.items()})


def subsets(players):
    for size in range(len(players) + 1):
        yield from (frozenset(s) for s in itertools.combinations(players, size))


def exact_shapley(players, value):
    """Factorial subset formula for a fully specified finite game."""
    n = len(players)
    result = []
    for player in players:
        others = [i for i in players if i != player]
        score = 0.
        for coalition in subsets(others):
            weight = math.factorial(len(coalition))*math.factorial(n-len(coalition)-1)/math.factorial(n)
            score += weight*(value[coalition | {player}]-value[coalition])
        result.append(score)
    return np.array(result)


def permutation_shapley(players, value):
    """Independent enumeration of marginal insertion along every ordering."""
    totals = {i: 0. for i in players}
    orderings = list(itertools.permutations(players))
    for ordering in orderings:
        before = frozenset()
        for player in ordering:
            after = before | {player}
            totals[player] += value[after]-value[before]
            before = after
    return np.array([totals[i]/len(orderings) for i in players])


def sign(values):
    values = np.asarray(values)
    return np.where(abs(values) > TOL, np.sign(values), 0).astype(int)


def allocation(values):
    positive = np.where(np.asarray(values) > TOL, values, 0.)
    if positive.sum() <= 0:
        return np.r_[np.zeros(len(positive)), 1.]
    return np.r_[positive/positive.sum(), 0.]


def compare(shapley, loo):
    identifiable = np.ptp(shapley) > TOL and np.ptp(loo) > TOL
    qphi, qloo = allocation(shapley), allocation(loo)
    return dict(spearman=float(spearmanr(shapley, loo).statistic) if identifiable else None,
        sign_agreement=float(np.mean(sign(shapley) == sign(loo))),
        allocation_TV=float(abs(qphi-qloo).sum()/2),
        mean_absolute_score_difference=float(np.mean(abs(shapley-loo))),
        shapley_positive_fraction=float(np.mean(sign(shapley) > 0)),
        loo_positive_fraction=float(np.mean(sign(loo) > 0)),
        shapley_reserved=int(qphi[-1] == 1), loo_reserved=int(qloo[-1] == 1))


def interval(values):
    x = np.array([v for v in values if v is not None], float)
    if not len(x):
        return dict(n=0, mean=None, sd=None, ci95_low=None, ci95_high=None)
    generator = np.random.default_rng(RNG)
    draws = np.empty(REPLICATES)
    for start in range(0, REPLICATES, 2000):
        weights = generator.multinomial(len(x), np.full(len(x), 1/len(x)), size=2000)
        draws[start:start+2000] = weights@x/len(x)
    return dict(n=len(x), mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else None,
                ci95_low=float(np.quantile(draws, .025)), ci95_high=float(np.quantile(draws, .975)))


def source_rules(source):
    enum = source/'source_snapshot/p6_coalition_enumeration.py'
    alpha = source/'source_snapshot/alpha_weight_experiments.py'
    host = source/'source_snapshot/host_server.py'
    tree = ast.parse(enum.read_text())
    uniform = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                       and t.value.id == 'optimizer' and t.attr == 'alphas' for t in node.targets)]
    assert len(uniform) == 1 and ast.unparse(uniform[0].value) == '(np.ones(size) / size).tolist()'
    parser = ast.parse(alpha.read_text())
    defaults = []
    for node in ast.walk(parser):
        if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == '--optimizer':
            defaults.extend(kw.value.value for kw in node.keywords if kw.arg == 'default')
    assert defaults == ['best_of']
    text = host.read_text()
    assert 'initial_betas = np.ones(len(candidates), dtype=float) / len(candidates)' in text
    assert 'method="SLSQP"' in text and 'options={"ftol": 1e-9, "maxiter": 200}' in text
    return dict(alpha_rule='uniform within each coalition',
        search_rule='historical best_of, own-coalition candidate-mean start',
        coefficient_rule='historical SLSQP: no explicit finite-difference eps',
        source_snapshot_sha256={p.name: sha(p) for p in [enum, alpha, host]})


def load_source(source):
    original = read(source/'run_manifest.json')
    assert original['status'] == 'complete' and original['seeds'] == [0, 1, 2, 3, 4]
    for name, expected in original['artifact_sha256'].items():
        path = (source/name).resolve()
        assert path.is_relative_to(source.resolve()) and sha(path) == expected, name
    raw = csvrows(source/'coalitions.csv')
    pool = csvrows(source/'candidate_pool.csv')
    assert len(raw) == 875 and len(pool) == 50
    lookup, compatibility = {}, []
    rules = source_rules(source)
    for seed in range(5):
        candidates = [r for r in pool if int(r['seed']) == seed]
        ids = [int(r['agent_id']) for r in candidates]
        assert len(ids) == len(set(ids)) == 10
        assert all(float(r['formal_target']) == .5 and float(r['qualification_target']) == 0 for r in candidates)
        rows = [r for r in raw if int(r['seed']) == seed]
        artifacts = {r['artifact_dir'] for r in rows}
        assert len(rows) == 175 and len(artifacts) == 1
        artifact = source/next(iter(artifacts))
        metadata = read(artifact/'metadata.json')
        assert metadata['scenario'] == dict(seed=seed, dimension=2, poison_ratio=0., target=.5)
        assert {p['agent_id'] for p in metadata['agent_data_profile']} == set(ids)
        statuses, solvers = {}, {}
        for row in rows:
            members = tuple(int(i) for i in json.loads(row['agent_ids']))
            indices = json.loads(row['coalition_indices'])
            assert tuple(ids[j] for j in indices) == members
            assert len(members) == int(row['coalition_size'])
            key = seed, frozenset(members)
            assert key not in lookup
            state = row['delivery_status']
            statuses[state] = statuses.get(state, 0)+1
            if state == 'valid':
                x = np.array(json.loads(row['final_solution']), float)
                assert x.shape == (2,) and np.isfinite(x).all() and max(abs(x)) <= 1+1e-8
                error = abs(float(x.sum()+x[0]*x[1])-.5)
                assert abs(error-float(row['truth_error'])) < 1e-12
                assert abs(abs(float(row['pooled_prediction'])-.5)-float(row['pooled_error'])) < 1e-12
                assert row['solver'] in ['custom', 'SLSQP']
                solvers[row['solver']] = solvers.get(row['solver'], 0)+1
                lookup[key] = dict(error=error, status=state)
            else:
                lookup[key] = dict(error=None, status=state)
        expected = {(seed, frozenset(s)) for size in [1, 2, 3] for s in itertools.combinations(ids, size)}
        assert {k for k in lookup if k[0] == seed} == expected
        compatibility.append(dict(seed=seed, candidate_ids=ids, coalitions=175, target=.5, dimension=2,
            local_model_sha256=sha(artifact/metadata['local_models_file']),
            evaluator_sha256=sha(artifact/metadata['global_evaluator_file']),
            artifact_dir=str(artifact.relative_to(source)), valid=statuses.get('valid', 0),
            statuses=statuses, chosen_solvers=solvers, alpha_rule=rules['alpha_rule'], search_rule=rules['search_rule']))
    return lookup, compatibility, rules


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    original = read(source/'run_manifest.json')
    files = [source/'run_manifest.json']+[source/p for p in original['artifact_sha256']]
    sources = {str(f.resolve()): sha(f) for f in files}
    runtime = {str(p.resolve()): sha(p) for p in [Path(__file__), Path(__file__).with_name('SHAPLEY.md')]}
    manifest = dict(status='running', source=str(source.resolve()), source_sha256=sources, runtime_sha256=runtime,
        created='2026-11-24', python=platform.python_version(), numpy=np.__version__,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        utility='v(empty)=0; v(S)=0.5-abs(h(saved X_S)-0.5)', baseline_point=[0., 0.],
        statistical_unit='five source seeds; first average 120 games within seed', replicates=REPLICATES, rng=RNG,
        new_model_queries=0, new_solver_calls=0)
    dump(output/'manifest.json', manifest)
    try:
        lookup, compatibility, rules = load_source(source)
        games, owners = [], []
        formula_gap = efficiency_gap = baseline_gap = 0.
        for context in compatibility:
            seed, ids = context['seed'], context['candidate_ids']
            for members in itertools.combinations(ids, 3):
                key = f"seed{seed}_ids"+'-'.join(map(str, members))
                coalitions = list(subsets(members))
                observed = [lookup[seed, s] for s in coalitions if s]
                valid = all(r['status'] == 'valid' for r in observed)
                row = dict(game=key, seed=seed, ids=list(members), status='valid' if valid else 'invalid_subcoalition',
                    nonempty_subsets=7, grand_utility=None, shapley=None, loo=None,
                    shapley_zero=0, loo_zero=0, **{metric: None for metric in METRICS})
                if valid:
                    utility = {s: .5-lookup[seed, s]['error'] if s else 0. for s in coalitions}
                    phi = exact_shapley(members, utility)
                    perm = permutation_shapley(members, utility)
                    full = frozenset(members)
                    loo = np.array([utility[full]-utility[full-{i}] for i in members])
                    shifted = {s: v+.3 if s else 0. for s, v in utility.items()}
                    shifted_phi = exact_shapley(members, shifted)
                    shifted_loo = np.array([shifted[full]-shifted[full-{i}] for i in members])
                    formula_gap = max(formula_gap, float(max(abs(phi-perm))))
                    efficiency_gap = max(efficiency_gap, abs(float(phi.sum())-utility[full]))
                    baseline_gap = max(baseline_gap, float(max(abs(shifted_phi-phi-.1))), float(max(abs(shifted_loo-loo))))
                    assert formula_gap < 1e-12 and efficiency_gap < 1e-12 and baseline_gap < 1e-12
                    row.update(grand_utility=utility[full], shapley=phi, loo=loo,
                        shapley_zero=int(sum(sign(phi) == 0)), loo_zero=int(sum(sign(loo) == 0)), **compare(phi, loo))
                    for i, ident in enumerate(members):
                        owners.append(dict(game=key, seed=seed, id=ident, status='valid', shapley=float(phi[i]),
                            loo=float(loo[i]), shapley_sign=int(sign(phi)[i]), loo_sign=int(sign(loo)[i]),
                            shapley_share=float(allocation(phi)[i]), loo_share=float(allocation(loo)[i])))
                else:
                    for ident in members:
                        owners.append(dict(game=key, seed=seed, id=ident, status='invalid_subcoalition', shapley=None,
                            loo=None, shapley_sign=None, loo_sign=None, shapley_share=None, loo_share=None))
                games.append(row)
        assert len(games) == 600 and len(owners) == 1800
        seed_rows = []
        for seed in range(5):
            rows = [r for r in games if r['seed'] == seed]
            record = dict(seed=seed, planned_games=120, valid_games=sum(r['status'] == 'valid' for r in rows),
                          rank_valid_games=sum(r['spearman'] is not None for r in rows),
                          shapley_zero_owners=sum(r['shapley_zero'] for r in rows), loo_zero_owners=sum(r['loo_zero'] for r in rows))
            for metric in METRICS:
                values = [r[metric] for r in rows if r[metric] is not None]
                record[metric] = float(np.mean(values)) if values else None
                record[metric+'_valid_games'] = len(values)
            seed_rows.append(record)
        summary = [dict(metric=metric, planned_seeds=5, scope='descriptive five-seed bootstrap',
                        **interval([r[metric] for r in seed_rows])) for metric in METRICS]
        for name, rows in [('games', games), ('owners', owners), ('seed_summary', seed_rows),
                           ('summary', summary), ('compatibility', compatibility)]:
            csvout(output/(name+'.csv'), rows)
        audit = dict(passed=True, original_rows=875, games=600, nonempty_subset_lookups=4200, owners=1800,
            valid_games=sum(r['status'] == 'valid' for r in games), invalid_games=sum(r['status'] != 'valid' for r in games),
            defined_ranks=sum(r['spearman'] is not None for r in games), max_formula_permutation_gap=formula_gap,
            max_efficiency_gap=efficiency_gap, max_baseline_shift_gap=baseline_gap,
            original_artifact_hashes_verified=len(original['artifact_sha256']), source_rules=rules,
            new_model_queries=0, new_solver_calls=0, files_sha256={p.name: sha(p) for p in output.glob('*.csv')},
            scope='Historical finite-procedure games, not direct comparison to 20 role teams or fairness ground truth')
        dump(output/'audit.json', audit)
        independent_acceptance(source, output)
        for path, expected in {**sources, **runtime}.items():
            assert sha(Path(path)) == expected, path
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='interrupted', exception=repr(exc))
        raise
    finally:
        dump(output/'manifest.json', manifest)
    print(json.dumps({k: v for k, v in audit.items() if k not in ['files_sha256', 'source_rules']}))


def independent_acceptance(source, output):
    """Raw CSV/permutation path; does not call the factorial or summary helpers."""
    errors = {}
    for row in csvrows(source/'coalitions.csv'):
        if row['delivery_status'] == 'valid':
            x, y = json.loads(row['final_solution'])
            errors[int(row['seed']), frozenset(json.loads(row['agent_ids']))] = abs(x+y+x*y-.5)
    checked = 0
    for row in csvrows(output/'games.csv'):
        if row['status'] != 'valid':
            continue
        seed = int(row['seed'])
        members = json.loads(row['ids'])
        marginal = {p: [] for p in members}
        for ordering in itertools.permutations(members):
            coalition = frozenset()
            before_value = 0.
            for person in ordering:
                coalition |= {person}
                after_value = .5-errors[seed, coalition]
                marginal[person].append(after_value-before_value)
                before_value = after_value
        phi = [sum(marginal[p])/6 for p in members]
        loo = [errors[seed, frozenset(members)-{p}]-errors[seed, frozenset(members)] for p in members]
        np.testing.assert_allclose(phi, json.loads(row['shapley']), atol=1e-12, rtol=0)
        np.testing.assert_allclose(loo, json.loads(row['loo']), atol=1e-12, rtol=0)
        x, y = np.array(phi), np.array(loo)
        sx = np.where(abs(x) <= TOL, 0, np.sign(x))
        sy = np.where(abs(y) <= TOL, 0, np.sign(y))
        assert abs(float(np.mean(sx == sy))-float(row['sign_agreement'])) < 1e-12
        normalized = []
        for values in [x, y]:
            positive = np.where(values > TOL, values, 0.)
            normalized.append(np.r_[positive/positive.sum(), 0.] if positive.sum() else np.r_[positive, 1.])
        assert abs(float(abs(normalized[0]-normalized[1]).sum()/2)-float(row['allocation_TV'])) < 1e-12
        if np.ptp(x) > TOL and np.ptp(y) > TOL:
            rank = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
            assert abs(rank-float(row['spearman'])) < 1e-12
        else:
            assert row['spearman'] == ''
        checked += 1
    seeds = csvrows(output/'seed_summary.csv')
    games = csvrows(output/'games.csv')
    for seed in seeds:
        selected = [r for r in games if r['seed'] == seed['seed']]
        assert len(selected) == 120
        for metric in METRICS:
            values = [float(r[metric]) for r in selected if r[metric] != '']
            assert len(values) == int(seed[metric+'_valid_games'])
            if values:
                assert abs(np.mean(values)-float(seed[metric])) < 1e-12
    for row in csvrows(output/'summary.csv'):
        values = [float(s[row['metric']]) for s in seeds if s[row['metric']] != '']
        assert len(values) == int(row['n'])
        if not values:
            continue
        assert abs(np.mean(values)-float(row['mean'])) < 1e-12
        rng = np.random.default_rng(RNG)
        weights = np.concatenate([rng.multinomial(len(values), [1/len(values)]*len(values), size=1000) for _ in range(200)])
        distribution = np.einsum('ij,j->i', weights, values)/len(values)
        np.testing.assert_allclose(np.quantile(distribution, [.025, .975]),
                                   [float(row['ci95_low']), float(row['ci95_high'])], atol=1e-12, rtol=0)
    dump(output/'independent_acceptance.json', dict(passed=True, raw_permutation_games=checked,
        seed_group_means=5*len(METRICS), summary_intervals=len(METRICS), bootstrap_replicates=REPLICATES,
        new_model_queries=0, new_solver_calls=0, validator_sha256=sha(Path(__file__))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output_dir)
