"""Replay peer-scored anchor allocation using sampled recorded rounds."""
import argparse
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def allocation(scores, threshold):
    positive = np.where(scores > threshold, scores, 0.)
    total = positive.sum(axis=-1, keepdims=True)
    shares = np.divide(positive, total, out=np.zeros_like(positive), where=total != 0)
    reserve = (total[..., 0] == 0).astype(float)
    return shares, reserve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'contribution_analysis/method5_cost_experiment_20261030_attempt_04')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.draws < 1 or (args.limit is not None and args.limit < 1):
        parser.error('draws and limit must be positive')
    if args.output_dir.exists():
        parser.error('Output directory must not exist')
    registration = json.loads((args.source/'registration.json').read_text())
    cases = registration['cases'][:args.limit]
    rates = [.1, .25, .5, .75, 1.]
    records = []
    for case in cases:
        team = json.loads((args.source/'records'/f"{case['key']}_New.json").read_text())
        count, rounds = len(team['owners']), len(team['full'])
        effects = np.zeros((rounds, count))
        tolerances = np.zeros_like(effects)
        weights = np.asarray(team['weights'])
        for row in team['rows']:
            owner = row['i']
            delta = np.asarray(row['minusA_losses']) - row['full_losses']
            if abs(weights @ delta - row['C_A']) > 1e-10:
                raise ValueError('Stored full-objective anchor effect is inconsistent')
            peers = np.arange(count) != owner
            effects[row['t'], owner] = weights[peers] @ delta[peers] / weights[peers].sum()
            tolerances[row['t'], owner] = row['rho']
        threshold = np.maximum(1e-8, tolerances.sum(0))
        baseline, base_reserve = allocation(effects.sum(0), threshold)
        permutations = [np.random.default_rng(seed).permutation(rounds) for seed in range(args.draws)]
        for rate in rates:
            size = max(1, math.ceil(rate * rounds))
            estimates = np.array([effects[p[:size]].sum(0)*rounds/size for p in permutations])
            shares, reserve = allocation(estimates, threshold)
            tv = (np.abs(shares-baseline).sum(1) + np.abs(reserve-base_reserve))/2
            records.append(dict(team=case['key'], rate=rate, actual_rate=size/rounds,
                                mean_allocation_tv=float(tv.mean())))
    summary = []
    for rate in rates:
        selected = [r for r in records if r['rate'] == rate]
        summary.append(dict(rate=rate, teams=len(selected),
            mean_actual_rate=float(np.mean([r['actual_rate'] for r in selected])),
            mean_allocation_tv=float(np.mean([r['mean_allocation_tv'] for r in selected]))))
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'team_results.json').write_text(json.dumps(records, indent=2)+'\n')
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (args.output_dir/'manifest.json').write_text(json.dumps(dict(source=str(args.source),draws=args.draws,
        teams=len(cases), model_calls=0, scope='Frozen-path allocation replay, not measured online query savings'),indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
