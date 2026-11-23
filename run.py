"""Run the supported decision workflow and its reproducible comparisons."""
import os
from pathlib import Path
import runpy
import sys
from project_support.legacy import prepare

ROOT = Path(__file__).resolve().parent
ENTRIES = {
    'nonzero-initialization': 'experiments/alpha/nonzero_targets/run.py',
    'nonzero-subspace': 'experiments/baselines/nonzero_subspace/run.py',
    'nonzero-roles': 'experiments/contribution/nonzero_targets/run.py',
    'stages': 'experiments/shared/run_experiments.py',
    'paired-benchmark': 'experiments/baselines/paired_benchmark/run.py',
    'baseline': 'experiments/baselines/compare_fedavg.py',
    'initialization': 'experiments/alpha/compare_initializations.py',
    'alpha-summary': 'experiments/alpha/analysis/summarize_ten_initializers.py',
    'coalition': 'experiments/coalition/p6_coalition_enumeration.py',
    'contribution': 'experiments/contribution/replay_cost_comparison.py',
    'roles': 'experiments/contribution/compare_roles.py',
    'sampling': 'experiments/contribution/sample_rounds.py',
    'calibration': 'experiments/alpha_calibration/run.py',
}


def main():
    if len(sys.argv) == 1 or sys.argv[1] in ('list', '--help', '-h'):
        print('Usage: python run.py <entry> [arguments]\n')
        for name, path in ENTRIES.items():
            print(f'{name:16} {path}')
        return
    entry = sys.argv[1]
    if entry not in ENTRIES:
        raise SystemExit(f'Unknown entry: {entry}. Run python run.py list.')
    prepare()
    os.chdir(ROOT)
    script = ROOT / ENTRIES[entry]
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
