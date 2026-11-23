"""Render all factorial cells with descriptive intervals from the saved table."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(analysis):
    outputs = [analysis/'factorial_cells.pdf', analysis/'factorial_cells.png', analysis/'plot_provenance.json']
    if any(p.exists() for p in outputs):
        raise FileExistsError('Plot output already exists')
    source = analysis/'factorial_groups.csv'
    with source.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 48
    fig, axes = plt.subplots(3, 2, figsize=(9, 9), constrained_layout=True)
    cells = ['UU', 'AU', 'UA', 'AA']
    for i, regime in enumerate(['control', 'single_shift', 'common_shift']):
        for j, target in enumerate([-.5, .5]):
            ax = axes[i, j]
            for method, offset, color in [('Q', -.04, '#24658a'), ('EU', .04, '#b85c38')]:
                selected = [next(r for r in rows if r['regime'] == regime and float(r['target']) == target
                                 and r['method'] == method and r['cell'] == cell) for cell in cells]
                y = np.array([float(r['mean']) for r in selected])
                low = np.array([float(r['ci95_low']) for r in selected])
                high = np.array([float(r['ci95_high']) for r in selected])
                ax.errorbar(np.arange(4)+offset, y, yerr=[y-low, high-y], fmt='o-', color=color,
                            capsize=3, linewidth=1.3, markersize=4, label=method)
            ax.set_xticks(range(4), ['U, U', 'A, U', 'U, A', 'A, A'])
            ax.set_title(f"{regime.replace('_', ' ')}; T={target:g}")
            ax.grid(axis='y', alpha=.2)
            ax.set_xlabel('Objective weights, direction weights')
            if j == 0:
                ax.set_ylabel('Mean true target error')
            ax.legend(frameon=False)
    fig.suptitle('Custom engine: 200-panel cap, 90 paired source seeds\n'
                 'Descriptive 95% intervals; A is Q or EU. UU is shared.', fontsize=12)
    fig.savefig(outputs[0])
    fig.savefig(outputs[1], dpi=150)
    plt.close(fig)
    report = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  outputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs[:2]},
                  plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='All 48 cells, descriptive intervals; see primary_effects.csv for Bonferroni36 inference')
    outputs[2].write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', type=Path, required=True)
    plot(p.parse_args().analysis)
