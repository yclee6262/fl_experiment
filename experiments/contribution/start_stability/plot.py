"""Plot all shadow-payment stability cells; no new inferential contrasts."""
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
    outputs = [analysis/'payment_stability.pdf', analysis/'payment_stability.png', analysis/'plot_provenance.json']
    if any(p.exists() for p in outputs):
        raise FileExistsError('Plot output exists')
    source = analysis/'payment_summary.csv'
    with source.open() as f:
        rows = [r for r in csv.DictReader(f) if r['metric'] == 'total_payment_TV']
    assert len(rows) == 16
    cells = [(-.5, 'inversion'), (-.5, 'solver'), (.5, 'inversion'), (.5, 'solver')]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, constrained_layout=True)
    for ax, scenario, title in zip(axes, ['clean', 'controlled_poison'], ['Clean, K=3', 'Controlled contamination, K=5']):
        for judge, offset, color in [('F', -.16, '#286485'), ('Fminus', .16, '#bb623f')]:
            rr = [next(r for r in rows if r['scenario'] == scenario and float(r['target']) == target
                       and r['axis'] == axis and r['judge'] == judge) for target, axis in cells]
            y = np.array([float(r['mean']) for r in rr])
            lo = np.array([float(r['ci95_low']) for r in rr])
            hi = np.array([float(r['ci95_high']) for r in rr])
            ax.bar(np.arange(4)+offset, y, .3, yerr=[y-lo, hi-y], capsize=3, color=color, label=judge)
        ax.set_xticks(range(4), ['T=-.5\nInverse', 'T=-.5\nSolver', 'T=.5\nInverse', 'T=.5\nSolver'])
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xlabel('Initialization factor varied within each pair')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
    axes[0].set_ylabel('Mean total shadow-payment TV, including reserve')
    fig.suptitle('Fixed-model initialization sensitivity\nDescriptive team-cluster 95% intervals; 10 teams per cohort', fontsize=12)
    fig.savefig(outputs[0])
    fig.savefig(outputs[1], dpi=150)
    plt.close(fig)
    report = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        outputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs[:2]},
        plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Descriptive 16 cells; no causal comparison between K3 and K5 or multiple-testing claim')
    outputs[2].write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', type=Path, required=True)
    plot(p.parse_args().analysis)
