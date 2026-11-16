"""Plot already locked development predictions; no model training or inference."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(root):
    with (root/'development_metrics.csv').open() as f:metrics=list(csv.DictReader(f))
    with (root/'development_predictions.csv').open() as f:predictions=list(csv.DictReader(f))
    names=['zero','constant','model_only','candidate_only','full']
    labels=['Zero','Constant','Model stats','Candidate stats','Full stats']
    colors={'clean':'#3074ad','controlled_poison':'#df8933'}
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.5),constrained_layout=True)
    for scenario,offset,label in [('clean',-.18,'Clean'),('controlled_poison',.18,'Controlled poison')]:
        values=[float(next(r['MAE'] for r in metrics if r['scenario']==scenario and r['method']==n)) for n in names]
        axes[0].bar(np.arange(len(names))+offset,values,width=.36,label=label,color=colors[scenario])
        rows=[r for r in predictions if r['scenario']==scenario]
        axes[1].scatter([float(r['C_full']) for r in rows],[float(r['full']) for r in rows],
                         color=colors[scenario],s=20,alpha=.6,label=label,edgecolors='none')
    axes[0].set_xticks(np.arange(len(names)),labels,rotation=16)
    axes[0].set_ylabel('Mean absolute error (lower is better)')
    axes[0].set_title('Prediction error vs. simple baselines')
    axes[0].legend(frameon=False)
    lo=min(float(r['C_full']) for r in predictions)-.006
    hi=max(float(r['C_full']) for r in predictions)+.006
    axes[1].plot([lo,hi],[lo,hi],'--',color='#777777',lw=1,label='Perfect prediction')
    axes[1].axhline(0,color='#aaaaaa',lw=.7);axes[1].axvline(0,color='#aaaaaa',lw=.7)
    axes[1].set_xlim(lo,hi);axes[1].set_ylim(lo,hi)
    axes[1].set_xlabel('Full-removal reference effect')
    axes[1].set_ylabel('Out-of-fold full-stats prediction')
    axes[1].set_title('Predictions miss larger positive / negative effects')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Development evidence only: 40 seed groups, 320 participant effects',fontsize=12)
    path=root/'development_summary.png'
    if path.exists():raise FileExistsError(path)
    fig.savefig(path,dpi=190);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();plot(a.root.resolve())
