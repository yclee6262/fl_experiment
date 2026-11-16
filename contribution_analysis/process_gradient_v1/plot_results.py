"""Reporting only: draw locked comparison values; never refit scores."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(root):
    with (root/'comparison_rows.csv').open() as f:rows=list(csv.DictReader(f))
    with (root/'metrics.csv').open() as f:metrics=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),constrained_layout=True)
    for scenario,color,label,offset in [('clean','#3074ad','Clean',-.18),('controlled_poison','#df8933','Controlled poison',.18)]:
        rs=[r for r in rows if r['scenario']==scenario]
        for ax,method,title in [(axes[0],'gradient_work','Local gradient work'),(axes[1],'endpoint_work','Exact endpoint work')]:
            x=[float(r['C_full']) for r in rs];y=[float(r[method]) for r in rs]
            ax.scatter(x,y,s=16,c=color,alpha=.65,label=label,edgecolors='none')
            ax.set_title(title);ax.set_xlabel('Full-removal reference effect')
            ax.set_yscale('symlog',linthresh=.01)
            ax.axhline(0,color='#999',lw=.7);ax.axvline(0,color='#999',lw=.7)
            ax.plot([-.06,.1],[-.06,.1],'--',c='#777',lw=1)
            ax.set_ylabel('Process score (symmetric-log scale)')
        methods=['gradient_work','endpoint_work','zero']
        vals=[float(next(r['MAE'] for r in metrics if r['scenario']==scenario and r['method']==m)) for m in methods]
        axes[2].bar(np.arange(3)+offset,vals,width=.36,color=color,label=label)
    axes[0].legend(frameon=False,fontsize=8)
    axes[2].set_xticks(np.arange(3),['Gradient','Endpoint','Zero']);axes[2].set_ylabel('Uncalibrated absolute gap vs. LOO')
    axes[2].set_title('Raw score mismatch (lower is better)')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Exploratory reused data: 40 seeds / 320 effects; no label fitting',fontsize=12)
    path=root/'comparison_summary.png'
    if path.exists():raise FileExistsError(path)
    fig.savefig(path,dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();plot(a.root.resolve())
