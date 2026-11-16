"""Post-analysis figures only; do not change the locked J/A scores."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(root):
    with (root/'team_comparison.csv').open() as f:teams=list(csv.DictReader(f))
    bench=json.loads((root/'benchmark_results.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.5),constrained_layout=True)
    for method,offset,color,label in [('S_spearman',-.13,'#347bb0','Previous direction S'),('A_spearman',.13,'#dc8837','Step gain A')]:
        for k,scenario in enumerate(['clean','controlled_poison']):
            values=np.array([float(t[method]) for t in teams if t['phase']=='development' and t['scenario']==scenario]);mean=values.mean()
            rng=np.random.default_rng(20261020);boot=values[rng.integers(0,len(values),(10000,len(values)))].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
            axes[0].errorbar(k+offset,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color=color,capsize=5,label=label if k==0 else None)
    axes[0].axhline(0,color='#999',lw=.8);axes[0].axhline(.5,color='#999',ls='--',lw=.8)
    axes[0].set_xticks([0,1],['Clean','Controlled poison']);axes[0].set_ylim(-.7,.85);axes[0].set_ylabel('Mean within-team Spearman vs. full-removal')
    axes[0].set_title('Ranking: exploratory 95% seed intervals');axes[0].legend(frameon=False,fontsize=9,loc='upper right')
    for method,offset,color,label in [('anchor',-.18,'#dc8837','Baseline + step gain'),('full_LOO',.18,'#347bb0','Baseline + full LOO')]:
        values=[np.median([t['pipeline_seconds'] for t in c['timings'] if t['method']==method])*1000 for c in bench['cases']]
        bars=axes[1].bar(np.arange(4)+offset,values,.36,color=color,label=label)
        axes[1].bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
    axes[1].set_xticks(np.arange(4),['Clean 1600','Clean 1601','Poison 1600','Poison 1601'],rotation=15)
    axes[1].set_ylabel('Reload + solve + attribution pipeline (ms)');axes[1].set_title('Paired pilot cost: 15 repeats per method');axes[1].legend(frameon=False,fontsize=9)
    axes[1].set_ylim(0,90)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Local anchor step gain: lower cost, no stable quality ranking established',fontsize=12)
    path=root/'results_summary.png'
    if path.exists():raise FileExistsError(path)
    fig.savefig(path,dpi=185);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();plot(a.root.resolve())
