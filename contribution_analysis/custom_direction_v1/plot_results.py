"""Post-analysis reporting only; read locked tables, no fitting/solving."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata


def plot(root):
    with (root/'team_comparison.csv').open() as f:teams=list(csv.DictReader(f))
    with (root/'participant_comparison.csv').open() as f:people=list(csv.DictReader(f))
    summary=json.loads((root/'summary.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.1),constrained_layout=True)
    colors={'clean':'#347bb0','controlled_poison':'#db8839'}
    for k,scenario in enumerate(colors):
        group=[r for r in teams if r['phase']=='development' and r['scenario']==scenario]
        values=np.array([float(r['spearman']) for r in group]);x=k+np.linspace(-.12,.12,len(values))
        axes[0].scatter(x,values,color=colors[scenario],s=25,alpha=.7)
        s=next(s for s in summary if s['phase']=='development' and s['scenario']==scenario);mean=s['mean_spearman'];lo,hi=s['exploratory_95pct_CI']
        axes[0].errorbar(k,mean,yerr=[[mean-lo],[hi-mean]],fmt='D',color='#222',capsize=5,ms=5)
    axes[0].set_xticks([0,1],['Clean','Controlled poison']);axes[0].set_ylim(-1.08,1.08)
    axes[0].axhline(0,color='#888',lw=.8);axes[0].axhline(.5,color='#888',ls='--',lw=.8)
    axes[0].set_ylabel('Within-team Spearman');axes[0].set_title('Mean and exploratory 95% seed CI')
    for ax,key,title in [(axes[1],'budget_clean_2026101616','Matching ranks: Clean / seed 1616'),
                         (axes[2],'budget_controlled_poison_2026101611','Reversed ranks: Poison / seed 1611')]:
        rs=[r for r in people if r['key']==key];n=len(rs)
        a=rankdata([-float(r['S']) for r in rs]);b=rankdata([-float(r['C_custom_full']) for r in rs])
        for i,r in enumerate(rs):
            color=plt.cm.tab10(i)
            ax.plot([0,1],[a[i],b[i]],'-o',color=color,lw=1.7,ms=4)
            ax.annotate('Agent '+r['agent_id'],(0,a[i]),xytext=(-6,0),textcoords='offset points',ha='right',va='center',fontsize=8,color=color)
        ax.set_xlim(-.5,1.2);ax.set_xticks([0,1],['Direction S','Full-removal C']);ax.set_yticks(range(1,n+1))
        ax.set_ylim(n+.5,.5);ax.set_title(title);ax.set_ylabel('Rank (1 = highest)')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True)
    fig.suptitle('Custom-only exploratory evidence: 10 stored seeds / 20 teams / 80 effects',fontsize=12)
    path=root/'results_summary.png'
    if path.exists():raise FileExistsError(path)
    fig.savefig(path,dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();plot(a.root.resolve())
