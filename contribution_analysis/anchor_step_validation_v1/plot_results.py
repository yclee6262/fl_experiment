"""Standalone scientific figure from locked team summaries; no new estimates."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from study import ROOT,read,save

def plot():
    rows=read(ROOT/'summary.json');fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    colors={'clean':'#2377a4','controlled_poison':'#c26431'}
    labels={'clean':'Clean','controlled_poison':'Controlled poison'}
    for sc in colors:
        rs=[next(s for s in rows if s['phase']=='development' and s['scenario']==sc and s['kind']=='persistence' and s['h']==h) for h in [0,1,3,5]]
        axes[0].plot([0,1,3,5],[s['retention']['mean'] for s in rs],marker='o',color=colors[sc],label=labels[sc])
        axes[0].fill_between([0,1,3,5],[s['retention']['CI95'][0] for s in rs],[s['retention']['CI95'][1] for s in rs],color=colors[sc],alpha=.13)
        axes[1].plot([0,1,3,5],[s['rank']['mean'] for s in rs],marker='o',color=colors[sc],label=labels[sc])
    axes[0].axhline(.8,color='gray',linestyle='--',linewidth=.9,label='Fixed screening threshold')
    axes[0].set(ylabel='Original sign retained (team mean)',ylim=(-.02,1.04),title='Local effects often fade or reverse')
    axes[1].axhline(0,color='gray',linewidth=.9)
    axes[1].set(ylabel='Mean within-team Spearman',ylim=(-.1,1.04),title='Anchor rankings change after continuation')
    for ax in axes:ax.set_xlabel('Continuation iterations (all anchors restored)');ax.set_xticks([0,1,3,5]);ax.grid(alpha=.15)
    axes[0].legend(fontsize=8,loc='lower left')
    fig.suptitle('First accepted step only | 20 reused teams | Internal loss | Exploratory',fontsize=11)
    fig.savefig(ROOT/'validation_summary.png',dpi=180);fig.savefig(ROOT/'validation_summary.pdf');plt.close(fig)

if __name__=='__main__':plot()
