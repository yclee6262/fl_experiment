"""Post-analysis figures; no model access or score modification."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


root=Path(sys.argv[1]);gate=json.loads((root/'gate.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
colors={'A':'#187e8a','M':'#ad5564'}
for ax,scenario,title in zip(axes,['clean','controlled_poison'],['Clean','Controlled poison']):
    rows=[x for x in gate['primary'] if x['scenario']==scenario]
    for j,r in enumerate(rows):
        mean=r['mean'];lo,hi=r['interval']
        ax.errorbar(j,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color=colors[r['score']],capsize=6,markersize=9)
        ax.text(j,hi+.06,f"mean {mean:.2f}; n={r['n']}/10",ha='center',fontsize=10)
    ax.axhline(0,color='#999',lw=.8);ax.axhline(.5,color='#a45a1c',ls='--',label='Mean gate: > 0.5')
    ax.set(xticks=[0,1],xticklabels=['Anchor score A','Model score M'],ylim=(-1,1),xlim=(-.6,1.6),title=title)
    ax.set_ylabel('Mean within-team Spearman');ax.grid(axis='y',alpha=.15);ax.legend(loc='lower left',fontsize=9)
fig.suptitle('Role-matched removal validation: all four ranking gates failed\nStored development data; 98.75% seed-bootstrap intervals',fontsize=12)
fig.savefig(root/'role_rank_validation.png',dpi=180);plt.close(fig)

signs=[x for x in json.loads((root/'sign_metrics.json').read_text()) if x['deadzone']==1e-4]
fig,ax=plt.subplots(figsize=(10,4.7),layout='constrained')
labels=[];correct=[];neutral=[];hurt=[];counts=[]
for s in signs:
    n=s['predicted_negative'];row=s['confusion'][0];counts.append(n)
    correct.append(row[0]/n);neutral.append(row[1]/n);hurt.append(row[2]/n)
    labels.append(('Clean' if s['scenario']=='clean' else 'Poison')+' / '+s['score'])
x=np.arange(len(signs))
ax.bar(x,correct,color='#318879',label='Removal improves quality')
ax.bar(x,neutral,bottom=correct,color='#d2d2d2',label='Reference near zero')
ax.bar(x,hurt,bottom=np.array(correct)+neutral,color='#bd6269',label='Removal harms quality')
for j,n in enumerate(counts):ax.text(j,1.025,f'n={n}',ha='center')
ax.set(xticks=x,xticklabels=labels,ylim=(0,1.13),ylabel='Fraction among negative-score cases')
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
ax.set_title('Negative scores frequently flag useful roles\nDead zone: +/- 1e-4; very small denominators, descriptive only')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,fontsize=9)
fig.savefig(root/'negative_score_validation.png',dpi=180);plt.close(fig)
