"""Static descriptive chart; no inference."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'j_local_quality_analysis_v1_20261027'
def main():
    s=json.loads((ROOT/'summary.json').read_text());fig,ax=plt.subplots(1,2,figsize=(10.5,4),layout='constrained')
    bottom=np.zeros(2)
    for field,label,color in [('sign','Same sign','#2b809c'),('reverse','Opposite sign','#d18458'),('external_zero','External near zero','#c8c8c8')]:
        y=np.array([r['metrics'][field]['mean'] for r in s]);ax[0].bar([0,1],y,bottom=bottom,label=label,color=color,width=.58)
        for i,v in enumerate(y):
            if v>.05:ax[0].text(i,bottom[i]+v/2,f'{100*v:.1f}%',ha='center',va='center',color='white')
        bottom+=y
    ax[0].set_xticks([0,1],['Clean','Controlled poison']);ax[0].set_ylim(0,1.02);ax[0].set_ylabel('Fraction among internally nonzero round scores');ax[0].set_title('Internal vs offline pooled round effect');ax[0].legend(loc='upper center',bbox_to_anchor=(.5,-.09),ncol=1,frameon=False)
    for k,(field,label,color) in enumerate([('agent_cancellation','Cancellation','#687a8c'),('agent_top1','Largest round / gross effect','#4e9c95'),('agent_top3','Top 3 rounds / gross effect','#d3ab54')]):
        values=[r['metrics'][field] for r in s];y=np.array([r['mean'] for r in values]);err=np.array([[r['mean']-r['ci'][0] for r in values],[r['ci'][1]-r['mean'] for r in values]])
        ax[1].bar(np.array([0,1])+(k-1)*.24,y,.23,color=color,label=label,yerr=err,capsize=3)
    ax[1].set_xticks([0,1],['Clean','Controlled poison']);ax[1].set_ylim(0,1);ax[1].set_title('Decomposition of the original signed J');ax[1].set_ylabel('Team-weighted mean (95% bootstrap CI)');ax[1].legend(loc='upper center',bbox_to_anchor=(.5,-.09),frameon=False)
    fig.suptitle('Historical data: 20 teams, 5 starts, 5,450 joint-removal branches',fontsize=12);fig.savefig(ROOT/'j_local_quality_overview.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
