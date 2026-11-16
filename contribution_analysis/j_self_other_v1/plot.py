"""Static diagnostic figure from locked results; zero inference."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]/'j_self_other_diagnosis_v2_20261027'
def main():
    s=json.loads((ROOT/'summary.json').read_text());fig,ax=plt.subplots(1,2,figsize=(11.5,4.7),layout='constrained');bottom=np.zeros(2)
    for field,label,color in [('self_direction_flip','Peers agree with pooled','#358e9c'),('peers_still_opposite','Peers still oppose pooled','#cf865b'),('peers_nearzero','Peers near zero','#b6bac3')]:
        y=np.array([r['metrics'][field]['mean'] for r in s]);ax[0].bar([0,1],y,bottom=bottom,color=color,label=label,width=.58)
        for i,v in enumerate(y):ax[0].text(i,bottom[i]+v/2,f'{v*100:.1f}%',ha='center',va='center',color='white')
        bottom+=y
    ax[0].set_xticks([0,1],['Clean','Controlled poison']);ax[0].set_ylim(0,1.02);ax[0].set_title('Among original internal / pooled sign reversals');ax[0].set_ylabel('Team-weighted fraction');ax[0].legend(loc='upper center',bbox_to_anchor=(.5,-.1),frameon=False)
    for k,(f,label,color) in enumerate([('j_sign','Original j','#456888'),('own_sign','Own weighted part','#a2aabd'),('other_mean_sign','Peer mean (diagnostic only)','#4b9993')]):
        m=[r['metrics'][f] for r in s];y=np.array([r['mean'] for r in m]);err=np.array([[r['mean']-r['ci'][0] for r in m],[r['ci'][1]-r['mean'] for r in m]])
        ax[1].bar(np.array([0,1])+(k-1)*.24,y,.23,label=label,color=color,yerr=err,capsize=3)
    ax[1].set_xticks([0,1],['Clean','Controlled poison']);ax[1].set_ylim(0,1);ax[1].set_title('Same external-reference denominator');ax[1].set_ylabel('Sign agreement (95% team bootstrap CI)');ax[1].legend(loc='upper center',bbox_to_anchor=(.5,-.1),frameon=False)
    fig.suptitle('Self / other scoring at fixed endpoints: 20 historical teams, 5 starts',fontsize=12);fig.savefig(ROOT/'self_other_overview.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
