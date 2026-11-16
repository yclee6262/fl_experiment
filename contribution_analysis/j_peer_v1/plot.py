"""Static cumulative comparison figure from saved results."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
    s=read(ROOT/'summary.json');fig,ax=plt.subplots(1,3,figsize=(12,4.2),layout='constrained');colors={'J':'#416f92','J_peer':'#4a9e92'}
    for k,field in enumerate(['D_L:rank','D_L:sign']):
        for mi,method in enumerate(['J','J_peer']):
            m=[r['metrics'][method+':'+field] for r in s];y=np.array([r['mean'] for r in m]);err=np.array([[r['mean']-r['ci'][0] for r in m],[r['ci'][1]-r['mean'] for r in m]])
            ax[k].bar(np.array([0,1])+(mi-.5)*.32,y,.3,color=colors[method],label=method,yerr=err,capsize=3)
        ax[k].set_xticks([0,1],['Clean','Controlled poison']);ax[k].set_ylim(-.1 if k==0 else 0,1);ax[k].legend(frameon=False)
    ax[0].set_title('Cumulative rank vs full internal LOO');ax[0].set_ylabel('Mean Spearman, 95% team bootstrap CI');ax[1].set_title('Cumulative sign vs full internal LOO');ax[1].set_ylabel('Reference-nonzero sign agreement')
    for mi,method in enumerate(['J','J_peer']):
        y=[r['counts'][method]/r['counts']['LOO'] for r in s];ax[2].bar(np.array([0,1])+(mi-.5)*.32,y,.3,color=colors[method],label=method)
        for i,v in enumerate(y):ax[2].text(i+(mi-.5)*.32,v+.02,f'{v:.3f}',ha='center',fontsize=9)
    ax[2].set_xticks([0,1],['Clean','Controlled poison']);ax[2].set_ylim(0,1.1);ax[2].axhline(1,color='black',lw=1);ax[2].set_title('Standalone model query cost / full LOO');ax[2].set_ylabel('Includes baseline, decisions, peer scoring');ax[2].legend(frameon=False)
    fig.suptitle('J_peer trial: 20 historical teams, 5 starts; fixed endpoint replay',fontsize=12);fig.savefig(ROOT/'j_peer_comparison.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
