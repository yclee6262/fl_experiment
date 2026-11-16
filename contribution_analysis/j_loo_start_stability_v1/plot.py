"""Static research figure, no new model calls."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
    s=read(ROOT/'internal_summary.json');fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    colors={'J':'#2878a0','LOO':'#dd8952'};scenarios=['clean','controlled_poison']
    for k,field in enumerate(['rank','flip']):
        for mi,m in enumerate(['J','LOO']):
            rr=[next(v for v in s['stability'] if v['scenario']==sc and v['method']==m)['metrics'][field] for sc in scenarios]
            ys=[r['mean'] for r in rr];err=np.array([[r['mean']-r['ci'][0] for r in rr],[r['ci'][1]-r['mean'] for r in rr]])
            axes[k].bar(np.arange(2)+(mi-.5)*.32,ys,.30,label=m,color=colors[m],yerr=err,capsize=3)
        axes[k].set_xticks([0,1],['Clean','Controlled poison']);axes[k].set_ylim(-.05,1.05);axes[k].legend(frameon=False)
    axes[0].set_title('Ranking across starts');axes[0].set_ylabel('Mean Spearman (95% team bootstrap CI)');axes[0].axhline(.8,color='gray',linestyle='--',lw=1)
    axes[1].set_title('Sign reversals across starts');axes[1].set_ylabel('Among pairs with both scores nonzero');axes[1].axhline(.1,color='gray',linestyle='--',lw=1)
    rr=s['costs'];axes[2].bar([0,1],[v['query_ratio'] for v in rr],color='#2878a0');axes[2].set_xticks([0,1],['Clean','Controlled poison']);axes[2].set_ylim(0,max(1.2,max(v['query_ratio'] for v in rr)*1.15));axes[2].axhline(1,color='black',lw=.8);axes[2].axhline(.7,color='gray',linestyle='--',lw=1)
    for i,v in enumerate(rr):axes[2].text(i,v['query_ratio']+.025,f"{v['query_ratio']:.3f}",ha='center')
    axes[2].set_title('J / full LOO query cost');axes[2].set_ylabel('Includes full baseline and common scoring')
    fig.suptitle('20 historical teams, five fixed starts per team; exploratory results',fontsize=12)
    fig.savefig(ROOT/'start_stability_overview.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
