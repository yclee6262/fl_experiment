"""Static research figures from locked results only."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import ROOT,read

def run():
    s=read(ROOT/'internal_summary.json');fig,axes=plt.subplots(2,3,figsize=(14,7.5),layout='constrained')
    qs=[0,.25,.5,.75];labels=['0%','25%','50%','75%']
    for row,sc in enumerate(['clean','controlled_poison']):
        rr=[m for m in s['metrics'] if m['scenario']==sc and m['rq']=='RQ1' and m['h']=='final']
        ax=axes[row,0];bottom=np.zeros(4)
        for field,color,label in [('retained','#3288bd','Same sign'),('reversed','#d6604d','Reversed'),('vanished','#bdbdbd','Near zero')]:
            y=np.array([next(m for m in rr if m['q']==q)['metrics'][field]['mean']*100 for q in qs])
            ax.bar(labels,y,bottom=bottom,color=color,label=label,width=.65);bottom+=y
        ax.set_ylim(0,103);ax.set_ylabel(('Clean' if row==0 else 'Controlled poison')+'\nTeam mean (%)');ax.set_title('One-round effect after restoration')
        ax.set_xticks(range(4),[f"{q:.0%}\n{next(m for m in rr if m['q']==q)['metrics']['retained']['n']}/10 teams" for q in qs])
        if row==0:ax.legend(fontsize=8,loc='lower left')
        ax=axes[row,1];rr=[m for m in s['metrics'] if m['scenario']==sc and m['rq']=='RQ3']
        means=[m['metrics']['rho']['mean'] for m in rr];low=[m['metrics']['rho']['interval'][0] for m in rr];high=[m['metrics']['rho']['interval'][1] for m in rr]
        ax.errorbar([1,2,3],means,yerr=[np.array(means)-low,np.array(high)-means],fmt='o',capsize=5,color='#4c72b0')
        ax.axhline(.8,color='#bf812d',ls='--',label='Mean target 0.8');ax.axhline(0,color='gray',lw=.6)
        ax.set_xticks([1,2,3],labels[1:]);ax.set_ylim(-1.05,1.05);ax.set_title('Late persistent effect vs full LOO\nSpearman; 99.8% exploratory CI')
        if row==0:ax.legend(fontsize=8)
        ax=axes[row,2];cost=[next(v for v in s['costs'] if v['scenario']==sc and v['q']==q)['ratio'] for q in qs]
        bars=ax.bar(labels,np.array(cost)*100,color=['#888888','#6baed6','#6baed6','#6baed6'],width=.65)
        for b,v in zip(bars,cost):ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{v*100:.1f}%',ha='center',fontsize=9)
        ax.axhline(70,color='#bf812d',ls='--',label='Cost target 70%');ax.set_ylim(0,115);ax.set_title('Standalone model-query cost\n% of full LOO, baseline included')
        if row==0:ax.legend(fontsize=8,loc='lower left')
    fig.suptitle('Path intervention diagnostics | 20 historical teams | no late candidate passed',fontsize=14)
    fig.savefig(ROOT/'path_intervention_overview.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for row,sc in enumerate(['clean','controlled_poison']):
        for col,key in enumerate(['max_temporal_change','max_restoration_gap']):
            r=s['examples'][sc][key];ax=axes[row,col]
            xx=np.arange(4);tau=[r['tau_1'],r['tau_3'],r['tau_5'],r['tau']];pi=[r['pi_1'],r['pi_3'],r['pi_5'],r['pi']]
            ax.plot(xx,tau,'o-',label='Temporary (restore)');ax.plot(xx,pi,'s-',label='Persistent (absent)')
            ax.axhline(r['D_L'],ls='--',color='gray',label='Full LOO internal');ax.axhline(0,lw=.5,color='black')
            ax.set_xticks(xx,['h=1','h=3','h=5','final']);ax.set_ylabel('Fixed full-panel loss difference')
            ax.set_title(f"{sc} | seed {r['seed']} | participant {r['agent_id']}\nintervene at {r['q']:.0%} (round index {r['t']})",fontsize=10)
            if row==0 and col==0:ax.legend(fontsize=8)
    fig.suptitle('Fixed-rule examples: effect at observation horizons (not spatial search paths)',fontsize=12)
    fig.savefig(ROOT/'path_intervention_examples.png',dpi=180);plt.close(fig)
if __name__=='__main__':run()
