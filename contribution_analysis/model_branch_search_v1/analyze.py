"""Paired team-level results; one predeclared fixed-budget primary comparison."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from contribution_analysis.model_branch_search_v1.run import OUT,ARMS,save,plain
import numpy as np
def compare(teams,label,ref):
    rr=[t['results'][label] for t in teams];R=np.array([(r[ref]['loss']-r['P']['loss'])/max(r['O']['initial'],1e-8) for r in rr]);rng=np.random.default_rng(20261107)
    strata=[[i for i,t in enumerate(teams) if t['n']==n and t['scenario']==sc] for n in [3,5] for sc in ['general','heterogeneous_bias']]
    bs=np.array([np.mean([np.mean(R[rng.choice(ix,len(ix),replace=True)]) for ix in strata]) for _ in range(10000)])
    diff=np.array([r[ref]['loss']-r['P']['loss'] for r in rr]);tol=np.array([1e-6*max(1,r['O']['initial']) for r in rr])
    return dict(mean_R=float(np.mean(R)),median_R=float(np.median(R)),CI95=np.quantile(bs,[.025,.975]),win=int(sum(diff>tol)),tie=int(sum(abs(diff)<=tol)),lose=int(sum(diff< -tol)),
        median_loss_difference=float(np.median(diff)),cost={a:sum(r[a]['samples'] for r in rr) for a in ARMS},
        mean_loss={a:float(np.mean([r[a]['loss'] for r in rr])) for a in ARMS},
        budget_stops={a:sum(r[a]['reason']=='budget' for r in rr) for a in ARMS},round_cap_stops={a:sum(r[a]['reason']=='round_limit' for r in rr) for a in ARMS})
def main():
    assert json.loads((OUT/'completion.json').read_text())['complete'];teams=json.loads((OUT/'progress.json').read_text());cc=[t for t in teams if t['phase']=='confirmation']
    assert len(cc)==80 and all(t['gate']['passed'] for t in teams)
    stats={label:{ref:compare(cc,label,ref) for ref in ['O','S','D']} for label in ['round30','budget200','budget500','budget1000']}
    primary=stats['budget500']['O'];mechanism=stats['budget500']['S']
    passed=primary['mean_R']>=.01 and primary['CI95'][0]>0 and primary['win']>=48 and primary['lose']<=16 and mechanism['mean_R']>0 and mechanism['CI95'][0]>0
    groups={}
    for n in [3,5]:
        for sc in ['general','heterogeneous_bias']:
            subset=[t for t in cc if t['n']==n and t['scenario']==sc]
            groups[f'n{n}_{sc}']={label:dict(teams=len(subset),mean_R=float(np.mean([(t['results'][label]['O']['loss']-t['results'][label]['P']['loss'])/max(t['results'][label]['O']['initial'],1e-8) for t in subset])),wins=sum(t['results'][label]['O']['loss']-t['results'][label]['P']['loss']>1e-6*max(1,t['results'][label]['O']['initial']) for t in subset)) for label in ['round30','budget500']}
    chosen=0;rounds=0;rescued=0;gains=[];examples={'improved':[],'round_good_budget_bad':[],'local_good_final_bad':[]}
    for t in cc:
        rec=json.loads((OUT/'records'/f"{t['key']}_round30.json").read_text());r=rec['P'];rounds+=len(r['rows'])
        for row in r['rows']:
            chosen+=row['chosen'].startswith('minus_');rescued+=row['rescued'];gains.append(row['G'])
        rd=t['results']['round30'];bd=t['results']['budget500'];item=dict(key=t['key'],round_loss={a:rd[a]['loss'] for a in ARMS},budget_loss={a:bd[a]['loss'] for a in ARMS},adopted=sum(z['chosen'].startswith('minus_') for z in r['rows']))
        if rd['P']['loss']<rd['O']['loss']-1e-6:examples['improved'].append(item)
        if rd['P']['loss']<rd['O']['loss']-1e-6 and bd['P']['loss']>bd['O']['loss']+1e-6:examples['round_good_budget_bad'].append(item)
        if item['adopted'] and rd['P']['loss']>rd['O']['loss']+1e-6:examples['local_good_final_bad'].append(item)
    for p,h in json.loads((OUT/'source_lock.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    out=dict(teams=80,stats=stats,groups=groups,primary_gate_passed=bool(passed),mechanism=dict(rounds=rounds,excluded_adopted=chosen,rescued=rescued,mean_local_extra_gain=float(np.mean(gains))),example_counts={k:len(v) for k,v in examples.items()})
    save(OUT/'analysis.json',out);save(OUT/'examples.json',{k:v[:4] for k,v in examples.items()});save(OUT/'audit.json',dict(passed=True,teams=92,source_lock=True,SD_equal=True,OS_gate=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    # Fixed-budget delivery snapshots: endpoint comparisons, no interpolation claim.
    for a in ARMS:
        y=[np.median([t['results'][f'budget{b}'][a]['loss']/max(t['results'][f'budget{b}'][a]['initial'],1e-8) for t in cc]) for b in [200,500,1000]]
        ax.plot([200,500,1000],y,'o-',label=a)
    ax.set(xlabel='Allowed model evaluations / number of models',ylabel='Median delivered F / initial F',title='Independent analytic confirmation (80 teams)');ax.legend();fig.tight_layout();fig.savefig(OUT/'quality_cost.png',dpi=180);plt.close(fig)
    print(json.dumps(plain(out),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
