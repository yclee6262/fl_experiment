"""Read-only recomputation of batch costs, allocations and participant diagnostics."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from contribution_analysis.model_qv_v1.run import OUT,save,TOL

def main():
    result=json.loads((OUT/'progress.json').read_text());timing=json.loads((OUT/'timing.json').read_text())
    assert json.loads((OUT/'completion.json').read_text())['complete']
    groups={}
    for name in ['historical','independent_analytic','clean','controlled_poison','general','duplicate','distinct_aligned','distinct_biased']:
        rr=[r for r in result if r['phase']==name or r['scenario']==name]
        costs={a:sum(r['cost'][a] for r in rr) for a in ['base','AV','QV','LOO']}
        saving=1-costs['QV']/costs['LOO'];cheaper=sum(r['cost']['QV']<r['cost']['LOO'] for r in rr)
        groups[name]=dict(teams=len(rr),cost=costs,saving_pct=100*saving,cheaper_teams=cheaper,
            gate='通過' if saving>=.1 and cheaper/len(rr)>=.8 else '部分通過' if saving>0 else '未達')
    stats={}; examples=[]
    for phase in ['historical','independent_analytic']:
        cases=[r for r in result if r['phase']==phase];allrows=[];owners=[];stages={}
        for team in cases:
            d=json.loads((OUT/'records'/(team['key']+'_QV.json')).read_text());allrows+=d['rows']
            for stage,val in d['stages'].items():stages[stage]=stages.get(stage,0)+val['samples']
            for i in sorted(set(r['i'] for r in d['rows'])):
                rows=[r for r in d['rows'] if r['i']==i]
                own=dict(key=team['key'],i=i,Q=sum(r['Q'] for r in rows),Cdir=sum(r['Cdir'] for r in rows),V=sum(r['V'] for r in rows),
                    Qpositive=sum(max(r['Q'],0) for r in rows),Qnegative=sum(min(r['Q'],0) for r in rows))
                owners.append(own)
                if own['Q']>TOL and own['Cdir']< -TOL:examples.append(own)
        allocation={}
        for lam in ['0.25','0.5','0.75']:
            ss=[r['allocations'][lam] for r in cases]
            allocation[lam]=dict(mean_max_share=float(np.mean([max(z['shares']) for z in ss])),max_share=max(max(z['shares']) for z in ss),
                mean_reserve=float(np.mean([z['reserved'] for z in ss])),Q_empty_teams=sum(max(z['Q'])<=TOL for z in ss))
        stats[phase]=dict(round_rows=len(allrows),owners=len(owners),Q_positive=sum(z['Q']>TOL for z in owners),Q_negative=sum(z['Q']< -TOL for z in owners),
            Q_positive_Cdir_negative=sum(z['Q']>TOL and z['Cdir']< -TOL for z in owners),Q_positive_V_zero=sum(z['Q']>TOL and z['V']<=TOL for z in owners),
            V_positive_Q_nonpositive=sum(z['V']>TOL and z['Q']<=TOL for z in owners),
            Cdir_positive_rows=sum(r['Cdir']>TOL for r in allrows),Cdir_negative_rows=sum(r['Cdir']< -TOL for r in allrows),
            allocation=allocation,physical_samples_by_stage=stages,
            median_team_core_seconds={a:float(np.median([np.median([t['core_seconds'] for t in timing if t['key']==r['key'] and t['arm']==a]) for r in cases])) for a in ['base','AV','QV','LOO']})
    save(OUT/'analysis.json',dict(groups=groups,stats=stats,Q_sign_disagreement_examples=examples))
    print(json.dumps(dict(groups=groups,stats=stats,examples=examples[:3]),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
