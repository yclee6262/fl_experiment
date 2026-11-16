"""Independent stored-result comparison of combinations and contribution roles."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from contribution_analysis.model_pool_combinations_v1.run import OUT,CA,ARMS,save
def main():
    assert json.loads((OUT/'completion.json').read_text())['complete']
    teams=json.loads((OUT/'progress.json').read_text());time=json.loads((OUT/'timing.json').read_text());groups={};checked=0
    for team in teams:
        records={a:json.loads((OUT/'records'/f"{team['key']}_{a}.json").read_text()) for a in ARMS}
        assert len({r['baseline_digest'] for r in records.values()})==1
        for a,r in records.items():assert sum(r['stages'].values())==r['samples']==team['cost'][a]
        if team['phase']=='historical':
            old=json.loads((CA/'combined_contribution_layer2_experiment_v1_20261021/development'/f"{team['key']}.json").read_text())
            assert len(old['groups'])==len(records['J']['rows'])
            for r,z in zip(records['J']['rows'],old['groups']):assert r['t']==z['t'] and r['i']==z['i'] and abs(r['J']-z['j'])<1e-10
        for p in ['','A_']:
            assert records[p+'CMN']['query_digest']==records[p+'VN']['query_digest']
            assert records[p+'CMN']['rows']==records[p+'VN']['rows']
        for pool in ['VU','CMN','VN']:
            for a,b in zip(records[pool]['rows'],records['A_'+pool]['rows']):
                for k in ['V','U']+(['CM','I','N'] if pool!='VU' else []):assert abs(a[k]-b[k])<1e-10
        for pool in ['VU','CMN','VN']:
            for beta,alloc in team['allocations'][pool].items():
                assert abs(sum(alloc['shares'])+alloc['reserve']-1)<1e-10
                assert all(x>=0 for x in alloc['shares'])
        checked+=1
    for path,h in json.loads((OUT/'source_lock.json').read_text()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    for phase in ['historical','known_analytic80']:
        rr=[r for r in teams if r['phase']==phase];cost={a:sum(r['cost'][a] for r in rr) for a in ARMS};allocation={}
        for pool in ['VU','CMN','VN']:
            bybeta={}
            for beta in ['0.1','0.2','0.3']:
                items=[r['allocations'][pool][beta] for r in rr];cm=[r['allocations']['CMN'][beta]['CM'] for r in rr]
                negowners=sum(v< -1e-10 for cs in cm for v in cs)
                paidnegative=sum(cs[i]< -1e-10 and x['shares'][i]>1e-10 for x,cs in zip(items,cm) for i in range(len(cs)))
                negative_share=sum(sum(x['shares'][i] for i in range(len(cs)) if cs[i]< -1e-10) for x,cs in zip(items,cm))/len(rr)
                negative_info=sum(sum(x['info_shares'][i] for i in range(len(cs)) if cs[i]< -1e-10) for x,cs in zip(items,cm))/len(rr)
                bybeta[beta]=dict(mean_max_share=float(np.mean([max(x['shares']) for x in items])),max_share=max(max(x['shares']) for x in items),
                    mean_reserve=float(np.mean([x['reserve'] for x in items])),CM_negative_owners=negowners,CM_negative_paid=paidnegative,
                    mean_budget_to_CM_negative=negative_share,mean_info_budget_to_CM_negative=negative_info,
                    mean_TV_vs_CMN=float(np.mean([sum(abs(a-b) for a,b in zip(x['shares'],r['allocations']['CMN'][beta]['shares']))/2 for x,r in zip(items,rr)])))
            allocation[pool]=bybeta
        groups[phase]=dict(teams=len(rr),cost=cost,extra_cost={a:v-cost['B'] for a,v in cost.items()},allocation=allocation,
            total_cost_ratio_to_LOO={a:cost[a]/cost['LOO'] for a in ARMS},
            cheaper_than_LOO={a:sum(r['cost'][a]<r['cost']['LOO'] for r in rr) for a in ARMS},
            cheaper_than_J={a:sum(r['cost'][a]<r['cost']['J'] for r in rr) for a in ARMS},
            median_team_core_seconds={a:float(np.median([np.median([t['core_seconds'] for t in time if t['key']==r['key'] and t['arm']==a]) for r in rr])) for a in ARMS})
    for n in [3,5]:
        for sc in ['general','heterogeneous_bias']:
            rr=[r for r in teams if r['phase']=='known_analytic80' and r['n']==n and r['scenario']==sc]
            groups[f'n{n}_{sc}']=dict(teams=len(rr),cost={a:sum(r['cost'][a] for r in rr) for a in ARMS})
    save(OUT/'analysis.json',groups);save(OUT/'audit.json',dict(passed=True,teams=checked,J_historical_replay=True,CMN_VN_same_model_queries=True,budget_conserved=True,source_lock=True))
    print(json.dumps(groups,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
