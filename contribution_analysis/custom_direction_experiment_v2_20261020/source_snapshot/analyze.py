"""Offline pooled scoring only after every baseline and removal is locked."""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from common import read,save,sha,csvsave,check_sources


def correlation(a,b):
    a=rankdata(a);b=rankdata(b)
    if np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])


def analyze(root):
    import torch
    from models import PretrainedModel
    torch.set_num_threads(1)
    check_sources(root)
    for folder,lockname in [('baselines','baseline_lock.json'),('removals','removal_lock.json')]:
        for n,h in read(root/lockname)['record_sha256'].items():
            if sha(root/folder/n)!=h:raise ValueError('unlocked_solution')
    rows=[];teams=[];diagnostics=[]
    for phase,cases in read(root/'cases.json').items():
        for case in cases:
            baseline=read(root/'baselines'/(case['key']+'.json'));s=baseline['scores'];ids=case['ids']
            evaluator=PretrainedModel(2);ep=torch.load(case['evaluator'],map_location='cpu',weights_only=True);evaluator.load_state_dict(ep['state_dict']);evaluator.eval()
            files=[root/'baselines'/(case['key']+'.json')]+[root/'removals'/(case['key']+'_'+str(i)+'.json') for i in ids]
            solutions=[read(p)['result']['solution'] for p in files]
            for x in solutions:
                if not np.isfinite(x).all() or np.max(abs(np.array(x)))>1+1e-8:raise ValueError('invalid_evaluated_delivery')
            with torch.no_grad():errors=[abs(float(evaluator(torch.tensor(x,dtype=torch.float32)[None,:]).item())) for x in solutions]
            C=np.array(errors[1:])-errors[0];status=s['status']
            if status=='available' and np.ptp(C)<=1e-4:status='near_flat_reference'
            primary=correlation(s['S'],C) if status=='available' else None
            if status=='available' and primary is None:status='score_tie'
            candidate=correlation(s['candidate_S'],C) if s['candidate_S'] is not None and np.ptp(C)>1e-4 else None
            alpha_corr=correlation(case['alphas'],C) if np.ptp(C)>1e-4 else None
            teams.append(dict(phase=phase,key=case['key'],scenario=case['scenario'],seed=case['seed'],m=len(ids),
                status=status,spearman=primary,candidate_spearman=candidate,alpha_spearman=alpha_corr,
                baseline_pooled_error=errors[0],path_length=s['path_length'],n_accepted=s['n_accepted'],
                C_range=float(np.ptp(C)),C_min=float(C.min()),C_max=float(C.max())))
            for j,i in enumerate(ids):
                rows.append(dict(phase=phase,key=case['key'],scenario=case['scenario'],seed=case['seed'],agent_id=i,
                    status=status,D=s['D'][j],D_plus=s['D_plus'][j],D_minus=s['D_minus'][j],
                    S_raw=None if s['S_raw'] is None else s['S_raw'][j],S=None if s['S'] is None else s['S'][j],
                    candidate_S=None if s['candidate_S'] is None else s['candidate_S'][j],
                    full_error=errors[0],removal_error=errors[j+1],C_custom_full=float(C[j])))
            accepted=[t for t in baseline['iterations'] if t['accepted']]
            trial_count=sum(len(t['trials']) for t in baseline['iterations']);rejected=sum(not a['accepted'] for t in baseline['iterations'] for a in t['trials'])
            x0=np.array(case['fixed_start']);xf=np.array(baseline['result']['solution']);net=float(np.linalg.norm(xf-x0))
            diagnostics.append(dict(phase=phase,key=case['key'],score_status=s['status'],n_iterations=s['n_iterations'],
                 n_dynamic=sum(t['mode']=='dynamic' for t in baseline['iterations']),n_accepted=len(accepted),
                 n_trials=trial_count,n_rejected=rejected,
                 min_accepted_rho=min((t['rho'] for t in accepted),default=None),
                 max_cancellation_amplification=max((t['cancellation_amplification'] for t in accepted),default=None),
                 max_g_error=s['max_g_error'],max_dx_error=s['max_dx_error'],max_projection_error=s['max_projection_error'],
                 net_displacement=net,path_length=s['path_length'],path_to_net_ratio=s['path_length']/net if net else None))
    summaries=[]
    for phase in ['pilot','development']:
        for scenario in ['clean','controlled_poison']:
            subset=[t for t in teams if t['phase']==phase and t['scenario']==scenario];valid=[t for t in subset if t['spearman'] is not None]
            values=np.array([t['spearman'] if t['spearman'] is not None else np.nan for t in subset]);rng=np.random.default_rng(20261020)
            sample=values[rng.integers(0,len(values),(10000,len(values)))];counts=np.isfinite(sample).sum(axis=1)
            boot=np.nansum(sample,axis=1)[counts>0]/counts[counts>0];ci=np.quantile(boot,[.025,.975]).tolist() if len(boot) else None
            mean=float(np.mean([t['spearman'] for t in valid])) if valid else None
            summaries.append(dict(phase=phase,scenario=scenario,total_teams=len(subset),valid_teams=len(valid),
                coverage=len(valid)/len(subset),mean_spearman=mean,median_spearman=float(np.median([t['spearman'] for t in valid])) if valid else None,
                exploratory_95pct_CI=ci,undefined_bootstraps=int(np.sum(counts==0)),
                mean_candidate_spearman=float(np.mean([t['candidate_spearman'] for t in subset if t['candidate_spearman'] is not None])) if any(t['candidate_spearman'] is not None for t in subset) else None,
                alpha_spearman_defined=sum(t['alpha_spearman'] is not None for t in subset),
                effect_screen_pass=bool(len(valid)/len(subset)>=.90 and mean is not None and mean>=.5 and ci[0]>0)))
    pilot=read(root/'pilot_results.json');development=[s for s in summaries if s['phase']=='development'];effect=all(s['effect_screen_pass'] for s in development)
    gate=dict(accepted=bool(effect and pilot['technical_accepted'] and pilot['cost_target_met']),technical_accepted=pilot['technical_accepted'],
        cost_target_met=pilot['cost_target_met'],effect_screen_pass=effect,
        status='exploratory_screen_passed' if effect and pilot['cost_target_met'] else 'exploratory_screen_not_passed',
        independent_confirmation_started=False,interpretation='reused-data exploratory evidence only; sign S is not sign of quality C')
    examples=[]
    for scenario in ['clean','controlled_poison']:
        subset=[t for t in teams if t['phase']=='development' and t['scenario']==scenario and t['spearman'] is not None]
        for label,reverse in [('best_rank',True),('worst_rank',False)]:
            if subset:
                t=sorted(subset,key=lambda t:((-t['spearman']) if reverse else t['spearman'],t['seed']))[0]
                examples.append(dict(category=label,team=t,participants=[r for r in rows if r['key']==t['key']]))
    csvsave(root/'participant_comparison.csv',rows);csvsave(root/'team_comparison.csv',teams);csvsave(root/'mechanism_diagnostics.csv',diagnostics)
    save(root/'summary.json',summaries);save(root/'gate.json',gate);save(root/'representative_cases.json',examples)
    print(__import__('json').dumps(dict(summary=summaries,gate=gate),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();analyze(a.root.resolve())
