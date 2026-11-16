"""Independent endpoint and diagnostic arithmetic; no participant queries."""
import argparse
import itertools
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from common import read,save,sha,csvread,check

def verify(root):
    import torch
    from models import PretrainedModel
    torch.set_num_threads(1);cfg=check(root);old=Path(cfg['previous_custom']);local=Path(cfg['previous_local']);errors=[]
    for folder,name in [('full_revaluation','full_revaluation_lock.json'),('anchor_removals','anchor_removal_lock.json')]:
        for n,h in read(root/name)['record_sha256'].items():
            if sha(root/folder/n)!=h:errors.append('lock:'+n)
    labels={(r['key'],int(r['agent_id'])):r for r in csvread(old/'participant_comparison.csv')}
    As={(r['key'],int(r['agent_id'])):r for r in csvread(local/'participant_comparison.csv')}
    values=csvread(root/'diagnostic_values.csv');by={(r['key'],int(r['agent_id'])):r for r in values};max_L_error=0.;max_E_error=0.
    if len(values)!=80 or len(by)!=80:errors.append('effect_coverage')
    for case in read(root/'cases.json'):
        b=read(old/'baselines'/(case['key']+'.json'));w=np.array(case['alphas']);w=w/w.sum();m=len(w);first={r['agent_id']:r for r in read(root/'full_revaluation'/(case['key']+'.json'))['rows']}
        ep=torch.load(case['evaluator'],map_location='cpu',weights_only=True);model=PretrainedModel(2);model.load_state_dict(ep['state_dict']);model.eval()
        def pooled(x):
            with torch.no_grad():return abs(float(model(torch.tensor(x,dtype=torch.float32)[None,:]).item()))
        E0=pooled(b['result']['solution'])
        for k,i in enumerate(case['ids']):
            f=first[i];r=read(root/'anchor_removals'/(case['key']+'_'+str(i)+'.json'));v=by[case['key'],i]
            Lfull=sum(float(a)*abs(float(p)) for a,p in zip(w,f['predictions']));Lanchor=sum(float(a)*abs(float(p)) for a,p in zip(w,r['final_predictions']))
            fullx=read(old/'removals'/(case['key']+'_'+str(i)+'.json'))['result']['solution']
            if not np.array_equal(f['solution'],fullx):errors.append('full_endpoint')
            for actual,reported in [(Lfull,f['loss']),(Lanchor,r['final_fixed_loss']),(Lanchor,r['result']['loss'])]:
                error=abs(actual-reported);max_L_error=max(max_L_error,error)
                if error>1e-10:errors.append('fixed_L_reconstruction')
            if r['scoring_model_ids']!=case['ids'] or not np.array_equal(r['scoring_alphas'],case['alphas']):errors.append('full_panel_changed')
            if r['remaining_anchor_ids']!=[a for j,a in enumerate(case['ids']) if j!=k]:errors.append('anchor_identity')
            if not np.array_equal(r['candidates'],np.delete(np.array(case['candidates']),k,axis=0)):errors.append('candidate_geometry')
            if not np.array_equal(r['fixed_start'],case['fixed_start']):errors.append('start_changed')
            if r['result']['history'][0]!=b['result']['history'][0]:errors.append('initial_full_objective_changed')
            if r['counter']['calls']%m or r['counter']['samples']!=r['counter']['calls']:errors.append('full_panel_call_count')
            if r['evaluator_loaded'] or r['truth_loaded']:errors.append('evaluator_in_solver')
            checks={'A':float(As[case['key'],i]['A']),'full_L':Lfull-b['result']['loss'],'anchor_L':Lanchor-b['result']['loss'],
                    'anchor_E':pooled(r['result']['solution'])-E0,'full_E':pooled(fullx)-E0}
            for n,expected in checks.items():
                delta=abs(expected-float(v[n]));max_E_error=max(max_E_error,delta if n.endswith('_E') else 0.)
                if delta>1e-10:errors.append('value:'+n)
            if float(v['full_E'])!=float(labels[case['key'],i]['C_custom_full']):errors.append('old_quality_reference_changed')
    pairs=csvread(root/'team_pair_comparisons.csv');scores=['A','anchor_L','anchor_E','full_L','full_E']
    if len(pairs)!=200:errors.append('all_ten_pairs_missing')
    for case in read(root/'cases.json'):
        rs=[r for r in values if r['key']==case['key']]
        for x,y in itertools.combinations(scores,2):
            t=next(t for t in pairs if t['key']==case['key'] and t['x']==x and t['y']==y)
            a=np.array([float(r[x]) for r in rs]);b=np.array([float(r[y]) for r in rs]);near=(x!='A' and np.ptp(a)<=1e-4) or (y!='A' and np.ptp(b)<=1e-4);tie=np.ptp(a)==0 or np.ptp(b)==0
            if near or tie:
                if t['spearman']:errors.append('undefined_rank_fabricated')
            else:
                expected=float(spearmanr(a,b).statistic)
                if not t['spearman'] or abs(expected-float(t['spearman']))>1e-12:errors.append('rank_rebuild')
    first=read(root/'full_revaluation_lock.json')['totals'];second=read(root/'anchor_removal_lock.json')['totals']
    if first['samples']!=340 or first['solves']!=0 or second['solves']!=80 or second['samples']>200000 or second['solver_seconds']>300:errors.append('resource_budget')
    result=dict(accepted=not errors,errors=errors,n_teams=20,n_effects=80,n_pairwise_team_comparisons=200,
        max_internal_L_rebuild_delta=max_L_error,max_pooled_reload_delta=max_E_error,
        original_A_unchanged=True,no_new_participant_queries=True,pooled_verification_points=180,
        interpretation='Diagnostic integrity, not estimator effectiveness or global optimality.')
    save(root/'verification.json',result);print(result)
    if errors:raise ValueError('verification_failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();verify(a.root.resolve())
