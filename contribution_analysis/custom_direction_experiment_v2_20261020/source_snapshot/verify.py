"""Independent loop reconstruction from raw replies; not the scoring function."""
import argparse
from pathlib import Path
import numpy as np
from common import read,save,sha,csvread,check_sources


def verify(root):
    import torch
    from models import PretrainedModel
    torch.set_num_threads(1);cfg=check_sources(root);errors=[];n_steps=0;max_error=0.
    locks={folder:read(root/name) for folder,name in [('baselines','baseline_lock.json'),('removals','removal_lock.json')]}
    for folder,lock in locks.items():
        for n,h in lock['record_sha256'].items():
            if sha(root/folder/n)!=h:errors.append('solution_lock:'+n)
    for case in read(root/'cases.json')['pilot']+read(root/'cases.json')['development']:
        b=read(root/'baselines'/(case['key']+'.json'));m=len(case['ids']);D=np.zeros(m);candidate=np.zeros(m);P=0.;sensitive=False
        w=np.array(case['alphas']);w=w/w.sum()
        for it in b['iterations']:
            g=np.array(it['g']);gn=float(np.linalg.norm(g));parts=np.zeros((m,len(g)));cols=np.zeros_like(parts)
            for direction in it['directions']:
                for i in range(m):
                    slope=(direction['reply_losses'][i]-it['current_losses'][i])/direction['width']
                    v=w[i]*direction['weight']*slope*np.array(direction['u'])
                    parts[i]+=v;cols[direction['k']]+=v
            err=float(np.linalg.norm(parts.sum(axis=0)-g));max_error=max(max_error,err)
            if err>1e-10*max(1.,gn):errors.append('g_rebuild:'+case['key'])
            accepted=[a for a in it['trials'] if a['accepted']]
            if not accepted:continue
            n_steps+=1;dx=np.array(accepted[0]['x'])-it['x'];s=float(np.linalg.norm(dx));P+=s
            unit=dx/s;delta_parts=-s*parts/gn;increments=delta_parts@unit
            D+=increments;candidate+=(-s*cols/gn)@unit
            denom=sum(float(np.linalg.norm(v)) for v in parts);rho=gn/denom if denom else 0.
            if rho<1e-6:sensitive=True
            tol=1e-10*max(1.,s)
            if rho>=1e-6 and (np.linalg.norm(delta_parts.sum(axis=0)-dx)>tol or abs(increments.sum()-s)>tol):errors.append('motion_rebuild:'+case['key'])
            if not np.allclose(increments,it['d'],atol=tol,rtol=1e-12):errors.append('per_step_projection:'+case['key'])
        status='no_accepted_motion' if P==0 else 'numerically_sensitive' if sensitive else 'available'
        if status!=b['scores']['status']:errors.append('status:'+case['key'])
        if not np.allclose(D,b['scores']['D'],atol=1e-9,rtol=1e-12):errors.append('D_rebuild:'+case['key'])
        if P and not np.allclose(D/P,b['scores']['S_raw'],atol=1e-8,rtol=1e-12):errors.append('S_rebuild:'+case['key'])
        if b['evaluator_loaded'] or b['removal_labels_loaded']:errors.append('baseline_leakage')
        for k,i in enumerate(case['ids']):
            r=read(root/'removals'/(case['key']+'_'+str(i)+'.json'))
            if r['remaining_ids']!=[j for j in case['ids'] if j!=i]:errors.append('removal_members')
            expected=np.delete(w,k);expected/=expected.sum()
            if not np.allclose(r['alphas'],expected,atol=1e-15,rtol=0):errors.append('removal_alpha')
            if not np.array_equal(r['fixed_start'],case['fixed_start']):errors.append('removal_start')
            if r['evaluator_loaded']:errors.append('removal_leakage')
    comparison=csvread(root/'participant_comparison.csv');by={(r['key'],int(r['agent_id'])):r for r in comparison}
    if len(by)!=96 or len(comparison)!=96:errors.append('effect_coverage')
    max_score_delta=0.
    for phase,cases in read(root/'cases.json').items():
        for case in cases:
            ep=torch.load(case['evaluator'],map_location='cpu',weights_only=True);evaluator=PretrainedModel(2);evaluator.load_state_dict(ep['state_dict']);evaluator.eval()
            def quality(x):
                with torch.no_grad():return abs(float(evaluator(torch.tensor(x,dtype=torch.float32)[None,:]).item()))
            baseline=read(root/'baselines'/(case['key']+'.json'));full=quality(baseline['result']['solution'])
            for i in case['ids']:
                removal=read(root/'removals'/(case['key']+'_'+str(i)+'.json'));C=quality(removal['result']['solution'])-full
                delta=abs(C-float(by[case['key'],i]['C_custom_full']));max_score_delta=max(max_score_delta,delta)
                if delta>1e-10:errors.append('pooled_reload')
    for team in csvread(root/'team_comparison.csv'):
        rs=[r for r in comparison if r['key']==team['key']]
        if team['spearman']:
            from scipy.stats import spearmanr
            corr=float(spearmanr([float(r['S']) for r in rs],[float(r['C_custom_full']) for r in rs]).statistic)
            if abs(corr-float(team['spearman']))>1e-12:errors.append('rank_rebuild')
    pilot=read(root/'pilot_results.json');ratio=sum(np.median(c['record_seconds']) for c in pilot['cases'])/sum(np.median(c['control_seconds']) for c in pilot['cases'])
    if abs(ratio-pilot['aggregate_ratio'])>1e-12:errors.append('timing_rebuild')
    result=dict(accepted=not errors,errors=errors,n_baselines=24,n_removals=96,n_effects=96,n_accepted_steps=n_steps,
        max_g_rebuild_error=max_error,max_pooled_reload_delta=max_score_delta,source_and_inputs_checked=True,
        no_extra_queries_in_verifier=True,not_validity_evidence=True)
    save(root/'verification.json',result);print(result)
    if errors:raise ValueError('verification_failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();verify(a.root.resolve())
