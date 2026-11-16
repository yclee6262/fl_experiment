"""Rebuild alternatives and J from saved replies, not the scoring kernel."""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from common import read,save,sha,csvread,check

def verify(root):
    cfg=check(root);prior=Path(cfg['previous']);errors=[];total_pairs=0;max_error=0.;new_samples=0
    for n,h in read(root/'scores_lock.json')['record_sha256'].items():
        if sha(root/'records'/n)!=h:errors.append('score_lock:'+n)
    for phase,cases in read(root/'cases.json').items():
        for case in cases:
            old=read(prior/'baselines'/(case['key']+'.json'));new=read(root/'records'/(case['key']+'.json'));m=len(case['ids']);weights=np.array(case['alphas']);weights/=weights.sum()
            by={(r['iteration'],r['anchor_index']):r for r in new['comparisons']};A=np.zeros(m);missing=np.zeros(m,int);samples=0
            for t in old['iterations']:
                accepted=[a for a in t['trials'] if a['accepted']]
                if not accepted:continue
                x=np.array(t['x']);dx=np.array(accepted[0]['x'])-x;s=np.linalg.norm(dx);g=np.array(t['g'])
                components=np.zeros((m,len(g)))
                for d in t['directions']:components[d['k']]=d['weight']*d['derivative']*np.array(d['u'])
                for k in range(m):
                    total_pairs+=1;r=by[t['iteration'],k];remaining=(g-components[k])/(1-weights[k]);norm=np.linalg.norm(remaining)
                    z=x if norm<=1e-8 else x-s*remaining/norm
                    if not np.allclose(z,r['alternative_x'],atol=1e-12,rtol=0):errors.append('alternative_point')
                    if np.max(abs(z))>1+1e-8:missing[k]+=1;continue
                    losses=np.array(r['reply_losses'])
                    if r['source']=='new':
                        if not np.allclose(losses,abs(np.array(r['responses'])),atol=1e-12,rtol=0):errors.append('new_reply_loss')
                        samples+=m
                    elif r['source']=='cache':
                        available=[(t['x'],t['current_losses'])]+[(d['point'],d['reply_losses']) for d in t['directions']]+[(a['x'],a['reply_losses']) for a in t['trials']]
                        available += [(q['alternative_x'],q['reply_losses']) for q in new['comparisons'] if q['iteration']==t['iteration'] and q['anchor_index']<k and q.get('reply_losses') is not None]
                        matches=[v for p,v in available if np.array_equal(np.asarray(p,dtype=np.float32),np.asarray(z,dtype=np.float32))]
                        if not matches or not any(np.allclose(losses,v,atol=1e-10,rtol=0) for v in matches):errors.append('cache_provenance')
                    elif r['source']=='no_motion':
                        if norm>1e-8 or not np.allclose(losses,t['current_losses'],atol=1e-12,rtol=0):errors.append('no_motion')
                    else:errors.append('unknown_source')
                    trial=sum(float(w)*float(v) for w,v in zip(weights,losses));j=min(t['loss'],trial)-accepted[0]['loss'];A[k]+=j
                    error=abs(j-r['J']);max_error=max(max_error,error)
                    if error>1e-10:errors.append('single_J')
            if not np.allclose(A,new['A_partial'],atol=1e-10,rtol=0):errors.append('A_sum')
            if not np.array_equal(missing,new['missing_steps']):errors.append('missing_steps')
            if samples!=new['counter']['samples'] or samples!=new['counter']['calls']:errors.append('samples')
            new_samples+=samples
            if new['evaluator_loaded'] or new['LOO_labels_loaded']:errors.append('information_leakage')
    rows=csvread(root/'participant_comparison.csv');labels={(r['key'],r['agent_id']):r for r in csvread(cfg['labels_path'])}
    if len(rows)!=96 or len({(r['key'],r['agent_id']) for r in rows})!=96:errors.append('row_coverage')
    for r in rows:
        if float(r['C_custom_full'])!=float(labels[r['key'],r['agent_id']]['C_custom_full']):errors.append('label_pairing')
    for t in csvread(root/'team_comparison.csv'):
        if t['A_spearman']:
            rs=[r for r in rows if r['key']==t['key']];corr=float(spearmanr([float(r['A']) for r in rs],[float(r['C_custom_full']) for r in rs]).statistic)
            if abs(corr-float(t['A_spearman']))>1e-12:errors.append('rank_recalculation')
    bench=read(root/'benchmark_results.json')
    for metric,agg in bench['aggregate'].items():
        a=sum(np.median([s[metric] for s in c['timings'] if s['method']=='anchor']) for c in bench['cases']);b=sum(np.median([s[metric] for s in c['timings'] if s['method']=='full_LOO']) for c in bench['cases'])
        if abs(a/b-agg['ratio'])>1e-12:errors.append('benchmark_aggregation')
    result=dict(accepted=not errors,errors=errors,n_cases=24,n_effects=96,n_anchor_step_pairs=total_pairs,
        new_attribution_samples=new_samples,max_J_rebuild_delta=max_error,
        source_and_inputs_checked=True,independent_arithmetic_checked=True,no_new_participant_queries=True,
        note='existing custom-only C labels reused after A lock; no new pooled evaluator loaded')
    save(root/'verification.json',result);print(result)
    if errors:raise ValueError('verification_failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();verify(a.root.resolve())
