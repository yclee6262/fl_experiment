"""Saved-data summaries and independent accounting/endpoint checks."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from contribution_analysis.senior_three_packages_v1.run import OUT,CA,save,read,sha,metrics,plain
import numpy as np
def mean(xs):
    xs=[v for v in xs if v is not None];return None if not xs else float(np.mean(xs))
def main():
    assert read(OUT/'completion.json')['complete'];pp=read(OUT/'participant_metrics.json');ranks=read(OUT/'rank_metrics.json');jj=read(OUT/'judge_comparisons.json');ints=read(OUT/'interactions.json');payments=read(OUT/'payments.json');groups={}
    for sc in ['clean','controlled_poison']:
        people=[p for p in pp if p['scenario']==sc];keys=sorted({p['key'] for p in people});ruler={};loo={}
        for role in ['A','M','J','AM']:
            ruler[role]={}
            for judge in ['Fminus','E']:
                zs=[z for z in jj if z['scenario']==sc and z['role']==role and z['other']==judge]
                ruler[role][judge]=dict(F_positive=sum(z['positive_F'] for z in zs),F_positive_other_negative=sum(z['positive_F_other_negative'] for z in zs),mean_TV=mean([z['TV'] for z in zs]),mean_spearman=mean([z['spearman'] for z in zs]),mean_sign_agreement=mean([z['sign_agreement'] for z in zs]),reserved=sum(z['other_reserved'] for z in zs))
            loo[role]={}
            for judge in ['F','Fminus','E']:
                zs=[z for z in ranks if z['scenario']==sc and z['score']==role and z['judge']==judge and z['reference']=='E' and z['tol']==1e-10]
                loo[role][judge]={m:mean([z[m] for z in zs]) for m in ['spearman','kendall','sign_agreement','top1','top2']};loo[role][judge]['valid_rank_teams']=sum(z['spearman'] is not None for z in zs)
        inter={}
        for judge in ['F','Fminus','E']:
            ratios=[];mismatch=0;count=0
            for k in keys:
                z=[z for z in ints if z['key']==k and z['judge']==judge];den=sum(abs(v['A'])+abs(v['M'])+abs(v['J']) for v in z)
                if den:ratios.append(sum(abs(v['residual']) for v in z)/den)
                mismatch+=sum(np.sign(v['A']+v['M'])!=np.sign(v['J']) for v in z if abs(v['A']+v['M'])>1e-10 and abs(v['J'])>1e-10);count+=len(z)
            inter[judge]=dict(mean_normalized_abs_residual=mean(ratios),nonzero_opposite=mismatch,owner_rounds=count)
        poison=[p for p in people if p['poisoned']];paid={}
        for label,sub in [('poisoned',poison),('not_poisoned',[p for p in people if not p['poisoned']])]:
            paid[label]=dict(people=len(sub),basic_received=sum(p['basic']>0 for p in sub),anchor_received=sum(p['anchor']>0 for p in sub),model_received=sum(p['model']>0 for p in sub),
                basic_total=sum(p['basic'] for p in sub),anchor_total=sum(p['anchor'] for p in sub),model_total=sum(p['model'] for p in sub),legacy_N_qualified=sum(p['legacy_N_paid'] for p in sub),
                LOO_E_negative=sum(p['D']['E']< -1e-10 for p in sub),LOO_E_negative_anchor_paid=sum(p['D']['E']< -1e-10 and p['anchor']>0 for p in sub),LOO_E_negative_model_paid=sum(p['D']['E']< -1e-10 and p['model']>0 for p in sub))
        groups[sc]=dict(teams=len(keys),people=len(people),ruler_comparison=ruler,LOO_pooled=loo,interaction=inter,payments=paid,alpha_identifiable_teams=sum(any(p['alpha_within_team_identifiable'] for p in people if p['key']==k) for k in keys))
    # Independent data reconstruction, not importing the score/settlement functions.
    checked=0;rounds=0;err=0.;pooled={t['key']:{tuple(v['x']):v['value'] for v in t['values']} for t in read(OUT/'pooled_values.json')}
    for c in read(OUT/'registration.json')['cases']:
        team=read(OUT/'records'/f"{c['key']}.json");a=np.array(c['alphas']);a/=a.sum();n=len(a);table=pooled[c['key']]
        parts=[p for p in pp if p['key']==c['key']];assert len(parts)==n
        for r in team['rows']:
            i=r['i'];lf=np.array(r['full_losses'])
            for role,v in r['roles'].items():
                delta=np.array(v['losses'])-lf;s=v['score'];er=max(abs(sum(float(a[j])*float(delta[j]) for j in range(n))-s['F']),abs(float(a[i]*delta[i])-s['self']),abs(sum(a[j]*delta[j] for j in range(n) if j!=i)/(1-a[i])-s['Fminus']),abs(table[tuple(v['x'])]-table[tuple(r['full'])]-s['E']))
                err=max(err,er);assert er<1e-10
            rounds+=1
        for p in parts:
            own=[r for r in team['rows'] if r['i']==p['i']]
            assert abs(sum(r['roles']['A']['score']['F'] for r in own)-p['SA'])<1e-10
            assert abs(sum(r['roles']['M']['score']['F'] for r in own)-p['SM'])<1e-10
            assert p['basic']==(0 if p['SA']< -p['tauA'] and p['SM']< -p['tauM'] else 100)
        for field,reserve,budget in [('basic','basic_reserved',100*n),('anchor','anchor_reserved',500),('model','model_reserved',500)]:
            assert all(p[field]>=0 and int(p[field])==p[field] for p in parts)
            assert sum(p[field] for p in parts)+team['allocation'][reserve]==budget
            if field!='basic':
                score='SA' if field=='anchor' else 'SM';tau='tauA' if field=='anchor' else 'tauM';weights=[p[score] if p[score]>p[tau] else 0 for p in parts];total=sum(weights)
                if total:
                    quotas=[budget*w/total for w in weights];base=[int(q) for q in quotas];order=sorted(range(n),key=lambda i:(-(quotas[i]-base[i]),str(parts[i]['id'])))
                    for i in order[:budget-sum(base)]:base[i]+=1
                    assert base==[p[field] for p in parts]
        checked+=1
    for path,h in read(OUT/'source_lock.json').items():assert sha(Path(path))==h
    assert sha(OUT/'endpoints_locked.json')==read(OUT/'endpoint_lock.json')['sha256']
    save(OUT/'analysis.json',groups);save(OUT/'audit.json',dict(passed=True,teams=checked,owner_rounds=rounds,max_reconstruction_error=err,source_unchanged=True,endpoint_lock=True,budget_conserved=True,shadow_only=True))
    save(OUT/'analysis_source_lock.json',{str(Path(__file__)):sha(Path(__file__))})
    # Archive the previous chat-only V/CM peer comparison for all 124 teams.
    vr=CA/'model_pool_v_cm_validation_20261107_attempt_02';reg=read(vr/'registration.json');peer=[]
    for phase in ['historical','known_analytic80','new_confirmation']:
        for judge in ['total','peer_normalized']:
            opposite=0;tvs=[];empty=[0,0];people=0
            for c in reg['known']+reg['fresh']:
                if c['phase']!=phase:continue
                r=read(vr/'records'/f"{c['key']}_30.json");v=np.array([[sum(z[m][judge] for z in r['rows'] if z['i']==i) for m in ['V','CM']] for i in range(r['n'])]);people+=r['n'];opposite+=sum((v[:,0]>1e-10)&(v[:,1]< -1e-10))+sum((v[:,0]< -1e-10)&(v[:,1]>1e-10));ss=[]
                for i in range(2):
                    w=np.where(v[:,i]>1e-10,v[:,i],0.);s=w/w.sum() if w.sum() else w;ss.append(np.r_[s,1-s.sum()]);empty[i]+=int(w.sum()==0)
                tvs.append(abs(ss[0]-ss[1]).sum()/2)
            peer.append(dict(phase=phase,judge=judge,people=people,opposite=int(opposite),mean_TV_with_reserve=mean(tvs),no_distribution=empty))
    save(OUT/'V_CM_peer_supplement.json',peer)
    print(json.dumps(plain(groups),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
