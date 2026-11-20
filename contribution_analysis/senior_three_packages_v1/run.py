"""Locked-endpoint three-ruler attribution review and shadow settlement."""
import sys,json,hashlib,time,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from contribution_analysis.model_pool_v_cm_validation_v1.run import CA,W,FROZEN,plain
from contribution_analysis.anchor_pool_end_to_end_v1.settlement import settle
import numpy as np
from scipy.stats import spearmanr,kendalltau
OUT=CA/'senior_three_packages_20261109_attempt_01'
TOL=1e-10
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(plain(x),ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(x):return np.asarray(x,np.float64).tobytes().hex()
def sign(x,tol=TOL):return np.where(np.abs(x)>tol,np.sign(x),0)
def vector_scores(full,minus,a,i):
    delta=np.asarray(minus)-full;self=float(a[i]*delta[i]);peer=float(sum(a[j]*delta[j] for j in range(len(a)) if j!=i));F=float(a@delta)
    assert abs(F-self-peer)<1e-10
    return dict(F=F,peer=peer,self=self,Fminus=peer/(1-a[i]))
def metrics(x,y,tol=TOL):
    x=np.array(x);y=np.array(y);valid=len(x)>=3 and np.ptp(x)>tol and np.ptp(y)>tol
    out=dict(spearman=float(spearmanr(x,y).statistic) if valid else None,kendall=float(kendalltau(x,y).statistic) if valid else None,sign_agreement=float(np.mean(sign(x,tol)==sign(y,tol))))
    for k in [1,2]:
        if not valid:out[f'top{k}']=None;continue
        sx=set(np.flatnonzero(x>=np.sort(x)[-min(k,len(x))]-tol));sy=set(np.flatnonzero(y>=np.sort(y)[-min(k,len(y))]-tol));out[f'top{k}']=len(sx&sy)/len(sx|sy)
    return out
def share(x,tau):
    w=np.where(np.array(x)>tau,x,0.);p=w/w.sum() if w.sum() else w
    return np.r_[p,max(0.,1-float(p.sum()))]
def integer_pay(p,budget,ids):
    # Final slot is reserve; largest remainder includes only a funded pool.
    if p[-1]>.5:return [0]*len(ids),budget
    quotas=np.array(p[:-1])*budget;v=np.floor(quotas).astype(int)
    order=sorted(range(len(ids)),key=lambda i:(-(quotas[i]-v[i]),str(ids[i])))
    for i in order[:budget-int(v.sum())]:v[i]+=1
    assert sum(v)==budget
    return v.tolist(),0
def main():
    import torch
    from models import PretrainedModel
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    OUT.mkdir(exist_ok=False);(OUT/'records').mkdir();(OUT/'source_snapshot').mkdir()
    cases=read(CA/'combined_contribution_layer2_experiment_v1_20261021/cases.json')['development'];paths=[Path(__file__),CA/'senior_three_packages_v1/README.md',CA/'anchor_pool_end_to_end_v1/settlement.py']+list(FROZEN.glob('*.py'))
    for c in cases:
        k=c['key'];paths.extend([CA/f'method5_cost_experiment_20261030_attempt_04/records/{k}_New.json',CA/f'method5_cost_experiment_20261030_attempt_04/records/{k}_LOO.json',CA/f'combined_contribution_layer2_experiment_v1_20261021/development/{k}.json',CA/f'model_pool_combinations_20261105_attempt_01/records/{k}_J.json',Path(c['models']),Path(c['evaluator']),Path(c['models']).parent/'metadata.json'])
    locks={str(p):sha(p) for p in paths};save(OUT/'source_lock.json',locks)
    for p in paths[:3]+list(FROZEN.glob('*.py')):(OUT/'source_snapshot'/'__'.join(p.relative_to(W).parts)).write_bytes(p.read_bytes())
    save(OUT/'registration.json',dict(cases=cases,historical_exploration=True,roles=['A','M','J','AM'],rulers=['F','Fminus','E'],tauM=TOL,basic_fee=100,anchor_budget=500,model_budget=500,pooled_cap=20000))
    start=time.perf_counter();locked=[];maxerr=0.
    for c in cases:
        k=c['key'];a=np.array(c['alphas']);a/=a.sum();old=read(CA/f'method5_cost_experiment_20261030_attempt_04/records/{k}_New.json');loo=read(CA/f'method5_cost_experiment_20261030_attempt_04/records/{k}_LOO.json');joint=read(CA/f'combined_contribution_layer2_experiment_v1_20261021/development/{k}.json');jc=read(CA/f'model_pool_combinations_20261105_attempt_01/records/{k}_J.json')
        assert old['base']['solution']==loo['base']['solution'] and old['base']['solution']==jc['base']['solution']
        assert len(old['rows'])==len(joint['groups'])==len(jc['rows'])
        profiles=read(Path(c['models']).parent/'metadata.json')['agent_data_profile'];poison={p['agent_id']:p['is_poisoned'] for p in profiles};rows=[]
        for r,g,j in zip(old['rows'],joint['groups'],jc['rows']):
            assert (r['t'],r['i'])==(g['t'],g['i'])==(j['t'],j['i']) and str(c['ids'][r['i']])==r['owner']
            assert np.allclose(r['full'],g['full']['post']['S_current'],rtol=0,atol=1e-12)
            assert np.allclose(r['full_losses'],g['full_losses'],rtol=0,atol=1e-12)
            out=dict(t=r['t'],i=r['i'],id=c['ids'][r['i']],full=r['full'],full_losses=r['full_losses'],rho=r['rho'],N=r['N'],roles={})
            for role,point,losses,ref in [('A',r['minusA'],r['minusA_losses'],r['C_A']),('M',r['minusM'],r['minusM_losses'],r['C_M']),('J',g['roles']['joint']['endpoint'],g['roles']['joint']['common_losses'],j['J'])]:
                v=vector_scores(np.array(r['full_losses']),losses,a,r['i']);maxerr=max(maxerr,abs(v['F']-ref));assert abs(v['F']-ref)<1e-10
                out['roles'][role]=dict(x=point,losses=losses,score=v)
            out['aligned_residual']=g['residual'];out['current_vs_aligned_A_difference']=r['C_A']-g['a2'];rows.append(out)
        ds=[]
        for r in loo['rows']:
            v=vector_scores(np.array(r['full_losses']),r['minus_losses'],a,r['i']);assert abs(v['F']-r['D'])<1e-10;ds.append(dict(i=r['i'],full=r['full'],minus=r['minus'],full_losses=r['full_losses'],minus_losses=r['minus_losses'],score=v))
        locked.append(dict(case=c,rows=rows,LOO=ds,poison=[poison[i] for i in c['ids']]))
    save(OUT/'endpoints_locked.json',locked);endpoint_sha=sha(OUT/'endpoints_locked.json');save(OUT/'endpoint_lock.json',dict(sha256=endpoint_sha,participant_calls=0,solver_calls=0,pooled_loaded=False))
    # No score is fed back to any solver: all search endpoints are fixed above.
    pooled_count=0;metrics_rows=[];participants=[];interactions=[];judge_comparisons=[];payments=[];pooled_tables=[]
    for team in locked:
        c=team['case'];k=c['key'];n=len(c['ids']);a=np.array(c['alphas']);a/=a.sum();table={}
        net=PretrainedModel(2);net.load_state_dict(torch.load(c['evaluator'],map_location='cpu',weights_only=True)['state_dict']);net.eval()
        def E(x):
            nonlocal pooled_count
            kk=key(x)
            if kk not in table:
                assert (pooled_count+1)*2<=20000
                with torch.no_grad():v=abs(float(net(torch.tensor(x,dtype=torch.float32)[None,:]).item()))
                table[kk]=dict(x=x,value=v);pooled_count+=1
            return table[kk]['value']
        for r in team['rows']:
            for v in r['roles'].values():v['score']['E']=E(v['x'])-E(r['full'])
            for judge in ['F','Fminus','E']:
                vals={role:r['roles'][role]['score'][judge] for role in ['A','M','J']};interactions.append(dict(key=k,scenario=c['scenario'],t=r['t'],i=r['i'],judge=judge,A=vals['A'],M=vals['M'],J=vals['J'],residual=vals['J']-vals['A']-vals['M'],aligned_F_residual=r['aligned_residual'] if judge=='F' else None))
        for d in team['LOO']:d['score']['E']=E(d['minus'])-E(d['full'])
        sumv={judge:{role:np.array([sum(r['roles'][role]['score'][judge] for r in team['rows'] if r['i']==i) for i in range(n)]) for role in ['A','M','J']} for judge in ['F','Fminus','E']}
        for judge in sumv:sumv[judge]['AM']=sumv[judge]['A']+sumv[judge]['M']
        tauA=np.array([max(1e-8,sum(r['rho'] for r in team['rows'] if r['i']==i)) for i in range(n)]);T=len(team['rows'])//n
        for judge in sumv:
            for role,x in sumv[judge].items():
                for reference in sorted({judge,'E'}):
                    y=[d['score'][reference] for d in team['LOO']]
                    for tol in [1e-10,1e-4]:metrics_rows.append(dict(key=k,scenario=c['scenario'],score=role,judge=judge,reference=reference,tol=tol,**metrics(x,y,tol)))
        for role in ['A','M','J','AM']:
            for judge in ['Fminus','E']:
                x=sumv['F'][role];y=sumv[judge][role];sx=share(x,tauA if role=='A' else TOL);sy=share(y,tauA if role=='A' else TOL)
                judge_comparisons.append(dict(key=k,scenario=c['scenario'],role=role,other=judge,**metrics(x,y),positive_F=int(sum(x>TOL)),positive_F_other_negative=int(sum((x>TOL)&(y< -TOL))),TV=float(abs(sx-sy).sum()/2),other_reserved=bool(sy[-1]>.5)))
        SA=sumv['F']['A'];SM=sumv['F']['M'];pb=np.where((SA< -tauA)&(SM< -TOL),0,100);pa,ra=integer_pay(share(SA,tauA),500,c['ids']);pm,rm=integer_pay(share(SM,TOL),500,c['ids'])
        owners=[str(i) for i in c['ids']];anchor=settle(owners,list(range(T)),[dict(owner=str(r['id']),t=r['t'],C=str(r['roles']['A']['score']['F']),rho=str(r['rho'])) for r in team['rows']])
        assert pa==[anchor['payments'][o] for o in owners]
        legacy_tau=max(1e-8,T*1e-10);sm_legacy=share(SM,legacy_tau);assert np.allclose(sm_legacy,share(SM,TOL),rtol=0,atol=1e-10)
        N=np.array([sum(r['N'] for r in team['rows'] if r['i']==i) for i in range(n)]);cn=.8*share(SM,legacy_tau)[:-1]+.2*share(np.where(SM>legacy_tau,N,0),legacy_tau)[:-1]
        for i in range(n):
            p=dict(key=k,scenario=c['scenario'],id=c['ids'][i],i=i,poisoned=team['poison'][i],alpha=float(a[i]),SA=float(SA[i]),SM=float(SM[i]),tauA=float(tauA[i]),tauM=TOL,basic=int(pb[i]),anchor=pa[i],model=pm[i],total=int(pb[i])+pa[i]+pm[i],legacy_CMN_share=float(cn[i]),legacy_N_paid=bool(SM[i]>legacy_tau and N[i]>legacy_tau))
            payments.append(p);participants.append(dict(**p,scores={judge:{role:float(vals[i]) for role,vals in roles.items()} for judge,roles in sumv.items()},D={judge:team['LOO'][i]['score'][judge] for judge in ['F','Fminus','E']},alpha_within_team_identifiable=bool(np.ptp(a)>TOL)))
        assert sum(pb)+(100*n-sum(pb))==100*n and sum(pa)+ra==500 and sum(pm)+rm==500
        team['allocation']=dict(assumed_complete_service=True,basic_budget=100*n,anchor_budget=500,model_budget=500,basic_reserved=int(100*n-sum(pb)),anchor_reserved=ra,model_reserved=rm,actual_payment=False)
        reload=PretrainedModel(2);reload.load_state_dict(torch.load(c['evaluator'],map_location='cpu',weights_only=True)['state_dict']);reload.eval()
        for v in table.values():
            with torch.no_grad():again=abs(float(reload(torch.tensor(v['x'],dtype=torch.float32)[None,:]).item()))
            assert again==v['value']
        pooled_tables.append(dict(key=k,values=list(table.values())));save(OUT/'records'/f'{k}.json',team)
        print(k,len(table),flush=True)
    save(OUT/'participant_metrics.json',participants);save(OUT/'rank_metrics.json',metrics_rows);save(OUT/'judge_comparisons.json',judge_comparisons);save(OUT/'interactions.json',interactions);save(OUT/'payments.json',payments);save(OUT/'pooled_values.json',pooled_tables)
    assert sha(OUT/'endpoints_locked.json')==endpoint_sha
    for p,h in locks.items():assert sha(Path(p))==h
    save(OUT/'completion.json',dict(complete=True,teams=len(cases),participants=len(participants),owner_rounds=sum(len(t['rows']) for t in locked),participant_model_calls=0,solver_calls=0,pooled_unique=pooled_count,pooled_with_recheck=2*pooled_count,max_score_reconstruction_error=maxerr,seconds=time.perf_counter()-start))
if __name__=='__main__':main()
