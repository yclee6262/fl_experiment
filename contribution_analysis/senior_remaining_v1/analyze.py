import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from contribution_analysis.senior_remaining_v1.run import *
from scipy.stats import spearmanr

def shp(x,tau=1e-10):
    w=np.where(x>tau,x,0.);den=w.sum(axis=-1,keepdims=True);p=np.divide(w,den,out=np.zeros_like(w),where=den>0)
    return np.concatenate([p,(den<=0).astype(float)],axis=-1)
def tv(x,y):return float(np.abs(x-y).sum()/2)
def avg(rows,k):return float(np.mean([r[k] for r in rows]))
def main():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    src=read(OUT/'summary.json');cases=read(OUT/'registration.json')['cases'];rows=[];costs=[];num=[];weight=[];policies=[];sampling=[];syb=[];devcheck=[]
    start=time.perf_counter()
    for team,c in zip(src,cases):
        k=c['key'];print(k,flush=True);b=read(OUT/'records'/f'{k}_base.json');n=b['n'];S=np.array(b['sums'])[:,:,0];tags=dict(key=k,scenario=c['scenario'],group=team['group'])
        # Clean exact reference arm has no diagnostics populating the cache.
        s=load_server(c);ref=execute(c,s);save(OUT/'records'/f'{k}_exact_cost.json',ref)
        assert np.max(np.abs(np.array(ref['sums'])-np.array(b['sums'])))<1e-10
        # Exact duplicate detection collapses the appended identity back to original inputs.
        assert np.array_equal(np.array(s.I_list),np.array(c['candidates']))
        restored=execute(c,load_server(c));assert np.array_equal(ref['sums'],restored['sums'])
        devcheck.append(dict(key=k,collapse_exact=True))
        for rate in team['sparse']:
            costs.append(dict(**tags,rate=rate['rate'],sparse=rate['calls'],baseline=rate['baseline'],added=rate['calls']-rate['baseline'],exact=ref['calls'],LOO=team['LOO_calls'],LOO_added=team['LOO_calls']-team['LOO_baseline']))
        for role in [0,1]:
            for variant in ['unnorm']+(['renorm'] if role==0 else []):
                val=np.zeros(n)
                for r in b['rows']:val[r['i']]+=r['diag'][('A' if role==0 else 'M')+'_'+variant][0]
                rows.append(dict(**tags,role=['A','M'][role],variant=variant,**metrics(S[:,role],val),TV=tv(shp(S[:,role]),shp(val)),max_score_delta=float(np.max(abs(S[:,role]-val)))))
        bs=alloc(S)
        for z,st in enumerate(team['stable']):
            ss=np.array(st['S'])[:,:,0];pp=alloc(ss)
            rows.append(dict(**tags,role='both',variant='start',start=z,A_TV=tv(bs['A'],pp['A']),M_TV=tv(bs['M'],pp['M']),basic_flip=float(np.mean((np.array(bs['basic'])>0)!=(np.array(pp['basic'])>0))),sign_agreement=float(np.mean(np.sign(S)==np.sign(ss)))))
        for w in team['weights']:
            i=w['owner'];fixed=np.array(w['fixed']);path=np.array(w['path']);pfix=alloc(fixed);pnew=alloc(path)
            for role in [0,1]:weight.append(dict(**tags,owner=i,role=['A','M'][role],fixed_TV=tv(bs[['A','M'][role]],pfix[['A','M'][role]]),path_TV=tv(bs[['A','M'][role]],pnew[['A','M'][role]]),fixed_owner_score_change=float(fixed[i,role]-S[i,role]),path_owner_score_change=float(path[i,role]-S[i,role])))
        for tau in [1e-12,1e-10,1e-8,1e-6,1e-4]:
            pp=alloc(S,tau);policies.append(dict(**tags,tau=tau,A_TV=tv(bs['A'],pp['A']),M_TV=tv(bs['M'],pp['M']),basic_flip=float(np.mean((bs['basic']>0)!=(pp['basic']>0))),reserve=pp['reserve']))
        for ratio in [[.2,.4,.4],[.4,.3,.3],[.6,.2,.2]]:
            budget=100*n+1000;bb=budget*ratio[0]/n;basic=np.where((S[:,0]<-1e-10)&(S[:,1]<-1e-10),0.,bb);tot=basic+budget*ratio[1]*bs['A'][:-1]+budget*ratio[2]*bs['M'][:-1];reserve=budget-tot.sum();assert reserve>=-1e-8
            policies.append(dict(**tags,ratio=ratio,total_budget=budget,reserve=float(reserve),total=tot))
        rng=np.random.default_rng(202611099000+c['seed'])
        for eps in [1e-10,1e-8,1e-6]:
            pert=S[None,:,:]+rng.uniform(-eps,eps,(1000,n,2));cancel=(pert[:,:,0]<-1e-10)&(pert[:,:,1]<-1e-10);orig=(S[:,0]<-1e-10)&(S[:,1]<-1e-10)
            num.append(dict(**tags,eps=eps,flip=float(np.mean(cancel!=orig)),nearzero=int(np.sum(np.abs(S)<=eps))))
        for sy in team['sybil']:syb.append(dict(**tags,**sy,gain_per_identity=sy['per_identity']-sy['original'],gain_fixed_budget=sy['fixed_budget']-sy['original']))
        ex=np.array([r['exact'] for r in b['rows']]);pr=np.array([r['proxy'] for r in b['rows']]);R=ex-pr;T=len(ex)//n
        for p in [.05,.1,.2,.3,.4,.5]:
            rng=np.random.default_rng(20261109000+c['seed']+int(p*100));ind=rng.random((1000,len(ex),2,1))<p
            est=(pr[None]+ind*R[None]/p).reshape(1000,T,n,2,2).sum(1)
            var=(ind*R[None]**2*(1-p)/p**2).reshape(1000,T,n,2,2).sum(1);se=np.sqrt(var);truth=np.array(b['sums']);lo=est-1.96*se;hi=est+1.96*se
            for ruler in [0,1]:
                true=truth[:,:,ruler];E=est[:,:,:,ruler];lower=lo[:,:,:,ruler];upper=hi[:,:,:,ruler];bc=(true[:,0]<-1e-10)&(true[:,1]<-1e-10);CIcancel=(upper[:,:,0]<0)&(upper[:,:,1]<0);pointcancel=(E[:,:,0]<-1e-10)&(E[:,:,1]<-1e-10)
                for role in [0,1]:
                    exactshare=shp(true[:,role]);ep=shp(E[:,:,role]);scale=max(float(np.sum(np.abs(true[:,role]))),1e-8)
                    sampling.append(dict(**tags,rate=p,ruler=['F','Fminus'][ruler],role=['A','M'][role],RMSE=float(np.sqrt(np.mean((E[:,:,role]-true[:,role])**2))),relative_l1=float(np.mean(np.sum(abs(E[:,:,role]-true[:,role]),axis=1))/scale),mean_bias=float(np.mean(E[:,:,role]-true[:,role])),CI_coverage=float(np.mean((lower[:,:,role]<=true[:,role])&(upper[:,:,role]>=true[:,role]))),zero_se=float(np.mean(se[:,:,:,ruler][:,:,role]==0)),TV=float(np.mean(np.sum(abs(ep-exactshare),axis=1)/2)),sign_agreement=float(np.mean(np.sign(E[:,:,role])==np.sign(true[:,role]))),basic_point_error=float(np.mean(pointcancel!=bc)),basic_CI_error=float(np.mean(CIcancel!=bc)),false_cancel=float(np.mean(CIcancel&~bc)),missed_cancel=float(np.mean(~CIcancel&bc))))
    save(OUT/'sensitivity.json',rows);save(OUT/'weights.json',weight);save(OUT/'policies.json',policies);save(OUT/'numerical.json',num);save(OUT/'sybil.json',syb);save(OUT/'costs.json',costs);save(OUT/'sampling.json',sampling);save(OUT/'duplicate_controls.json',devcheck)
    report={}
    for group in ['all','validation_existing_models']:
        filt=lambda rr:[r for r in rr if group=='all' or r['group']==group]
        gg=filt(rows);ww=filt(weight);ss=filt(syb);nn=filt(num);pp=filt(policies)
        report[group]=dict(teams=20 if group=='all' else 16,normalization={role:{v:dict(mean_TV=avg([r for r in gg if r['role']==role and r['variant']==v],'TV'),max_delta=max(r['max_score_delta'] for r in gg if r['role']==role and r['variant']==v)) for v in (['unnorm','renorm'] if role=='A' else ['unnorm'])} for role in ['A','M']},starts={k:avg([r for r in gg if r['variant']=='start'],k) for k in ['A_TV','M_TV','basic_flip','sign_agreement']},weights={role:{k:avg([r for r in ww if r['role']==role],k) for k in ['fixed_TV','path_TV','fixed_owner_score_change','path_owner_score_change']} for role in ['A','M']},sybil={pol:dict(tested=len([r for r in ss if r['policy']==pol]),profitable_fixed=sum(r['gain_fixed_budget']>1e-8 for r in ss if r['policy']==pol),profitable_per_identity=sum(r['gain_per_identity']>1e-8 for r in ss if r['policy']==pol),mean_gain_fixed=avg([r for r in ss if r['policy']==pol],'gain_fixed_budget'),max_gain_fixed=max(r['gain_fixed_budget'] for r in ss if r['policy']==pol)) for pol in ['split_weight','equal_identity']},nearzero={str(e):avg([r for r in nn if r['eps']==e],'flip') for e in [1e-10,1e-8,1e-6]},threshold={str(t):{k:avg([r for r in pp if r.get('tau')==t],k) for k in ['A_TV','M_TV','basic_flip','reserve']} for t in [1e-12,1e-10,1e-8,1e-6,1e-4]},sampling=[],cost=[])
        for p in [.05,.1,.2,.3,.4,.5]:
            cc=[r for r in filt(costs) if r['rate']==p];tot={k:sum(r[k] for r in cc) for k in ['baseline','added','sparse','exact','LOO','LOO_added']};tot.update(rate=p,saving_vs_LOO=1-tot['sparse']/tot['LOO']);report[group]['cost'].append(tot)
            for ruler in ['F','Fminus']:
                for role in ['A','M']:
                    rr=[r for r in filt(sampling) if r['rate']==p and r['ruler']==ruler and r['role']==role];report[group]['sampling'].append(dict(rate=p,ruler=ruler,role=role,**{k:avg(rr,k) for k in ['RMSE','relative_l1','mean_bias','CI_coverage','zero_se','TV','sign_agreement','basic_point_error','basic_CI_error','false_cancel','missed_cancel']}))
    save(OUT/'analysis.json',report);save(OUT/'analysis_completion.json',dict(complete=True,extra_physical=Q.total,seconds=time.perf_counter()-start,source_sha=sha(Path(__file__))))
if __name__=='__main__':main()
