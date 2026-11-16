import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from contribution_analysis.senior_remaining_v1.run import *
from scipy.stats import rankdata,spearmanr

def main():
    teams=read(OUT/'summary.json');locks=read(OUT/'source_lock.json');maxerr=0.;ranks=[];alpha=[];proxy=[]
    for team in teams:
        k=team['key'];b=read(OUT/'records'/f'{k}_base.json');n=b['n'];old=read(CA/f'method5_cost_experiment_20261030_attempt_04/records/{k}_New.json');assert len(b['rows'])==len(old['rows']) and b['base']==old['base'];weights=np.array(b['weights']);sums=np.zeros((n,2,2));tags=dict(key=k,group=team['group'],scenario=team['scenario'])
        for r in b['rows']:
            for role in [0,1]:
                d=np.array(r['deltas'][role]);v=[sum(float(w)*float(x) for w,x in zip(weights,d)),sum(float(weights[j])*float(d[j]) for j in range(n) if j!=r['i'])/(1-weights[r['i']])]
                maxerr=max(maxerr,float(np.max(abs(np.array(v)-r['exact'][role]))));sums[r['i'],role]+=v
        assert np.max(abs(sums-np.array(b['sums'])))<1e-8
        for w in team['weights']:
            a=np.array(w['alpha'])
            for mode in ['fixed','path']:
                for role in [0,1]:
                    val=np.array(w[mode])[:,role];corr=float(spearmanr(a,val).statistic) if np.ptp(val)>1e-10 else None
                    alpha.append(dict(**tags,mode=mode,role=['A','M'][role],rho=corr))
        ex=np.array([r['exact'] for r in b['rows']]);pr=np.array([r['proxy'] for r in b['rows']]);res=ex-pr;T=len(ex)//n
        for role in [0,1]:
            rr=abs(res[:,role,0]);proxy.append(dict(**tags,role=['A','M'][role],raw_proxy_RMSE=float(np.sqrt(np.mean(res[:,role,0]**2))),largest_residual_share=float(max(rr)/rr.sum()) if rr.sum() else 0,deficient_rounds=sum(r['rank']<2 for r in b['rows'])//n))
        for p in [.05,.1,.2,.3,.4,.5]:
            sparse=read(OUT/'records'/f'{k}_sparse{p}.json');rngmask=np.random.default_rng(202611091000+int(k[-10:])+int(p*100));assert np.array_equal(rngmask.random((len(ex),2))<p,np.array([r['sampled'] for r in sparse['rows']]));rebuilt=np.zeros((n,2,2))
            for r,full in zip(sparse['rows'],b['rows']):
                for role in [0,1]:
                    if r['sampled'][role]:assert np.max(abs(np.array(r['exact'][role])-full['exact'][role]))<1e-8
                    proxyval=np.array(r['proxy'][role]);rebuilt[r['i'],role]+=proxyval+((np.array(r['exact'][role])-proxyval)/p if r['sampled'][role] else 0.)
            assert np.max(abs(rebuilt-np.array(sparse['sums'])))<1e-8
            # Independent conditional expectation of each row, algebraic audit.
            assert np.max(abs(pr+(p*res/p)-ex))<1e-10
            rng=np.random.default_rng(20261109000+int(k[-10:])+int(p*100));ind=rng.random((1000,len(ex),2,1))<p
            est=(pr[None]+ind*res[None]/p).reshape(1000,T,n,2,2).sum(1)
            for ruler in [0,1]:
                for role in [0,1]:
                    vals=est[:,:,role,ruler];truth=sums[:,role,ruler];rv=rankdata(vals,axis=1);rt=rankdata(truth);center=rv-rv.mean(1,keepdims=True);ct=rt-rt.mean();den=np.sqrt((center**2).sum(1)*(ct**2).sum());ok=den>0;corr=np.divide(center@ct,den,out=np.zeros(1000),where=ok)
                    a=vals>=vals.max(1,keepdims=True)-1e-10;btop=truth>=truth.max()-1e-10;inter=(a&btop).sum(1);union=(a|btop).sum(1)
                    ranks.append(dict(**tags,rate=p,ruler=['F','Fminus'][ruler],role=['A','M'][role],spearman=float(corr[ok].mean()) if ok.any() else None,rankable=int(ok.sum()),top1=float((inter/union).mean())))
    assert maxerr<1e-8 and all(sha(Path(p))==h for p,h in locks.items())
    # Small exact finite-population fixture checks HT mean/variance independently.
    import itertools
    z=np.array([.2,-.4,.8]);p=.2;mean=second=0.
    for bits in itertools.product([0,1],repeat=3):
        prob=np.prod([p if bit else 1-p for bit in bits]);value=sum(bit*v/p for bit,v in zip(bits,z));mean+=prob*value;second+=prob*value*value
    assert abs(mean-z.sum())<1e-12 and abs(second-mean**2-sum(z*z)*(1-p)/p)<1e-12
    save(OUT/'rank_supplement.json',ranks);save(OUT/'alpha_correlations.json',alpha);save(OUT/'proxy_diagnostics.json',proxy)
    save(OUT/'audit.json',dict(passed=True,max_reconstruction_error=maxerr,source_unchanged=True,teams=len(teams),sparse_reconstructions=120,exact_HT_fixture=True,audit_sha=sha(Path(__file__)),limitations='Conditional on fixed trajectories; no external quality, new-model confirmation, identity security or valid-normal-CI guarantee.'))
if __name__=='__main__':main()
