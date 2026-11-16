"""Seed-paired inference; no selector or optimization code."""
import numpy as np
from scipy.stats import norm

def intervals(matrix, replicates=200000, rng_seed=2026101501, family=11):
    a=np.asarray(matrix,float)
    if a.ndim!=2 or not len(a) or not np.isfinite(a).all():raise ValueError('Finite complete seed matrix required')
    rng=np.random.default_rng(rng_seed);n,k=a.shape
    samples=np.empty((replicates,k))
    for start in range(0,replicates,1000):
        size=min(1000,replicates-start);w=rng.multinomial(n,np.full(n,1/n),size=size)
        samples[start:start+size]=w@a/n
    return {'ordinary95':np.quantile(samples,[.025,.975],axis=0).T.tolist(),
            'simultaneous':np.quantile(samples,[.05/(2*family),1-.05/(2*family)],axis=0).T.tolist()}

def stable_advantage(simultaneous_ci,batch_means,batch_cis,complete,failures_not_higher):
    return bool(complete and failures_not_higher and simultaneous_ci is not None and simultaneous_ci[1]<0 and len(batch_means)==3
                and all(x is not None and x<0 for x in batch_means)
                and sum(c is not None and c[1]<0 for c in batch_cis)>=2)

def power_screen(residual_matrix, n_values=(90,180,360), delta=.001, simulations=5000, family=11):
    """Approximate empirical-residual simulation with normal-CI surrogates for bootstrap gates."""
    a=np.asarray(residual_matrix,float)
    if a.ndim!=2 or len(a)<3 or not np.isfinite(a).all():raise ValueError('Invalid development residuals')
    a=a-a.mean(axis=0);rng=np.random.default_rng(2026101502);z=norm.ppf(1-.05/(2*family));out=[]
    for n in n_values:
        if n%3:raise ValueError('Three equal batches required')
        bn=n//3;bmeans=[];bvars=[]
        for _ in range(3):
            counts=rng.multinomial(bn,np.full(len(a),1/len(a)),size=simulations)
            mean=counts@a/bn;var=(counts@(a*a)-bn*mean*mean)/(bn-1)
            bmeans.append(mean-delta);bvars.append(np.maximum(var,0))
        means=np.stack(bmeans,axis=1);variances=np.stack(bvars,axis=1)
        overall=means.mean(axis=1)
        overall_var=((bn-1)*variances.sum(axis=1)+bn*((means-overall[:,None,:])**2).sum(axis=1))/(n-1)
        simultaneous=overall+z*np.sqrt(overall_var/n)<0
        batch95=means+norm.ppf(.975)*np.sqrt(variances/bn)<0
        stable=simultaneous & (means<0).all(axis=1) & (batch95.sum(axis=1)>=2)
        out.append({'n':n,'approx_stable_power_by_contrast':stable.mean(axis=0).tolist(),
                    'approx_family_ci_detection_by_contrast':simultaneous.mean(axis=0).tolist()})
    return out
