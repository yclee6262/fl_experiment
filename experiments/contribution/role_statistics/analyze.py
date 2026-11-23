"""Recompute fixed-team role statistics and cluster intervals without model calls."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.stats import spearmanr,kendalltau

REPS=200000
SEED=20261123
TOL=1e-10
ROLES=['A','M','J','AM']
JUDGES=['F','Fminus','E']
ROOT=Path(__file__).resolve().parents[3]


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x

def dump(p,x):Path(p).write_text(json.dumps(plain(x),indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(plain(rows))


def sign(x):return np.where(np.abs(x)>TOL,np.sign(x),0)
def rank_metrics(x,y):
    x=np.asarray(x);y=np.asarray(y)
    valid=len(x)>=3 and np.ptp(x)>TOL and np.ptp(y)>TOL
    out=dict(spearman=float(spearmanr(x,y).statistic) if valid else None,
             kendall=float(kendalltau(x,y).statistic) if valid else None,
             sign_agreement=float(np.mean(sign(x)==sign(y))))
    for k in (1,2):
        if not valid:out[f'top{k}']=None;continue
        a=set(np.flatnonzero(x>=np.sort(x)[-min(k,len(x))]-TOL))
        b=set(np.flatnonzero(y>=np.sort(y)[-min(k,len(y))]-TOL))
        out[f'top{k}']=len(a&b)/len(a|b)
    out['x_zero']=int(sum(sign(x)==0));out['y_zero']=int(sum(sign(y)==0))
    out['x_tied_pairs']=sum(abs(x[i]-x[j])<=TOL for i in range(len(x)) for j in range(i))
    out['y_tied_pairs']=sum(abs(y[i]-y[j])<=TOL for i in range(len(y)) for j in range(i))
    return out


def weights(n):return np.random.default_rng(SEED).multinomial(n,np.full(n,1/n),size=REPS)

def interval(values, w=None, family=1):
    valid=[v for v in values if v is not None and np.isfinite(v)]
    n=len(valid)
    if not n:return dict(n=0,missing=len(values),mean=None,ci95_low=None,ci95_high=None,family_low=None,family_high=None)
    w=weights(n) if w is None or w.shape[1]!=n else w
    bs=w@np.asarray(valid)/n
    lo,hi=np.quantile(bs,[.025,.975]);fl,fh=np.quantile(bs,[.05/(2*family),1-.05/(2*family)])
    return dict(n=n,missing=len(values)-n,mean=float(np.mean(valid)),ci95_low=float(lo),ci95_high=float(hi),family_low=float(fl),family_high=float(fh))


def ratio_interval(numerators,denominators,w=None):
    num=np.asarray(numerators);den=np.asarray(denominators)
    if (den<0).any() or (num<0).any() or (num>den).any():raise ValueError('Invalid event counts')
    w=weights(len(num)) if w is None else w
    bn=w@num;bd=w@den;keep=bd>0
    bs=bn[keep]/bd[keep]
    lo,hi=np.quantile(bs,[.025,.975]) if len(bs) else (None,None)
    return dict(numerator=int(num.sum()),denominator=int(den.sum()),teams=len(num),positive_denominator_teams=int(sum(den>0)),
                ratio=float(num.sum()/den.sum()) if den.sum() else None,
                ci95_low=None if lo is None else float(lo),ci95_high=None if hi is None else float(hi),zero_denominator_draws=int(sum(~keep)))


def allocate(scores,tau,budget,ids):
    scores=np.asarray(scores);positive=np.where(scores>tau,scores,0.)
    if not positive.sum():return [0]*len(scores),int(budget),np.r_[np.zeros(len(scores)),1.]
    share=positive/positive.sum();q=budget*share;pay=np.floor(q).astype(int)
    order=sorted(range(len(ids)),key=lambda i:(-(q[i]-pay[i]),str(ids[i])))
    for i in order[:budget-int(pay.sum())]:pay[i]+=1
    return pay.tolist(),0,np.r_[share,0.]


def unique(data,fields):
    seen=set()
    for row in data:
        k=tuple(row[f] for f in fields)
        if k in seen:raise ValueError(f'Duplicate {k}')
        seen.add(k)


def analyze(source,output):
    output.mkdir(parents=True,exist_ok=False)
    files=['participant_metrics.json','rank_metrics.json','judge_comparisons.json','payments.json','registration.json','audit.json','completion.json','source_lock.json','endpoint_lock.json','endpoints_locked.json']
    hashes={str((source/f).resolve()):sha(source/f) for f in files}
    assert read(source/'audit.json')['passed'] and read(source/'completion.json')['complete']
    assert sha(source/'endpoints_locked.json')==read(source/'endpoint_lock.json')['sha256']
    people=read(source/'participant_metrics.json');ranks=read(source/'rank_metrics.json');judges=read(source/'judge_comparisons.json');payments=read(source/'payments.json')
    unique(people,['key','id']);unique(ranks,['key','score','judge','reference','tol']);unique(judges,['key','role','other'])
    assert len(people)==80
    grouped=defaultdict(list)
    for p in people:grouped[p['key']].append(p)
    assert len(grouped)==20
    rankindex={(r['key'],r['score'],r['judge']):r for r in ranks if r['reference']=='E' and r['tol']==TOL}
    judgeindex={(r['key'],r['role'],r['other']):r for r in judges}
    payindex={(r['key'],r['id']):r for r in payments}
    team_metrics=[];team_flips=[];team_alloc=[];team_pay=[];owner_pay=[];comparisons=0
    for key,parts in sorted(grouped.items()):
        parts=sorted(parts,key=lambda p:p['i']);n=len(parts);scenario=parts[0]['scenario'];ids=[p['id'] for p in parts]
        assert n==(3 if scenario=='clean' else 5)
        assert sorted(p['i'] for p in parts)==list(range(n))
        for p in parts:
            for f in ['basic','anchor','model','total']:assert p[f]==payindex[key,p['id']][f]
        for role in ROLES:
            y=[p['D']['E'] for p in parts]
            for judge in JUDGES:
                x=[p['scores'][judge][role] for p in parts];metrics=rank_metrics(x,y)
                old=rankindex[key,role,judge]
                for f in ['spearman','kendall','sign_agreement','top1','top2']:
                    assert (metrics[f] is None and old[f] is None) or (metrics[f] is not None and old[f] is not None and abs(metrics[f]-old[f])<1e-12)
                    comparisons+=1
                team_metrics.append(dict(key=key,scenario=scenario,role=role,judge=judge,owners=n,**metrics))
            full=np.array([p['scores']['F'][role] for p in parts])
            for judge in ['Fminus','E']:
                other=np.array([p['scores'][judge][role] for p in parts]);num=int(sum((full>TOL)&(other< -TOL)));den=int(sum(full>TOL))
                old=judgeindex[key,role,judge];assert num==old['positive_F_other_negative'] and den==old['positive_F']
                tau=np.array([p['tauA'] for p in parts]) if role=='A' else TOL
                _,rf,sf=allocate(full,tau,500,ids);_,ro,so=allocate(other,tau,500,ids)
                tv=float(abs(sf-so).sum()/2);assert abs(tv-old['TV'])<1e-12 and bool(ro)==old['other_reserved']
                team_flips.append(dict(key=key,scenario=scenario,role=role,other=judge,numerator=num,denominator=den))
                team_alloc.append(dict(key=key,scenario=scenario,role=role,other=judge,TV=tv,other_reserved=bool(ro)))
        for judge in JUDGES:
            sa=np.array([p['scores'][judge]['A'] for p in parts]);sm=np.array([p['scores'][judge]['M'] for p in parts]);ta=np.array([p['tauA'] for p in parts]);tm=np.array([p['tauM'] for p in parts])
            basic=np.where((sa< -ta)&(sm< -tm),0,100);anchor,ar,_=allocate(sa,ta,500,ids);model,mr,_=allocate(sm,tm,500,ids)
            br=100*n-int(basic.sum())
            if judge=='F':
                assert basic.tolist()==[p['basic'] for p in parts] and anchor==[p['anchor'] for p in parts] and model==[p['model'] for p in parts]
            record=dict(key=key,scenario=scenario,judge=judge,owners=n,basic_budget=100*n,anchor_budget=500,model_budget=500,basic_paid=int(basic.sum()),anchor_paid=sum(anchor),model_paid=sum(model),basic_reserved=br,anchor_reserved=ar,model_reserved=mr,basic_canceled=int(sum(basic==0)),marked_owners=sum(p['poisoned'] for p in parts))
            assert record['basic_paid']+br==100*n and record['anchor_paid']+ar==500 and record['model_paid']+mr==500
            for flag,label in [(True,'marked'),(False,'unmarked')]:
                ix=[i for i,p in enumerate(parts) if p['poisoned']==flag]
                for field,values in [('basic',basic),('anchor',anchor),('model',model)]:record[label+'_'+field+'_paid']=int(sum(values[i] for i in ix))
            team_pay.append(record)
            for i,p in enumerate(parts):owner_pay.append(dict(key=key,scenario=scenario,judge=judge,id=p['id'],marked=p['poisoned'],basic=int(basic[i]),anchor=anchor[i],model=model[i],total=int(basic[i])+anchor[i]+model[i]))
    rank_summary=[];rank_changes=[];flip_summary=[];allocation_summary=[];pay_summary=[];pay_changes=[];cancellations=[]
    for sc in ['clean','controlled_poison']:
        keys=sorted(k for k,p in grouped.items() if p[0]['scenario']==sc);assert len(keys)==10;w=weights(10)
        for role in ROLES:
            for judge in JUDGES:
                rs=[r for r in team_metrics if r['scenario']==sc and r['role']==role and r['judge']==judge]
                for metric in ['spearman','kendall','sign_agreement','top1','top2']:
                    rank_summary.append(dict(scenario=sc,role=role,judge=judge,reference='pooled_whole_task_LOO',metric=metric,planned_teams=10,x_zero=sum(r['x_zero'] for r in rs),y_zero=sum(r['y_zero'] for r in rs),x_tied_pairs=sum(r['x_tied_pairs'] for r in rs),y_tied_pairs=sum(r['y_tied_pairs'] for r in rs),**interval([r[metric] for r in rs],w)))
            for metric in ['spearman','kendall','sign_agreement','top1','top2']:
                by={(r['key'],r['judge']):r for r in team_metrics if r['scenario']==sc and r['role']==role};delta=[]
                for k in keys:
                    f=by[k,'F'][metric];p=by[k,'Fminus'][metric];delta.append(None if f is None or p is None else p-f)
                rank_changes.append(dict(scenario=sc,role=role,metric=metric,contrast='peer_minus_full',family='exploratory_8_spearman_changes' if metric=='spearman' else 'descriptive',**interval(delta,w,family=8 if metric=='spearman' else 1)))
            for other in ['Fminus','E']:
                rs=[r for r in team_flips if r['scenario']==sc and r['role']==role and r['other']==other]
                flip_summary.append(dict(scenario=sc,role=role,other=other,**ratio_interval([r['numerator'] for r in rs],[r['denominator'] for r in rs],w)))
                rs=[r for r in team_alloc if r['scenario']==sc and r['role']==role and r['other']==other]
                allocation_summary.append(dict(scenario=sc,role=role,other=other,reserved_teams=sum(r['other_reserved'] for r in rs),**interval([r['TV'] for r in rs],w)))
        metrics=['basic_paid','anchor_paid','model_paid','basic_reserved','anchor_reserved','model_reserved','basic_canceled','marked_basic_paid','marked_anchor_paid','marked_model_paid','unmarked_basic_paid','unmarked_anchor_paid','unmarked_model_paid']
        for judge in JUDGES:
            for marked in [True,False]:
                nums=[];dens=[]
                for k in keys:
                    owners=[r for r in owner_pay if r['key']==k and r['judge']==judge and r['marked']==marked]
                    nums.append(sum(r['basic']==0 for r in owners));dens.append(len(owners))
                cancellations.append(dict(scenario=sc,judge=judge,marked=marked,**ratio_interval(nums,dens,w)))
            rs=[r for r in team_pay if r['scenario']==sc and r['judge']==judge]
            for metric in metrics:pay_summary.append(dict(scenario=sc,judge=judge,metric=metric,total=sum(r[metric] for r in rs),**interval([r[metric] for r in rs],w)))
        by={(r['key'],r['judge']):r for r in team_pay if r['scenario']==sc}
        for metric in metrics:
            d=[by[k,'Fminus'][metric]-by[k,'F'][metric] for k in keys]
            pay_changes.append(dict(scenario=sc,contrast='peer_minus_full_derived_shadow_policy',metric=metric,total_delta=sum(d),**interval(d,w)))
    outputs={'team_metrics':team_metrics,'team_sign_flips':team_flips,'team_allocations':team_alloc,'team_payments':team_pay,'owner_payments':owner_pay,'rank_summary':rank_summary,'rank_changes':rank_changes,'sign_flips':flip_summary,'allocation_summary':allocation_summary,'payment_summary':pay_summary,'payment_changes':pay_changes,'basic_cancellations':cancellations}
    for name,rs in outputs.items():csvout(output/(name+'.csv'),rs)
    audit=dict(passed=True,teams=20,owners=80,owner_rounds=read(source/'completion.json')['owner_rounds'],replicates=REPS,rng=SEED,recomputed_rank_metric_checks=comparisons,budget_checks=20*3*3,source_sha256=hashes,analysis_sha256=sha(Path(__file__)),design_sha256=sha(Path(__file__).with_name('design.md')),git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),new_model_queries=0,new_solver_calls=0,scope='Historical fixed-endpoint team-clustered reanalysis, not independent confirmation or across-start stability. Ordinary intervals descriptive; eight Spearman changes form separate exploratory family.')
    dump(output/'audit.json',audit);print(json.dumps(plain({k:v for k,v in audit.items() if k!='source_sha256'})))
    return audit

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();analyze(a.source,a.output_dir)
