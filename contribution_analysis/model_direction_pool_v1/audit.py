"""Post-run independent formula/receipt/legacy endpoint audit; no solver calls."""
import sys,json,math,copy,subprocess
from pathlib import Path
from fractions import Fraction as F
from contribution_analysis.custom_realized_gain_v1.schema import save,sha
from contribution_analysis.model_peer_pool_v1.reference import branch
W=Path(__file__).resolve().parents[2]

def independent_settle(scores,budget,n):
    tau=max(1e-8,n*1e-10);positive={o:max(v,0) if v>tau else 0 for o,v in scores.items()};den=sum(positive.values());amounts=dict.fromkeys(scores,0)
    if den:
        raw={o:budget*v/den for o,v in positive.items()};amounts={o:math.floor(v) for o,v in raw.items()}
        for o in sorted(raw,key=lambda o:(-(raw[o]-amounts[o]),o))[:budget-sum(amounts.values())]:amounts[o]+=1
    return amounts,budget-sum(amounts.values())

def main():
    root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=False);read=lambda p:json.loads(Path(p).read_text());summary=read(root/'summary.json');nvalues=0;legacy=0;oracle=0;rawchecks=0;maxerr=0.;maxbranch=0.;evalcalls=0
    for rel,h in read(root/'completion_receipt.json')['file_sha256'].items():assert sha(root/rel)==h
    historical=W/'contribution_analysis/model_peer_pool_experiment_v1_20261028_attempt_02'
    for rel,h in read(historical/'completion_receipt.json')['file_sha256'].items():assert sha(historical/rel)==h
    def value(p,x):
        nonlocal evalcalls
        evalcalls+=1;return abs(float(F(p['offset']))+sum(float(F(z))*a for z,a in zip(p['linear'],x))+sum(float(F(p['quadratic'][i][j]))*x[i]*x[j] for i in range(2) for j in range(2)))
    def close(a,b):
        nonlocal nvalues,maxerr
        error=abs(a-b);maxerr=max(maxerr,error);nvalues+=1;assert error<=1e-10+1e-9*abs(b),(a,b)
    allrows=[]
    for cc in summary['cases']:
        rec=read(root/'cases'/(cc['id']+'.json'));sc=read(root/'scores'/(cc['id']+'.json'))['score'];c=rec['fixture'];n=len(c['models']);w=[float(F(z)) for z in c['weights']];den=sum(w);w=[z/den for z in w]
        scales=[]
        for i,m in enumerate(c['models']):
            vals=sorted(value(m,[x,y]) for x in [-1,-.5,0,.5,1] for y in [-1,-.5,0,.5,1])
            def q(p):h=24*p;j=int(h);return vals[j]+(h-j)*(vals[j+1]-vals[j])
            scales.append(q(.9)-q(.1));close(scales[-1],rec['scale'][i])
        for ev in rec['events']:close(ev['loss'],value(c['models'][ev['model']],ev['x']));rawchecks+=1
        computed=[]
        for t,fe in enumerate(rec['full']):
            x=fe['pre']['S_current'];mode=fe['pre']['current_method'];g=[[] for i in range(n)]
            for anchor in c['anchors']:
                a=[float(F(z)) for z in anchor];d=math.dist(a,x)
                if d<1e-8:continue
                u=[(aa-xx)/d for aa,xx in zip(a,x)];h=d
                if mode!='secant':
                    h=fe['pre']['delta']
                    for xx,uu in zip(x,u):
                        if uu>0:h=min(h,(1-xx)/uu)
                        if uu<0:h=min(h,(-1-xx)/uu)
                    h=max(0,h)
                if h<=1e-12:continue
                point=a if mode=='secant' else [xx+h*uu for xx,uu in zip(x,u)]
                for i,m in enumerate(c['models']):g[i].append((value(m,point)-value(m,x))/(scales[i]*h))
            for rr in [r for r in sc['rows'] if r['round']==t]:
                i=rr['model'];U=min(math.sqrt(sum((a-b)**2 for a,b in zip(g[i],g[j]))/len(g[i])) for j in range(n) if j!=i);I=math.dist(rr['full'],rr['minus'])/fe['pre']['eta'];Ub=.1*U/(1+.1*U);N=Ub*I/(1+I)
                P=sum(w[j]*(value(c['models'][j],rr['minus'])-value(c['models'][j],rr['full']))/scales[j] for j in range(n) if j!=i)/sum(w[j] for j in range(n) if j!=i)
                for k,v in dict(U=U,I=I,Ub=Ub,N=N,P=P).items():close(v,rr[k])
                computed.append(dict(rr))
        for method,got in sc['methods'].items():
            beta=0 if method=='P_only' else float(method.rsplit('_',1)[1]);extra='Ub' if 'Ub' in method else 'N';amounts=dict.fromkeys(c['owners'],0);reserve=0
            for field,budget in [('P',round(500*(1-beta)))]+([(extra,round(500*beta))] if beta else []):
                scores={o:sum(r[field] for r in computed if r['owner']==o) for o in c['owners']};aa,rr=independent_settle(scores,budget,len(rec['full']));reserve+=rr
                for o in amounts:amounts[o]+=aa[o]
            assert amounts==got['amounts'] and reserve==got['reserve'],(c['id'],method)
        if c['split']=='historical_development':
            old=read(historical/'cases'/(c['id']+'.json'))
            for b in rec['branches']:
                run=old['owner_runs'][c['owners'][b['model']]];assert b['full']==run['lock']['full'] and b['minus']==run['lock']['minus'];legacy+=1
            for ids,endpoint in [(list(range(n)),rec['full'][0]['post']['S_current'])]+[([j for j in range(n) if j!=b['model']],b['minus']) for b in rec['branches']]:
                ref=branch(c,ids);err=math.dist(ref['x'],endpoint);maxbranch=max(maxbranch,err);assert err<1e-10;oracle+=1
        # Per-round diagnostics do not replicate final owner allocations.
        for rr in sc['rows']:
            D=value(c['reference'],rr['minus'])-value(c['reference'],rr['full']);allrows.append(dict(id=c['id'],split=c['split'],category=c['category'],owner=rr['owner'],round=rr['round'],D=D,P=rr['P'],Ub=rr['Ub'],N=rr['N']))
    tests=subprocess.run([sys.executable,'-m','unittest','contribution_analysis.model_direction_pool_v1.test_core','-v'],cwd=W,capture_output=True,text=True);save(out/'technical_tests.json',dict(returncode=tests.returncode,stdout=tests.stdout,stderr=tests.stderr));assert tests.returncode==0
    save(out/'round_diagnostics.json',allrows)
    save(out/'audit.json',dict(passed=True,artifact_receipts_valid=True,historical_receipt_sha256=sha(historical/'completion_receipt.json'),legacy_endpoints_exact=legacy,rational_1d_branches=oracle,max_rational_endpoint_error=maxbranch,formula_values_checked=nvalues,max_formula_error=maxerr,raw_reply_checks=rawchecks,independent_settlements=len(summary['cases'])*7,analytic_evaluations=evalcalls,solver_calls=0,diagnostics_postrun=True))
    save(out/'completion_receipt.json',dict(file_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}));print('AUDIT PASS',nvalues,legacy,oracle,evalcalls)
if __name__=='__main__':main()
