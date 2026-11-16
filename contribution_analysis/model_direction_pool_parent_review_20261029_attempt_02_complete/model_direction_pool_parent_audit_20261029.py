"""Independent saved-output audit; imports no experiment scoring implementation."""
import json, math, hashlib, sys
from pathlib import Path
from fractions import Fraction as F

def read(p): return json.loads(p.read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def num(v): return float(F(str(v)))
def value(p,x):
    return num(p['offset'])+sum(num(a)*b for a,b in zip(p['linear'],x))+sum(num(p['quadratic'][i][j])*x[i]*x[j] for i in range(len(x)) for j in range(len(x)))
def norm(x): return math.sqrt(sum(a*a for a in x))
def distance(a,b): return norm([x-y for x,y in zip(a,b)])
def close(a,b):
    assert math.isfinite(a) and math.isfinite(b)
    assert abs(a-b)<=1e-8+1e-8*abs(b),(a,b)
def quantile(v,p):
    a=sorted(v); h=(len(a)-1)*p; j=int(h)
    return a[j]+(a[min(j+1,len(a)-1)]-a[j])*(h-j)
def settle(vals,budget,rounds):
    w={o:F(str(v)) if v>max(1e-8,rounds*1e-10) else F(0) for o,v in vals.items()}
    if not sum(w.values()): return dict.fromkeys(w,0),budget
    raw={o:budget*v/sum(w.values()) for o,v in w.items()}; out={o:int(v) for o,v in raw.items()}
    for o in sorted(w,key=lambda o:(-(raw[o]-out[o]),o))[:budget-sum(out.values())]:out[o]+=1
    return out,0

def audit(root):
    receipt=read(root/'completion_receipt.json')
    for rel,h in receipt['file_sha256'].items():assert digest(root/rel)==h,rel
    result=dict(files_verified=len(receipt['file_sha256']),cases=0,owner_rounds=0,events=0,settlements=0,full_rounds=0,mutation_checks=0)
    tables={}; examples=[]; costs={}; extra_scale_evaluations=0; reference_evaluations=0
    has_endpoint_scale_control='scale_control_evaluations' in read(root/'summary.json')
    for path in sorted((root/'cases').glob('*.json')):
        rec=read(path); c=rec['fixture']; models=c['models']; owners=c['owners']; n=len(models)
        sc=read(root/'scores'/path.name)['score']; diag=read(root/'diagnostics'/path.name)['rows']
        assert read(root/'scores'/path.name)['case_sha256']==digest(path)
        assert read(root/'diagnostics'/path.name)['scores_sha256']==digest(root/'scores'/path.name)
        scales=[]
        for i,m in enumerate(models):
            probe=[abs(value(m,[x,y])) for x in [-1,-.5,0,.5,1] for y in [-1,-.5,0,.5,1]]
            scales.append(quantile(probe,.9)-quantile(probe,.1));close(scales[-1],rec['scale'][i])
        for e in rec['events']:close(abs(value(models[e['model']],e['x'])),e['loss']); result['events']+=1
        nr=len(rec['full']); assert len(sc['rows'])==nr*n
        expected={(t,o) for t in range(nr) for o in owners};assert {(r['round'],r['owner']) for r in sc['rows']}==expected
        computed=[]
        for t,f in enumerate(rec['full']):
            pre=f['pre']; x=pre['S_current']; ds=[]; g=[[] for _ in models]
            for a in c['anchors']:
                a=list(map(num,a)); d=distance(a,x)
                if d<1e-8:continue
                u=[(aa-xx)/d for aa,xx in zip(a,x)];h=d
                if pre['current_method']!='secant':
                    h=pre['delta']
                    for xx,uu in zip(x,u):
                        if uu>0:h=min(h,(1-xx)/uu)
                        elif uu<0:h=min(h,(-1-xx)/uu)
                    h=max(0,h)
                if h<=1e-12:continue
                point=a if pre['current_method']=='secant' else [xx+h*uu for xx,uu in zip(x,u)]
                ds.append(point)
                for i,m in enumerate(models):g[i].append((abs(value(m,point))-abs(value(m,x)))/(scales[i]*h))
            assert len(ds)==len(rec['directions'][t]['directions']) and ds
            for i in range(n):
                for a,b in zip(g[i],rec['directions'][t]['g'][i]):close(a,b)
                r=next(r for r in sc['rows'] if r['round']==t and r['model']==i)
                U=min(distance(g[i],g[j])/math.sqrt(len(ds)) for j in range(n) if j!=i)
                I=distance(r['full'],r['minus'])/pre['eta']; Ub=.1*U/(1+.1*U); N=Ub*I/(1+I)
                w=list(map(num,c['weights'])); P=sum(w[j]*(abs(value(models[j],r['minus']))-abs(value(models[j],r['full'])))/scales[j] for j in range(n) if j!=i)/sum(w[j] for j in range(n) if j!=i)
                for key,v in dict(U=U,I=I,Ub=Ub,N=N,P=P).items():close(r[key],v)
                D=value(c['reference'],r['minus'])-value(c['reference'],r['full'])
                computed.append(dict(owner=owners[i],P=P,Ub=Ub,N=N,D=D))
            extra_scale_evaluations+=3*n*n*(1+len(ds))
            if has_endpoint_scale_control:extra_scale_evaluations+=6*n*n*n
        reference_evaluations+=2*n*nr
        for o in owners:close(sum(r['D'] for r in computed if r['owner']==o),next(d['D'] for d in diag if d['owner']==o))
        # Settlement uses archived decimal scores, avoiding near-tie changes from independent arithmetic.
        for name,meth in sc['methods'].items():
            beta=0 if name=='P_only' else float(name.rsplit('_',1)[1]); extra='Ub' if 'Ub' in name else 'N'
            a,ra=settle({o:sum(r['P'] for r in sc['rows'] if r['owner']==o) for o in owners},round(500*(1-beta)),nr)
            b,rb=settle({o:sum(r[extra] for r in sc['rows'] if r['owner']==o) for o in owners},round(500*beta),nr)
            assert meth['amounts']=={o:a[o]+b[o] for o in owners} and meth['reserve']==ra+rb
            result['settlements']+=1
            for scope in ['all',c['split']]:
                z=tables.setdefault(scope,{}).setdefault(name,dict(beneficial=0,harmful=0,zero=0,beneficial_paid=0,harmful_paid=0,zero_paid=0,harmful_units=0,zero_units=0,allocated_units=0,reserve=0))
                z['reserve']+=meth['reserve']
                for d in diag:
                    cls='beneficial' if d['D']>1e-10 else 'harmful' if d['D']< -1e-10 else 'zero'; pay=meth['amounts'][d['owner']]
                    z[cls]+=1;z[cls+'_paid']+=int(pay>0);z['allocated_units']+=pay
                    if cls!='beneficial':z[cls+'_units']+=pay
        if c['split']=='new_heldout':examples.append(dict(id=c['id'],rounds=nr,diagnostics=diag))
        for st,v in rec['cost'].items():
            z=costs.setdefault(st,dict(requests=0,evaluations=0,hits=0))
            for k in z:z[k]+=v[k]
        result['cases']+=1;result['owner_rounds']+=n*nr;result['full_rounds']+=nr
    result.update(tables=tables,costs=costs,run_scale_extra_model_evaluations=extra_scale_evaluations,reference_endpoint_evaluations=reference_evaluations,heldout_examples=examples)
    return result

if __name__=='__main__':
    root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=False)
    r=audit(root);(out/'independent_audit.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(input=str(root),receipt_sha256=digest(root/'completion_receipt.json'),audit_source_sha256=digest(Path(__file__)),scope='post-run independent formula, raw reply, settlement and receipt audit; not independent solver replay'),indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ['tables','costs','heldout_examples']},indent=2))
