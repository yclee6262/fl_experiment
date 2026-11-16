"""Independent archived raw-loss, allocation and scale audit; no production imports."""
import json,hashlib,math,sys,time
from pathlib import Path
from fractions import Fraction as F
W=Path(__file__).resolve().parents[1]
SOURCE=W/'contribution_analysis/model_direction_pool_experiment_v1_20261029_attempt_02'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert abs(a-b)<=1e-10+1e-9*abs(b),(a,b)
def settle(vals,budget,T):
    weights={o:F(str(v)) if v>max(1e-8,T*1e-10) else F(0) for o,v in vals.items()};den=sum(weights.values());out=dict.fromkeys(weights,0)
    if den:
        raw={o:budget*v/den for o,v in weights.items()};out={o:int(v) for o,v in raw.items()}
        for o in sorted(out,key=lambda o:(-(raw[o]-out[o]),o))[:budget-sum(out.values())]:out[o]+=1
    return out,budget-sum(out.values())
def verify(root):
    receipt=read(root/'completion_receipt.json');files=receipt.get('output_sha256',receipt.get('file_sha256'))
    for rel,v in files.items():assert sha(root/rel)==v
    return len(files)
def run(root,out):
    out.mkdir(exist_ok=False);start=time.perf_counter();nfiles=verify(root);verify(SOURCE)
    counts=dict(cases=0,owner_rounds=0,settlements=0,scale_variants=0,scale_C_values=0);maxerror=0.;examples=[]
    for path in sorted((root/'cases').glob('*.json')):
        got=read(path);src=read(SOURCE/'cases'/path.name);c=src['fixture'];owners=c['owners'];T=len(src['full']);lookup={}
        for ev in src['events']:
            k=ev['model'],tuple(ev['x']);v=ev['loss']
            if k in lookup:assert lookup[k]==v
            lookup[k]=v
        w=[float(F(v)) for v in c['weights']];w=[v/sum(w) for v in w]
        assert len(got['rows'])==len(owners)*T
        for r in got['rows']:
            lf=[lookup[i,tuple(r['full'])] for i in range(len(w))];lm=[lookup[i,tuple(r['minus'])] for i in range(len(w))]
            assert lf==r['loss_full'] and lm==r['loss_minus']
            C=math.fsum(a*b for a,b in zip(w,lm))-math.fsum(a*b for a,b in zip(w,lf));close(C,r['C']);maxerror=max(maxerror,abs(C-r['C']))
            d=src['directions'][r['round']];keys={}
            for k,geom in enumerate(d['directions']):
                key=json.dumps([geom['point'],geom['h'],geom['unit']]);keys[key]=k
            ix=list(keys.values());g=d['g'];i=r['model']
            U=min(math.sqrt(sum((g[i][k]-g[j][k])**2 for k in ix)/len(ix)) for j in range(len(w)) if j!=i)
            I=math.dist(r['full'],r['minus'])/src['full'][r['round']]['pre']['eta'];Ub=.1*U/(1+.1*U);Ib=I/(1+I)
            for key,v in dict(U=U,I=I,Ub=Ub,Ib=Ib,N=Ub*Ib).items():close(r[key],v)
            counts['owner_rounds']+=1
        for name,m in got['methods'].items():
            extra=None if name=='C_only' else name.split('_')[0][2:];beta=0 if extra is None else float(name.rsplit('_',1)[1]);amount=dict.fromkeys(owners,0);reserve=0
            for key,B in [('C',round(500*(1-beta)))]+([] if extra is None else [(extra,round(500*beta))]):
                vals={o:sum((F(str(r[key])) for r in got['rows'] if r['owner']==o),F(0)) for o in owners}
                a,b=settle(vals,B,T)
                for o in owners:amount[o]+=a[o]
                reserve+=b
            assert amount==m['amounts'] and reserve==m['reserve'];counts['settlements']+=1
        for control in read(root/'scale_controls'/path.name):
            fac=F(str(control['factor']));j=control['model'];multipliers=[fac if j is None or k==j else F(1) for k in range(len(w))]
            original_w=[F(v) for v in c['weights']];original_w=[v/sum(original_w) for v in original_w]
            ws=[v/multipliers[k] if control['kind']=='single_compensated' else v for k,v in enumerate(original_w)];Z=sum(ws);ws=[v/Z for v in ws]
            for a,b in zip(ws,control['v']):close(float(a),b)
            totals=dict.fromkeys(owners,0.)
            for old,new in zip(got['rows'],control['transformed_endpoints']):
                lf=[F(str(v))*multipliers[k] for k,v in enumerate(old['loss_full'])];lm=[F(str(v))*multipliers[k] for k,v in enumerate(old['loss_minus'])]
                value=sum(v*(b-a) for v,a,b in zip(ws,lf,lm));close(float(value),new['C'])
                for a,b in zip(lf,new['loss_full']):close(float(a),b)
                for a,b in zip(lm,new['loss_minus']):close(float(a),b)
                counts['scale_C_values']+=1
            # Use recorded per-row decimal C, exactly as public settlement input.
            totals={o:sum((F(str(r['C'])) for r in control['transformed_endpoints'] if r['owner']==o),F(0)) for o in owners}
            a,b=settle(totals,500,T);assert a==control['allocation']['amounts'] and b==control['allocation']['reserve']
            counts['scale_variants']+=1
        for o,z in got['owners'].items():
            rr=[r for r in got['rows'] if r['owner']==o];cs=[r['C'] for r in rr];close(sum(cs),z['net']);close(sum(max(v,0) for v in cs),z['Gplus']);close(sum(max(-v,0) for v in cs),z['Gminus'])
            if z['classification']=='negative' or (z['classification']=='positive' and got['methods']['C_only']['amounts'][o]==0):examples.append(dict(case=path.stem,owner=o,net=z['net'],baseline=got['methods']['C_only']['amounts'][o],N=got['methods']['C+N_0.2']['amounts'][o]))
        counts['cases']+=1
    counts.update(files_verified=nfiles,max_direct_F_error=maxerror,examples=examples,model_calls=0,solver_calls=0,reference_calls=0,seconds=time.perf_counter()-start)
    (out/'audit.json').write_text(json.dumps(counts,indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(input=str(root),input_receipt_sha256=sha(root/'completion_receipt.json'),source_sha256=sha(Path(__file__))),indent=2)+'\n');print(counts)
if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
