"""Independent frozen-endpoint extreme-share audit; no production imports."""
import json,hashlib,math,time
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal,localcontext
W=Path(__file__).resolve().parents[1]
ROOT=W/'contribution_analysis/model_cn_extreme_20261030_attempt_01'
OUT=W/'contribution_analysis/model_cn_extreme_parent_review_20261030'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert abs(a-b)<=1e-10+1e-9*abs(b),(a,b)
def settle(v,B,tau):
    w={o:x if x>tau else F(0) for o,x in v.items()};s=sum(w.values());out=dict.fromkeys(w,0);q=None
    if s:
        q={o:x/s for o,x in w.items()};raw={o:B*x for o,x in q.items()};out={o:int(x) for o,x in raw.items()}
        for o in sorted(out,key=lambda o:(-(raw[o]-out[o]),o))[:B-sum(out.values())]:out[o]+=1
    return out,B-sum(out.values()),q
def main():
    OUT.mkdir(exist_ok=False);start=time.perf_counter();receipt=read(ROOT/'completion_receipt.json')
    for rel,h in receipt['file_sha256'].items():assert sha(ROOT/rel)==h
    result=dict(files=len(receipt['file_sha256']),cases=0,owner_rounds=0,settlements=0,relative=[],threshold=[],linear_oracle=[],max_direct_F_error=0.,cost={})
    for p in sorted((ROOT/'cases').glob('*.json')):
        c=read(p);src=read(W/c['source']);n=len(src['fixture']['models']);owners=list(c['owners']);T=c['T'];tau=max(F(1,10**8),T*F(1,10**10));w=list(map(F,src['fixture']['weights']));w=[v/sum(w) for v in w];lookup={}
        for e in src['events']:
            k=(e['model'],tuple(e['x']))
            if k in lookup:assert lookup[k]==e['loss']
            lookup[k]=e['loss']
        for row in c['rows']:
            a=[lookup[j,tuple(row['full'])] for j in range(n)];b=[lookup[j,tuple(row['minus'])] for j in range(n)]
            value=math.fsum(float(v)*x for v,x in zip(w,b))-math.fsum(float(v)*x for v,x in zip(w,a));close(value,row['C']);result['max_direct_F_error']=max(result['max_direct_F_error'],abs(value-row['C']))
            d=src['directions'][row['round']];g=d['g'];i=row['model'];keys={json.dumps([z['point'],z['h'],z['unit']]):j for j,z in enumerate(d['directions'])};ix=list(keys.values())
            U=min(math.sqrt(sum((g[i][k]-g[j][k])**2 for k in ix)/len(ix)) for j in range(n) if j!=i);I=math.dist(row['full'],row['minus'])/src['full'][row['round']]['pre']['eta'];N=.1*U/(1+.1*U)*I/(1+I)
            for key,v in dict(U=U,I=I,N=N).items():close(v,row[key])
        S={o:sum((F(str(r['C'])) for r in c['rows'] if r['owner']==o),F(0)) for o in owners};R={o:sum((F(str(r['N'])) for r in c['rows'] if r['owner']==o),F(0)) if S[o]>tau else F(0) for o in owners}
        for label,beta in [('C_only',F(0)),('0.1',F(1,10)),('0.2',F(1,5)),('0.3',F(3,10))]:
            a,ra,qc=settle(S,int(500*(1-beta)),tau);b,rb,qn=settle(R,int(500*beta),tau) if beta else (dict.fromkeys(S,0),0,None);method=c['methods'][label]
            assert method['amounts']=={o:a[o]+b[o] for o in owners} and method['reserve']==ra+rb
            if label=='0.2':
                for o,z in c['owners'].items():
                    assert (qc is None)==(z['qC'] is None);assert (qn is None)==(z['qN'] is None)
                    if qc:close(float(qc[o]),z['qC'])
                    if qn:close(float(qn[o]),z['qN'])
                    relative=bool(qc and qn and qc[o]<=F(1,100) and qn[o]>=F(9,10) and b[o]>0)
                    threshold=bool(qn and 1<S[o]/tau<=10 and qn[o]>=F(9,10) and b[o]>0)
                    assert relative==z['E_relative'] and threshold==z['E_threshold']
                    for key,flag in [('relative',relative),('threshold',threshold)]:
                        if flag:result[key].append(dict(case=c['id'],scope=c['scope'],owner=o,C=float(S[o]),qC=float(qc[o]),qN=float(qn[o]),C_units=a[o],N_units=b[o]))
                    assert z['other_C_signs']=={k:sum((v>tau if k=='positive' else v< -tau if k=='negative' else abs(v)<=tau) for who,v in S.items() if who!=o) for k in ['positive','negative','zero']}
            result['settlements']+=1
        if c['scope']!='historical_94':
            for stage,v in src['cost'].items():
                acc=result['cost'].setdefault(stage,dict(requests=0,evaluations=0,hits=0))
                for k in acc:acc[k]+=v[k]
            if c['fixture']['category'] in ['exact_overlap','opposing_unique'] and c['fixture']['iterations']==1:
                with localcontext() as ctx:
                    ctx.prec=60;delta=Decimal(c['fixture']['delta']);lx=1-delta if c['fixture']['category']=='exact_overlap' else 1-2*delta;ly=delta;norm=(lx*lx+ly*ly).sqrt();expected=Decimal('.1')*(norm-lx)
                    row=next(r for r in c['rows'] if r['owner']=='c');err=abs(float(expected)-row['C']);assert err<1e-10
                    target=[float(-Decimal('.1')*lx/norm),float(-Decimal('.1')*ly/norm)]
                    assert math.dist(target,row['full'])<1e-12 and row['minus']==[-.1,0.]
                    result['linear_oracle'].append(dict(case=c['id'],C_expected=float(expected),C_error=err,endpoint_pass=True))
        result['cases']+=1;result['owner_rounds']+=len(c['rows'])
    result.update(model_calls=0,solver_calls=0,reference_calls=0,seconds=time.perf_counter()-start)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n');(OUT/'provenance.json').write_text(json.dumps(dict(input_receipt_sha256=sha(ROOT/'completion_receipt.json'),source_sha256=sha(Path(__file__)),scope='frozen raw loss/features, independent Fraction allocation and Decimal linear first-round oracle'),indent=2)+'\n')
    print({k:(len(v) if isinstance(v,list) else v) for k,v in result.items() if k!='cost'})
if __name__=='__main__':main()
