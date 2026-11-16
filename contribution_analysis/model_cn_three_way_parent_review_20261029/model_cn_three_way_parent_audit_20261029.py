"""Independent three-way saved-record and ledger audit without production imports."""
import json,math,hashlib,time
from pathlib import Path
from fractions import Fraction as F
W=Path(__file__).resolve().parents[1]
ROOT=W/'contribution_analysis/model_cn_three_way_20261029_attempt_01'
OLD=W/'contribution_analysis/model_direction_pool_experiment_v1_20261029_attempt_02'
OUT=W/'contribution_analysis/model_cn_three_way_parent_review_20261029'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert abs(a-b)<=1e-10+1e-9*abs(b),(a,b)
def settle(values,B,T):
    tau=max(F(1,10**8),T*F(1,10**10));v={o:x if x>tau else F(0) for o,x in values.items()};s=sum(v.values());out=dict.fromkeys(v,0)
    if s:
        raw={o:B*x/s for o,x in v.items()};out={o:int(x) for o,x in raw.items()}
        for o in sorted(out,key=lambda o:(-(raw[o]-out[o]),o))[:B-sum(out.values())]:out[o]+=1
    return out,B-sum(out.values())
def main():
    OUT.mkdir(exist_ok=False);start=time.perf_counter();receipt=read(ROOT/'completion_receipt.json')
    for p,h in receipt['file_sha256'].items():assert sha(ROOT/p)==h
    count=dict(files=len(receipt['file_sha256']),cases=0,solver_cases=0,synthetic_cases=0,owner_rounds=0,settlements=0,new_full_rounds=0,new_minus_rounds=0,new_control_rounds=0)
    examples=[];maxC=0.;cost={}
    for path in sorted((ROOT/'cases').glob('*.json')):
        c=read(path);T=c['T'];owners=list(c['owners']);tau=max(F(1,10**8),T*F(1,10**10));rows=c['rows']
        assert len(rows)==len(owners)*T
        if c['scope']!='synthetic_ledger':
            isnew=c['scope']!='historical_78';rec=read((ROOT/'solver_records' if isnew else OLD/'cases')/path.name)
            lookup={}
            for e in rec['events']:
                k=e['model'],tuple(e['x'])
                if k in lookup:assert lookup[k]==e['loss']
                lookup[k]=e['loss']
            w=[F(x) for x in rec['fixture']['weights']];w=[v/sum(w) for v in w]
            for row in rows:
                lf=[lookup[j,tuple(row['full'])] for j in range(len(w))];lm=[lookup[j,tuple(row['minus'])] for j in range(len(w))]
                val=math.fsum(float(v)*x for v,x in zip(w,lm))-math.fsum(float(v)*x for v,x in zip(w,lf));close(val,row['C']);maxC=max(maxC,abs(val-row['C']))
                d=rec['directions'][row['round']];geometry={json.dumps([a['point'],a['h'],a['unit']]):j for j,a in enumerate(d['directions'])};ix=list(geometry.values());g=d['g'];i=row['model']
                U=min(math.sqrt(sum((g[i][k]-g[j][k])**2 for k in ix)/len(ix)) for j in range(len(w)) if i!=j)
                I=math.dist(row['full'],row['minus'])/rec['full'][row['round']]['pre']['eta'];N=.1*U/(1+.1*U)*I/(1+I)
                for key,value in dict(U=U,I=I,N=N).items():close(value,row[key])
            count['solver_cases']+=1
            if isnew:
                count['new_full_rounds']+=T;count['new_control_rounds']+=T;count['new_minus_rounds']+=len(rec['branches'])
                for k,v in rec['cost'].items():
                    z=cost.setdefault(k,dict(requests=0,evaluations=0,hits=0))
                    for key in z:z[key]+=v[key]
        else:count['synthetic_cases']+=1
        S={o:sum((F(str(r['C'])) for r in rows if r['owner']==o),F(0)) for o in owners}
        N={o:sum((F(str(r['N'])) for r in rows if r['owner']==o),F(0)) for o in owners}
        for method,got in c['methods'].items():
            b=F(0) if method=='C_only' else F(method.rsplit('_',1)[1]);filtered={o:v if not 'positive' in method or S[o]>tau else F(0) for o,v in N.items()}
            a,r=settle(S,int(500*(1-b)),T);supp,res=settle(filtered,int(500*b),T) if b else (dict.fromkeys(owners,0),0)
            assert got['amounts']=={o:a[o]+supp[o] for o in owners} and got['reserve']==r+res,(path.name,method)
            if 'positive' in method:assert all(got['amounts'][o]==0 for o in owners if S[o]<=tau)
            count['settlements']+=1
        for o,v in S.items():close(float(v),c['owners'][o]['net'])
        if c['scope']=='synthetic_ledger' or c['id'].startswith('unique_harmful'):
            examples.append(dict(id=c['id'],scope=c['scope'],net={o:float(v) for o,v in S.items()},N={o:float(v) for o,v in N.items()},amounts={m:c['methods'][m]['amounts'] for m in ['C_only','C+N_0.2','C+N_positive_0.2']}))
        # Independent closed-form oracle for new harmful single-round family.
        if c['id'] in ['unique_harmful_0_1','unique_harmful_1_1']:
            a=next(r for r in rows if r['owner']=='owner_a');assert a['full']==[.1,0.] and a['minus']==[-.1,0.];close(a['C'],-.2);assert a['N']>0
        count['cases']+=1;count['owner_rounds']+=len(rows)
    count.update(max_direct_F_error=maxC,new_cost=cost,examples=examples,new_solver_calls=0,new_model_calls=0,new_reference_calls=0,seconds=time.perf_counter()-start)
    (OUT/'audit.json').write_text(json.dumps(count,indent=2)+'\n')
    (OUT/'provenance.json').write_text(json.dumps(dict(input_receipt_sha256=sha(ROOT/'completion_receipt.json'),source_sha256=sha(Path(__file__)),scope='independent raw losses, frozen directions, exact settlement and two closed-form first-round endpoint checks; not general solver oracle'),indent=2)+'\n')
    print({k:v for k,v in count.items() if k not in ['examples','new_cost']})
if __name__=='__main__':main()
