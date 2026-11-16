"""Audit all accepted/rejected strata and start summaries independently."""
from verify import *
def q(rows):
    nz=[r for r in rows if sign(r['j'])];both=[r for r in nz if sign(r['j_E'])];large=[r for r in rows if abs(r['j'])>.001];zero=[r for r in rows if not sign(r['j'])]
    v=dict(sign=avg([sign(r['j'])==sign(r['j_E']) for r in nz]),reverse=avg([sign(r['j'])==-sign(r['j_E']) for r in nz]),external_zero=avg([not sign(r['j_E']) for r in nz]),
        both_nonzero_sign=avg([sign(r['j'])==sign(r['j_E']) for r in both]),positive_sign=avg([sign(r['j_E'])==1 for r in nz if sign(r['j'])==1]),
        negative_sign=avg([sign(r['j_E'])==-1 for r in nz if sign(r['j'])==-1]),zero_external_nonzero=avg([sign(r['j_E'])!=0 for r in zero]),
        severe_reverse=avg([abs(r['j_E'])>.001 and r['j']*r['j_E']<0 for r in large]))
    rounds=sorted(set(r['t'] for r in rows));rs=[]
    for t in rounds:
        rr=[r for r in rows if r['t']==t];x=rank([r['j'] for r in rr],[r['j_E'] for r in rr])
        if x is not None:rs.append(x)
    v.update(rank=avg(rs),rank_coverage=None if not rounds else len(rs)/len(rounds));return v
def main():
    rows=read(ROOT/'round_rows.json');strata=read(ROOT/'team_acceptance_strata.json');starts=read(ROOT/'start_quality.json')
    for target in strata:
        values=[]
        for si in range(5):
            rr=[r for r in rows if r['key']==target['key'] and r['start']==si and r['full_accepted']==target['full_accepted']]
            if rr:values.append(q(rr))
        for f in q(rows[:1]):match(avg([v[f] for v in values if v[f] is not None]),target[f])
    for name,source,filters in [('acceptance_summary.json',strata,['scenario','full_accepted']),('start_summary.json',starts,['scenario','start'])]:
        for target in read(ROOT/name):
            rr=[r for r in source if all(r[f]==target[f] for f in filters)]
            for f,expected in target['metrics'].items():
                actual=interval([r[f] for r in rr if r[f] is not None]);assert actual['n']==expected['n'];match(actual['mean'],expected['mean']);match(actual['ci'],expected['ci'])
    import verify
    with (ROOT/'supplemental_verification.json').open('x') as f:json.dump(dict(accepted=True,max_error=verify.error,new_model_inferences=0),f,indent=2)
    print('Accepted',verify.error)
if __name__=='__main__':main()
