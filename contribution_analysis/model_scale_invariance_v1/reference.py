"""Independent rational quantiles and score; no score module import."""
from fractions import Fraction as F
import math


def rational(v):return F(str(v))


def calculate(p):
    def quantile(v,q):
        vs=sorted(map(rational,v));idx=(len(vs)-1)*q;k=idx.numerator//idx.denominator;part=idx-k
        return vs[k]*(1-part)+vs[min(k+1,len(vs)-1)]*part
    scales=[quantile(v,F(9,10))-quantile(v,F(1,10))for v in p['probes']];ws=list(map(rational,p['weights']));total=sum(ws);ws=[w/total for w in ws]
    result=dict(scales=scales,C=None,own=None,others=None)
    if p['minus']is None:return dict(result,status='undefined_no_remaining_weight')
    if any(s<=F('1e-12')and w>0 for s,w in zip(scales,ws)):return dict(result,status='undefined_scale')
    terms=[w*(rational(b)-rational(a))/s if w else F(0)for w,a,b,s in zip(ws,p['full'],p['minus'],scales)]
    own=sum((v for i,v in enumerate(terms)if i in p['target']),F(0));others=sum(terms)-own
    return dict(scales=scales,C=sum(terms),own=own,others=others,status='available')


def verify(p,got):
    want=calculate(p);errors=[]
    if want['status']!=got['status']:raise ValueError('reference_status')
    if len(want['scales'])!=len(got['scales']):raise ValueError('reference_scale_length')
    for k in ('scales','C','own','others'):
        pairs=zip(want[k],got[k])if k=='scales'else[(want[k],got[k])]
        for a,b in pairs:
            if a is None:
                if b is not None:raise ValueError('reference_null')
            else:
                err=abs(float(a)-float(b));errors.append(err)
                if not math.isfinite(float(b))or err>1e-10+1e-9*abs(float(a)):raise ValueError('reference_numeric:'+k)
    return dict(passed=True,max_error=max(errors,default=0))


def polynomial(p,z):
    x=list(map(rational,z));l=list(map(rational,p['linear']));Q=[[rational(v)for v in row]for row in p['quadratic']]
    return rational(p['offset'])+sum(a*b for a,b in zip(l,x))+sum(x[i]*Q[i][j]*x[j]for i in range(len(x))for j in range(len(x)))
