"""Only normal-search information enters these estimators; no oracle imports."""
import numpy as np

ROLES=('anchor','model','joint')
def key(x):return np.asarray(x,dtype=np.float64).tobytes().hex()
def bound(x,d,h):
    h=float(h)
    for v,u in zip(x,d):
        if u>0:h=min(h,(1-v)/u)
        elif u<0:h=min(h,(-1-v)/u)
    return max(0.,h)

class View:
    FIELDS={'x','mode','eta','delta','anchors','a','alpha','trials','full_endpoint','round','case'}
    def __init__(self,data,normal):
        if set(data)!=self.FIELDS:raise ValueError('unapproved_round_fields')
        if any(v.get('source')!='baseline' or v.get('available_round',-1)>data['round'] for v in normal.values()):
            raise ValueError('non_baseline_or_future_reply')
        self.data=data;self.normal=normal
        self.x=np.asarray(data['x'],float);self.I=np.asarray(data['anchors'],float)
        self.alpha=np.asarray(data['alpha'],float);self.alpha=self.alpha/self.alpha.sum()
        self.a=np.asarray(data['a'],float);self.a=self.a/self.a.sum();self.n=len(self.alpha)
    def losses(self,x):
        return np.array([abs(self.normal[str(i)+':'+key(x)]['y']) for i in range(self.n)])
    def panel(self,i,role):return [j for j in range(self.n) if role=='anchor' or j!=i]
    def weights(self,panel):
        w=self.alpha[panel]
        if not len(w) or w.sum()<=0:raise ValueError('undefined_panel')
        return w/w.sum()

def n0(v,i):
    ids=v.panel(i,'model');w=v.weights(ids);lx=v.losses(v.x)
    # Direct remaining sum avoids cancellation in L-alpha_i*ell_i.
    current=float(np.dot(w,lx[ids]));end=v.x.copy();attempt=None
    for k,z in enumerate(v.data['trials']):
        if float(np.dot(w,v.losses(z)[ids]))<current:end=np.asarray(z,float);attempt=k;break
    score=float(np.dot(v.alpha,v.losses(end))-np.dot(v.alpha,v.losses(v.data['full_endpoint'])))
    return dict(score=score,endpoint=end.tolist(),accepted=attempt is not None,attempt=attempt)

def direction(v,i,role):
    ids=v.panel(i,role);w=v.weights(ids);ks=[k for k in range(v.n) if role=='model' or k!=i]
    a=v.a[ks]
    if not len(a) or a.sum()<=0:raise ValueError('undefined_anchors')
    a=a/a.sum();lx=float(np.dot(w,v.losses(v.x)[ids]));g=np.zeros_like(v.x)
    for k,ak in zip(ks,a):
        d=v.I[k]-v.x;dist=np.linalg.norm(d)
        if dist<1e-8:continue
        u=d/dist
        if v.data['mode']=='secant':z=v.I[k];h=dist
        else:
            h=bound(v.x,u,v.data['delta'])
            if h<=1e-12:continue
            z=v.x+h*u
        derivative=(float(np.dot(w,v.losses(z)[ids]))-lx)/h
        g+=ak*derivative*u
    return g

def n1(v,i,role,query):
    ids=v.panel(i,role);w=v.weights(ids);lx=v.losses(v.x);g=direction(v,i,role)
    norm=np.linalg.norm(g);end=v.x.copy();candidate=None;accept=False;losses=lx
    if norm>1e-8:
        d=g/norm;candidate=v.x-bound(v.x,-d,v.data['eta'])*d
        # Decisions receive only the permitted panel; supplemental excluded reply later.
        partial={j:abs(query(j,candidate)) for j in ids}
        accept=float(np.dot(w,[partial[j] for j in ids]))<float(np.dot(w,lx[ids]))
        if accept:
            end=candidate.copy()
            losses=np.array([partial[j] if j in partial else abs(query(j,candidate)) for j in range(v.n)])
    xf=v.x.copy()
    if v.data['trials']:
        z=v.data['trials'][0];lz=v.losses(z)
        if float(np.dot(v.alpha,lz))<float(np.dot(v.alpha,lx)):xf=np.asarray(z,float)
    score=float(np.dot(v.alpha,losses)-np.dot(v.alpha,v.losses(xf)))
    return dict(score=score,endpoint=end.tolist(),candidate=None if candidate is None else candidate.tolist(),
                full_one_endpoint=xf.tolist(),g=g.tolist(),accepted=bool(accept))
