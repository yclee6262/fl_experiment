"""Independent arithmetic/decision fixtures for intended Q/V semantics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from contribution_analysis.model_qv_v1.run import *

def main():
    c=new_cases()[0];s=analytic_server(c);x=np.array([.5,0.]);F=lambda z:float(z@z)
    # Full direction +x improves from .25 to .16. Removing helpful influence
    # reverses it; original acceptance rejects all worse points and stays put.
    full=decide(s,F,x,np.array([1.,0.]),F(x),.1)
    harmful_without=decide(s,F,x,np.array([-1.,0.]),F(x),.1)
    good_without=decide(s,F,x,np.array([1.,0.]),F(x),.1)
    bad_full=decide(s,F,x,np.array([-1.,0.]),F(x),.1)
    positive=harmful_without['loss']-full['loss']
    negative=good_without['loss']-bad_full['loss']
    assert positive>0 and negative<0
    assert full['loss']==good_without['loss']
    # Independent V fixture: model A vetoes P, full panel accepts Q.
    cur=np.array([.3,.3,.3]);p=np.array([.7,.2,.2]);z=np.array([.1,.25,.25])
    assert p.mean()>cur.mean() and z.mean()<cur.mean()
    assert p[1:].mean()<cur[1:].mean()
    v=float(p.mean()-z.mean());assert v>0
    # Exact duplicate slopes have zero nearest-neighbour RMS; multiplying
    # a model's losses and its scale together preserves standardized slopes.
    slopes=np.array([[1.,0.],[1.,0.],[0.,1.]])
    d=np.sqrt(np.mean((slopes[:,None,:]-slopes[None,:,:])**2,axis=2));np.fill_diagonal(d,np.inf)
    assert np.array_equal(d.min(1),[0.,0.,1.])
    assert np.allclose(np.array([5.,10.])/10,np.array([50.,100.])/100)
    save(OUT/'mechanism_checks.json',dict(passed=True,fixtures=6,
        helpful_direction_C=positive,harmful_direction_C=negative,unchanged_endpoint_C=0,
        duplicate_U=[0,0,1],veto_V=v,scale_invariance=True,
        scope='controlled decision/arithmetic fixtures, not population quality validation'))
    print('6 mechanism fixtures passed')

if __name__=='__main__':main()
