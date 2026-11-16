import numpy as np
from four_way_bayes import propose,metrics

def test_proposal_is_feasible_novel_and_data_driven():
    w=np.vstack([np.eye(4),np.full(4,.25),.2+.2*np.eye(4)])
    target=np.array([.15,.35,.3,.2]);y=((w-target)**2).sum(1)
    x,a=propose(w,y,0)
    assert min(x)>=.05-1e-8 and abs(sum(x)-1)<1e-8
    assert min(np.linalg.norm(w-x,axis=1))>=.01-1e-8 and a['EI']>0
    z,_=propose(w,y[::-1],0)
    assert np.linalg.norm(x-z)>.001

def test_metrics_groups_scenarios_by_seed():
    rr=[{'seed':s,'scenario':c,'pooled_error':e} for s,e in [(1,1.),(2,3.)] for c in range(3)]
    base={(s,c):{'pooled_error':0.} for s in [1,2] for c in range(3)}
    d,j=metrics(rr,base);np.testing.assert_equal(d,[1,3]);assert j==3
