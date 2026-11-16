import unittest,copy
import numpy as np
from core import *

def make(mode='secant',alpha=None,a=None,eta=.1,fn=None,trials=None):
    fn=fn or (lambda j,x:float((j+1)*x[0]+(.1 if j==0 else -.1)))
    data=dict(x=[0.],mode=mode,eta=eta,delta=.0001,anchors=[[-.5],[.5]],a=a or [.4,.6],
        alpha=alpha or [.7,.3],trials=trials or [],full_endpoint=[0.],round=0,case='synthetic')
    xs=[[0.],[-.5],[.5]]+(trials or [])
    if mode!='secant':xs += [[-.0001],[.0001]]
    normal={str(j)+':'+key(x):dict(i=j,x=x,y=fn(j,x),source='baseline',available_round=0) for x in xs for j in range(2)}
    return View(data,normal),fn

class Tests(unittest.TestCase):
    def test_n0_no_trials(self):
        v,_=make();self.assertEqual(n0(v,0)['score'],0.)
    def test_n0_first_not_best(self):
        v,_=make(trials=[[-.02],[-.04]],fn=lambda j,x:1+x[0]);r=n0(v,0)
        self.assertEqual(r['attempt'],0);self.assertEqual(r['endpoint'],[-.02])
    def test_n0_original_early_stop(self):
        v,_=make(trials=[[-.02]],fn=lambda j,x:1-x[0]);self.assertFalse(n0(v,0)['accepted'])
    def test_forbidden_field(self):
        v,_=make();d=copy.deepcopy(v.data);d['oracle']=0
        with self.assertRaises(ValueError):View(d,v.normal)
    def test_future(self):
        v,_=make();normal=copy.deepcopy(v.normal);next(iter(normal.values()))['available_round']=1
        with self.assertRaises(ValueError):View(v.data,normal)
    def test_oracle_source(self):
        v,_=make();normal=copy.deepcopy(v.normal);next(iter(normal.values()))['source']='oracle'
        with self.assertRaises(ValueError):View(v.data,normal)
    def test_missing_not_zero(self):
        v,_=make();v.normal={}
        with self.assertRaises(KeyError):n0(v,0)
    def test_role_difference(self):
        v,_=make();self.assertGreater(np.linalg.norm(direction(v,0,'anchor')-direction(v,0,'model')),1e-3)
    def test_dynamic(self):
        v,f=make(mode='dynamic');r=n1(v,0,'joint',lambda j,x:f(j,x));self.assertEqual(len(r['g']),1)
    def test_no_remaining_weight(self):
        v,_=make(alpha=[1.,0.])
        with self.assertRaises(ValueError):n0(v,0)
    def test_symmetry_null_effect(self):
        v,f=make(fn=lambda j,x:1+x[0],trials=[[-.1]])
        r=n1(v,0,'model',lambda j,x:f(j,x));self.assertAlmostEqual(r['score'],0.)
    def test_rejected_one_step(self):
        v,f=make(eta=.5,fn=lambda j,x:float((x[0]-.1)**2),trials=[[.5],[.25],[.125]])
        r=n1(v,0,'model',lambda j,x:f(j,x));self.assertFalse(r['accepted']);self.assertEqual(r['score'],0.)
    def test_bounded(self):self.assertEqual(bound(np.array([1.]),np.array([1.]),.1),0.)
    def test_panel_does_not_query_removed_on_rejection(self):
        v,f=make(eta=.5,fn=lambda j,x:float((x[0]-.1)**2));calls=[]
        n1(v,0,'model',lambda j,x:(calls.append(j) or f(j,x)))
        self.assertEqual(calls,[1])

if __name__=='__main__':unittest.main()
