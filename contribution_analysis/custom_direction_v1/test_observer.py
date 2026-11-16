import unittest
import numpy as np
from host_server import HostServer
from observer import solve,Collector,calculate,point_key


class Agent:
    def __init__(self,fn):self.fn=fn
    def api_predict(self,x):return np.array([self.fn(np.asarray(x))])


def server():
    s=HostServer(0.,1,constraint_mode='box_constrained')
    s.trusted_agents=[Agent(lambda x:float((x[0]-.3)**2+1)),Agent(lambda x:float((x[0]+.1)**2+.5))]
    s.alphas=[.5,.5];s.I_list=np.array([[-.8],[.9]])
    return s


class Tests(unittest.TestCase):
    def test_original_behavior_and_full_reply_record(self):
        s=server();a=solve(s,False,[.6]);b=solve(s,True,[.6])
        np.testing.assert_array_equal(a['result']['solution'],b['result']['solution'])
        self.assertEqual(a['query_order_sha256'],b['query_order_sha256']);self.assertEqual(a['counter'],b['counter'])
        self.assertEqual(a['result']['history'],b['result']['history'])
        self.assertAlmostEqual(sum(b['scores']['S_raw']),1.,places=6)
        self.assertTrue(any(t['mode']=='dynamic' for t in b['iterations']))
        self.assertTrue(any(not p['accepted'] for t in b['iterations'] for p in t['trials']))
        for t in b['iterations']:
            if not t['accepted']:self.assertTrue(np.all(t['d']==0))

    def make(self,slopes,u,step=.1):
        c=Collector();m=len(slopes);dim=len(u);x=np.zeros(dim);u=np.array(u,float);u/=np.linalg.norm(u)
        c.cache[point_key(x)]=np.ones(m)*10;c.cache[point_key(u)]=10+np.array(slopes)
        c.begin(0,x,'secant',10.);c.direction(0,u,1.,u,float(np.mean(slopes)),1.)
        g=float(np.mean(slopes))*u;c.combined(g,np.linalg.norm(g))
        if step:c.trial(-step*g/np.linalg.norm(g),9.,10.)
        return c,calculate(c,[1/m]*m)

    def test_opposing_example(self):
        _,r=self.make([6.,-2.],[1.]);np.testing.assert_allclose(r['D'],[.15,-.05]);np.testing.assert_allclose(r['S'],[1.5,-.5])

    def test_noncollinear_and_orthogonal_components(self):
        c=Collector();x=np.zeros(2);c.cache[point_key(x)]=np.array([10.,10.]);c.begin(0,x,'secant',10.)
        for k,(u,slopes) in enumerate([([1.,0.],[0.,4.]),([0.,1.],[4.,-4.])]):
            u=np.array(u);c.cache[point_key(u)]=10+np.array(slopes);c.direction(k,u,1.,u,float(np.mean(slopes)),.5)
        c.combined(np.array([1.,0.]),1.);c.trial(np.array([-.1,0.]),9.,10.)
        r=calculate(c,[.5,.5]);np.testing.assert_allclose(r['D'],[0.,.1]);np.testing.assert_allclose(c.iterations[0]['model_delta_x'].sum(axis=0),[-.1,0.])

    def test_strong_cancellation_retains_raw_but_abstains(self):
        _,r=self.make([2.,-2.+2**-22],[1.]);self.assertEqual(r['status'],'numerically_sensitive');self.assertIsNone(r['S']);self.assertIsNotNone(r['S_raw'])

    def test_no_motion_undefined(self):
        _,r=self.make([1.,1.],[1.],step=0);self.assertEqual(r['status'],'no_accepted_motion');self.assertIsNone(r['S']);np.testing.assert_array_equal(r['D'],[0.,0.])

    def test_missing_reply_fails(self):
        c=Collector();c.begin(0,np.zeros(1),'secant',0.);c.combined(np.ones(1),1.)
        with self.assertRaises(KeyError):calculate(c,[1.])

    def test_box_boundary_zero_step(self):
        s=HostServer(0.,1,constraint_mode='box_constrained');s.trusted_agents=[Agent(lambda x:float(x[0]+2))];s.alphas=[1.];s.I_list=np.array([[0.]])
        a=solve(s,False,[-1.]);b=solve(s,True,[-1.]);self.assertEqual(a['query_order_sha256'],b['query_order_sha256']);self.assertEqual(b['scores']['status'],'no_accepted_motion')


if __name__=='__main__':unittest.main()
