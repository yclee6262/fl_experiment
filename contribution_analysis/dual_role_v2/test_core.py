import unittest
import numpy as np
from core import attribute,quality,run
from host_server import HostServer


class Agent:
    def __init__(self,i,fn):self.agent_id=i;self.fn=fn
    def api_predict(self,x):return np.array([self.fn(np.asarray(x))])


class Tests(unittest.TestCase):
    def trace(self,slopes):
        g=np.mean(slopes);x=np.array([0.]);u=np.array([1.])
        return [dict(g=np.array([g]),x=x,loss=10.,current_losses=np.ones(len(slopes))*10,iteration=0,
            directions=[dict(k=0,width=1.,weight=1.,u=u,reply_losses=10+np.array(slopes))],
            trials=[dict(x=np.array([-.1*np.sign(g)]),loss=9.,accepted=True)])]
    def test_signed_shares(self):
        r=attribute(self.trace([6.,-2.]),[.5,.5],[1.,.5])
        np.testing.assert_allclose(r['A'],[1.,0.]);np.testing.assert_allclose(r['B'],[1.5,-.5]);np.testing.assert_allclose(r['M'],[1.5,-.25])
    def test_no_quality(self):self.assertIsNone(attribute(self.trace([1.,1.]),[.5,.5],None)['M'])
    def test_rejected_not_counted(self):
        t=self.trace([1.,1.]);t[0]['trials'][0]['accepted']=False
        self.assertEqual(attribute(t,[.5,.5],[1.,1.])['status'],'no_accepted_motion')
    def test_bad_motion(self):
        t=self.trace([1.,1.]);t[0]['trials'][0]['x']=[.1]
        with self.assertRaises(ValueError):attribute(t,[.5,.5],[1.,1.])
    def test_bad_gain(self):
        t=self.trace([1.,1.]);t[0]['trials'][0]['loss']=11
        with self.assertRaises(ValueError):attribute(t,[.5,.5],[1.,1.])
    def test_quality_scale_invariance(self):np.testing.assert_allclose(quality([1,3],2),quality([100,300],200))
    def test_invalid_quality(self):
        self.assertIsNone(quality([1],0));self.assertIsNone(quality([-1],1));self.assertIsNone(quality([float('nan')],1))
    def test_quality_boundaries(self):np.testing.assert_allclose(quality([0,2],2),[1,.5])
    def test_sensitivity(self):
        r=attribute(self.trace([2.,-2.+2**-22]),[.5,.5],[1.,1.]);self.assertEqual(r['status'],'numerically_sensitive')
    def make_server(self):
        s=HostServer(0.,1,constraint_mode='box_constrained');s.alphas=[.7,.3];s.I_list=np.array([[-.8],[.9]])
        s.trusted_agents=[Agent(1,lambda x:float((x[0]-.3)**2+1)),Agent(2,lambda x:float((x[0]+.1)**2+.5))];return s
    def test_model_only_keeps_direction_weights(self):
        r=run(self.make_server(),[.6],role='model',excluded=0)
        np.testing.assert_array_equal(r['direction_weights'],[.7,.3]);self.assertEqual(r['model_ids'],[2]);self.assertEqual(r['anchor_ids'],[1,2])
    def test_anchor_only_keeps_models(self):
        r=run(self.make_server(),[.6],role='anchor',excluded=0)
        self.assertEqual(r['model_ids'],[1,2]);self.assertEqual(r['anchor_ids'],[2]);np.testing.assert_array_equal(r['model_weights'],[.7,.3])
    def test_record_identical(self):
        s=self.make_server();a=run(s,[.6]);b=run(s,[.6],True,[.9,.7])
        self.assertEqual(a['query_order_sha256'],b['query_order_sha256']);self.assertEqual(a['result']['history'],b['result']['history'])
        self.assertEqual(a['result']['states'],b['result']['states']);np.testing.assert_array_equal(a['result']['solution'],b['result']['solution'])


if __name__=='__main__':unittest.main()
