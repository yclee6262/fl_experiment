import copy
import unittest
from .score import score
from .reference import verify


class Tests(unittest.TestCase):
    def panel(self):return dict(weights=[.5,.5],probes=[[1.,2.,3.],[10.,20.,30.]],full=[1.,10.],minus=[2.,20.],target=[0])
    def test_score(self):
        p=self.panel();v=score(p);verify(p,v);self.assertAlmostEqual(v['C'],.625);self.assertAlmostEqual(v['own']+v['others'],v['C'])
    def test_consistent_scale(self):
        p=self.panel();q=copy.deepcopy(p);q['probes'][0]=[v*10 for v in q['probes'][0]];q['full'][0]*=10;q['minus'][0]*=10
        self.assertAlmostEqual(score(p)['C'],score(q)['C'])
    def test_zero_scale(self):
        p=self.panel();p['probes'][0]=[1.,1.,1.];self.assertEqual(score(p)['status'],'undefined_scale')
    def test_nearzero_scale(self):
        p=self.panel();p['probes'][0]=[0.,1e-14,2e-14];self.assertEqual(score(p)['status'],'undefined_scale')
    def test_no_minus(self):
        p=self.panel();p['minus']=None;self.assertIsNone(score(p)['C'])
    def test_zero_weight_scale(self):
        p=self.panel();p['weights']=[0,1];p['probes'][0]=[1.,1.,1.];v=score(p);verify(p,v);self.assertEqual(v['status'],'available')
    def test_nonfinite(self):
        p=self.panel();p['full'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'invalid_loss'):score(p)
    def test_negative_loss(self):
        p=self.panel();p['full'][0]=-1
        with self.assertRaisesRegex(ValueError,'invalid_loss'):score(p)
    def test_missing_probe(self):
        p=self.panel();p['probes']=[[1],[2]]
        with self.assertRaisesRegex(ValueError,'invalid_probes'):score(p)
    def test_wrong_endpoint_length(self):
        p=self.panel();p['minus']=[1]
        with self.assertRaisesRegex(ValueError,'invalid_endpoint_shape'):score(p)
    def test_prediction_scale_not_loss_scale(self):
        target=5;prediction=6
        self.assertNotEqual(abs(10*prediction-target),10*abs(prediction-target))
    def test_absolute_scale_gate_boundary(self):
        p=self.panel();p['probes'][0]=[0.,1e-14,2e-14];a=score(p)
        p['probes'][0]=[v*1000 for v in p['probes'][0]];b=score(p)
        self.assertEqual(a['status'],'undefined_scale');self.assertEqual(b['status'],'available')


if __name__=='__main__':unittest.main()
