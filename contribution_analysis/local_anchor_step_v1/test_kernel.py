import unittest
import numpy as np
from kernel import alternative,cache_for,score

class Agent:
    def __init__(self,fn):self.fn=fn;self.calls=0
    def api_predict(self,x):self.calls+=1;return np.array([self.fn(np.asarray(x))])

def fixture(fn):
    # Original g=2 causes left motion; removing anchor0 leaves -1/.5 (right),
    # removing anchor1 leaves 3/.5 (left, the already observed accepted point).
    agents=[Agent(fn),Agent(fn)];x=np.array([0.]);nxt=np.array([-.1])
    before=fn(x);after=fn(nxt)
    directions=[dict(k=0,weight=.5,derivative=6.,u=[1.],point=[1.],reply_losses=[fn(np.array([1.]))]*2),
                dict(k=1,weight=.5,derivative=-2.,u=[1.],point=[.5],reply_losses=[fn(np.array([.5]))]*2)]
    it=dict(iteration=0,mode='secant',x=x,g=[2.],loss=before,current_losses=[before]*2,
            directions=directions,trials=[dict(x=nxt,loss=after,accepted=True,reply_losses=[after]*2)])
    return dict(ids=[1,2],alphas=[.5,.5]),dict(iterations=[it]),agents

class Tests(unittest.TestCase):
    def test_removal_and_weight_normalization(self):
        a=alternative([0.],[-.1],[2.],[3.],.5);np.testing.assert_allclose(a['g_minus'],[-2.]);np.testing.assert_allclose(a['x'],[.1])
    def test_rejected_alternative_stays_at_old_point_and_cache(self):
        case,b,agents=fixture(lambda x:1+float(x[0]));r=score(case,b,agents)
        np.testing.assert_allclose(r['A'],[.1,0.]);self.assertEqual(r['counter']['samples'],2);self.assertEqual(r['counter']['cache_hits'],1)
        self.assertFalse(r['comparisons'][0]['alternative_accepted'])
    def test_better_alternative_gets_negative_value(self):
        case,b,agents=fixture(lambda x:1-float(x[0])**2-.05*float(x[0]));r=score(case,b,agents)
        self.assertLess(r['A'][0],0.);self.assertAlmostEqual(r['A'][1],0.)
    def test_zero_direction_no_query(self):
        a=alternative([0.],[-.1],[2.],[2.],.5);self.assertEqual(a['status'],'no_motion')
    def test_boundary_not_clipped(self):
        a=alternative([.95],[.85],[2.],[3.],.5);self.assertEqual(a['status'],'infeasible_equal_step');self.assertGreater(a['x'][0],1.)
    def test_no_remaining_anchor(self):self.assertEqual(alternative([0.],[.1],[1.],[1.],1.)['status'],'no_remaining_anchor')
    def test_float32_cache_and_missing_reply(self):
        _,b,_=fixture(lambda x:1+float(x[0]));it=b['iterations'][0];self.assertTrue(cache_for(it,2))
        it['trials'][0]['reply_losses']=None
        with self.assertRaises(ValueError):cache_for(it,2)
    def test_signed_sum_no_extra_step_weight(self):
        case,b,agents=fixture(lambda x:1+float(x[0]));b['iterations']=b['iterations']*2;r=score(case,b,agents)
        np.testing.assert_allclose(r['A'],[.2,0.]);np.testing.assert_allclose(r['A_positive']+r['A_negative'],r['A'])
    def test_skipped_anchor_has_zero_one_step_value(self):
        fn=lambda x:1+float(x[0]);case,b,agents=fixture(fn);case=dict(ids=[1,2,3],alphas=[1/3]*3);agents.append(Agent(fn))
        it=b['iterations'][0];it['current_losses']=[1.]*3;it['trials'][0]['reply_losses']=[.9]*3
        for d,derivative in zip(it['directions'],[9.,-3.]):
            d['weight']=1/3;d['derivative']=derivative;d['reply_losses']*=1;d['reply_losses'].append(d['reply_losses'][0])
        r=score(case,b,agents);self.assertAlmostEqual(r['A'][2],0.)
    def test_no_accepted_steps_not_fabricated_as_valid_score(self):
        case,b,agents=fixture(lambda x:1+float(x[0]));b['iterations'][0]['trials'][0]['accepted']=False
        r=score(case,b,agents);self.assertIsNone(r['A']);self.assertEqual(r['status'],'no_accepted_steps')
    def test_conflicting_float32_cache_fails(self):
        _,b,_=fixture(lambda x:1+float(x[0]));it=b['iterations'][0]
        it['trials'].append(dict(x=[np.nextafter(-.1,-1.)],reply_losses=[2.,2.]))
        with self.assertRaises(ValueError):cache_for(it,2)

if __name__=='__main__':unittest.main()
