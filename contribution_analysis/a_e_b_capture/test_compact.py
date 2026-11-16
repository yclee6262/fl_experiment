import unittest
import numpy as np
from contribution_analysis.a_e_b_capture.compact import record_compact_run
from contribution_analysis.a_e_b_validation.run import server,score
from contribution_analysis.custom_realized_gain_v1.runtime import manifest_for,equivalent
from contribution_analysis.custom_realized_gain_v1.observer import record_custom_run
from contribution_analysis.custom_realized_gain_v1.schema import plain
from contribution_analysis.custom_realized_gain_v1.verify import verify_run
from contribution_analysis.custom_realized_gain_v1.verify_balance import verify_balance


class CompactTests(unittest.TestCase):
    def compare(self,s,start,n=3):
        m=manifest_for(s,start,dict(num_iterations=n,use_annealing=True,allow_tangent=True))
        old=record_custom_run(s,start,m,True);new=record_compact_run(s,start,m);equivalent(old,new)
        a,b=score(m,old);aa,bb=score(m,new)
        self.assertEqual(plain(a),plain(aa));self.assertEqual(plain(b),plain(bb))
        verify_run(m,plain(new),plain(aa));verify_balance(m,plain(new),plain(bb))
    def test_linear(self):self.compare(server([lambda x:2+x[0]],[[-.8],[.8]]),[.5])
    def test_zero_direction(self):self.compare(server([lambda x:1.],[[-.8],[.8]]),[0.])
    def test_zero_budget(self):self.compare(server([lambda x:2+x[0]],[[.8]]),[0.],0)
    def test_duplicate_anchor(self):self.compare(server([lambda x:2+x[0]],[[.8],[.8]]),[0.])
    def test_boundary(self):self.compare(server([lambda x:2-x[0]],[[0.]]),[.99])
    def test_rejection_and_switch(self):self.compare(server([lambda x:1+x[0]**2],[[.8]]),[0.])
    def test_backtrack(self):self.compare(server([lambda x:1+(x[0]-.01)**2],[[.01]]),[0.],1)
    def test_stochastic_rejected_and_wrappers_restored(self):
        counter=[0]
        def f(x):counter[0]+=1;return 1+counter[0]*.001
        s=server([f],[[.8]]);original=s.trusted_agents[0].api_predict;m=manifest_for(s,[0.])
        with self.assertRaisesRegex(ValueError,'stochastic'):record_compact_run(s,[0.],m)
        self.assertEqual(s.trusted_agents[0].api_predict,original)
    def test_nan_rejected(self):
        s=server([lambda x:float('nan')],[[.8]]);m=manifest_for(s,[0.])
        with self.assertRaisesRegex(ValueError,'nonfinite'):record_compact_run(s,[0.],m)
    def test_model_exception_restored(self):
        def f(x):raise RuntimeError('intentional')
        s=server([f],[[.8]]);m=manifest_for(s,[0.]);original=s.trusted_agents[0].api_predict
        with self.assertRaisesRegex(RuntimeError,'intentional'):record_compact_run(s,[0.],m)
        self.assertEqual(s.trusted_agents[0].api_predict,original)


if __name__=='__main__':unittest.main()
