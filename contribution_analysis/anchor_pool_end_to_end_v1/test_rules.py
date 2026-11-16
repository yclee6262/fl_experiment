import unittest
from .settlement import settle
from .reference import verify_settlement


class Rules(unittest.TestCase):
    def es(self,a,b):return [dict(owner=o,t=t,C=str(v)if v is not None else None,rho='1e-12')for o,vs in [('A',a),('B',b)]for t,v in enumerate(vs)]
    def check(self,es,ts=(0,1,2)):
        v=settle(['A','B'],list(ts),es);verify_settlement(['A','B'],list(ts),es,v);return v
    def test_signed(self):self.assertEqual(self.check(self.es([3,-2,1],[1,1,1]))['payments'],dict(A=200,B=300))
    def test_tiny_positive_no_economic_gate(self):self.assertEqual(self.check(self.es(['0.00000002',0,0],[0,0,0]))['payments'],dict(A=500,B=0))
    def test_deadzone(self):self.assertEqual(self.check(self.es(['1e-8',0,0],[0,0,0]))['status'],'reserved')
    def test_accumulate_before_deadzone(self):self.assertEqual(self.check(self.es(['9e-9','9e-9',0],[0,0,0]))['payments']['A'],500)
    def test_missing(self):self.assertEqual(self.check(self.es([1,1,1],[1,1,1])[1:])['status'],'held')
    def test_null(self):self.assertIsNone(self.check(self.es([1,None,1],[1,1,1]))['payments']['A'])
    def test_replay(self):
        es=self.es([3,-2,1],[1,1,1]);self.assertEqual(self.check(es),self.check(list(reversed(es+es))))
    def test_conflict(self):
        es=self.es([3,-2,1],[1,1,1])
        with self.assertRaisesRegex(ValueError,'conflicting'):self.check(es+[dict(es[0],C='4')])
    def test_nonfinite(self):
        with self.assertRaisesRegex(ValueError,'nonfinite'):self.check(self.es(['NaN',0,0],[0,0,0]))
    def test_all_negative(self):self.assertEqual(self.check(self.es([-1,-1,-1],[-1,-1,-1]))['status'],'reserved')
    def test_zero_rounds(self):self.assertEqual(self.check([],())['status'],'reserved')
    def test_ties(self):
        os=['A','B','C'];es=[dict(owner=o,t=0,C='1',rho='1e-12')for o in os];v=settle(os,[0],es);verify_settlement(os,[0],es,v)
        self.assertEqual(v['payments'],dict(A=167,B=167,C=166))


if __name__=='__main__':unittest.main()
