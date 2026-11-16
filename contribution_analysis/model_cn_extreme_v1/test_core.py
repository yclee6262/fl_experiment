import unittest
from fractions import Fraction as Q
from .run import fixtures,allocation
class Tests(unittest.TestCase):
    def test_grid_locked(self):
        fs=fixtures();self.assertEqual(len(fs),64);self.assertEqual(len({f['id'] for f in fs}),64)
        self.assertEqual(sum(f['split']=='new_development' for f in fs),32)
        for f in fs:
            self.assertEqual(sum(map(Q,f['weights'])),1);self.assertTrue(all(Q(w)>0 for w in f['weights']));self.assertNotIn('reference',f)
    def test_signed_threshold(self):
        for c,yes in [('0.00000001',False),('0.0000000101',True)]:
            rows=[dict(owner='a',round=0,C=float(c),N=.9),dict(owner='b',round=0,C=1,N=.01)]
            a=allocation(rows,['a','b'],1,.2,True);self.assertEqual(a['eligible']['a'],yes);self.assertEqual(a['amounts']['a']>0,yes)
    def test_supplement_deadzone(self):
        rs=[dict(owner='a',round=0,C=1,N=1e-8),dict(owner='b',round=0,C=-1,N=.9)]
        a=allocation(rs,['a','b'],1,.2,True);self.assertEqual(a['arms']['N']['reserve'],100);self.assertEqual(a['amounts']['b'],0)
    def test_cancellation_and_budget(self):
        rs=[dict(owner=o,round=t,C=c,N=.2) for o,seq in [('a',[1,-2]),('b',[1,1])] for t,c in enumerate(seq)]
        for b in [.1,.2,.3]:
            a=allocation(rs,['a','b'],2,b,True);self.assertEqual(a['amounts']['a'],0);self.assertEqual(sum(a['amounts'].values())+a['reserve'],500)
    def test_missing_held(self):self.assertEqual(allocation([],['a','b'],1,.2,True)['status'],'held')
if __name__=='__main__':unittest.main()
