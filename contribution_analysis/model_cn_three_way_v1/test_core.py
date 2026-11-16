import math,unittest
from .run import allocation,fixtures,synthetic
class Tests(unittest.TestCase):
    def rows(self,c=1e-8):return [dict(owner='a',round=0,C=c,N=.9),dict(owner='b',round=0,C=1,N=.01)]
    def test_exact_tau(self):
        for c,eligible in [(.99e-8,False),(1e-8,False),(1.01e-8,True)]:
            a=allocation(self.rows(c),['a','b'],1,.2,True);self.assertEqual(a['eligible']['a'],eligible);self.assertEqual(a['amounts']['a']>0,eligible)
    def test_signed_accumulation(self):
        for f in synthetic()[:6]:
            a=allocation(f['rows'],f['owners'],2,.2,True);net=sum(r['C'] for r in f['rows'] if r['owner']=='a');self.assertEqual(a['eligible']['a'],net>1e-8)
    def test_missing_duplicate_nonfinite(self):
        for rs in [[],self.rows()[:1],self.rows()+self.rows()[:1],[dict(owner='a',round=0,C=1,N=math.nan),self.rows()[1]]]:
            a=allocation(rs,['a','b'],1,.2,True);self.assertEqual(a['status'],'held');self.assertIsNone(a['reserve'])
    def test_unqualified_supplement_reserved(self):
        rs=[dict(owner=o,round=0,C=-1,N=1) for o in ['a','b']];a=allocation(rs,['a','b'],1,.2,True);self.assertEqual(a['reserve'],500)
        a=allocation(rs,['a','b'],1,.2,False);self.assertEqual(a['reserve'],400)
    def test_nontransfer_and_budget(self):
        rs=[dict(owner='a',round=0,C=1,N=0),dict(owner='b',round=0,C=0,N=0)]
        for b in [.1,.2,.3]:
            a=allocation(rs,['a','b'],1,b,True);self.assertEqual(a['reserve'],round(500*b));self.assertEqual(sum(a['amounts'].values())+a['reserve'],500)
    def test_micro_C_large_N(self):
        a=allocation(self.rows(2e-8),['a','b'],1,.2,True);self.assertGreaterEqual(a['amounts']['a'],98)
    def test_fixture_counts_no_reference(self):
        fs=fixtures();self.assertEqual(len(fs),16);self.assertEqual(len(synthetic()),10);self.assertTrue(all('reference' not in f for f in fs))
if __name__=='__main__':unittest.main()
