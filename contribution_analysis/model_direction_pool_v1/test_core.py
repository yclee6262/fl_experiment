import unittest,copy
import numpy as np
from .run import allocation
from .core import features,Replies
from contribution_analysis.three_pool_t3_model_v1.core import server_for
from contribution_analysis.three_pool_t3_model_v1.fixtures import poly
class Tests(unittest.TestCase):
    def rows(self):return [dict(owner=o,round=0,P=p,N=n) for o,p,n in [('a',.1,.2),('b',-.1,.3)]]
    def test_budget(self):
        a=allocation(self.rows(),.2,'N');self.assertEqual(sum(a['amounts'].values())+a['reserve'],500)
    def test_missing_matrix_held_null(self):
        a=allocation(self.rows()[:1],.2,'N',owners=['a','b'],rounds=[0]);self.assertEqual(a['status'],'held');self.assertIsNone(a['amounts']);self.assertIsNone(a['reserve'])
    def test_duplicate_held(self):self.assertEqual(allocation([self.rows()[0]]*2,.2,'N',owners=['a','b'],rounds=[0])['status'],'held')
    def test_nonfinite_held(self):
        for value in [float('nan'),float('inf'),None]:
            rows=self.rows();rows[0]['P']=value;self.assertEqual(allocation(rows,.2,'N')['status'],'held')
    def test_p_only_independent_undefined_N(self):
        rows=self.rows();rows[0]['N']=None;self.assertEqual(allocation(rows,0,'N')['status'],'allocated');self.assertEqual(allocation(rows,.2,'N')['status'],'held')
    def test_all_nonpositive_reserved(self):
        rows=self.rows()
        for r in rows:r.update(P=-1,N=0)
        a=allocation(rows,.2,'N');self.assertEqual(a['reserve'],500);self.assertEqual(a['amounts'],dict(a=0,b=0))
    def test_signed_accumulation(self):
        rows=[dict(owner=o,round=t,P=p,N=0) for t,p in [(0,1),(1,-2)] for o in ['a','b']];self.assertEqual(allocation(rows,0,'N')['reserve'],500)
    def test_empty_directions(self):
        c=dict(models=[poly(1),poly(-1)],weights=['1/2','1/2'],anchors=[['0','0']],anchor_weights=['1']);s=server_for(c);r=Replies(c)
        self.assertEqual(features(s,dict(S_current=np.zeros(2),current_method='secant'),r,np.ones(2))['status'],'undefined_empty_directions')
    def test_scale_undefined(self):
        c=dict(models=[poly(1),poly(-1)],weights=['1/2','1/2'],anchors=[['1','0']],anchor_weights=['1']);s=server_for(c);r=Replies(c)
        self.assertEqual(features(s,dict(S_current=np.zeros(2),current_method='secant'),r,np.array([0,1]))['status'],'undefined_scale')
    def test_clone_zero_u(self):
        c=dict(models=[poly(1),poly(1),poly(-1)],weights=['1/3']*3,anchors=[['1','0']],anchor_weights=['1']);s=server_for(c);r=Replies(c)
        self.assertEqual(features(s,dict(S_current=np.zeros(2),current_method='secant'),r,np.ones(3))['U'].tolist(),[0,0,2])
if __name__=='__main__':unittest.main()
