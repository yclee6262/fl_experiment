import unittest
from contribution_analysis.model_raw_c_validation_v1.run import allocate,classify,index_events,raw,scale_controls
class Tests(unittest.TestCase):
    def test_full_objective_and_cancel_before_clip(self):
        self.assertAlmostEqual(float(raw([1000,1000],[1001,999],['3/4','1/4'])),.5)
        rows=[dict(owner=o,round=t,C=c,distance=1) for o,cs in [('a',[2,-3]),('b',[1,1])] for t,c in enumerate(cs)]
        a=allocate(rows,['a','b'],2);self.assertEqual(a['amounts'],{'a':0,'b':500});self.assertEqual(classify(rows[:2],2)['kind'],'cancellation_negative')
    def test_missing_duplicate_nonfinite_held(self):
        for rows in [[],[dict(owner='a',round=0,C=1)]*2,[dict(owner='a',round=0,C=float('nan')),dict(owner='b',round=0,C=1)]]:
            self.assertEqual(allocate(rows,['a','b'],1)['status'],'held');self.assertIsNone(allocate(rows,['a','b'],1)['reserve'])
    def test_reserve_and_subbudget(self):
        rows=[dict(owner='a',round=0,C=-1,N=1),dict(owner='b',round=0,C=0,N=0)]
        self.assertEqual(allocate(rows,['a','b'],1)['reserve'],500)
        a=allocate(rows,['a','b'],1,.2,'N');self.assertEqual(a['reserve'],400);self.assertEqual(a['amounts']['a'],100)
    def test_equal_objective_different_endpoint(self):
        self.assertEqual(classify([dict(C=0,distance=1)],1)['kind'],'different_endpoint_equal_F')
    def test_threshold_unit_dependence(self):
        rows=[dict(owner='a',round=0,C=5e-9),dict(owner='b',round=0,C=0)]
        self.assertEqual(allocate(rows,['a','b'],1)['reserve'],500);rows[0]['C']*=10;self.assertEqual(allocate(rows,['a','b'],1)['reserve'],0)
    def test_conflicting_events(self):
        with self.assertRaises(ValueError):index_events([dict(model=0,x=[0],loss=1),dict(model=0,x=[0],loss=2)])
    def test_compensated_units(self):
        rows=[dict(owner='a',round=0,C=1,loss_full=[10,20],loss_minus=[12,20]),dict(owner='b',round=0,C=1,loss_full=[10,20],loss_minus=[10,22])]
        controls=scale_controls(rows,['1/2','1/2'],['a','b'],1)
        self.assertTrue(all(not c['amounts_changed'] for c in controls if c['kind']!='single_uncompensated'))
if __name__=='__main__':unittest.main()
