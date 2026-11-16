import copy
import unittest
from .analysis import sign,agreement,summarize,matched_pairs


class DiagnosticsTests(unittest.TestCase):
    def test_deadzone_and_null(self):
        self.assertEqual([sign(v) for v in [None,1e-11,-1e-11,.1,-.1]],[None,0,0,1,-1])
    def test_undefined_not_zero(self):
        a=agreement([dict(C=None,D=0),dict(C=0,D=0)],'C')
        self.assertEqual((a['total'],a['available'],a['undefined'],a['matches']),(2,1,1,1))
    def test_harmful_reward_cannot_pass(self):
        r=dict(id='bad',category='x',split='heldout_parameter',C=.1,D=-.1,raw_C=.1,own=.2,others=-.1,peer=-.2)
        a=summarize([r]);self.assertFalse(a['quality_gate_passed']);self.assertEqual(a['harmful_positive_ids'],['bad'])
    def test_matched_pair_tamper_detected(self):
        f=dict(models=[1],weights=[1],target=[0],anchors=[1],anchor_weights=[1],start=[0],eta='.1')
        docs=[]
        for a,b in [('effective_information','erroneous_information'),('effective_restraint','erroneous_block')]:
            for i in range(6):
                for cat,d in [(a,.1),(b,-.1)]:
                    docs.append(dict(fixture=dict(f,id=f'{cat}_{i}'),decision=dict(events=[],lock={}),panel={},normalized=dict(C=.1),D=d))
        self.assertEqual(len(matched_pairs(docs)),12)
        bad=copy.deepcopy(docs);bad[1]['normalized']['C']=.2
        with self.assertRaisesRegex(ValueError,'matched_pair_control'):matched_pairs(bad)
