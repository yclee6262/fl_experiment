import copy
import unittest
import numpy as np
from .core import Panel,execute
from .fixtures import cases,poly


class ModelTests(unittest.TestCase):
    def test_grouped_owner_and_K_not_M(self):
        c=copy.deepcopy(cases()[0]);c.update(models=[poly(3),poly(1),poly(-1)],weights=['1/4','1/4','1/2'],target=[0,1],reference=poly(1))
        r=execute(c);self.assertAlmostEqual(r['score']['C'],.1);self.assertAlmostEqual(r['score']['own'],.2);self.assertAlmostEqual(r['score']['others'],-.1)
        self.assertEqual(r['remaining_model_indices'],[2]);np.testing.assert_array_equal(r['anchor_direction_weights'],[.5,.5])
    def test_no_remaining_weight(self):
        c=next(c for c in cases() if c['id']=='concentrated_weights_2');r=execute(c)
        self.assertIsNone(r['score']['C']);self.assertIsNone(r['minus']);self.assertEqual(r['score']['status'],'undefined_no_remaining_weight')
    def test_zero_weight_model(self):
        c=next(c for c in cases() if c['id']=='concentrated_weights_5');r=execute(c)
        self.assertEqual(r['score']['C'],0);self.assertEqual(r['score']['E_target'],0)
    def test_common_scoring_requires_lock(self):
        p=Panel(cases()[0]);p.stage='common'
        with self.assertRaisesRegex(ValueError,'score_before_lock'):p.losses([0,0])
    def test_excluded_model_never_queried(self):
        p=Panel(cases()[0]);p.stage='minus'
        with self.assertRaisesRegex(ValueError,'excluded_model_called'):p.reply(0,[0,0])
    def test_no_reaccept_by_common_reference(self):
        c=next(c for c in cases() if c['id']=='prevent_bad_step_0');r=execute(c)
        self.assertFalse(r['full']['accepted']);self.assertTrue(r['minus']['accepted']);self.assertGreater(r['score']['C'],0)
        self.assertTrue(any(e['stage']=='common' and e['model']==0 for e in r['events']))
    def test_anchor_weights_unchanged_with_model_weights(self):
        c=next(c for c in cases() if c['id']=='useful_information_0');r=execute(c)
        np.testing.assert_array_equal(r['anchor_direction_weights'],[.5,.5]);np.testing.assert_array_equal(r['model_weights'],[.75,.25])


if __name__=='__main__':unittest.main()
