"""Parent's post-run hand-computed tests; not a new research dataset."""
import copy
import unittest
from fractions import Fraction as F
from .core import peer_score
from .reference import score as reference_score
from contribution_analysis.model_consensus_pool_v1.core import canonical_hash


def fixture():
    m=dict(schema_version='1',owners=['a','b','c'],rounds=['t'],budget=500,probe_id='manual',models=[
        dict(id=i,owner=o,v=v,version='v1',probe=[0,1.25])for i,o,v in [('a1','a',.2),('a2','a',.3),('b','b',.4),('c','c',.1)]])
    losses={'a':dict(a1=9,a2=8,b=1.1,c=.8),'b':dict(a1=1.2,a2=.9,b=99,c=1.1),'c':dict(a1=1,a2=.9,b=.98,c=99)}
    ss=[dict(owner=o,round='t',manifest_hash=canonical_hash(m),versions={r['id']:r['version']for r in m['models']},
        full=dict(id='full',losses={r['id']:1 for r in m['models']}),minus=dict(id='minus-'+o,losses=losses[o]))for o in m['owners']]
    return m,ss


def relock(m,s):
    for row in s:row['manifest_hash']=canonical_hash(m)


class ParentTests(unittest.TestCase):
    def test_weighted_mean_and_whole_owner_exclusion(self):
        m,s=fixture();r=peer_score(m,s)['peer']
        self.assertEqual(r['scores']['a'],.04)
        self.assertAlmostEqual(r['scores']['b'],1/30)
        self.assertAlmostEqual(r['scores']['c'],-19/450)
        self.assertEqual(r['amounts'],dict(a=273,b=227,c=0))
        self.assertAlmostEqual(r['q']['a'],6/11)
        self.assertEqual(r['scores'],reference_score(m,s)['peer']['scores'])

    def test_all_self_models_removed(self):
        m,s=fixture();before=peer_score(m,s)['peer']
        s[0]['minus']['losses'].update(a1=999,a2=2000)
        self.assertEqual(peer_score(m,s)['peer']['scores']['a'],before['scores']['a'])

    def test_invalid_version_and_endpoint(self):
        for kind in ['version','hash','full','same_id']:
            m,s=fixture()
            if kind=='version':s[0]['versions']['a1']='new'
            elif kind=='hash':s[0]['manifest_hash']='bad'
            elif kind=='full':s[1]['full']['losses']['a1']=2
            else:s[0]['minus']['id']='full'
            r=peer_score(m,s)['peer'];self.assertEqual(r['status'],'held');self.assertIsNone(r['amounts'])

    def test_exact_scale_boundary(self):
        m,s=fixture();m['models'][0]['probe']=[0,1.25e-12];relock(m,s)
        self.assertEqual(peer_score(m,s)['peer']['status'],'held')

    def test_finite_input_output_overflow(self):
        m,s=fixture();m['models'][2]['probe']=[0,2e-12];s[0]['minus']['losses']['b']=1e308;relock(m,s)
        r=peer_score(m,s)['peer'];self.assertEqual(r['status'],'held');self.assertIsNone(r['q'])

    def test_zero_weight_model_retained_but_not_scored(self):
        m,s=fixture();before=peer_score(m,s)['peer'];m['models'].append(dict(id='zero',owner='a',v=0,version='v1',probe=[0,0]))
        for slot in s:
            slot['versions']['zero']='v1'
            for endpoint in ['full','minus']:slot[endpoint]['losses']['zero']=0
        relock(m,s)
        self.assertEqual(peer_score(m,s)['peer']['scores'],before['scores'])

    def test_undefined_not_zero_and_no_truth_field(self):
        m,s=fixture();m['truth']={'a':1}
        r=peer_score(m,s)['peer'];self.assertEqual(r['status'],'held');self.assertIsNone(r['scores'])

    def test_reorder_preserves_all_outputs(self):
        m,s=fixture();before=peer_score(m,s);m['owners'].reverse();m['models'].reverse();s.reverse();relock(m,s)
        self.assertEqual(before,peer_score(m,s))


if __name__=='__main__':unittest.main()
