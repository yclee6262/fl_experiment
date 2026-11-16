import unittest,copy
from fractions import Fraction as F
from contribution_analysis.model_consensus_pool_v1.core import canonical_hash,settle
from .core import peer_score
from .reference import score

def fixture():
    m=dict(schema_version='1',owners=['a','b'],models=[dict(id=o,owner=o,v=1,version='v1',probe=[0,1.25]) for o in ['a','b']],rounds=['t'],budget=500,probe_id='p')
    slots=[dict(owner=o,round='t',manifest_hash=canonical_hash(m),versions={o:'v1' for o in ['a','b']},full=dict(id='f',losses=dict(a=1,b=1)),minus=dict(id=o,losses=dict(a=1.1,b=1.1))) for o in ['a','b']]
    return m,slots
class Tests(unittest.TestCase):
    def test_reference_and_tie(self):
        m,s=fixture();a=peer_score(m,s);b=score(m,s)
        for k in b:
            for key in b[k]:self.assertEqual(a[k][key],b[k][key])
        self.assertEqual(a['peer']['amounts'],dict(a=250,b=250))
    def test_held(self):
        for kind in ['missing','duplicate','minus','nonfinite','scale','remaining']:
            m,s=fixture()
            if kind=='missing':s.pop()
            if kind=='duplicate':s.append(s[0])
            if kind=='minus':s[0]['minus']=None
            if kind=='nonfinite':s[0]['minus']['losses']['b']=float('nan')
            if kind=='scale':m['models'][0]['probe']=[1,1]
            if kind=='remaining':m['models'][1]['v']=0
            self.assertEqual(peer_score(m,s)['peer']['status'],'held')
    def test_settlement(self):
        self.assertEqual(settle(dict(a=F(1,10**8),b=F(0)),500,1)['status'],'reserved')
        self.assertEqual(settle(dict(a=F(2),b=F(-1)),500,1)['amounts'],dict(a=500,b=0))
        self.assertEqual(settle(dict(a=F(1),b=F(1)),501,1)['amounts'],dict(a=251,b=250))
    def test_signed_rounds(self):
        m,s=fixture();m['rounds']=['t','u'];other=copy.deepcopy(s)
        for z in other:z['round']='u';z['minus']['id']+='u';z['minus']['losses']=dict(a=.9,b=.9)
        s+=other
        for z in s:z['manifest_hash']=canonical_hash(m)
        self.assertEqual(peer_score(m,s)['peer']['status'],'reserved')
    def test_multimodel_and_sorting(self):
        m,s=fixture();base=peer_score(m,s)['peer'];m['models'][0]['v']=.5;n=copy.deepcopy(m['models'][0]);n['id']='a2';m['models'].append(n)
        for z in s:
            z['manifest_hash']=canonical_hash(m);z['versions']['a2']='v1'
            for arm in ['full','minus']:z[arm]['losses']['a2']=z[arm]['losses']['a']
        self.assertEqual(peer_score(m,list(reversed(s)))['peer']['scores'],base['scores'])
        s[0]['minus']['losses']['a']=10;s[0]['minus']['losses']['a2']=20
        self.assertEqual(peer_score(m,s)['peer']['scores']['a'],base['scores']['a'])

    def test_invalid_manifest_wrapper(self):
        for m in [None,[],{},dict(budget=True),dict(budget=-1)]:
            result=peer_score(m,[])['peer'];self.assertEqual(result['status'],'held');self.assertIsNone(result['reserve'])
        m,s=fixture();m['models'][0]['probe']=[0,10**1000]
        self.assertEqual(peer_score(m,s)['peer']['status'],'held')
