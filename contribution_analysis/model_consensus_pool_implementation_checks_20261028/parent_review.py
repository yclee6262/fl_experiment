import copy,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,'/Users/yclee/Documents/GitHub/fl_experiment')
from contribution_analysis.model_consensus_pool_v1.core import score
from contribution_analysis.model_consensus_pool_v1.reference import audit
m=dict(schema_version='1',owners=['a','b','c'],rounds=['r1','r2'],budget=501,probe_id='review-probe',models=[dict(id=o,owner=o,v=1,version='v1',probe=[0,1]) for o in ['a','b','c']])
def seal(m,ss):
 h=hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 for s in ss:s['manifest_hash']=h;s['versions']={x['id']:x['version'] for x in m['models']}
def fixture():
 # These endpoint deltas were calculated by hand from scale=.8; not by core.
 ds={('a','r1'):[.032,.04,.048],('a','r2'):[-.008,.008,.016],('b','r1'):[.016,.024,.032],('b','r2'):[.008,.016,.024],('c','r1'):[-.008,-.004,0],('c','r2'):[.004,.008,.016]}
 from decimal import Decimal
 ss=[dict(owner=o,round=r,full=dict(id='full-'+r,losses={k:1 for k in m['owners']}),minus=dict(id=o+'-'+r,losses={k:float(Decimal('1')+Decimal(str(v))) for k,v in zip(m['owners'],vs)}))for(o,r),vs in ds.items()]
 seal(m,ss);return ss
ss=fixture();r=score(m,ss)
assert r['status']=='allocated',r
assert r['amounts']==dict(a=251,b=250,c=0),r
assert r['q']==dict(a=.5,b=.5,c=0),r
assert r['scores']==dict(a=.03,b=.03,c=-.005),r
assert not audit(m,ss,r)
checks=['hand_computed_signed_totals','501_lexicographic_tie','independent_reference']
for name,change in [('missing',lambda z:z.pop()),('duplicate',lambda z:z.append(copy.deepcopy(z[0]))),('undefined',lambda z:z[0].update(minus=None)),('bad_owner_type',lambda z:z[0].update(owner=[])),('same_id_changed_loss',lambda z:z[0]['minus'].update(id=z[0]['full']['id']))]:
 z=copy.deepcopy(ss);change(z);v=score(m,z)
 assert v['status']=='held'and v['amounts']is None and v['q']is None,(name,v)
 checks.append(name)
# Same zero effect for every model: all budget is retained.
z=copy.deepcopy(ss)
for s in z:s['minus']=copy.deepcopy(s['full'])
v=score(m,z);assert v['status']=='reserved'and v['reserve']==501 and v['q']is None and not any(v['amounts'].values());checks.append('zero_effect_reserve')
# Identical observations cannot acquire a different payment by a hidden truth label.
mm=copy.deepcopy(m);mm['truth']='positive';vv=score(mm,ss);assert vv['status']=='held';checks.append('truth_not_admitted')
Path('/tmp/model_consensus_pool_parent_input.json').write_text(json.dumps(dict(manifest=m,slots=ss),indent=2))
print(json.dumps(dict(passed=True,checks=checks,result=r),indent=2))
