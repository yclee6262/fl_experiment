"""Independent per-request cold-cache accounting and scorer checks."""
import numpy as np


def audit_events(events):
    seen={};counts={};owners_ok=True
    for e in events:
        expected=e['stage']!='baseline' and e['key'] in seen
        if e['hit']!=expected:raise ValueError('Cache sequence mismatch')
        if expected and seen[e['key']]!=e['y']:raise ValueError('Cache value mismatch')
        seen[e['key']]=e['y']
        row=counts.setdefault(e['stage'],dict(requests=0,samples=0,hits=0))
        row['requests']+=1;row['samples']+=not e['hit'];row['hits']+=e['hit']
        if e['stage']=='generation' and e['owner']==e['model']:owners_ok=False
    if not owners_ok:raise ValueError('Removed model used in generation')
    return counts


def compare_saved(result, saved, baseline):
    np.testing.assert_array_equal(result['base']['solution'],baseline['result']['solution'])
    if result['base']['history']!=baseline['result']['history']:raise ValueError('Baseline history mismatch')
    if result['x'] is not None:
        np.testing.assert_allclose(result['x'],saved['x'],atol=1e-10,rtol=0)
        for field in ['peer','full']:np.testing.assert_allclose(result[field],saved[field],atol=1e-10,rtol=0)


def independent_scores(result, alpha, target):
    """Reconstruct losses from raw model replies keyed at locked endpoints."""
    if result['x'] is None:return
    replies={e['key']:abs(e['y']-target) for e in result['events']}
    # Keys are version:model:float64 bytes; never infer float32 equivalence.
    prefix=result['events'][0]['key'].split(':')[0]
    def loss(i,x):return replies[f'{prefix}:{i}:'+np.asarray(x,np.float64).tobytes().hex()]
    for i,x in enumerate(result['x']):
        delta=[loss(j,x)-loss(j,result['base']['solution']) for j in range(len(alpha))]
        full=sum(float(alpha[j])*delta[j] for j in range(len(alpha)))
        peer=sum(float(alpha[j])*delta[j] for j in range(len(alpha)) if j!=i)/sum(float(alpha[j]) for j in range(len(alpha)) if j!=i)
        np.testing.assert_allclose([full,peer],[result['full'][i],result['peer'][i]],atol=1e-12,rtol=0)
