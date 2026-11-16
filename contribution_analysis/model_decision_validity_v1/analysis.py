"""Signed agreement diagnostics; no information from D feeds C."""
import math


def sign(v):
    return None if v is None else 1 if v>1e-10 else -1 if v < -1e-10 else 0


def agreement(rows,field):
    usable=[r for r in rows if r[field] is not None and r['D'] is not None]
    confusion={f'{a},{b}':sum(sign(r['D'])==a and sign(r[field])==b for r in usable)
               for a in [-1,0,1] for b in [-1,0,1]}
    matches=sum(sign(r[field])==sign(r['D']) for r in usable)
    return dict(total=len(rows),available=len(usable),undefined=len(rows)-len(usable),matches=matches,
                agreement=None if not usable else matches/len(usable),confusion_D_then_score=confusion)


def summarize(rows):
    fields=['raw_C','C','own','others','peer']
    groups={}
    for split in ['all','development','heldout_parameter']:
        subset=rows if split=='all' else [r for r in rows if r['split']==split]
        groups[split]={cat:{f:agreement(subset if cat=='all' else [r for r in subset if r['category']==cat],f)
                           for f in fields} for cat in ['all']+sorted(set(r['category'] for r in rows))}
    # All five categories must pass; overall accuracy alone cannot establish validity.
    quality=all(r['C'] is not None and sign(r['C'])==sign(r['D']) for r in rows)
    return dict(groups=groups,quality_gate_passed=quality,
        harmful_positive_ids=[r['id'] for r in rows if sign(r['D'])==-1 and sign(r['C'])==1],
        helpful_negative_ids=[r['id'] for r in rows if sign(r['D'])==1 and sign(r['C'])==-1],
        nonzero_missed_ids=[r['id'] for r in rows if sign(r['D']) in [-1,1] and sign(r['C'])==0])


def matched_pairs(docs):
    by={d['fixture']['id']:d for d in docs};pairs=[]
    for first,second in [('effective_information','erroneous_information'),('effective_restraint','erroneous_block')]:
        for i in range(6):
            a=by[f'{first}_{i}'];b=by[f'{second}_{i}']
            # Exclude only reference and descriptive labels; every model-visible input is checked.
            same=all(a['fixture'][k]==b['fixture'][k] for k in ['models','weights','target','anchors','anchor_weights','start','eta'])
            same= same and a['decision']['events']==b['decision']['events'] and a['decision']['lock']==b['decision']['lock']
            same= same and a['panel']==b['panel'] and a['normalized']==b['normalized']
            if not same or sign(a['D'])!=-sign(b['D']) or sign(a['D'])==0:raise ValueError('matched_pair_control')
            pairs.append(dict(ids=[a['fixture']['id'],b['fixture']['id']],observations_identical=True,
                              C=a['normalized']['C'],D=[a['D'],b['D']]))
    return pairs
