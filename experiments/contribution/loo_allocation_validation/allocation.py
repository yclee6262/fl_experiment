"""Specified bonus rule; missing evidence and a retained pool are distinct."""
import numpy as np


def allocate(scores):
    if scores is None or not np.isfinite(scores).all():
        return dict(status='invalid', q=None, augmented=None, positive_sum=None, fragile=None)
    p = np.maximum(scores, 0.)
    total = float(p.sum())
    q = p / total if total > 1e-8 else None
    return dict(status='distributed' if q is not None else 'reserved',
                q=None if q is None else q.tolist(),
                augmented=([*q, 0.] if q is not None else [*np.zeros(len(p)), 1.]),
                positive_sum=total, fragile=total <= 1e-4)


def compare(pred, reference):
    a, b = allocate(pred), allocate(reference)
    both = a['q'] is not None and b['q'] is not None
    delta = abs(np.array(a['q']) - b['q']) if both else None
    aug = (.5 * sum(abs(np.array(a['augmented']) - b['augmented']))
           if a['augmented'] is not None and b['augmented'] is not None else None)
    return dict(predicted=a, reference=b, tv=float(delta.sum()/2) if both else None,
                max_share_error=float(delta.max()) if both else None,
                augmented_tv=None if aug is None else float(aug),
                both_reserved=a['status'] == b['status'] == 'reserved',
                status_disagreement=a['status'] != b['status'])


def panel_weights(alpha, owner=None):
    a = np.asarray(alpha, float)
    if not np.isfinite(a).all() or (a < 0).any() or not np.isclose(a.sum(), 1., atol=1e-12):
        raise ValueError('Invalid alpha')
    if owner is not None:
        a = a.copy(); a[owner] = 0
    if a.sum() <= 0:
        raise ValueError('No remaining model weight')
    return a/a.sum()
