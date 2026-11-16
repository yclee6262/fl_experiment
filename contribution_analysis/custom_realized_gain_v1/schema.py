"""Shared artifact format and strict numerical/input checks."""
import hashlib
import json
from pathlib import Path
import numpy as np

VERSION = 'custom_realized_gain_v1'
ATOL = RTOL = 1e-10


def plain(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    return x


def save(path, obj):
    with Path(path).open('x') as f:
        json.dump(plain(obj), f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n')


def save_lines(path, rows):
    with Path(path).open('x') as f:
        for row in rows:
            f.write(json.dumps(plain(row), ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')


def read(path): return json.loads(Path(path).read_text())
def lines(path): return [json.loads(v) for v in Path(path).read_text().splitlines() if v]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def key(x): return np.asarray(x, dtype=np.float64).tobytes().hex()


def check_close(a, b, label):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(label+':shape_or_finite')
    if not np.all(np.abs(a-b) <= ATOL+RTOL*np.maximum(np.abs(a), np.abs(b))):
        raise ValueError(label+':numeric')
    return float(np.max(np.abs(a-b))) if a.size else 0.


def weights(values):
    w = np.asarray(values, dtype=float)
    if w.ndim != 1 or len(w) == 0 or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('invalid_weights')
    return w/w.sum()


def validate_manifest(m):
    if m['method_version'] != VERSION or m['objective_kind'] != 'fixed_weighted_absolute_error':
        raise ValueError('unsupported_objective')
    if m['attribution_unit'] != 'owner' or not m.get('deterministic', False):
        raise ValueError('unsupported_attribution_or_stochastic')
    for name, owners in [('anchor_ids', 'anchor_owners'), ('model_ids', 'model_owners')]:
        if not m[name] or len(set(m[name])) != len(m[name]) or len(m[name]) != len(m[owners]):
            raise ValueError('invalid_ids_or_mapping')
        if any(not isinstance(o, str) or not o for o in m[owners]): raise ValueError('invalid_owner')
    a, v = weights(m['direction_weights_raw']), weights(m['model_weights_raw'])
    if len(a) != len(m['anchor_ids']) or len(v) != len(m['model_ids']): raise ValueError('weight_length')
    check_close(a, m['direction_weights'], 'direction_weights')
    check_close(v, m['model_weights'], 'model_weights')
    anchors, start = np.asarray(m['anchors'], float), np.asarray(m['initial_solution'], float)
    if anchors.shape != (len(a), len(start)) or not np.isfinite(anchors).all() or not np.isfinite(start).all():
        raise ValueError('invalid_geometry')
    if not np.isfinite(m['target']): raise ValueError('invalid_target')
    return a, v
