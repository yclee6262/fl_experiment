"""Shared seed-level acceptance and paired-bootstrap helpers."""

from collections import Counter

import numpy as np


BOOTSTRAP_SEED = 20260924
BOOTSTRAP_REPLICATES = 10_000


def finite_float(value):
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def valid_delivery(row, metric="external_global_error"):
    solution = row.get("final_solution")
    return (
        row.get("delivery_status") == "valid"
        and (solution is None or str(solution).strip() not in {"", "[]", "null"})
        and str(row.get("final_box_feasible", row.get("box_feasible", ""))).lower()
        == "true"
        and finite_float(row.get(metric)) is not None
    )


def audit_coverage(rows, expected_keys, key_fields):
    """Return duplicate, missing, and unexpected keys without hiding failures."""
    expected = {tuple(key) for key in expected_keys}
    keys = [tuple(row.get(field, "") for field in key_fields) for row in rows]
    counts = Counter(keys)
    observed = set(keys)
    return {
        "duplicate_keys": sorted(key for key, count in counts.items() if count > 1),
        "missing_keys": sorted(expected - observed),
        "unexpected_keys": sorted(observed - expected),
        "complete": expected == observed and all(count == 1 for count in counts.values()),
    }


def paired_bootstrap_mean_ci(differences, confidence=0.95, seed=BOOTSTRAP_SEED,
                             replicates=BOOTSTRAP_REPLICATES):
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None, None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    estimates = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1-tail))
