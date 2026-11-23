"""Reprice accepted Task 11 runs without invoking any model or solver."""

import argparse
from collections import defaultdict
import csv
from fractions import Fraction
import gzip
import hashlib
from itertools import combinations, product
import json
from math import isfinite
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
SOURCE_SUMMARY = HERE.parent / "start_stability/results/acceptance_summary.json"
IDENTITY = ("key", "target", "r", "w")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def allocate_pool(scores, thresholds, ids, amount):
    """Keep the entire pool in reserve when the required scores are unusable."""
    if not (len(scores) == len(thresholds) == len(ids)) or len(set(map(str, ids))) != len(ids):
        raise ValueError("Scores and thresholds must match distinct participant IDs")
    if any(score is None or not isfinite(score) for score in scores):
        return [0] * len(ids), amount
    if any(not isfinite(tau) or tau < 0 for tau in thresholds):
        raise ValueError("Invalid numerical threshold")
    weights = [Fraction(s) if s > tau else Fraction(0) for s, tau in zip(scores, thresholds)]
    total = sum(weights)
    if not total:
        return [0] * len(ids), amount
    quotas = [amount * weight / total for weight in weights]
    payments = [int(quota) for quota in quotas]
    order = sorted(range(len(ids)), key=lambda i: (payments[i] - quotas[i], str(ids[i])))
    for i in order[:amount - sum(payments)]:
        payments[i] += 1
    return payments, 0


def ledger(basic, anchor, model, reserve):
    return tuple(b + a + m for b, a, m in zip(basic, anchor, model)) + (reserve,)


def reprice(raw, amounts):
    ids = raw["ids"]
    people = raw["participants"]
    if [p["id"] for p in people] != ids:
        raise ValueError("Roster mismatch")
    base = amounts["basic_per_completed_service_bundle"]
    anchor_pool, model_pool = amounts["anchor_pool"], amounts["model_pool"]
    anchor_scores = [p["scores"]["Fminus"]["A"] for p in people]
    model_scores = [p["scores"]["Fminus"]["M"] for p in people]
    anchor_tau = [p["tauA"] for p in people]
    model_tau = 1e-10
    anchor, ar = allocate_pool(anchor_scores, anchor_tau, ids, anchor_pool)
    model, mr = allocate_pool(model_scores, [model_tau] * len(ids), ids, model_pool)
    old_basic = [0 if a < -tau and m < -model_tau else base
                 for a, tau, m in zip(anchor_scores, anchor_tau, model_scores)]
    old = ledger(old_basic, anchor, model, len(ids) * base - sum(old_basic) + ar + mr)
    saved = raw["payments"]["Fminus"]
    if (old != tuple(saved["total_with_reserve"]) or old_basic != saved["basic"]
            or anchor != saved["anchor"] or model != saved["model"]):
        raise ValueError("Reconstructed peer shadow ledger differs from accepted archive")
    unlocked = ledger([base] * len(ids), anchor, model, ar + mr)
    main = (base,) * len(ids) + (anchor_pool + model_pool,)
    budget = len(ids) * base + anchor_pool + model_pool
    if any(sum(v) != budget or min(v) < 0 for v in (old, unlocked, main)):
        raise ValueError("Payment and reserve must conserve the budget")
    return dict(old=old, unlocked=unlocked, main=main, budget=budget,
                old_basic=old_basic, anchor_reserve=ar, model_reserve=mr)


def pairs(axis):
    if axis == "inversion":
        return [((a, w), (b, w)) for w in range(3) for a, b in combinations(range(5), 2)]
    return [((r, a), (r, b)) for r in range(5) for a, b in combinations(range(3), 2)]


def tv(left, right, policy):
    return sum(abs(a - b) for a, b in zip(left[policy], right[policy])) / (2 * left["budget"])


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def replay(source, output):
    if output.exists():
        raise FileExistsError(output)
    protocol = read(HERE / "protocol.json")
    accepted = read(SOURCE_SUMMARY)
    manifest_path = source / "manifest.json"
    if (digest(manifest_path) != accepted["manifest_sha256"]
            or read(source / "acceptance.json") != accepted["acceptance"]
            or not accepted["acceptance"]["passed"]):
        raise ValueError("This replay requires the accepted formal Task 11 archive")
    index = read(source / "index.json")
    cases, roster, scenarios = {}, {}, {}
    index_hash = digest(source / "index.json")
    for item in index:
        path = source / item["file"]
        if digest(path) != item["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")
        with gzip.open(path, "rt") as f:
            raw = json.load(f)
        key = tuple(raw[field] for field in IDENTITY)
        if key in cases or key != tuple(item[field] for field in IDENTITY) or raw["status"] != "valid":
            raise ValueError(f"Unexpected record: {key}")
        team = raw["key"]
        if (raw["ids"] != roster.setdefault(team, raw["ids"])
                or raw["scenario"] != scenarios.setdefault(team, raw["scenario"])):
            raise ValueError(f"Team mismatch: {team}")
        cases[key] = reprice(raw, protocol["stress_ledger"])
    if len(roster) != 20 or set(cases) != set(product(roster, [-.5, .5], range(5), range(3))):
        raise ValueError("Incomplete crossed-start coverage")

    grouped = defaultdict(list)
    owner_rows = []
    for team, target in product(sorted(roster), [-.5, .5]):
        runs = {(r, w): cases[team, target, r, w] for r, w in product(range(5), range(3))}
        for i, owner in enumerate(roster[team]):
            row = dict(team=team, scenario=scenarios[team], target=target, owner=owner)
            for policy in ("old", "unlocked", "main"):
                values = [v[policy][i] for v in runs.values()]
                row.update({f"{policy}_min": min(values), f"{policy}_max": max(values)})
            owner_rows.append(row)
        for axis in ("inversion", "solver"):
            matched = [(runs[a], runs[b]) for a, b in pairs(axis)]
            entry = {f"{p}_tv": mean(tv(a, b, p) for a, b in matched)
                     for p in ("old", "unlocked", "main")}
            entry["old_base_flip"] = mean(
                sum(x != y for x, y in zip(a["old_basic"], b["old_basic"])) / len(roster[team])
                for a, b in matched
            )
            grouped[scenarios[team], target, axis].append(entry)
    summary = [dict(scenario=s, target=t, axis=ax, teams=len(entries), pairs_per_team=len(pairs(ax)),
                    **{k: mean(v[k] for v in entries) for k in entries[0]})
               for (s, t, ax), entries in sorted(grouped.items())]
    if any(row["teams"] != 10 for row in summary):
        raise ValueError("Unexpected cohort size")

    output.mkdir(parents=True)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "owner_ranges.csv", owner_rows)
    audit = dict(passed=True, runs=len(cases), comparisons=sum(r["teams"] * r["pairs_per_team"] for r in summary),
                 owner_configurations=sum(len(v["old_basic"]) for v in cases.values()),
                 old_basic_zeros=sum(v["old_basic"].count(0) for v in cases.values()),
                 owner_ranges=len(owner_rows), model_calls=0, solver_calls=0,
                 manifest_sha256=digest(manifest_path), index_sha256=index_hash,
                 acceptance_summary_sha256=digest(SOURCE_SUMMARY),
                 protocol_sha256=digest(HERE / "protocol.json"), replay_sha256=digest(Path(__file__)),
                 results_sha256={f: digest(output / f) for f in ("summary.csv", "owner_ranges.csv")})
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    replay(args.source, args.output_dir)


if __name__ == "__main__":
    main()
