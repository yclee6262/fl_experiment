"""Replay AV, ACM, J and full LOO with independent caches into a fresh directory."""
import argparse
import json
from pathlib import Path
from contribution_analysis.peer_exclusion_cost_v1 import run as historical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional first N historical teams; omit for all 24")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    cases = json.loads((historical.OLD / "registration.json").read_text())["fresh"]
    if args.limit is not None:
        cases = cases[:args.limit]
    historical.save(output / "manifest.json", {"status": "running", "teams": len(cases), "source": str(historical.OLD)})
    for case in cases:
        old = json.loads((historical.OLD / "records" / f"{case['key']}_cost.json").read_text())
        historical.old_acm = old["ACM"]  # Existing endpoint cross-check uses this reference.
        for arm in ("AV", "ACM", "J", "LOO"):
            result = historical.execute(case, arm, old[arm])
            historical.save(output / f"{case['key']}_{arm}.json", result)
    historical.save(output / "manifest.json", {"status": "complete", "teams": len(cases),
                    "arms": ["AV", "ACM", "J", "LOO"], "scope": "historical analytic-model cold-cache replay; no payment or external-validity claim"})


if __name__ == "__main__":
    main()
