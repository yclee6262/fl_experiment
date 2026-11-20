"""Recompute role scores and payment diagnostics from saved team endpoints."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        parser.error('Output directory must not exist')
    output.parent.mkdir(parents=True, exist_ok=True)
    from contribution_analysis.senior_three_packages_v1 import run
    run.OUT = output
    run.main()
    from contribution_analysis.senior_three_packages_v1 import analyze
    analyze.OUT = output
    analyze.main()


if __name__ == '__main__':
    main()
