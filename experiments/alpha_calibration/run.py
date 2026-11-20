"""Run the preserved calibration research in a new output directory."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from project_support.legacy import prepare


def main():
    prepare()
    from experiments.alpha_calibration.experiment import (
        build_parser, run, validate_args, write_run_manifest,
    )
    parser = build_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest == 'output_dir':
            action.required = True
            action.default = None
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    output = Path(args.output_dir)
    if output.exists():
        parser.error('Output directory must not exist; use a new directory')
    try:
        run(args)
    except Exception:
        if (output/'run_manifest.json').exists():
            write_run_manifest(output, args, 'failed')
        raise


if __name__ == '__main__':
    main()
