"""Sequential all-method screen on one saved paired source, never pooled-tuned."""
import argparse, json
from pathlib import Path
import oob_shrinkage_research as study
from unified_alpha_methods import METHODS, DEFINITIONS, initial_alpha

def run(args):
    study.METHODS = METHODS
    study.initial_alpha = initial_alpha
    args.sizes='20'; args.targets='0'; args.draws=1; args.original_blind=True
    study.run(args)
    path=Path(args.output)/'manifest.json'; m=json.loads(path.read_text())
    m.update(experiment_kind='exploratory_unified_screen',
        definitions=DEFINITIONS, stage3_solver='best_of, custom iterations 30',
        information_budget='original 20 qualification labels, no additional labels',
        historical_semantics='all alpha vectors use original weighted absolute-error objective; historical ensemble-target/sparse objective is not replayed',
        aliases={'uniform_stacked_half':'ensemble_shrink', 'sparse_top_two_alpha':'top_two_expertise'},
        analysis_rule='pooled quality first; within-scenario ranking; no posthoc parameter tuning; paired seed-bootstrap vs A and B')
    study.dump(path,m)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--seeds',required=True)
    run(p.parse_args())
