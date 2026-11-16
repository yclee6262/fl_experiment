"""Frozen three-method clean replication, evaluator is post-delivery only."""
import argparse
import oob_shrinkage_research as study
from adaptive_correlation_alpha import initial_alpha
METHODS=['B_stage0','corr_precision_0p5','adaptive_corr_precision']
study.METHODS=METHODS
study.SCENARIOS=['reference_clean']
study.initial_alpha=initial_alpha
original_dump=study.dump
def dump(path,value):
    if str(path).endswith('manifest.json') and 'methods' in value:
        value['expected']=len(value['seeds'])*len(value['scenarios'])*len(value['draws'])*len(value['sizes'])*len(value['targets'])*len(value['methods'])
        value['experiment']='independent clean three-method correlation replication v1'
        value['matrix_rule']={'grid':[0,.25,.5,.75,1],'folds':5,'fit_rows':16,'heldout_rows':4,'tie_tolerance':1e-12}
    original_dump(path,value)
study.dump=dump
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--seeds',required=True)
    args=p.parse_args();args.sizes='20';args.targets='0';args.draws=1;args.original_blind=True
    study.run(args)
