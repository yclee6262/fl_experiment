import argparse
import oob_shrinkage_research as study
from conservative_shrinkage import METHODS,initial_alpha
study.METHODS=METHODS;study.initial_alpha=initial_alpha
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--seeds',required=True)
 args=p.parse_args();args.sizes='20';args.targets='0';args.draws=1;args.original_blind=True;study.run(args)
