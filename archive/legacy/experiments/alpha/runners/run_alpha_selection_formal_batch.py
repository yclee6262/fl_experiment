"""Use the unchanged pilot runner but suppress interim descriptive quality analysis."""
import torch
import argparse
from unittest.mock import patch
import run_alpha_initialization_selection as runner
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('registration');a.add_argument('batch');x=a.parse_args()
 with patch.object(runner,'analyze',lambda *args:None):runner.main_run(x.registration,x.batch)
