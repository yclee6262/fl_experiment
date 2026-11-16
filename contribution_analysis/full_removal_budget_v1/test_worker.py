import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from host_server import HostServer
import worker


class Agent:
    def __init__(self,agent_id): self.agent_id=agent_id
    def api_predict(self,x): return np.atleast_1d(np.sum(x,axis=-1)+0.01*self.agent_id)


class WorkerTests(unittest.TestCase):
    def test_inner_break_at_iteration_cap_is_budget_exhaustion(self):
        outer={479:'direction_gradient_below_threshold',517:'no_improving_step_after_search'}
        self.assertEqual(worker.classify_custom_stop(dict(break_line=499,iteration=29,num_iterations=30),outer),'iteration_budget_exhausted')
        self.assertEqual(worker.classify_custom_stop(dict(break_line=517,iteration=10,num_iterations=30),outer),'no_improving_step_after_search')
        self.assertEqual(worker.classify_custom_stop(dict(break_line=479,iteration=0,num_iterations=30),outer),'direction_gradient_below_threshold')

    def fixture(self,path,candidates=None):
        server=HostServer(0,2,constraint_mode='box_constrained')
        server.trusted_agents=[Agent(1),Agent(2),Agent(3)]
        server.alphas=[1/3]*3
        server.I_list=np.array(candidates or [[0.2,0.1],[-0.3,0.4],[0.1,-0.2]])
        state=dict(seed=1,scenario='clean',ids=[1,2,3],artifact='artifact',fixed_start=[0.1,0.1])
        (path/'artifact').mkdir()
        (path/'artifact/local_models.pt').write_bytes(b'test')
        (path/'coalition.json').write_text('{}')
        args=argparse.Namespace(directory=path,output=path/'result.json',excluded=-1,iterations=30,
                                calls=20000,samples=1000000,seconds=60)
        return server,state,args

    def test_instrumentation_preserves_solution_and_counts(self):
        with tempfile.TemporaryDirectory() as td:
            server,state,args=self.fixture(Path(td))
            expected=server.optimize_candidate_subspace(strategy='best_of',custom_iterations=30,
                                                       initial_solution=state['fixed_start'])
            with patch.object(worker,'load_server',return_value=(server,state)):
                worker.solve(args)
            result=json.loads(args.output.read_text())
            self.assertEqual(result['status'],'valid_delivery')
            np.testing.assert_array_equal(result['result']['solution'],expected['solution'])
            self.assertEqual(result['result']['loss'],expected['loss'])
            self.assertFalse(result['evaluator_loaded'])
            self.assertEqual(result['calls'],sum(result['engines'][e]['calls'] for e in ['bfgs','custom']))
            self.assertEqual(result['calls'],result['input_samples'])
            self.assertIn('nit',result['engines']['scipy'])
            self.assertIn('stop_reason',result['engines']['custom'])
            self.assertNotEqual(result['engines']['custom']['stop_reason'],'unclassified_stop')

    def test_budget_stop_preserves_failure(self):
        with tempfile.TemporaryDirectory() as td:
            server,state,args=self.fixture(Path(td)); args.calls=1
            with patch.object(worker,'load_server',return_value=(server,state)):
                worker.solve(args)
            result=json.loads(args.output.read_text())
            self.assertEqual(result['status'],'budget_stop')
            self.assertEqual(result['calls'],1)
            self.assertIsNone(result['result'])

    def test_rank_mapping_failure_not_zero_effect(self):
        with tempfile.TemporaryDirectory() as td:
            server,state,args=self.fixture(Path(td),[[1,0],[2,0],[3,0]])
            with patch.object(worker,'load_server',return_value=(server,state)):
                worker.solve(args)
            result=json.loads(args.output.read_text())
            self.assertEqual(result['status'],'semantic_qualification_failure')
            self.assertNotIn('result',result)


if __name__=='__main__': unittest.main()
