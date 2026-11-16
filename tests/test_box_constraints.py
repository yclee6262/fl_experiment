import unittest
import numpy as np
from host_server import HostServer


class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def api_predict(self, x):
        values = np.asarray(x, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        return values.sum(axis=1)

    def get_minimum_bid(self):
        return 1.0

class BoxTests(unittest.TestCase):
    def server(self, lower=-1, upper=1):
        return HostServer(0, 2, constraint_mode="box_constrained", feasible_lower=lower, feasible_upper=upper)

    def test_beta_boundary_and_rank_deficiency(self):
        server = self.server()
        for candidates in ([[1.,0.]], [[1.,0.],[1.,0.]], [[1.,0.],[0.,1.]]):
            result = server._solve_subspace_bfgs(candidates, lambda x: float(np.sum((x-2)**2)))
            self.assertLessEqual(server.box_violation(result["solution"]), 1e-8)
            self.assertEqual(result["actual_solver"], "SLSQP")

    def test_custom_never_queries_outside_box(self):
        server = self.server()
        def objective(x):
            self.assertLessEqual(server.box_violation(x), 1e-8)
            return float(np.sum((x-2)**2))
        result = server._solve_subspace_custom([[1.,0.],[0.,1.]], objective, num_iterations=30)
        self.assertLessEqual(server.box_violation(result["solution"]), 1e-8)

    def test_infeasible_span(self):
        server = self.server(0.5, 1.)
        with self.assertRaisesRegex(ValueError, "No feasible"):
            server._solve_subspace_bfgs([[0.,0.]], lambda x: float(np.sum(x*x)))

    def test_bad_custom_start(self):
        with self.assertRaisesRegex(ValueError, "warm start"):
            self.server()._solve_subspace_custom([[0.,0.]], lambda x: 0., initial_solution=[2.,0.])

    def test_feasible_warm_start(self):
        result = self.server()._solve_subspace_bfgs([[1.,0.],[0.,1.]], lambda x: float(np.sum(x*x)), initial_solution=[0.5,0.5])
        self.assertLessEqual(np.max(np.abs(result["solution"])),1.)

    def test_exclusion_reuses_box_constraint(self):
        server = self.server()
        server.trusted_agents = [Agent(1), Agent(2), Agent(3)]
        server.alphas = [1 / 3] * 3
        server.I_list = [np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]),
                         np.asarray([-1.0, -1.0])]
        full = server.optimize_candidate_subspace(strategy="best_of")["solution"]
        _, reports, _ = server._compute_exclusion_reports(
            full, evaluation_mode="uniform", return_summary=True
        )
        for report in reports:
            self.assertLessEqual(
                server.box_violation(report["restricted_solution"]), 1e-8
            )

    def test_best_of_discards_infeasible_engine_result(self):
        server = self.server()
        server.trusted_agents = [Agent(1)]
        server.alphas = [1.0]
        server.I_list = [np.asarray([0.0, 0.0])]
        server._solve_subspace_bfgs = lambda *args, **kwargs: {
            "solution": np.asarray([2.0, 0.0]), "loss": 0.0, "success": True,
            "message": "", "history": [], "states": [], "actual_solver": "mock",
        }
        server._solve_subspace_custom = lambda *args, **kwargs: {
            "solution": np.asarray([0.5, 0.0]), "loss": 1.0, "success": True,
            "message": "", "history": [], "states": [], "actual_solver": "mock",
        }
        result = server.optimize_candidate_subspace(strategy="best_of")
        np.testing.assert_allclose(result["solution"], [0.5, 0.0])
        self.assertTrue(result["valid_delivery"])

if __name__ == "__main__": unittest.main()
