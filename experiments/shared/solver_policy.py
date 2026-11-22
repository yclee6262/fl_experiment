"""Explicit numerical policy selection; historical callers stay unchanged."""
import hashlib
import json
from pathlib import Path

POLICY_PATH = Path(__file__).with_name('solver_policy_v1.json')
POLICIES = ('historical', 'float32_slsqp_v1')


def apply_solver_policy(server, name):
    if name not in POLICIES:
        raise ValueError(f'Unknown solver policy: {name}')
    spec = json.loads(POLICY_PATH.read_text())
    if name == 'float32_slsqp_v1':
        if server.constraint_mode != 'box_constrained':
            raise ValueError('Corrected policy requires explicit box constraints')
        if (server.feasible_lower, server.feasible_upper, server.feasibility_tolerance) != (-1., 1., 1e-8):
            raise ValueError('Corrected v1 policy requires [-1,1] and feasibility tolerance 1e-8')
        import torch
        if any(p.dtype != torch.float32 for a in server.trusted_agents for p in a.model.parameters()):
            raise ValueError('Corrected v1 policy requires float32 model parameters')
        server.coefficient_eps = spec['coefficient']['eps']
    else:
        server.coefficient_eps = None
    return {'name': name, 'coefficient_eps': server.coefficient_eps,
            'protocol_sha256': hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
            'budget_kind': 'iteration_limited_actual_queries_reported',
            'coefficient_maxiter': 200, 'custom_iterations': 30}
