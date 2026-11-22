import numpy as np
import pytest
import core
from host_server import HostServer
from experiments.shared.solver_policy import apply_solver_policy


def test_invalid_step_and_policy_rejected():
    for eps in [0,-1,np.nan,np.inf]:
        with pytest.raises(ValueError): HostServer(0,2,coefficient_eps=eps)
    s=HostServer(0,2)
    with pytest.raises(ValueError): apply_solver_policy(s,'float32_slsqp_v1')
    with pytest.raises(ValueError): apply_solver_policy(s,'unknown')


def test_explicit_step_recovers_quantized_objective():
    candidates=np.array([[1.]])
    objective=lambda x: float((np.float32(x[0])-np.float32(.25))**2)
    old=HostServer(0,1,constraint_mode='box_constrained')
    new=HostServer(0,1,constraint_mode='box_constrained')
    apply_solver_policy(new,'float32_slsqp_v1')
    a=old._solve_subspace_bfgs(candidates,objective,initial_solution=np.array([.75]))
    b=new._solve_subspace_bfgs(candidates,objective,initial_solution=np.array([.75]))
    assert a['loss']>.1
    assert b['loss']<1e-7
    assert new.box_violation(b['solution'])<=1e-8
    apply_solver_policy(new,'historical')
    c=new._solve_subspace_bfgs(candidates,objective,initial_solution=np.array([.75]))
    np.testing.assert_array_equal(a['solution'],c['solution'])
    assert a['loss']==c['loss']
