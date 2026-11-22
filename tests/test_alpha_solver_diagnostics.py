"""Regression checks for diagnostic isolation and precision adapters."""
import sys
import numpy as np
import pytest
import torch
from experiments.alpha.solver_diagnostics.run import difference_step, prepare_precision, host_module
from agent_client import AgentNode
from host_server import HostServer


def test_scoped_step_restores_after_error():
    original=host_module.minimize
    with pytest.raises(RuntimeError):
        with difference_step(1e-4):
            assert host_module.minimize is not original
            raise RuntimeError('test')
    assert host_module.minimize is original


def test_precision_adapter_matches_api_and_double_model():
    torch.manual_seed(19)
    agent=AgentNode(1,None,n_features=2)
    x=np.array([[0.1,-0.3],[0.2,0.8]])
    expected=agent.api_predict(x)
    server=HostServer(0,2)
    server.trusted_agents=[agent]
    count=prepare_precision(server,'float32')
    np.testing.assert_array_equal(agent.api_predict(x),expected)
    assert count['requests']==1 and count['input_samples']==2
    with torch.no_grad():
        expected_double=agent.model.double()(torch.tensor(x,dtype=torch.float64)).numpy().flatten()
    prepare_precision(server,'float64')
    np.testing.assert_array_equal(agent.api_predict(x),expected_double)


def test_existing_output_rejected(tmp_path, monkeypatch):
    sentinel=tmp_path/'keep'
    sentinel.write_text('unchanged')
    from experiments.alpha.solver_diagnostics.run import main
    monkeypatch.setattr(sys, 'argv', ['diagnostic', '--source', str(tmp_path/'missing'), '--output-dir', str(tmp_path)])
    with pytest.raises(FileExistsError):
        main()
    assert sentinel.read_text()=='unchanged'
