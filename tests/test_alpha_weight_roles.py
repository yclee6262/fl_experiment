import copy
import numpy as np
import torch
from core.models import PretrainedModel
from core.host_server import HostServer
from experiments.alpha.weight_roles.run import ShiftedModel, custom_search
from experiments.baselines.paired_benchmark.search import run_search


def test_output_shift_preserves_original_parameters_and_input_gradient():
    torch.manual_seed(11)
    base = PretrainedModel(2).eval()
    saved = copy.deepcopy(base.state_dict())
    shifted = ShiftedModel(base, .5)
    x = torch.tensor([[.25, -.4]], requires_grad=True)
    expected = base(x)
    actual = shifted(x)
    np.testing.assert_allclose(actual.detach().numpy() - expected.detach().numpy(), .5, atol=2e-7)
    g1 = torch.autograd.grad(expected.sum(), x, retain_graph=True)[0]
    g2 = torch.autograd.grad(actual.sum(), x)[0]
    torch.testing.assert_close(g1, g2)
    for k, v in base.state_dict().items():
        torch.testing.assert_close(v, saved[k], rtol=0, atol=0)


def example_server():
    class Agent:
        def __init__(self, offset): self.offset = offset
        def api_predict(self, x):
            x = np.asarray(x)
            return np.array([x[0] + 2*x[1] + self.offset])
    server = HostServer(.5, 2, constraint_mode='box_constrained', coefficient_eps=1e-5)
    server.trusted_agents = [Agent(0), Agent(.4), Agent(-.2)]
    server.I_list = [np.array([-.5, .1]), np.array([.7, .4]), np.array([.2, -.4])]
    server.alphas = [.6, .3, .1]
    return server


def test_separate_weights_reproduce_coupled_branch_and_enforce_objective():
    s = example_server()
    coupled = run_search(s, 'best_of', 80)
    branch = next(r for r in coupled['engine_details'] if r['method'] == 'custom')
    same = custom_search(s, s.alphas, s.alphas, 40)
    assert same['X'] == branch['X'] and same['loss'] == branch['loss']
    assert same['model_queries'] == branch['model_queries']
    split = custom_search(s, [1/3]*3, [.9, .05, .05], 40)
    assert split['panel_evaluations'] <= 40
    assert s.alphas == [.6, .3, .1]
    for e in split['events']:
        if e.get('loss') is not None:
            assert np.isclose(e['loss'], np.mean(abs(np.array(e['predictions']) - s.target_T)))


def test_factorial_shared_branches_are_decoupled_from_best_of_selection():
    s = example_server()
    result = run_search(s, 'best_of', 80)
    custom = next(r for r in result['engine_details'] if r['method'] == 'custom')
    events = [e for e in result['events'] if e['engine'] == 'custom' and e.get('loss') is not None]
    assert custom['loss'] == min(e['loss'] for e in events)
    # The custom factorial endpoint is recovered even if another branch wins best_of.
    assert custom['X'] is not None and custom['panel_budget'] == 40
    assert result['model_queries'] == sum(r['model_queries'] for r in result['engine_details'])


def test_direction_changes_path_with_identical_objective_and_anchors():
    s = example_server()
    u = [1/3]*3
    first = custom_search(s, u, [.9, .05, .05], 80)
    second = custom_search(s, u, [.05, .9, .05], 80)
    x0 = np.mean(s.I_list, axis=0).tolist()
    assert first['events'][0]['X'] == second['events'][0]['X'] == x0
    # Five initial panels are current point, three anchors, current point again.
    assert first['events'][5]['X'] != second['events'][5]['X']
    for result in [first, second]:
        for event in result['events']:
            if event.get('loss') is not None:
                assert np.isclose(event['loss'], np.mean(abs(np.array(event['predictions'])-.5)))
