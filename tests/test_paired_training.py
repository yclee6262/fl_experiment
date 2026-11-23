import copy
import numpy as np
import torch
from core.models import PretrainedModel
from experiments.baselines.paired_benchmark.training import (
    initial_weights, train_paired, train_fedavg, epoch_permutations, invert_model)


def records():
    rng=np.random.default_rng(8)
    return [dict(agent_id=i+1,X=rng.uniform(-1,1,(20,2)),observed_y=rng.normal(size=20)) for i in range(3)]


def test_batched_epoch_matches_separate_models_and_fedavg():
    torch.set_num_threads(1)
    rr=records();state=initial_weights(2,37)
    actual=train_paired(rr,37,epochs=1,batch_size=8,initial_state=state)
    expected=[]
    for r,perm in zip(rr,epoch_permutations(rr,37,0)):
        model=PretrainedModel(2);model.load_state_dict(state)
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        x=torch.tensor(r['X'],dtype=torch.float32);y=torch.tensor(r['observed_y'],dtype=torch.float32)
        for b in range(0,len(x),8):
            ix=perm[b:b+8];optimizer.zero_grad()
            loss=((model(x[ix]).flatten()-y[ix])**2).mean();loss.backward();optimizer.step()
        expected.append(model.state_dict())
    for got,expected_state in zip(actual['local_states'],expected):
        for key in got:torch.testing.assert_close(got[key],expected_state[key],atol=2e-6,rtol=2e-5)
        left=PretrainedModel(2);left.load_state_dict(got)
        right=PretrainedModel(2);right.load_state_dict(expected_state)
        probe=torch.tensor(rr[0]['X'],dtype=torch.float32)
        torch.testing.assert_close(left(probe),right(probe),atol=2e-6,rtol=2e-5)
    for key in state:
        mean=torch.stack([s[key] for s in expected]).mean(0)
        torch.testing.assert_close(actual['fedavg_state'][key],mean,atol=2e-6,rtol=2e-5)
    metrics=actual['metrics']
    assert metrics['training_forward_samples']==60 and metrics['optimizer_steps_all_agents']==9


def test_pairing_determinism_and_subset_schedule():
    rr=records();first=train_paired(rr,41,epochs=2,batch_size=8)
    second=train_fedavg(rr,41,first['initial_state'],epochs=2,batch_size=8)
    for k in first['fedavg_state']:torch.testing.assert_close(first['fedavg_state'][k],second['state'][k],atol=0,rtol=0)
    np.testing.assert_array_equal(epoch_permutations(rr,41,1)[2],epoch_permutations([rr[2]],41,1)[0])


def test_inversion_preserves_model_and_selects_model_loss():
    model=PretrainedModel(2);before=copy.deepcopy(model.state_dict());model.train()
    flags=[p.requires_grad for p in model.parameters()]
    a=invert_model(model,0.,32,starts=3,steps=5)
    b=invert_model(model,0.,32,starts=3,steps=5)
    assert a==b and a['status']=='valid'
    assert a['loss']==min(v['loss'] for v in a['start_results'])
    assert a['forward_samples']==18 and a['backward_steps']==15
    assert max(abs(np.array(a['X'])))<=1
    assert model.training and [p.requires_grad for p in model.parameters()]==flags
    for key in before:torch.testing.assert_close(before[key],model.state_dict()[key],atol=0,rtol=0)
