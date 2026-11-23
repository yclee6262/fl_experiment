"""Paired CPU training and bounded multi-start inversion for the task-8 study.

Each batched parameter slice is one ordinary 64-64 ReLU MLP.  The summed
per-agent MSE gives each slice the same gradient as a separately trained model.
"""
from collections import OrderedDict
import copy
import time
import numpy as np
import torch
from torch import nn
from core.models import PretrainedModel


KEYS = ('fc1.weight','fc1.bias','fc2.weight','fc2.bias','fc3.weight','fc3.bias')


def clone_state(state):
    return OrderedDict((k,v.detach().cpu().clone()) for k,v in state.items())


def initial_weights(dimension, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return clone_state(PretrainedModel(in_features=dimension).state_dict())


def epoch_permutations(records, seed, epoch):
    """Per-owner schedule remains identical when an owner subset is selected."""
    return [torch.from_numpy(np.random.default_rng(np.random.SeedSequence(
        [int(seed), int(epoch), int(record['agent_id']), 98137])).permutation(len(record['X'])))
        for record in records]


class BatchedMLP(nn.Module):
    def __init__(self, states):
        super().__init__()
        self.values = nn.ParameterList([nn.Parameter(torch.stack([s[k].detach().cpu() for s in states])) for k in KEYS])

    def forward(self, x):
        w1,b1,w2,b2,w3,b3 = self.values
        x=torch.relu(torch.bmm(x,w1.transpose(1,2))+b1[:,None,:])
        x=torch.relu(torch.bmm(x,w2.transpose(1,2))+b2[:,None,:])
        return (torch.bmm(x,w3.transpose(1,2))+b3[:,None,:]).squeeze(-1)

    def states(self):
        return [OrderedDict((k,v[i].detach().clone()) for k,v in zip(KEYS,self.values)) for i in range(len(self.values[0]))]

    def set_states(self, states):
        with torch.no_grad():
            for k,v in zip(KEYS,self.values):v.copy_(torch.stack([s[k] for s in states]))


def _data(records):
    if not records:raise ValueError('Training requires at least one participant')
    counts={len(r['X']) for r in records}
    dims={np.asarray(r['X']).shape[1] for r in records}
    if len(counts)!=1 or len(dims)!=1:raise ValueError('Batched study requires equal sample counts and dimensions')
    x=torch.stack([torch.as_tensor(np.asarray(r['X']),dtype=torch.float32) for r in records])
    y=torch.stack([torch.as_tensor(np.asarray(r['observed_y']),dtype=torch.float32) for r in records])
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('Training inputs must be finite')
    return x,y


def train_epoch(model, x, y, permutations, batch_size, lr):
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    perm=torch.stack(permutations)
    rows=torch.arange(len(x))[:,None]
    steps=0
    for begin in range(0,x.shape[1],batch_size):
        ix=perm[:,begin:begin+batch_size]
        xb=x[rows,ix];yb=y[rows,ix]
        optimizer.zero_grad(set_to_none=True)
        loss=((model(xb)-yb)**2).mean(dim=1).sum()
        loss.backward();optimizer.step();steps+=1
    return steps


def _metrics(records, epochs, batch_size, seconds):
    n=len(records);samples=sum(len(r['X']) for r in records)
    steps=(len(records[0]['X'])+batch_size-1)//batch_size
    return dict(participants=n,epochs=epochs,rounds=epochs,batch_size=batch_size,
                training_forward_samples=samples*epochs,training_backward_samples=samples*epochs,
                optimizer_steps_per_agent=steps*epochs,optimizer_steps_all_agents=n*steps*epochs,
                batched_optimizer_steps=steps*epochs,optimizer_reset_each_epoch=True,
                aggregation='equal_weight',seconds=float(seconds),precision='float32',device='cpu')


def _controls(epochs,batch_size,lr):
    if epochs<1 or batch_size<1 or not np.isfinite(lr) or lr<=0:raise ValueError('Invalid training controls')


def train_paired(records, seed, epochs=30, batch_size=32, lr=.01, initial_state=None):
    _controls(epochs,batch_size,lr)
    start=time.perf_counter();x,y=_data(records)
    state=initial_weights(x.shape[-1],seed) if initial_state is None else clone_state(initial_state)
    local=BatchedMLP([state]*len(records));fed=BatchedMLP([state]*len(records))
    for epoch in range(epochs):
        permutations=epoch_permutations(records,seed,epoch)
        train_epoch(local,x,y,permutations,batch_size,lr)
        train_epoch(fed,x,y,permutations,batch_size,lr)
        individual=fed.states()
        average=OrderedDict((k,torch.stack([s[k] for s in individual]).mean(0)) for k in KEYS)
        fed.set_states([average]*len(records))
    metrics=_metrics(records,epochs,batch_size,time.perf_counter()-start)
    metrics['training_forward_samples_combined']=2*metrics['training_forward_samples']
    metrics['training_backward_samples_combined']=2*metrics['training_backward_samples']
    metrics['optimizer_steps_all_agents_combined']=2*metrics['optimizer_steps_all_agents']
    metrics['initialization']='same_state_for_all_local_models_and_fedavg'
    return dict(local_states=local.states(),fedavg_state=clone_state(average),initial_state=state,metrics=metrics)


def train_fedavg(records, seed, initial_state, epochs=30, batch_size=32, lr=.01):
    _controls(epochs,batch_size,lr)
    start=time.perf_counter();x,y=_data(records);state=clone_state(initial_state)
    model=BatchedMLP([state]*len(records))
    for epoch in range(epochs):
        train_epoch(model,x,y,epoch_permutations(records,seed,epoch),batch_size,lr)
        individual=model.states()
        state=OrderedDict((k,torch.stack([s[k] for s in individual]).mean(0)) for k in KEYS)
        model.set_states([state]*len(records))
    return dict(state=clone_state(state),metrics=_metrics(records,epochs,batch_size,time.perf_counter()-start))


def invert_model(model, target, seed, starts=5, steps=500, lr=.01):
    """Run fixed starts, select final endpoint by this model's target residual.

    The caller's model, parameter gradients, requires_grad flags and mode are
    preserved. All start points are box feasible before their first evaluation.
    """
    if starts<1 or steps<1 or not np.isfinite(target) or not np.isfinite(lr) or lr<=0:
        raise ValueError('Invalid inversion controls')
    net=copy.deepcopy(model).cpu().eval()
    for p in net.parameters():p.requires_grad_(False)
    dimension=net.fc1.in_features
    rng=np.random.default_rng(seed)
    results=[];forward=0;backward=0
    for start in range(starts):
        x=torch.tensor(np.clip(rng.normal(size=(1,dimension)),-1,1),dtype=torch.float32,requires_grad=True)
        initial=x.detach().numpy().reshape(-1).tolist()
        optimizer=torch.optim.Adam([x],lr=lr)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            output=net(x);forward+=1
            loss=((output-target)**2).mean()
            if not torch.isfinite(loss):break
            loss.backward();backward+=1;optimizer.step()
            with torch.no_grad():x.clamp_(-1,1)
        with torch.no_grad():
            value=float(net(x).item());forward+=1
        xx=x.detach().numpy().reshape(-1)
        valid=bool(np.isfinite(xx).all() and np.isfinite(value) and np.max(abs(xx))<=1)
        results.append(dict(start=start,initial_X=initial,X=xx.tolist(),prediction=value,
                            loss=abs(value-target) if np.isfinite(value) else None,valid=valid))
    valid=[r for r in results if r['valid']]
    best=min(valid,key=lambda r:r['loss']) if valid else None
    return dict(X=None if best is None else best['X'],loss=None if best is None else best['loss'],
                status='valid' if best else 'no_valid_inversion',start_results=results,
                forward_samples=forward,backward_steps=backward,
                selection_forward_samples=starts,starts=starts,steps_per_start=steps,
                scope='local_gradient_inversion_not_remote_black_box_queries')
