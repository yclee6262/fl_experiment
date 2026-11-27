"""Clean local datasets and fixed qualification for the scaling exploration."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from core.models import PretrainedModel


def labels(x, gamma):
    x=np.asarray(x)
    d=x.shape[-1]
    return x.sum(-1)/np.sqrt(d)+gamma*(x[...,:-1]*x[...,1:]).sum(-1)/np.sqrt(d-1)


class LocalAPI:
    def __init__(self, model):self.model=model
    def api_predict(self,x):
        with torch.no_grad():
            t=torch.as_tensor(np.asarray(x),dtype=torch.float32)
            if t.ndim==1:t=t[None,:]
            return self.model(t).numpy().reshape(-1)


def build_pool(seed, dimension, gamma, directory, budget):
    import time
    from .run import save
    accepted=[];anchors=[];audit=[];started=time.perf_counter()
    for aid in range(40):
        budget.check();budget.trained+=1
        model_seed=int((seed*101+aid*1009+dimension*17)%(2**31-1))
        rng=np.random.default_rng(model_seed);torch.manual_seed(model_seed)
        shift=rng.uniform(-.15,.15,dimension)
        x=np.clip(rng.uniform(-1,1,(768,dimension))+shift,-1,1).astype(np.float32)
        y=(labels(x,gamma)+rng.normal(0,.02,768)).astype(np.float32)
        model=PretrainedModel(dimension).cpu();opt=torch.optim.Adam(model.parameters(),lr=.005)
        tx=torch.tensor(x[:512]);ty=torch.tensor(y[:512,None]);model.train()
        for epoch in range(40):
            perm=rng.permutation(512)
            for start in range(0,512,128):
                ids=perm[start:start+128];opt.zero_grad();loss=((model(tx[ids])-ty[ids])**2).mean();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():rmse=float(torch.sqrt(((model(torch.tensor(x[512:])).reshape(-1)-torch.tensor(y[512:]))**2).mean()))
        for p in model.parameters():p.requires_grad_(False)
        z=torch.tensor(rng.uniform(-.8,.8,dimension),dtype=torch.float32,requires_grad=True)
        inv=torch.optim.Adam([z],lr=.03);best=None;best_loss=float('inf')
        for step in range(301):
            value=model(z[None,:]).reshape(())
            error=float(abs(value.detach()))
            if error<best_loss:best_loss=error;best=z.detach().numpy().copy()
            if step==300:break
            inv.zero_grad();(value**2).backward();inv.step()
            with torch.no_grad():z.clamp_(-1,1)
        reasons=[]
        if not np.isfinite(rmse) or rmse>.20:reasons.append('validation_rmse')
        if not np.isfinite(best_loss) or best_loss>.02:reasons.append('anchor_loss')
        model_path=directory/f'candidate_{aid}.pt'
        torch.save(dict(state_dict=model.state_dict(),dimension=dimension,seed=model_seed),model_path)
        np.savez_compressed(directory/f'candidate_{aid}_data.npz',X=x,y=y)
        row=dict(agent_id=aid,model_seed=model_seed,rmse=rmse,anchor_loss=best_loss,anchor=best,
                 accepted=not reasons,reasons=reasons,model_path=str(model_path.resolve()))
        audit.append(row)
        if not reasons:accepted.append((aid,LocalAPI(model)));anchors.append(best)
        if len(accepted)==20:break
    save(directory/'qualification.json',audit)
    save(directory/'pool_completion.json',dict(attempted=len(audit),accepted=len(accepted),seconds=time.perf_counter()-started))
    return accepted,np.asarray(anchors),audit
