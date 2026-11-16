"""Separate post-lock evaluator reload; no decision/score functions imported."""
import time
import numpy as np
from study import ROOT,read,save,setup,check,sha

def recheck():
    import torch
    from models import PretrainedModel
    check();start=time.perf_counter();records={r['key']:r for r in read(ROOT/'pooled_evaluation.json')}
    maximum=0.;points=0
    for c in read(ROOT/'cases.json')['development']:
        net=PretrainedModel(len(c['fixed_start']));net.load_state_dict(torch.load(c['evaluator'],map_location='cpu',weights_only=True)['state_dict']);net.eval()
        for value in records[c['key']]['values'].values():
            with torch.no_grad():pred=float(net(torch.as_tensor(np.asarray(value['x'],np.float32)).reshape(1,-1)).flatten()[0])
            error=abs(abs(pred)-value['E']);maximum=max(maximum,error);assert error==0;points+=1
    assert points*3<=12000
    save(ROOT/'pooled_independent_recheck.json',dict(accepted=True,points=points,max_error=maximum,
        all_pooled_evaluations=points*3,wall_seconds=time.perf_counter()-start,source_sha256=sha(__file__),
        postscore_verification_only=True))
    print('INDEPENDENT_POOLED_RECHECK',points,maximum,flush=True)

if __name__=='__main__':setup();recheck()
