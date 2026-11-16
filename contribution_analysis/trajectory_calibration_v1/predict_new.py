"""One normal solve -> locked predictions; no evaluator or LOO data loaded."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main(args):
    root=args.root.resolve();sys.path.insert(0,str(root/'source_snapshot'))
    import torch
    from recorder import solve,features,FEATURES
    from experiment import load_server,read,save,sha
    import rules
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    state=read(args.directory/'preparation.json');model_manifest=read(root/'fitted_models_manifest.json')
    if model_manifest['features']!=FEATURES:raise ValueError('feature_schema_mismatch')
    for name,h in model_manifest['model_sha256'].items():
        if sha(root/name)!=h:raise ValueError('prediction_model_hash_mismatch')
    case=dict(ids=state['ids'],alphas=state['alphas'],candidates=state['candidates'],
              models=str(args.directory/state['artifact']/'local_models.pt'))
    server=load_server(case)
    r=solve(server,True,'objective',initial_solution=state['fixed_start'],
            max_calls=args.calls,max_samples=args.calls,max_seconds=args.seconds)
    begin=time.perf_counter();z=features(r['trace'],server.I_list,server.alphas)
    predictions={'zero':np.zeros(len(z)),
       'constant':rules.constant_predict(read(root/'constant_development_model.json'),z[:,5].astype(int))}
    for name in ['full','model_only','candidate_only']:
        predictions[name]=rules.predict(read(root/(name+'_development_model.json')),z)
    feature_and_prediction_seconds=time.perf_counter()-begin
    # This immutable file must exist before any full-removal counterfactual launches.
    save(args.directory/'predictions.json',dict(status='locked',seed=state['seed'],scenario=state['scenario'],
        ids=state['ids'],features=z,predictions=predictions,evaluator_loaded=False,LOO_loaded=False,
        fitted_models_sha256=model_manifest['model_sha256']))
    save(args.output,dict(status='valid_delivery',seed=state['seed'],scenario=state['scenario'],
          result=r['result'],counter=r['counter'],calls=r['counter']['calls'],
          input_samples=r['counter']['samples'],seconds=r['seconds'],
          feature_and_prediction_seconds=feature_and_prediction_seconds,trace=r['trace'],
          fixed_start=state['fixed_start'],remaining_ids=state['ids'],alphas=state['alphas'],
          evaluator_loaded=False,predictions_sha256=sha(args.directory/'predictions.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--calls',type=int,default=20000);p.add_argument('--seconds',type=float,default=60.)
    args=p.parse_args()
    try:main(args)
    except Exception as exc:
        if not args.output.exists():
            with args.output.open('x') as f:json.dump(dict(status='worker_failure',error=f'{type(exc).__name__}:{exc}'),f)
        raise
