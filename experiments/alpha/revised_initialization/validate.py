"""Independently rescore saved endpoints and audit source/decision integrity."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from project_support.legacy import prepare
prepare()
from experiments.alpha.replay import load_base
from research_evaluation import load_evaluator_artifact, score_solution


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def lines(p): return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    out=args.run_dir/'endpoint_acceptance.json'
    if out.exists():raise FileExistsError(out)
    root=Path(__file__).resolve().parents[3]
    reg=read(args.run_dir/'registration.json')
    torch.set_num_threads(1)
    failures=[];checked=0;truths=[];metadata={};maxloss=0.;maxpool=0.;api=0
    for f,h in {**reg['input_sha256'],**reg['code_sha256']}.items():
        if sha(root/f)!=h: failures.append('hash:'+f)
    for b in reg['batches']:
        directory=Path(b['output'])
        manifest=read(directory/'manifest.json')
        if manifest['status']!='complete':failures.append('incomplete batch')
        rows=lines(directory/'results.jsonl');decisions=lines(directory/'decisions.jsonl')
        assert len(rows)==900 and len(decisions)==900
        assert len({(r['scenario'],r['seed'],r['method']) for r in rows})==900
        for row,decision in zip(rows,decisions):
            assert row['X']==decision['X'] and decision['pooled_error'] is None
            assert row['solver_policy']['name']=='float32_slsqp_v1' and row['solver_policy']['coefficient_eps']==1e-5
        for scenario in sorted({r['scenario'] for r in rows}):
            for seed in b['seeds']:
                server,item,artifact=load_base(b['source'],scenario,seed)
                evaluator,_=load_evaluator_artifact(artifact)
                metadata[str((artifact/'metadata.json').relative_to(root))]=sha(artifact/'metadata.json')
                selected=[r for r in rows if r['scenario']==scenario and r['seed']==seed]
                assert len(selected)==10
                for row in selected:
                    assert row['status']=='valid'
                    x=np.array(row['X']);a=np.array(row['alpha'])
                    assert np.isfinite(x).all() and server.box_violation(x)<=1e-8
                    assert np.isfinite(a).all() and min(a)>=0 and abs(a.sum()-1)<1e-10
                    server.alphas=a.tolist()
                    loss=server._consensus_loss(x);api+=len(a)
                    pool=score_solution(evaluator,x,0.)[1]
                    truth=server._true_target_error(x)
                    maxloss=max(maxloss,abs(loss-row['solver']['loss']));maxpool=max(maxpool,abs(pool-row['pooled_error']))
                    truths.append(dict(scenario=scenario,seed=seed,method=row['method'],truth_error=truth))
                    checked+=1
    assert checked==2700 and maxloss<1e-12 and maxpool<1e-12
    assert not failures,failures
    result=dict(passed=True,records=checked,source_hashes=len(reg['input_sha256']),code_hashes=len(reg['code_sha256']),
                max_loss_difference=maxloss,max_pooled_difference=maxpool,metadata_sha256=metadata,
                validation_participant_queries=api,validation_evaluator_queries=checked,
                validator_sha256=sha(Path(__file__)))
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='metadata_sha256'}))
if __name__=='__main__': main()
