"""Compare one isolated P5 replay with its original seed artifacts and outputs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def load_rows(path, seed=None):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["method_label"], int(row["seed"])): row for row in rows
            if seed is None or int(row["seed"]) == seed}


def _artifact_for_seed(root, filename, seed):
    for metadata_path in (root / "artifacts").glob("scenario_*/metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        if int(metadata.get("scenario", {}).get("seed", -1)) == seed:
            artifact = metadata_path.parent / filename
            if artifact.is_file():
                return artifact
    raise FileNotFoundError(f"missing {filename} artifact for seed {seed} in {root}")


def compare(original, replay, tolerance=1e-7, seed=None):
    original,replay=Path(original),Path(replay)
    left=load_rows(original/"summary.csv")
    right=load_rows(replay/"summary.csv")
    replay_seeds={key[1] for key in right}
    if seed is None:
        if len(replay_seeds)!=1:raise ValueError("Replay must contain exactly one seed or pass --seed")
        seed=next(iter(replay_seeds))
    right=load_rows(replay/"summary.csv",seed)
    left={key:value for key,value in left.items() if key[1]==seed}
    if not right:raise ValueError(f"Replay has no rows for seed {seed}")
    original_art=_artifact_for_seed(original,"pooled_data.npz",seed)
    replay_art=_artifact_for_seed(replay,"pooled_data.npz",seed)
    with np.load(original_art) as original_data,np.load(replay_art) as replay_data:
        if set(original_data.files)!=set(replay_data.files):raise AssertionError("pooled-data array schema differs")
        data_equal={name:bool(np.array_equal(original_data[name],replay_data[name])) for name in original_data.files}
    max_model_delta=0.0
    for filename in ("local_models.pt","global_evaluator.pt"):
        pa=_artifact_for_seed(original,filename,seed)
        pb=_artifact_for_seed(replay,filename,seed)
        a=torch.load(pa,map_location="cpu",weights_only=True);b=torch.load(pb,map_location="cpu",weights_only=True)
        def walk(x,y):
            if isinstance(x,dict):
                if set(x)!=set(y):raise AssertionError(f"{filename} model state keys differ")
                for k in x:walk(x[k],y[k])
            elif torch.is_tensor(x):
                nonlocal max_model_delta
                if x.shape!=y.shape:raise AssertionError(f"{filename} model shape differs")
                if x.numel():max_model_delta=max(max_model_delta,float(torch.max(torch.abs(x.float()-y.float()))))
        state_a=a.get("agents",a.get("state_dict"));state_b=b.get("agents",b.get("state_dict"));walk(state_a,state_b)
    if set(left)!=set(right):raise AssertionError(f"condition keys differ: {set(left)^set(right)}")
    x_alpha_deltas=[];score_deltas=[];discrete=[]
    for key,a in left.items():
        b=right[key]
        for field in ("final_solution","final_optimization_weights","final_contribution_weights"):
            va=np.asarray(json.loads(a[field]),dtype=float);vb=np.asarray(json.loads(b[field]),dtype=float)
            if va.shape!=vb.shape:raise AssertionError(f"{key} {field} shape differs")
            x_alpha_deltas.append(float(np.max(np.abs(va-vb))) if va.size else 0.0)
        for field in ("external_global_error","final_target_error"):
            score_deltas.append(abs(float(a[field])-float(b[field])))
        for field in ("delivery_status","stop_reason","converged","final_accepted_round","final_stage3_actual_solver"):
            if a.get(field)!=b.get(field):discrete.append({"condition":key,"field":field,"original":a.get(field),"replay":b.get(field)})
    result={"original":str(original),"replay":str(replay),"seed":seed,"pooled_data_exactly_equal":all(data_equal.values()),
        "pooled_data_arrays":data_equal,"max_model_weight_delta":max_model_delta,
        "max_x_alpha_q_delta":max(x_alpha_deltas,default=0.0),"max_scoring_delta":max(score_deltas,default=0.0),
        "discrete_decision_differences":discrete,
        "passed":all(data_equal.values()) and max_model_delta<=tolerance and max(x_alpha_deltas,default=0)<=tolerance
            and max(score_deltas,default=0)<=tolerance and not discrete,
        "tolerance":tolerance,"time_comparison":"not required"}
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("original");parser.add_argument("replay");parser.add_argument("--output",required=True);parser.add_argument("--seed",type=int)
    args=parser.parse_args();result=compare(args.original,args.replay,seed=args.seed)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
