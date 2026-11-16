"""Offline JSON CLI. Run with python -m contribution_analysis.model_consensus_pool_v1.runner."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from .core import score

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True); p.add_argument('--output',required=True)
    args=p.parse_args(); started=time.perf_counter()
    raw=Path(args.input).read_bytes()
    # Refuse duplicate JSON keys rather than silently accepting the last value.
    def pairs(items):
        d={}
        for k,v in items:
            if k in d: raise ValueError('duplicate JSON key: '+k)
            d[k]=v
        return d
    data=json.loads(raw,object_pairs_hook=pairs)
    if not isinstance(data,dict) or set(data) != {'manifest','slots'}: raise ValueError('input must contain exactly manifest and slots')
    result=score(data.get('manifest'),data.get('slots'))
    sources={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path(__file__).parent.glob('*.py'))}
    output={'result':result,'provenance':{'input_sha256':hashlib.sha256(raw).hexdigest(),'source_sha256':sources},'cost':{'elapsed_seconds':time.perf_counter()-started,'solver_calls':0,'model_api_calls':0,'input_bytes':len(raw)}}
    with open(args.output,'x',encoding='utf8') as f: json.dump(output,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'output':args.output}))

if __name__=='__main__': main()
