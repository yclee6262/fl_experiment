"""Actual frozen-search controls: identical and duplicate scorer predictions."""
import sys,json,copy,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from contribution_analysis.model_pool_v_cm_validation_v1.run import OUT,execute,analytic_server,save
import numpy as np
def main():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    dest=OUT/'controlled';dest.mkdir(exist_ok=False)
    seed=json.loads((OUT/'registration.json').read_text())['fresh'][0];results={}
    for label in ['identical','duplicate']:
        c=copy.deepcopy(seed);c['key']='control_'+label;c['phase']='controlled'
        c['coefficients'][1]=copy.deepcopy(c['coefficients'][0])
        if label=='identical':c['coefficients'][2]=copy.deepcopy(c['coefficients'][0])
        r=execute(c,analytic_server(c),30);save(dest/f'{label}.json',r)
        by={(z['t'],z['i']):z for z in r['rows']}
        if label=='identical':
            error=max(abs(z[m]['total']) for z in r['rows'] for m in ['V','CM'])
        else:
            error=max(abs(by[t,0][m]['total']-by[t,1][m]['total']) for t in range(len(r['rows'])//3) for m in ['V','CM'])
        assert error<1e-10,(label,error)
        results[label]=dict(passed=True,max_error=error,owner_rounds=len(r['rows']),samples=r['samples'])
    save(dest/'audit.json',results);save(dest/'source_lock.json',{str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(results)
if __name__=='__main__':main()
