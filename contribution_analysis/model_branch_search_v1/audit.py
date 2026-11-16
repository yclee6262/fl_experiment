"""Independent saved-ledger replay; no model API calls."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from contribution_analysis.model_branch_search_v1.run import OUT,save
import numpy as np
def main():
    teams=json.loads((OUT/'progress.json').read_text());reg=json.loads((OUT/'registration.json').read_text());cases={c['key']:c for c in reg['development']+reg['confirmation']}
    checked=0;calls=0;fullrounds=0
    for t in teams:
        c=cases[t['key']];a=np.array(c['alphas']);a/=a.sum()
        for label in ['round30','budget200','budget500','budget1000']:
            rr=json.loads((OUT/'records'/f"{t['key']}_{label}.json").read_text())
            for arm,r in rr.items():
                cache={};miss=0
                for e in r['events']:
                    k=(e['i'],np.array(e['x'],np.float64).tobytes());assert e['hit']==(k in cache)
                    if e['hit']:assert cache[k]==e['loss']
                    else:cache[k]=e['loss'];miss+=1
                assert miss==r['samples']==sum(r['stages'].values())<=r['budget'];calls+=miss
                def F(x):return float(a@np.array([cache[i,np.array(x,np.float64).tobytes()] for i in range(len(a))]))
                assert abs(F(r['x'])-r['loss'])<1e-10
                if arm!='O':
                    for z in r['rows']:
                        eps=1e-10*max(1,abs(z['before']));assert abs(F(z['x'])-z['loss'])<1e-10
                        assert z['loss']<=z['before']+eps
                        candidates=[z['before'],z['full_loss']]+([b['loss'] for b in z['branches']] if arm=='P' else [])
                        assert z['loss']<=min(candidates)+eps+1e-14
                        for b in z['branches']:assert abs(F(b['x'])-b['loss'])<1e-10
                        assert z['state']['history'][-1]==z['loss'] and z['state']['global_best_loss']==z['loss']
                        fullrounds+=1
                checked+=1
    completion=json.loads((OUT/'completion.json').read_text());controls=json.loads((OUT/'edge_controls.json').read_text())
    assert completion['physical']==calls+controls['physical'] and completion['complete']
    for p,h in json.loads((OUT/'source_lock.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    save(OUT/'independent_audit.json',dict(passed=True,executions=checked,controller_rounds=fullrounds,ledger_physical=calls,controls_physical=controls['physical'],no_model_calls=True,source_lock=True))
    save(OUT/'audit_source_lock.json',{str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4));cc=[t for t in teams if t['phase']=='confirmation']
    for arm in ['O','S','D','P']:
        xs=[];ys=[]
        for b in [200,500,1000]:
            xs.append(np.median([t['results'][f'budget{b}'][arm]['samples']/t['n'] for t in cc]));ys.append(np.median([t['results'][f'budget{b}'][arm]['loss']/max(t['results'][f'budget{b}'][arm]['initial'],1e-8) for t in cc]))
        ax.plot(xs,ys,'o-',label=arm)
    ax.set(xlabel='Median actual model evaluations / n',ylabel='Median delivered F / initial F',title='Budget snapshots (not a continuous trajectory)');ax.legend();fig.tight_layout();fig.savefig(OUT/'quality_actual_cost.png',dpi=180);plt.close(fig)
    print('Independent ledger audit passed',checked)
if __name__=='__main__':main()
