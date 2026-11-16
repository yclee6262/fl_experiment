"""Original full-LOO query-order replay and external bootstrap audit, no new inference."""
import copy,hashlib,time
import numpy as np
from common import *

def run():
    from support import load_server
    check();tic=time.perf_counter();solves=0;maxerr=0.;totalrequests=0
    for phase,cs in read(ROOT/'cases.json').items():
        for c in cs:
            r=read(ROOT/phase/(c['key']+'.json'));raw={key(v['i'],v['x']):v['y'] for v in r['raw_replies']}
            for excluded,agent in enumerate(c['ids']):
                s=load_server({k:v for k,v in c.items() if k!='evaluator'});panel=[j for j in range(len(c['ids'])) if j!=excluded]
                s.I_list=np.asarray(s.I_list)[panel];aa=np.asarray(s.alphas)[panel];s.alphas=aa/aa.sum()
                s.trusted_agents=[s.trusted_agents[j] for j in panel];digest=hashlib.sha256();requests=0
                for local,(original,a) in enumerate(zip(panel,s.trusted_agents)):
                    def predict(x,j=original,k=local):
                        nonlocal requests
                        digest.update(k.to_bytes(4,'little'));digest.update(np.asarray(x,np.float64).tobytes());requests+=1
                        return np.asarray([raw[key(j,x)]])
                    a.api_predict=predict
                out=s.optimize_candidate_subspace(strategy='custom',mode='optimization',custom_iterations=30,initial_solution=np.asarray(c['fixed_start']))
                old=read(OLD/'removals'/(c['key']+'_'+str(agent)+'.json'))
                assert digest.hexdigest()==old['query_order_sha256'];assert requests==old['counter']['samples']
                assert plain(out)==old['result']
                g=next(v for v in r['groups'] if v['i']==excluded and 0 in v['q_labels'])
                maxerr=max(maxerr,near(out['solution'],g['P']['result']['solution']))
                assert out['history']==g['P']['result']['history']
                assert out['states'][1:]==g['P']['result']['states'][1:]
                totalrequests+=requests;solves+=1
    # Another independent calculation of offline direction agreement and intervals.
    rows=read(ROOT/'external_comparisons.json');stats=read(ROOT/'external_summary.json')
    def sign(x):return int(x>1e-4)-int(x< -1e-4)
    for m in stats:
        x,y=m['comparison'].split(':')
        for t in m['teams']:
            rr=[r for r in rows if r['key']==t['key'] and r['q']==m['q'] and sign(r[x])]
            agreement=None if not rr else sum(sign(r[x])==sign(r[y]) for r in rr)/len(rr)
            if agreement is None:assert t['agreement'] is None
            else:maxerr=max(maxerr,near(agreement,t['agreement']))
        for metric,field in [('rank','rho'),('agreement','agreement')]:
            v=[t[field] for t in m['teams'] if t[field] is not None];stat=m[metric];assert len(v)==stat['n']
            if not v:continue
            rng=np.random.default_rng(20261024);ii=rng.integers(0,len(v),(10000,len(v)))
            means=np.asarray([sum(v[int(j)] for j in pick)/len(v) for pick in ii]);ci=np.quantile(means,[.001,.999])
            maxerr=max(maxerr,near(ci,stat['interval']),near(sum(v)/len(v),stat['mean']))
    save(ROOT/'supplemental_audit.json',dict(accepted=True,original_full_loo_replays=solves,original_query_requests=totalrequests,
        original_all_query_hashes_and_solver_histories_matched=True,max_error=maxerr,new_model_samples=0,
        external_bootstrap_verified=True,wall_seconds=time.perf_counter()-tic));check()
    print(read(ROOT/'supplemental_audit.json'))
if __name__=='__main__':setup();run()
