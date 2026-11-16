"""Independent original-solver replay, manual joint branches, and exact billing."""
import copy
from common import *

def verify(phase):
    from study import load_server
    from host_server import HostServer
    from independent_interpreter import walk,reset,compare
    check();lock=read(ROOT/(phase+'_locked.json'))
    for p,h in lock['hashes'].items():assert sha(ROOT/p)==h
    assert sha(ROOT/(phase+'_scores.json'))==lock['scores_sha256']
    previous=read(ROOT/'pilot_verification.json')['model_reload_samples'] if phase=='development' else 0
    budget=Budget(1000000-previous,600);error=0.;cases=0;branches=0;expected=[]
    historical={r['key']:r for r in read(FOUR/'scores_and_costs.json')}
    for c in read(ROOT/'cases.json')[phase]:
        print('VERIFY',phase,c['key'],flush=True);s=load_server({k:v for k,v in c.items() if k!='evaluator'});ids=list(range(len(c['ids'])))
        for si,x in enumerate(starts(c)):
            for method in ['J','LOO']:
                r=read(ROOT/phase/(c['key']+f'_s{si}_{method}.json'));raw={key(v['i'],v['x']):v['y'] for v in r['raw']}
                for v in r['raw']:
                    budget.before();y=float(s.trusted_agents[v['i']].api_predict(np.asarray(v['x']))[0]);budget.samples+=1;assert y==v['y']
                calls=[]
                def obj(panel):
                    weights=np.asarray(c['alphas'])[panel];weights=weights/weights.sum()
                    def f(point):
                        keys=[key(i,point) for i in panel];calls.extend(keys)
                        return float(np.dot(weights,[abs(raw[k]) for k in keys]))
                    return f
                f=obj(ids);calls.clear()
                baseline=HostServer._solve_subspace_custom(s,s.I_list,f,num_iterations=30,direction_weights=s.alphas,initial_solution=x)
                assert plain(baseline)==r['baseline'];assert calls==[e['key'] for e in r['events'] if e['stage']=='baseline']
                if method=='J':
                    loss=f(x);initial=dict(S_current=x,best_loss=loss,global_best_S=x,global_best_loss=loss,eta=.1,current_method='secant',delta=.0001,
                        history=[loss],states=['Start'],anchor_losses=[f(a) for a in s.I_list],next_iteration=0,halted=False)
                    full=walk(s,f,initial,30);error=max(error,compare(full,r['baseline_trace']))
                totals=np.zeros(len(ids))
                for b in r['branches']:
                    i=b['i'];panel=[j for j in ids if j!=i];ss=copy.copy(s);ss.I_list=np.asarray(s.I_list)[panel];ww=np.asarray(s.alphas)[panel];ss.alphas=ww/ww.sum();g=obj(panel)
                    if method=='J':
                        state=reset(full['steps'][b['t']]['pre'],g,ss.I_list);state['states']=['Local Start']
                        branch=walk(ss,g,state,1);error=max(error,compare(branch,b['trace']));endpoint=branch['result']['solution']
                        original=full['steps'][b['t']]['post']['S_current'];error=max(error,near(original,b['x_full']))
                    else:
                        calls.clear();br=HostServer._solve_subspace_custom(ss,ss.I_list,g,num_iterations=30,direction_weights=ss.alphas,initial_solution=x)
                        assert plain(br)==b['result'];endpoint=br['solution'];original=baseline['solution']
                        assert calls==[e['key'] for e in r['events'] if e['stage']=='decision' and e['i']==i]
                    error=max(error,near(endpoint,b['x_minus']));score=f(endpoint)-f(np.array(original));error=max(error,near(score,b['score']));totals[i]+=score;branches+=1
                error=max(error,near(totals,r['values']));seen=set();charged=0;base=0
                for e in r['events']:
                    hit=e['stage']!='baseline' and e['key'] in seen;assert hit==e['hit'];charged+=not hit;seen.add(e['key'])
                    if e['stage']=='baseline':base+=1
                    if e['stage']=='decision':assert int(e['key'].split(':')[0])!=e['i']
                assert charged==r['samples'] and base==r['baseline_samples'] and charged-base==r['extra_samples']
                if phase=='development' and si==0:
                    old=historical[c['key']]['schemes']['S3' if method=='J' else 'S4'];field='J' if method=='J' else 'D_L'
                    error=max(error,near(totals,[v['values'][field] for v in old['rows']]));assert old['samples']==r['samples']
                expected.append(dict(key=c['key'],start=si,method=method,values=totals,samples=charged));cases+=1
    save(ROOT/(phase+'_verification.json'),dict(accepted=True,cases=cases,branches=branches,max_error=error,model_reload_samples=budget.samples,
        raw_reload_max_error=0,wall_seconds=time.perf_counter()-budget.start,pooled_loaded=False,independent_results=expected));check()
if __name__=='__main__':setup();verify(sys.argv[1])
