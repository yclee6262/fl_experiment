"""Edge controls for actual frozen branch/controller integration."""
import sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from contribution_analysis.model_branch_search_v1.run import *
def main():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    dev,_=cases();c=copy.deepcopy(dev[0]);s=analytic_server(c)
    q=Replies(s,10**9)
    native=HostServer._solve_subspace_custom(s,s.I_list,q.objective(range(3)),num_iterations=30,direction_weights=s.alphas,initial_solution=np.array(c['fixed_start']))
    observed=execute(c,s,'O',30)
    assert abs(native['loss']-observed['loss'])<1e-12 and np.array_equal(native['solution'],observed['x'])
    # Tie priority, stay, tolerance, and feasible boundary trajectories.
    opts=[dict(label='stay',loss=1.,priority=-1),dict(label='full',loss=.5,priority=0),dict(label='minus_0',loss=.5,priority=1)]
    assert choose(opts,1.,1e-10)['label']=='full'
    assert choose([opts[0],dict(label='full',loss=1.-1e-12,priority=0)],1.,1e-10)['label']=='stay'
    for kind in ['identical','zero_gradient','boundary']:
        d=copy.deepcopy(c)
        if kind=='identical':d['coefficients']=[d['coefficients'][0]]*3
        if kind=='zero_gradient':d['coefficients']=[[.3,0,0,0,0]]*3
        if kind=='boundary':d['fixed_start']=[1.,-1.]
        rr={a:execute(d,analytic_server(d),a,30) for a in ARMS};gates(rr)
        if kind=='zero_gradient':assert all(abs(r['loss']-.3)<1e-10 for r in rr.values())
    # Every possible small budget cut, including model-branch full-F completion.
    for budget in range(12,90):
        for a in ARMS:
            r=execute(c,s,a,10,budget);assert r['samples']<=budget
            exact=sum(w*abs(float(s.trusted_agents[i].api_predict(np.array(r['x']))[0])-s.target_T) for i,w in enumerate(s.alphas))
            assert abs(exact-r['loss'])<1e-10
    result=dict(passed=True,native_observer_equal=True,edge_cases=['identical','zero_gradient','boundary','tie','near_zero_improvement','budget_cuts_12_to_89'],physical=Replies.total)
    save(Path('/private/tmp/branch_search_controls_20261107.json'),result)
    print(result)
if __name__=='__main__':main()
