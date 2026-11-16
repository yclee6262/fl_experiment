import run as r
from engine import run as trace_run
import copy,numpy as np

def main():
 records=r.read(r.OUT/'scored.json');selected=[('budget_controlled_poison_2026101611',3),('budget_controlled_poison_2026101613',15),('budget_clean_2026101616',None)];out=[]
 for key,aid in selected:
  rec=next(x for x in records if x['case']['key']==key);c=rec['case'];aid=aid or c['ids'][1];i=c['ids'].index(aid)
  base,model,groups,rem=r.prep(c);s=r.load_server(c);q=r.Q(s);q.stage='diagnostic';ids=[j for j in range(len(c['ids'])) if j!=i];b=copy.copy(s);b.I_list=s.I_list[ids];b.alphas=s.alphas[ids]/sum(s.alphas[ids]);result,tr=trace_run(b,q.objective(ids),30,initial=np.array(c['fixed_start']))
  np.testing.assert_array_equal(result['solution'],rem[i]['result']['solution'])
  original=[v['pre']['S_current'] for v in model['full']]+[base['result']['solution']]
  removed=[v['S_current'].tolist() for v in tr.pre]+[result['solution'].tolist()]
  dist=[float(np.linalg.norm(np.array(a)-z)) for a,z in zip(original,removed)]
  out.append(dict(key=key,agent_id=aid,base_path=original,loo_path=removed,matched_round_distances=dist,
   first_divergent_state=next((t for t,x in enumerate(dist) if x>1e-10),None),
   full=base['result']['solution'],loo=result['solution'],cum=rec['methods']['CUM']['x'][i],
   last=rec['methods']['LAST_ANCHOR_REPLAY']['x'][i],D=rec['checks']['diagnostics'][i]['cumulative_vector']))
 r.save('path_diagnostics.json',out);r.save('path_diagnostics_cost.json',dict(local_physical_samples=r.Q.total_new))
if __name__=='__main__':main()
