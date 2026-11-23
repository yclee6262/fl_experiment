import numpy as np
from experiments.baselines.nonzero_subspace.run import witness,h,geometry,targets


def test_nonzero_witnesses_are_box_feasible_and_exact():
 for d in [5,10]:
  for label,target in targets(d).items():
   x=witness(d,target)
   assert np.max(abs(x))<=1
   assert abs(h(x)-target)<1e-10
  assert h(np.ones(d))==2*d-1


def test_linear_and_affine_geometry_are_distinct():
 a=np.array([[1.,0.,0.,0.,0.]])
 g=geometry(a,np.zeros(5))
 assert g['linear_rank']==1 and g['affine_rank']==0
 assert g['witness_linear_distance']==0 and g['witness_affine_distance']==1
 assert g['linear_box_feasible'] and g['affine_box_feasible']
 g=geometry(np.r_[a,a],np.array([0.,1.,0.,0.,0.]))
 assert g['linear_rank']==1 and g['affine_rank']==0


def test_single_anchor_affine_and_linear_search_spaces():
 from core.host_server import HostServer
 from experiments.baselines.paired_benchmark.search import run_search
 class Agent:
  def api_predict(self,x):
   x=np.asarray(x);return np.array([np.sum((x-np.array([0.,1.,0.,0.,0.]))**2)])
 s=HostServer(0.,5,constraint_mode='box_constrained',coefficient_eps=1e-5)
 s.trusted_agents=[Agent()];s.alphas=[1.];s.I_list=[np.array([1.,0.,0.,0.,0.])]
 custom=run_search(s,'custom',100);coeff=run_search(s,'coefficient',100);direct=run_search(s,'direct_slsqp',100)
 np.testing.assert_array_equal(custom['X'],s.I_list[0])
 assert geometry(s.I_list,coeff['X'])['witness_linear_distance']<1e-10
 assert geometry(s.I_list,direct['X'])['witness_linear_distance']>.5


def test_single_anchor_exact_range_and_geometric_error_floor():
 from experiments.baselines.nonzero_subspace.analyze import one_anchor_truth_range
 r=one_anchor_truth_range([1.,1.,0.,0.,0.],3.5)
 assert r['truth_lower']==-1 and r['truth_upper']==3
 assert r['linear_truth_error_floor']==.5
 assert r['affine_truth_error_floor']==.5
 r=one_anchor_truth_range([1.,0.,0.,0.,0.],.5)
 assert r['linear_truth_error_floor']==0 and r['affine_truth_error_floor']==.5
 r=one_anchor_truth_range([0.,0.,0.,0.,0.],.5)
 assert r['linear_truth_error_floor']==r['affine_truth_error_floor']==.5
