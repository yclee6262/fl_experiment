import sys
import unittest
import subprocess
from pathlib import Path
import numpy as np
from study import WORK,save,geometry,execute

sys.path.insert(0,str(WORK))
sys.path.insert(0,str(WORK/'contribution_analysis/model_step_decision_v1'))
from host_server import HostServer
from engine import run,branch_state

def server():
    s=HostServer.__new__(HostServer);s.I_list=np.array([[1.,0.],[0.,1.]])
    s.alphas=np.array([.5,.5]);s.constraint_mode='box_constrained'
    s.feasible_lower=-1.;s.feasible_upper=1.;s.feasibility_tolerance=1e-8
    return s

class ToyReplies:
    def __init__(self):self.events=[]
    def objective(self,ids):
        def f(x):
            for i in ids:self.events.append(dict(i=i))
            return float(np.mean([2+x[i] for i in ids]))
        return f

def state():
    return dict(S_current=np.array([0.,0.]),best_loss=2.,global_best_S=np.array([0.,0.]),global_best_loss=2.,
        eta=.1,current_method='secant',delta=.0001,history=[2.],states=['Start'],
        anchor_losses=[2.25,2.25],next_iteration=0,halted=False)

class Roles(unittest.TestCase):
    def test_anchor_preserves_scoring_panel(self):
        s=server();r=ToyReplies();b=execute(s,r,state(),0,'anchor')
        self.assertEqual(b['panel'],[0,1]);self.assertEqual(b['anchor_indices'],[1]);np.testing.assert_array_equal(b['direction_weights'],[1.])
    def test_model_preserves_anchor_weights(self):
        s=server();r=ToyReplies();b=execute(s,r,state(),0,'model')
        self.assertEqual(b['panel'],[1]);self.assertEqual(b['anchor_indices'],[0,1]);np.testing.assert_array_equal(b['direction_weights'],[.5,.5])
        self.assertTrue(all(e['i']==1 for e in r.events))
    def test_joint_removes_both(self):
        s=server();r=ToyReplies();b=execute(s,r,state(),0,'joint')
        self.assertEqual(b['panel'],[1]);self.assertEqual(b['anchor_indices'],[1])
        self.assertTrue(all(e['i']==1 for e in r.events));np.testing.assert_array_equal(b['endpoint'],[0.,-.1])
    def test_interaction_residual_not_forced_zero(self):
        s=server();r=ToyReplies();f=r.objective([0,1]);st=branch_state(state(),f,s.I_list)
        _,full=run(s,f,1,resume=st);L=f(full.final['S_current'])
        diffs={role:f(execute(s,r,state(),0,role)['endpoint'])-L for role in ['anchor','model','joint']}
        expected=.05-.1/np.sqrt(2)
        self.assertAlmostEqual(diffs['joint']-diffs['anchor']-diffs['model'],expected)
    def test_original_geometry_unchanged(self):
        s=server();before=s.I_list.copy();r=ToyReplies()
        for role in ['anchor','model','joint']:execute(s,r,state(),0,role)
        np.testing.assert_array_equal(s.I_list,before);np.testing.assert_array_equal(s.alphas,[.5,.5])
    def test_nonuniform_weight_separation(self):
        s=server();s.alphas=np.array([.7,.3])
        m,ids,_=geometry(s,0,'model');j,jids,_=geometry(s,0,'joint')
        np.testing.assert_array_equal(m.alphas,[.7,.3]);np.testing.assert_array_equal(j.alphas,[1.]);self.assertEqual(ids,jids)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Roles))
    old=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(WORK/'contribution_analysis/model_step_decision_v1'),'-p','test_engine.py','-v'],capture_output=True,text=True)
    print(old.stderr)
    save(Path(__file__).with_name('technical_results.json'),dict(passed=result.wasSuccessful() and old.returncode==0,
        new_tests=result.testsRun,legacy_tests=12,legacy_returncode=old.returncode,legacy_output=old.stderr,
        failures=len(result.failures),errors=len(result.errors)))
    sys.exit(0 if result.wasSuccessful() and old.returncode==0 else 1)
