import unittest
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from host_server import HostServer
from engine import run,branch_state
from study import plain,save

def server(anchors=((-1.,),(.8,)),weights=(.5,.5)):
    s=HostServer.__new__(HostServer);s.target_T=0.;s.n_features=1
    s.constraint_mode='box_constrained';s.feasible_lower=-1.;s.feasible_upper=1.;s.feasibility_tolerance=1e-8
    s.I_list=np.asarray(anchors);s.alphas=np.asarray(weights);return s

def pre(s,f,x=0.,mode='secant',eta=.1):
    return dict(S_current=np.array([x]),best_loss=f(np.array([x])),global_best_S=np.array([x]),
        global_best_loss=f(np.array([x])),eta=eta,current_method=mode,delta=.0001,
        history=[f(np.array([x]))],states=['Start'],anchor_losses=[f(a) for a in s.I_list],
        next_iteration=0,halted=False)

class TestEngine(unittest.TestCase):
    def test_original_query_order(self):
        s=server();f=lambda x:float((x[0]-.2)**2)
        a=[];b=[]
        def fa(x):a.append(np.asarray(x).copy());return f(x)
        def fb(x):b.append(np.asarray(x).copy());return f(x)
        orig=s._solve_subspace_custom(s.I_list,fa,num_iterations=30,direction_weights=s.alphas,initial_solution=np.array([0.]))
        new,tr=run(s,fb,30,initial=[0.])
        self.assertEqual(plain(orig),plain(new));np.testing.assert_array_equal(a,b)
        self.assertGreater(len(tr.pre),0)
    def test_positive_negative_and_no_effect(self):
        # Same state; the different decision endpoints can help or hurt the common target.
        s=server();ref=lambda x:float(1+x[0])
        st=pre(s,ref);_,full=run(s,ref,1,resume=st)
        fullx=full.final['S_current'];self.assertAlmostEqual(fullx[0],-.1)
        # Reverse preference creates a worse common-panel endpoint.
        alt=lambda x:float(1-x[0]);bst=branch_state(st,alt,s.I_list)
        _,bt=run(s,alt,1,resume=bst);self.assertGreater(ref(bt.final['S_current'])-ref(fullx),0)
        # A different direction can improve a nonlinear common target more than its secant estimate.
        s2=server(anchors=((.5,),(.8,)));ref2=lambda x:float(20+x[0]-20*x[0]**2)
        _,ft=run(s2,ref2,1,resume=pre(s2,ref2))
        alt2=lambda x:float(2+x[0]);_,at=run(s2,alt2,1,resume=branch_state(pre(s2,ref2),alt2,s2.I_list))
        self.assertLess(ref2(at.final['S_current'])-ref2(ft.final['S_current']),0)
        _,same=run(s,ref,1,resume=branch_state(st,ref,s.I_list))
        np.testing.assert_array_equal(same.final['S_current'],fullx)
    def test_full_reject_alt_accept(self):
        s=server();flat=lambda x:1.;alt=lambda x:float(1+x[0])
        st=pre(s,flat);_,full=run(s,flat,1,resume=st)
        _,bt=run(s,alt,1,resume=branch_state(st,alt,s.I_list))
        self.assertEqual(len(full.trials),0);self.assertTrue(any(t['accepted'] for t in bt.trials))
    def test_full_accept_alt_reject(self):
        s=server();f=lambda x:float(1+x[0]);flat=lambda x:1.
        st=pre(s,f);_,ft=run(s,f,1,resume=st);_,at=run(s,flat,1,resume=branch_state(st,flat,s.I_list))
        self.assertTrue(any(t['accepted'] for t in ft.trials));self.assertTrue(at.final['halted'])
    def test_both_no_motion(self):
        s=server();f=lambda x:1.;_,t=run(s,f,1,resume=pre(s,f))
        np.testing.assert_array_equal(t.final['S_current'],[0.]);self.assertTrue(t.final['halted'])
    def test_shrink(self):
        s=server();f=lambda x:float((x[0]-.03)**2)
        _,t=run(s,f,1,resume=pre(s,f,mode='dynamic',eta=.5))
        self.assertGreater(len(t.trials),1);self.assertTrue(t.trials[-1]['accepted'])
    def test_failed_secant_does_not_execute_dynamic(self):
        s=server();f=lambda x:float(1+x[0]**2)
        _,t=run(s,f,1,resume=pre(s,f))
        self.assertEqual(len(t.pre),1);self.assertEqual(len(t.trials),10)
        self.assertEqual(t.final['current_method'],'dynamic');self.assertEqual(t.final['eta'],.5)
        np.testing.assert_array_equal(t.final['S_current'],[0.])
    def test_dynamic_failure_terminal(self):
        s=server(anchors=((.5,),(.8,)));f=lambda x:float(1+x[0]**2)
        _,t=run(s,f,1,resume=pre(s,f,mode='dynamic'))
        self.assertTrue(t.final['halted']);self.assertEqual(len(t.trials),10)
    def test_box_bound(self):
        s=server();f=lambda x:float(2-x[0])
        _,t=run(s,f,1,resume=pre(s,f,x=.99))
        self.assertAlmostEqual(t.final['S_current'][0],1.);self.assertLessEqual(s.box_violation(t.final['S_current']),1e-8)
    def test_goal_reset_keeps_geometry_controls(self):
        s=server();f=lambda x:float(1+x[0]);st=pre(s,f);g=lambda x:float(10-x[0])
        b=branch_state(st,g,s.I_list)
        self.assertEqual(b['best_loss'],10.);self.assertEqual(b['global_best_loss'],10.)
        for name in ['eta','delta','current_method','next_iteration']:self.assertEqual(st[name],b[name])
        np.testing.assert_array_equal(s.I_list,[[-1.],[.8]])
    def test_zero_weight_and_undefined(self):
        w=np.array([.5,.5,0.]);vals=np.array([1.,2.,100.])
        self.assertEqual(np.dot(w,vals),np.dot(w[:2]/w[:2].sum(),vals[:2]))
        self.assertEqual(np.delete(np.array([1.,0.]),0).sum(),0.)
    def test_common_scoring_no_reaccept(self):
        # A branch accepted by its own objective remains its endpoint, even if common loss worsens.
        s=server();ref=lambda x:float(1+x[0]);alt=lambda x:float(1-x[0]);st=pre(s,ref)
        _,t=run(s,alt,1,resume=branch_state(st,alt,s.I_list))
        self.assertGreater(ref(t.final['S_current']),ref(st['S_current']))
        self.assertAlmostEqual(t.final['S_current'][0],.1)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(TestEngine)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    out=Path(__file__).with_name('technical_results.json')
    if out.exists():
        import time
        out.rename(out.with_name('technical_attempt_'+str(time.time_ns())+'.json'))
    save(out,dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors)))
    sys.exit(0 if result.wasSuccessful() else 1)
