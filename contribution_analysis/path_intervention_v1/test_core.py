import copy,sys,unittest
from types import SimpleNamespace
import numpy as np
from common import WORK,MOD,ROOT,save
sys.path.insert(0,str(MOD/'source_snapshot'))
from core import rebase,advance,run_pair,point_at
from host_server import HostServer

def server():
    s=HostServer.__new__(HostServer);s.I_list=np.asarray([[-.8],[.8]]);s.alphas=np.asarray([.5,.5])
    s.constraint_mode='box_constrained';s.feasible_lower=-1.;s.feasible_upper=1.;s.feasibility_tolerance=1e-8
    return s
def state(x=.3,mode='secant',iteration=0):
    f=lambda z:float(np.asarray(z)[0]**2)
    return dict(S_current=np.asarray([x]),best_loss=f([x]),global_best_S=np.asarray([x]),global_best_loss=f([x]),eta=.1,
        current_method=mode,delta=.0001,history=[f([x])],states=['Start'],anchor_losses=[f([-.8]),f([.8])],next_iteration=iteration,halted=False)
class Fake:
    def __init__(self):self.events=[];self.stage=''
    def objective(self,panel):
        def f(x):
            for j in panel:self.events.append(dict(key=str(j)+':fake'))
            return 1.
        return f
class Tests(unittest.TestCase):
    def test_rebase_resets_objective(self):
        a=state();a['global_best_S']=np.array([-.4]);a['halted']=True;b=rebase(a,lambda x:7.,[[-.8]])
        self.assertEqual(b['best_loss'],7);self.assertFalse(b['halted']);self.assertEqual(len(b['anchor_losses']),1)
        np.testing.assert_array_equal(b['global_best_S'],a['S_current']);self.assertEqual(b['eta'],a['eta'])
    def test_source_unchanged(self):
        a=state();saved=copy.deepcopy(a);rebase(a,lambda x:2.,[[-.8]])
        np.testing.assert_array_equal(a['S_current'],saved['S_current']);self.assertEqual(a['history'],saved['history'])
    def test_zero_direction_halts(self):
        a=rebase(state(),lambda x:1.,server().I_list);r=advance(server(),lambda x:1.,a,1)
        self.assertTrue(r['final']['halted']);self.assertEqual(r['steps'][0]['stop_reason'],'zero_direction')
    def test_rejected_point_not_delivered(self):
        a=state();a['best_loss']=0.;a['global_best_loss']=0.;a['anchor_losses']=[1.,2.]
        r=advance(server(),lambda x:1.,a,1);np.testing.assert_array_equal(r['result']['solution'],a['S_current'])
        self.assertEqual(len(r['steps'][0]['trials']),10);self.assertFalse(r['steps'][0]['accepted'])
    def test_secant_failure_switches(self):
        a=state();a['anchor_losses']=[1.,2.];r=advance(server(),lambda x:1.,a,1)
        self.assertEqual(r['final']['current_method'],'dynamic');self.assertEqual(r['final']['eta'],.5)
    def test_dynamic_failure_stops(self):
        a=state(mode='dynamic');s=server();s.I_list=np.asarray([[.7],[.9]])
        r=advance(s,lambda x:1.,a,1)
        self.assertTrue(r['final']['halted']);self.assertEqual(r['steps'][0]['stop_reason'],'dynamic_no_improvement')
    def test_halted_does_not_run(self):
        a=state();a['halted']=True;r=advance(server(),lambda x:1.,a,20);self.assertEqual(len(r['steps']),0)
    def test_restore_restart_and_tabs(self):
        s=server();q=Fake();o=run_pair(s,q,state(),0)
        self.assertTrue(o['restart']);self.assertTrue(o['first']['final']['halted']);self.assertGreater(len(o['T']['steps']),0)
        self.assertEqual(len(o['P']['steps']),0);np.testing.assert_array_equal(o['T_abs_x'],o['first']['result']['solution'])
    def test_no_budget_no_restart(self):
        o=run_pair(server(),Fake(),state(iteration=29),0)
        self.assertFalse(o['restart']);self.assertEqual(len(o['T']['steps']),0)
    def test_horizon_absorbs(self):
        r=advance(server(),lambda x:1.,rebase(state(),lambda x:1.,server().I_list),1)
        p=point_at([r],5);self.assertTrue(p['stopped']);self.assertEqual(p['state_iteration'],1)
    def test_exact_original_continuation(self):
        s=server();f=lambda x:float(np.asarray(x)[0]**2);a=rebase(state(),f,s.I_list)
        whole=advance(s,f,a,6);one=advance(s,f,a,1);tail=advance(s,f,one['final'],5)
        np.testing.assert_array_equal(whole['result']['solution'],tail['result']['solution'])
        self.assertEqual(whole['result']['history'],tail['result']['history'])
    def test_query_cache_is_exact(self):
        from common import key
        self.assertNotEqual(key(0,[.1]),key(1,[.1]));self.assertNotEqual(key(0,[.1]),key(0,[.10000000000001]))

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    save(__import__('pathlib').Path(__file__).parent/'technical_results.json',dict(accepted=r.wasSuccessful(),tests=r.testsRun,failures=len(r.failures),errors=len(r.errors)))
    if not r.wasSuccessful():raise SystemExit(1)
