import numpy as np
from four_alpha_initialization import initializer,CONTRASTS,METHODS
from initial_alpha_methods import initial_alpha as legacy
from peer_support_alpha import support_signal
class Agent:
 def __init__(self,i,p):self.agent_id=i;self.p=p
 def api_predict(self,x):return self.p.copy()
def test_same_information_and_exact_four_formulas():
 y=np.array([0.,.2,-.1,.4]);p=np.array([[.1,-.1,.2],[.1,.3,.4],[-.2,.1,-.1],[.3,.6,.5]])
 x=np.arange(8).reshape(4,2);agents=[Agent(i,p[:,i]) for i in range(3)];mse=((p-y[:,None])**2).mean(axis=0);proposals=np.zeros((3,2));peer=np.array([[0.,.4,.1],[.2,0.,.3],[.3,.6,0.]])
 init=initializer(peer);b=1/(mse+1e-5);b/=b.sum();u=np.ones(3)/3;s,_=support_signal(peer,0)
 _,audit=legacy('ensemble_shrink',agents,mse,x,y,proposals,0,cached_blind_predictions=p);w=np.array(audit['raw_scores'])
 assert not audit['fallback_reason']
 expected=[u,b,.5*w+.5*u,.5*w+.5*b,.5*s+.5*u,.5*s+.5*b]
 for method,target in zip(METHODS,expected):
  actual,_=init(method,agents,mse,x,y,proposals,0,cached_blind_predictions=p)
  np.testing.assert_allclose(actual,target,atol=1e-12)
  assert min(actual)>=0 and abs(sum(actual)-1)<1e-8
 assert len(CONTRASTS)==12 and len(set(CONTRASTS))==12
