import numpy as np
from pure_es_initialization import initializer
from four_alpha_initialization import initializer as mixed
from peer_support_alpha import support_signal
class Agent:
 def __init__(self,i,p):self.agent_id=i;self.p=p
 def api_predict(self,x):return self.p.copy()
def test_same_signal_as_shrunk_controls():
 y=np.array([0.,.2,-.1,.4]);p=np.array([[.1,-.1,.2],[.1,.3,.4],[-.2,.1,-.1],[.3,.6,.5]])
 x=np.arange(8).reshape(4,2);agents=[Agent(i,p[:,i]) for i in range(3)];mse=((p-y[:,None])**2).mean(axis=0);proposals=np.zeros((3,2));peer=np.array([[0.,.4,.1],[.2,0.,.3],[.3,.6,0.]])
 pure=initializer(peer);mix=mixed(peer);kw=dict(cached_blind_predictions=p)
 e,_=pure('stacked_full',agents,mse,x,y,proposals,0,**kw);eu,_=mix('ensemble_shrink',agents,mse,x,y,proposals,0,**kw)
 s,_=pure('pure_support',agents,mse,x,y,proposals,0,**kw);su,_=mix('peer_support_shrink',agents,mse,x,y,proposals,0,**kw)
 np.testing.assert_allclose(eu,.5*e+.5/3,atol=1e-12);np.testing.assert_allclose(su,.5*s+.5/3,atol=1e-12)
 np.testing.assert_allclose(s,support_signal(peer,0)[0],atol=1e-12)
 assert min(e)>=0 and abs(sum(e)-1)<1e-8 and min(s)>=0 and abs(sum(s)-1)<1e-8
