import numpy as np

def fit_local(X,y,pred,center):
 X=np.asarray(X);y=np.asarray(y);pred=np.asarray(pred);center=np.asarray(center);d=np.linalg.norm(X-center,axis=1);order=np.argsort(d,kind='stable')[:32]
 if len(order)<32:raise ValueError('insufficient_client_observations')
 h=max(float(d[order[-1]]),1e-6);w=np.exp(-.5*(d[order]/h)**2);r=y[order]-pred;phi=np.column_stack([np.ones(32),(X[order]-center)/h]);gram=phi.T@(w[:,None]*phi);ridge=1e-3*float(np.mean(np.diag(gram)[1:]));pen=np.diag([0]+[ridge]*(phi.shape[1]-1));beta=np.linalg.solve(gram+pen,phi.T@(w*r));intercept=float(w@r/w.sum())
 return dict(indices=order,center=center,h=h,residuals=r,weights=w,beta=beta,intercept=intercept,ridge=ridge,condition_number=float(np.linalg.cond(gram+pen)))
def corrected(model_predictions,summary,x,mode):
 x=np.asarray(x);return float(model_predictions+(summary['intercept'] if mode=='intercept' else np.r_[1,(x-np.array(summary['center']))/summary['h']]@summary['beta']))
