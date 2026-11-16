import itertools
import numpy as np

def features(x,degree):
 x=np.atleast_2d(x);parts=[np.ones(len(x))]+[x[:,j] for j in range(x.shape[1])]
 if degree==2:parts += [x[:,j]*x[:,k] for j,k in itertools.combinations_with_replacement(range(x.shape[1]),2)]
 return np.stack(parts,axis=1)
def fit(X,y,center,degree):
 X=np.asarray(X);y=np.asarray(y);center=np.asarray(center);dist=np.linalg.norm(X-center,axis=1);order=np.argsort(dist,kind='stable')[:128]
 if len(order)<128:raise ValueError('insufficient_client_observations')
 h=max(float(dist[order[-1]]),1e-6);phi=features((X[order]-center)/h,degree)
 if len(order)<6*phi.shape[1]:raise ValueError('insufficient_observations_per_dimension')
 w=np.exp(-.5*(dist[order]/h)**2);gram=phi.T@(w[:,None]*phi);ridge=1e-6*float(np.diag(gram)[1:].mean());pen=np.diag([0]+[ridge]*(gram.shape[0]-1));beta=np.linalg.solve(gram+pen,phi.T@(w*y[order]))
 if not np.isfinite(beta).all():raise ValueError('nonfinite_client_fit')
 return dict(indices=order.tolist(),center=center.tolist(),h=h,degree=degree,beta=beta.tolist(),ridge=ridge,condition_number=float(np.linalg.cond(gram+pen)),n_observations=128,n_coefficients=len(beta))
def predict(summary,x):return float(features((np.array(x)-summary['center'])/summary['h'],summary['degree'])[0]@summary['beta'])
