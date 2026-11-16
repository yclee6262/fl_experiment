"""Depth-two weighted rules with group support; no live LOO or evaluator."""
import numpy as np


def fit(X,y,groups,weights,columns=None,depth=2,min_groups=5):
    X=np.asarray(X);y=np.asarray(y);groups=np.asarray(groups);weights=np.asarray(weights)
    columns=list(range(X.shape[1])) if columns is None else list(columns)
    def node(indices,remaining):
        w=weights[indices];v=y[indices]
        mean=float(np.dot(w,v)/w.sum());base=float(np.dot(w,(v-mean)**2))
        out=dict(value=mean,n_rows=len(indices),n_groups=len(np.unique(groups[indices])))
        if remaining==0:return out
        best_gain=0.;best=None
        for col in columns:
            values=np.unique(X[indices,col])
            for threshold in values[:-1]+np.diff(values)/2:
                mask=X[indices,col]<=threshold;left=indices[mask];right=indices[~mask]
                if min(len(np.unique(groups[left])),len(np.unique(groups[right])))<min_groups:continue
                cost=0.
                for idx in [left,right]:
                    a=weights[idx];b=y[idx];mu=np.dot(a,b)/a.sum();cost+=np.dot(a,(b-mu)**2)
                gain=float(base-cost)
                if gain>best_gain:
                    best_gain=gain;best=(col,float(threshold),left,right)
        if best is not None:
            col,threshold,left,right=best
            out.update(feature=int(col),threshold=threshold,gain=best_gain,
                       left=node(left,remaining-1),right=node(right,remaining-1))
        return out
    return node(np.arange(len(y)),depth)


def predict(tree,X):
    result=[]
    for x in X:
        node=tree
        while 'feature' in node:
            node=node['left'] if x[node['feature']]<=node['threshold'] else node['right']
        result.append(node['value'])
    return np.array(result)


def constant_fit(y,m,weights):
    return {str(k):float(np.average(y[m==k],weights=weights[m==k])) for k in np.unique(m)}


def constant_predict(model,m):return np.array([model[str(k)] for k in m])
