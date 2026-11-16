"""Black-box gradient work on an already locked accepted trajectory.

No evaluator, labels, fitting, parameter gradients, or new solver is used.
"""
import time
import numpy as np


def attribute(agents, trace, alpha, target=0., h=.001, lower=-1., upper=1.):
    start = time.perf_counter()
    x = np.asarray(trace['states'], dtype=float)
    y = np.asarray(trace['responses'], dtype=float)
    w = np.asarray(alpha, dtype=float)
    if x.ndim != 2 or len(x) == 0 or y.shape != (len(x), len(agents)):
        raise ValueError('trajectory_shape')
    if w.shape != (len(agents),) or np.any(w < 0) or not np.isclose(w.sum(), 1., atol=1e-12, rtol=0):
        raise ValueError('alpha')
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(w).all()):
        raise ValueError('nonfinite_trajectory')
    if h <= 0 or np.any(x < lower-1e-8) or np.any(x > upper+1e-8):
        raise ValueError('domain')
    delta = np.diff(x, axis=0)
    n, d = delta.shape
    gradients = np.zeros((n, len(agents), d))
    replies = []
    points = np.repeat(x[:-1, None, None, :], d, axis=1)
    points = np.repeat(points, 2, axis=2)
    for j in range(d):
        points[:, j, 0, j] = np.minimum(upper, x[:-1, j]+h)
        points[:, j, 1, j] = np.maximum(lower, x[:-1, j]-h)
    # Match the public API's float32 visible coordinates and denominator.
    points = points.astype(np.float32)
    widths = np.stack([points[:, j, 0, j].astype(float)-points[:, j, 1, j].astype(float)
                       for j in range(d)], axis=1)
    if n and np.any(widths <= 0):
        raise ValueError('finite_difference_resolution')
    calls = 0
    for i, agent in enumerate(agents):
        if n:
            if 2*d*n > 10000:
                raise RuntimeError('case_gradient_sample_budget')
            pred = np.asarray(agent.api_predict(points.reshape(-1, d)), dtype=float)
            calls += 1
            if pred.shape != (2*d*n,) or not np.isfinite(pred).all():
                raise ValueError('gradient_reply')
            pred = pred.reshape(n, d, 2)
            losses = np.abs(pred-target)
            gradients[:, i, :] = (losses[:, :, 0]-losses[:, :, 1])/widths
            replies.append(pred)
        else:
            replies.append(np.empty((0, d, 2)))
    steps = -np.einsum('tmd,td,m->tm', gradients, delta, w)
    score = steps.sum(axis=0)
    exact = w*(np.abs(y[0]-target)-np.abs(y[-1]-target))
    return dict(gradient_work=score, endpoint_work=exact, gradients=gradients,
                step_work=steps, fd_points=points, fd_responses=np.asarray(replies),
                fd_widths=widths, calls=calls, samples=int(2*d*n*len(agents)),
                n_steps=n, seconds=time.perf_counter()-start,
                integration_gap=score-exact,
                internal_loss_decrease=float(exact.sum()))
