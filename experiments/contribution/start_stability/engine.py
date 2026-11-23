"""Crossed-start interventions; reuse maintained frozen numerical primitives."""
import copy
import itertools
import math
import time

import numpy as np
from experiments.contribution.nonzero_targets.run import (
    Replies, split, truth, HostServer, run, endpoint, construct_directions, decide, branch_state)

TOL = 1e-10


def allocation(scores, threshold):
    values = np.where(np.asarray(scores) > threshold, scores, 0.)
    total = values.sum()
    return np.r_[values/total, 0.] if total else np.r_[values, 1.]


def integer_pool(shares, budget, ids):
    if shares[-1] > .5:
        return [0]*len(ids), int(budget)
    quotas = np.asarray(shares[:-1])*budget
    pay = np.floor(quotas).astype(int)
    order = sorted(range(len(ids)), key=lambda i: (-(quotas[i]-pay[i]), str(ids[i])))
    for i in order[:budget-int(pay.sum())]:
        pay[i] += 1
    assert int(pay.sum()) == budget
    return pay.tolist(), 0


def shadow_payment(participants, ids, judge):
    sa = np.array([p['scores'][judge]['A'] for p in participants])
    sm = np.array([p['scores'][judge]['M'] for p in participants])
    ta = np.array([p['tauA'] for p in participants])
    basic = np.where((sa < -ta) & (sm < -TOL), 0, 100)
    anchor, ar = integer_pool(allocation(sa, ta), 500, ids)
    model, mr = integer_pool(allocation(sm, TOL), 500, ids)
    total = basic + anchor + np.array(model)
    br = 100*len(ids)-int(basic.sum())
    assert int(total.sum())+br+ar+mr == 100*len(ids)+1000
    return dict(basic=basic.tolist(), anchor=anchor, model=model, total=total.tolist(),
        basic_reserved=br, anchor_reserved=ar, model_reserved=mr,
        reserved=br+ar+mr, budget=100*len(ids)+1000,
        total_with_reserve=np.r_[total, br+ar+mr].astype(int).tolist(),
        normalized_total=(np.r_[total, br+ar+mr]/(100*len(ids)+1000)).tolist(), actual_payment=False)


def exact_shapley(values, n):
    if len(values) != 2**n or abs(values[0]) > 1e-12:
        raise ValueError('Complete characteristic function with zero empty utility required')
    result = np.zeros(n)
    for i in range(n):
        for mask in range(2**n):
            if mask & (1 << i):
                continue
            size = mask.bit_count()
            weight = math.factorial(size)*math.factorial(n-size-1)/math.factorial(n)
            result[i] += weight*(values[mask | (1 << i)]-values[mask])
    assert abs(result.sum()-(values[-1]-values[0])) < 1e-10
    return result


def matched_game(server, replies, full, loo, initial, iterations):
    n = len(server.alphas)
    all_mask = 2**n-1
    replies.stage = 'matched_game'
    base = replies.losses(initial)
    baseline = float(np.asarray(server.alphas)@base)
    baseline_truth = truth(initial, server.target_T)
    coalitions = []
    for mask in range(2**n):
        ids = [i for i in range(n) if mask & (1 << i)]
        if mask == 0:
            point, origin = np.asarray(initial), 'empty_fixed_start'
        elif mask == all_mask:
            point, origin = np.asarray(full['solution']), 'reuse_full'
        elif len(ids) == n-1:
            removed = next(i for i in range(n) if not mask & (1 << i))
            point, origin = np.asarray(loo[removed]['X']), 'reuse_LOO'
        else:
            sub = copy.copy(server)
            sub.I_list = server.I_list[ids]
            sub.alphas = np.asarray(server.alphas)[ids]
            sub.alphas = sub.alphas/sub.alphas.sum()
            result = HostServer._solve_subspace_custom(sub, sub.I_list, replies.objective(ids),
                num_iterations=iterations, direction_weights=sub.alphas, initial_solution=initial)
            point, origin = np.asarray(result['solution']), 'new_subset_search'
        losses = replies.losses(point)
        fixed_loss = float(np.asarray(server.alphas)@losses)
        error = truth(point, server.target_T)
        coalitions.append(dict(mask=mask, members=ids, X=point, losses=losses,
            fixed_panel_loss=fixed_loss, truth_error=error, vF=baseline-fixed_loss,
            vTruth=baseline_truth-error, origin=origin))
    values_f = [c['vF'] for c in coalitions]
    values_truth = [c['vTruth'] for c in coalitions]
    sf, st = exact_shapley(values_f, n), exact_shapley(values_truth, n)
    for i in range(n):
        assert abs(values_f[-1]-values_f[all_mask ^ (1 << i)]-loo[i]['score']['F']) < 1e-10
        assert abs(values_truth[-1]-values_truth[all_mask ^ (1 << i)]-loo[i]['truth_effect']) < 1e-10
    return dict(baseline_F=baseline, baseline_truth=baseline_truth, coalitions=coalitions,
        phi_F=sf, phi_truth=st, empty_utility=0., common_initial=initial,
        scorer='fixed_full_panel_uniform_F_N', actual_payment=False)


def execute(server, case, target, inversions, initial, include_game=False, iterations=30):
    start = time.perf_counter()
    s = copy.copy(server)
    n = len(s.trusted_agents)
    a = np.ones(n)/n
    s.target_T, s.alphas = target, a
    s.I_list = np.array([v['X'] for v in inversions])
    fixed = np.asarray(initial, float)
    q, ids = Replies(s), list(range(n))
    q.stage = 'full'
    full, tr = run(s, q.objective(ids), iterations, initial=fixed)
    observer = Replies(s)
    control = HostServer._solve_subspace_custom(s, s.I_list, observer.objective(ids),
        num_iterations=iterations, direction_weights=a, initial_solution=fixed)
    np.testing.assert_allclose(control['solution'], full['solution'], atol=1e-12, rtol=0)
    assert abs(control['loss']-full['loss']) < 1e-12
    rows = []
    for t in range(len(tr.pre)):
        feature = endpoint(tr, t)
        pre, xf = feature['pre'], np.asarray(feature['post']['S_current'])
        q.stage = 'directions'
        _, _, h, _ = construct_directions(s, q, a, feature)
        lf = q.losses(xf)
        for i in ids:
            peer = [j for j in ids if j != i]
            q.stage = 'anchor'
            alt = decide(s, q.objective(ids), pre['S_current'], np.asarray(feature['g'])-h[i],
                         pre['best_loss'], pre['eta'], pre['current_method'])
            la = q.losses(alt['x'])
            rho = max(1e-12, 32*np.finfo(float).eps*(1+abs(float(a@la))+abs(float(a@lf))))
            roles = {'A': dict(X=alt['x'], losses=la, score=split(la-lf, a, i))}
            for label in ['M', 'J']:
                sub = copy.copy(s)
                if label == 'J':
                    sub.I_list, sub.alphas = s.I_list[peer], a[peer]/a[peer].sum()
                q.stage = label
                objective = q.objective(peer)
                state = branch_state(pre, objective, sub.I_list)
                _, branch = run(sub, objective, 1, resume=state)
                point = np.asarray(branch.final['S_current'])
                losses = q.losses(point)
                roles[label] = dict(X=point, losses=losses, score=split(losses-lf, a, i))
            rows.append(dict(t=t, i=i, full_X=xf, full_losses=lf, rho=rho, roles=roles))
    q.stage = 'full_LOO'
    loo = []
    for i in ids:
        peer = [j for j in ids if j != i]
        sub = copy.copy(s)
        sub.I_list, sub.alphas = s.I_list[peer], a[peer]/a[peer].sum()
        result = HostServer._solve_subspace_custom(sub, sub.I_list, q.objective(peer),
            num_iterations=iterations, direction_weights=sub.alphas, initial_solution=fixed)
        losses, base = q.losses(result['solution']), q.losses(full['solution'])
        loo.append(dict(i=i, X=result['solution'], losses=losses, full_losses=base,
            score=split(losses-base, a, i), truth_effect=truth(result['solution'], target)-truth(full['solution'], target)))
    participants = []
    for i in ids:
        own = [r for r in rows if r['i'] == i]
        scores = {j:{role:sum(r['roles'][role]['score'][j] for r in own) for role in ['A', 'M', 'J']}
                  for j in ['F', 'Fminus']}
        for judge in scores:
            scores[judge]['AM'] = scores[judge]['A']+scores[judge]['M']
            scores[judge]['LOO'] = loo[i]['score'][judge]
        participants.append(dict(i=i, id=case['ids'][i], tauA=max(1e-8, sum(r['rho'] for r in own)),
            scores=scores, truth_LOO=loo[i]['truth_effect']))
    role_misses, role_requests = len(q.cache), len(q.events)
    game = matched_game(s, q, full, loo, fixed, iterations) if include_game else None
    endpoints = [full['solution']]+[v['X'] for r in rows for v in r['roles'].values()]+[v['X'] for v in loo]
    if game:
        endpoints += [v['X'] for v in game['coalitions']]
    assert all(np.isfinite(x).all() and max(abs(np.asarray(x))) <= 1+1e-8 for x in endpoints)
    return dict(status='valid', key=case['key'], scenario=case['scenario'], seed=case['seed'], target=target,
        n=n, ids=case['ids'], anchors=s.I_list, start=fixed, alpha=a, full=full,
        truth_error=truth(full['solution'], target), rows=rows, LOO=loo, participants=participants,
        actual_rounds=len(tr.pre), payments={j:shadow_payment(participants, case['ids'], j) for j in ['F', 'Fminus']},
        game=game, cache_miss_calls=len(q.cache), requests=len(q.events), events=q.events,
        role_cache_misses=role_misses, role_requests=role_requests,
        matched_game_incremental_misses=len(q.cache)-role_misses,
        observer_equivalence_cache_misses=len(observer.cache), observer_equivalence_requests=len(observer.events),
        seconds=time.perf_counter()-start)
