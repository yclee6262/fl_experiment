"""Reload every cache-miss reply and independently verify scores and payments."""
import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from core.models import PretrainedModel
from experiments.contribution.role_statistics.analyze import allocate

ROOT = Path(__file__).resolve().parents[3]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def unpack(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def truth(point, target):
    x = np.array(point)
    return abs(float(x[0]+x[1]+x[0]*x[1])-target)


def validate(source, smoke=False, reference=None):
    output = source/'acceptance.json'
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(1)
    manifest = read(source/'manifest.json')
    assert manifest['status'] == 'complete' and manifest['smoke'] == smoke
    for name, expected in {**manifest['runtime_sha256'], **manifest['input_sha256']}.items():
        assert sha(Path(name)) == expected, name
    cases = read(ROOT/'contribution_analysis/combined_contribution_layer2_experiment_v1_20261021/cases.json')['development']
    if smoke: cases = cases[:1]
    case_map = {c['key']:c for c in cases}
    models, preparations, initials = {}, {}, {}
    inverse_checks = prep_forward = 0
    for p in read(source/'preparations.json'):
        assert sha(source/p['file']) == p['sha256']
        prep = read(source/p['file'])
        key, r, target = prep['key'], prep['r'], prep['target']
        c = case_map[key]
        if key not in models:
            payload = torch.load(c['models'], map_location='cpu', weights_only=True)
            models[key] = []
            for ident in c['ids']:
                net = PretrainedModel(2)
                net.load_state_dict(payload['agents'][ident])
                models[key].append(net.eval())
        n = len(c['ids'])
        rng = np.random.default_rng(c['seed']+11000000)
        expected = np.r_[np.full((1,n),1/n), rng.dirichlet(np.ones(n),size=2)]
        np.testing.assert_array_equal(expected, prep['warm_coefficients'])
        for i, inv in enumerate(prep['inversions']):
            assert inv['status'] == 'valid' and inv['steps_per_start'] == 500
            rng = np.random.default_rng(c['seed']*100+c['ids'][i]+1000000*r)
            first = np.clip(rng.normal(size=(1,2)), -1, 1).astype(np.float32).reshape(-1).tolist()
            assert inv['start_results'][0]['initial_X'] == first
            ikey = key, r, i
            if ikey in initials: assert initials[ikey] == first
            else: initials[ikey] = first
            with torch.no_grad():
                value = float(models[key][i](torch.tensor(inv['X'], dtype=torch.float32)[None,:]).item())
            assert abs(abs(value-target)-inv['loss']) < 1e-12
            assert np.isfinite(inv['X']).all() and max(abs(np.array(inv['X']))) <= 1
            inverse_checks += 1
        preparations[key,target,r] = prep
        prep_forward += prep['inversion_forward_samples']
    reference_index = {(v['key'],v['target']):v for v in read(reference/'index.json')} if reference else {}
    identities = set()
    misses = hits = scores = payments = game_checks = baseline_checks = 0
    total_requests = role_misses = game_misses = observer_misses = endpoint_checks = 0
    for entry in read(source/'index.json'):
        assert sha(source/entry['file']) == entry['sha256']
        raw = unpack(source/entry['file'])
        assert raw['status'] == entry['status'] == 'valid'
        key,target,r,w = raw['key'],raw['target'],raw['r'],raw['w']
        identity = key,target,r,w
        assert identity not in identities
        identities.add(identity)
        c = case_map[key]
        n = len(c['ids'])
        a = np.ones(n)/n
        prep = preparations[key,target,r]
        assert sha(source/raw['preparation_file']) == raw['preparation_sha256']
        anchors = np.array([v['X'] for v in prep['inversions']])
        np.testing.assert_array_equal(raw['anchors'], anchors)
        np.testing.assert_array_equal(raw['alpha'], a)
        np.testing.assert_array_equal(raw['warm_coefficients'], prep['warm_coefficients'][w])
        initial = anchors.mean(0) if w == 0 else np.array(prep['warm_coefficients'][w])@anchors
        np.testing.assert_array_equal(raw['start'], initial)
        cache = {}
        for event in raw['events']:
            x = np.asarray(event['X'], float)
            cachekey = event['i'], x.tobytes()
            assert np.isfinite(x).all() and max(abs(x)) <= 1+1e-8
            if event['hit']:
                assert cachekey in cache and cache[cachekey] == event['loss']
                hits += 1
            else:
                assert cachekey not in cache
                with torch.no_grad():
                    value = abs(float(models[key][event['i']](torch.tensor(x,dtype=torch.float32)[None,:]).item())-target)
                assert abs(value-event['loss']) < 1e-12
                cache[cachekey] = value
                misses += 1
        assert len(cache) == raw['cache_miss_calls'] and len(raw['events']) == raw['requests']
        role_events = raw['events'][:raw['role_requests']]
        assert sum(not e['hit'] for e in role_events) == raw['role_cache_misses']
        assert raw['role_cache_misses']+raw['matched_game_incremental_misses'] == raw['cache_miss_calls']
        def loss_vector(point):
            nonlocal endpoint_checks
            x = np.asarray(point,float)
            assert np.isfinite(x).all() and max(abs(x)) <= 1+1e-8
            endpoint_checks += 1
            return np.array([cache[i,x.tobytes()] for i in range(n)])
        base = loss_vector(raw['full']['solution'])
        assert abs(a@base-raw['full']['loss']) < 1e-12
        assert abs(truth(raw['full']['solution'],target)-raw['truth_error']) < 1e-12
        totals = {j:{role:np.zeros(n) for role in ['A','M','J']} for j in ['F','Fminus']}
        rho = np.zeros(n)
        for row in raw['rows']:
            i = row['i']
            lf = loss_vector(row['full_X'])
            np.testing.assert_allclose(lf,row['full_losses'],atol=1e-12,rtol=0)
            peer = [j for j in range(n) if j != i]
            for role,v in row['roles'].items():
                losses = loss_vector(v['X'])
                np.testing.assert_allclose(losses,v['losses'],atol=1e-12,rtol=0)
                delta = losses-lf
                f, p = float(a@delta), float(delta[peer].mean())
                assert abs(f-v['score']['F']) < 1e-12 and abs(p-v['score']['Fminus']) < 1e-12
                totals['F'][role][i] += f
                totals['Fminus'][role][i] += p
                scores += 2
                if role == 'A':
                    expected_rho = max(1e-12,32*np.finfo(float).eps*(1+abs(float(a@losses))+abs(float(a@lf))))
                    assert expected_rho == row['rho']
                    rho[i] += expected_rho
        for i, item in enumerate(raw['LOO']):
            loss = loss_vector(item['X'])
            np.testing.assert_allclose(loss,item['losses'],atol=1e-12,rtol=0)
            delta = loss-base
            peer = [j for j in range(n) if j != i]
            assert abs(float(delta.mean())-item['score']['F']) < 1e-12
            assert abs(float(delta[peer].mean())-item['score']['Fminus']) < 1e-12
            assert abs(truth(item['X'],target)-raw['truth_error']-item['truth_effect']) < 1e-12
            part = raw['participants'][i]
            assert part['tauA'] == max(1e-8,float(rho[i])) and part['id'] == c['ids'][i]
            for judge in ['F','Fminus']:
                for role in ['A','M','J']:
                    assert abs(totals[judge][role][i]-part['scores'][judge][role]) < 1e-10
                assert part['scores'][judge]['AM'] == part['scores'][judge]['A']+part['scores'][judge]['M']
                assert part['scores'][judge]['LOO'] == item['score'][judge]
        for judge in ['F','Fminus']:
            pay = raw['payments'][judge]
            ta = np.array([p['tauA'] for p in raw['participants']])
            sa, sm = totals[judge]['A'], totals[judge]['M']
            basic = np.where((sa < -ta)&(sm < -1e-10),0,100)
            ap,ar,_ = allocate(sa,ta,500,c['ids'])
            mp,mr,_ = allocate(sm,1e-10,500,c['ids'])
            assert pay['basic'] == basic.tolist() and pay['anchor'] == ap and pay['model'] == mp
            total = basic+ap+np.array(mp)
            reserve = 100*n-int(basic.sum())+ar+mr
            assert pay['total'] == total.tolist() and pay['reserved'] == reserve
            assert sum(pay['total'])+pay['reserved'] == pay['budget'] == n*100+1000
            np.testing.assert_array_equal(pay['normalized_total'],np.r_[total,reserve]/pay['budget'])
            payments += 1
        if raw['game']:
            assert r == w == 0
            game = raw['game']
            assert len(game['coalitions']) == 2**n
            vf,vt = [],[]
            baseline = loss_vector(initial)
            for mask,coalition in enumerate(game['coalitions']):
                assert coalition['mask'] == mask
                losses = loss_vector(coalition['X'])
                np.testing.assert_allclose(losses,coalition['losses'],atol=1e-12,rtol=0)
                vf.append(float(a@baseline-a@losses))
                vt.append(truth(initial,target)-truth(coalition['X'],target))
                assert abs(vf[-1]-coalition['vF']) < 1e-12 and abs(vt[-1]-coalition['vTruth']) < 1e-12
                if mask == 0: np.testing.assert_array_equal(coalition['X'],initial)
            for values,phi in [(vf,game['phi_F']),(vt,game['phi_truth'])]:
                independent = np.zeros(n)
                permutations = list(itertools.permutations(range(n)))
                for order in permutations:
                    mask = 0
                    for i in order:
                        newmask = mask | (1 << i)
                        independent[i] += (values[newmask]-values[mask])/len(permutations)
                        mask = newmask
                np.testing.assert_allclose(independent,phi,atol=1e-10,rtol=0)
                assert abs(sum(phi)-values[-1]) < 1e-10
            for i in range(n):
                without = (2**n-1) ^ (1 << i)
                assert abs(vf[-1]-vf[without]-raw['LOO'][i]['score']['F']) < 1e-12
                assert abs(vt[-1]-vt[without]-raw['LOO'][i]['truth_effect']) < 1e-12
            game_checks += 1
        if reference and r == w == 0:
            oldentry = reference_index[key,target]
            assert sha(reference/oldentry['file']) == oldentry['sha256']
            old = unpack(reference/oldentry['file'])
            for name in ['full','rows','LOO','anchors','start','alpha','truth_error','actual_rounds']:
                assert raw[name] == old[name], (key,target,name)
            assert role_events == old['events']
            assert raw['role_cache_misses'] == old['cache_miss_calls']
            baseline_checks += 1
        total_requests += raw['requests']
        role_misses += raw['role_cache_misses']
        game_misses += raw['matched_game_incremental_misses']
        observer_misses += raw['observer_equivalence_cache_misses']
    expected = set(itertools.product(case_map,[.5] if smoke else [-.5,.5],[0,1] if smoke else range(5),[0,1] if smoke else range(3)))
    assert identities == expected and manifest['completed_jobs'] == len(expected)
    assert game_checks == (1 if smoke else 40)
    report = dict(passed=True,smoke=smoke,jobs=len(expected),preparations=len(preparations),
        inverse_endpoint_checks=inverse_checks,inversion_forward_samples=prep_forward,
        independent_cache_miss_evaluations=misses,verified_hits=hits,requests=total_requests,
        score_checks=scores,cached_endpoint_panels_verified=endpoint_checks,shadow_payments_verified=payments,
        matched_games_verified=game_checks,task9_baseline_jobs_identical=baseline_checks,
        role_cache_misses=role_misses,matched_game_incremental_misses=game_misses,
        observer_equivalence_cache_misses=observer_misses,
        validator_sha256=sha(Path(__file__)),canonical_payment_source_sha256=sha(ROOT/'experiments/contribution/role_statistics/analyze.py'))
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--reference',type=Path)
    args = p.parse_args()
    validate(args.source,args.smoke,args.reference)
