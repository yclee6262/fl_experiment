"""Independent scalar reconstruction of balance statistics from raw replies.

Does not import ledger.py or read its intermediate projections as reference.
"""
import math


def verify_balance(manifest,run,ledger):
    errors=[]
    def close(a,b):
        if a is None or b is None:
            if a is not b:raise ValueError('balance_verify_missing')
            return
        if not math.isfinite(a) or not math.isfinite(b) or abs(a-b)>1e-10+1e-10*max(abs(a),abs(b)):raise ValueError('balance_verify_numeric')
        errors.append(abs(a-b))
    owners=list(dict.fromkeys(manifest['anchor_owners']))
    if ledger['owners']!=owners or len(ledger['rounds'])!=len(run['iterations']):raise ValueError('balance_verify_coverage')
    ev=run['evaluations'];queries=run['queries'];v=manifest['model_weights'];a=manifest['direction_weights'];D=len(manifest['initial_solution'])
    raw=[]
    for e in ev:
        raw.append([abs(float(queries[qid]['output'][0])-manifest['target']) for qid in e['model_query_ids']])
    refs=[]
    for it,saved in zip(run['iterations'],ledger['rounds']):
        norm=math.sqrt(math.fsum(z*z for z in it['g_solver']));eligible=norm>1e-8
        perowner={o:[[] for _ in range(D)] for o in owners}
        if eligible:
            before=raw[it['current_eval_id']]
            for d in it['directions']:
                if d['skip_reason']:continue
                k=d['index'];o=manifest['anchor_owners'][k];source=raw[d['source_eval_id']]
                vec=[manifest['anchors'][k][j]-it['x_before'][j] for j in range(D)];dist=math.sqrt(math.fsum(z*z for z in vec))
                for j in range(D):
                    for m in range(len(v)):
                        perowner[o][j].append(a[k]*v[m]*(source[m]-before[m])/d['actual_h']*vec[j]/dist)
            p=[math.fsum(math.fsum(perowner[o][j])*it['g_solver'][j]/norm for j in range(D)) for o in owners]
            den=math.fsum(abs(z) for z in p);b=[max(-z,0)/den for z in p]
            for j in range(len(owners)):
                close(p[j],saved['p_owner'][j]);close(b[j],saved['b_owner'][j])
                if bool(saved['opposing'][j])!=(p[j]<0):raise ValueError('balance_verify_sign')
            close(den,saved['denominator_abs_projection'])
        else:
            p=b=None
            if saved['b_owner'] is not None or saved['p_owner'] is not None:raise ValueError('balance_verify_zero_direction')
        trials=it['trials']; accepted=[i for i,z in enumerate(trials) if z['accepted']]
        flags={
            'accepted_first_trial':bool(accepted and accepted[0]==0),
            'accepted_after_halving':bool(accepted and accepted[0]>0),
            'all_trials_rejected':bool(trials and not accepted),
            'zero_direction':not eligible,
            'has_rejected_trial':sum(not t['accepted'] for t in trials)>0,
            'no_update':not accepted,
            'mode_switch':it['mode_after']!=it['mode_before'],
            'solver_stop':it['stop_reason'] is not None,
            'box_limited_first_trial':bool(trials and trials[0]['requested_eta']-trials[0]['actual_bounded_step']>1e-12)}
        if saved['contexts']!=flags or saved['eligible']!=eligible or saved['t']!=it['t']:raise ValueError('balance_verify_context')
        outcome=next(name for name in ['accepted_first_trial','accepted_after_halving','all_trials_rejected','zero_direction'] if flags[name])
        if saved['outcome']!=outcome or saved['protection_effect_status']!='not_established':raise ValueError('balance_verify_semantics')
        refs.append(dict(p=p,b=b,eligible=eligible,flags=flags))
    def check_summary(indices,owner_index,saved):
        elig=[r for r in indices if r['eligible']];neg=[r for r in elig if r['p'][owner_index]<0]
        for field,value in [('rounds',len(indices)),('eligible_rounds',len(elig)),('opposing_rounds',len(neg)),
                            ('tiny_projection_rounds',sum(abs(r['p'][owner_index])<=1e-10 for r in elig))]:
            if saved[field]!=value:raise ValueError('balance_verify_count:'+field)
        close(saved['eligible_coverage'],len(elig)/len(indices) if indices else None)
        close(saved['opposing_rate'],len(neg)/len(elig) if elig else None)
        close(saved['mean_b'],math.fsum(r['b'][owner_index] for r in elig)/len(elig) if elig else None)
        close(saved['mean_b_when_opposing'],math.fsum(r['b'][owner_index] for r in neg)/len(neg) if neg else None)
    for i,s in enumerate(ledger['owner_summary']):
        if s['owner_id']!=owners[i] or s['protection_effect_status']!='not_established':raise ValueError('balance_verify_owner')
        check_summary(refs,i,s)
        for context,sub in s['contexts'].items():check_summary([r for r in refs if r['flags'][context]],i,sub)
    return dict(accepted=True,independent_raw_reply_reconstruction=True,max_absolute_error=max(errors,default=0),
                rounds=len(refs),owners=len(owners),new_model_predictions=0,protection_effect_established=False)
