"""Descriptive opposing-projection ledger. No valuation or reward conversion."""
import numpy as np
from .schema import check_close

VERSION='custom_balance_ledger_v1'
CONTEXTS=('accepted_first_trial','accepted_after_halving','all_trials_rejected','zero_direction',
          'has_rejected_trial','no_update','mode_switch','solver_stop','box_limited_first_trial')


def build_balance_ledger(owners,iterations,diagnostics):
    if not owners or len(set(owners))!=len(owners):raise ValueError('invalid_owner_ids')
    if len(iterations)!=len(diagnostics):raise ValueError('missing_round')
    if [r['t'] for r in iterations]!=list(range(len(iterations))) or [r['t'] for r in diagnostics]!=list(range(len(diagnostics))):raise ValueError('duplicate_or_missing_round')
    rows=[]
    for it,d in zip(iterations,diagnostics):
        if not it.get('closed'):raise ValueError('unclosed_round')
        norm=float(it['g_norm'])
        if not np.isfinite(norm) or norm<0:raise ValueError('invalid_norm')
        trials=it['trials'];accepted=[j for j,z in enumerate(trials) if z['accepted']]
        if len(accepted)>1 or (accepted and accepted[0]!=len(trials)-1):raise ValueError('invalid_trial_order')
        if bool(accepted)!=d['accepted'] or it['accepted_eval_id']!=(trials[accepted[0]]['eval_id'] if accepted else None):raise ValueError('acceptance_mismatch')
        valid=norm>1e-8
        if not valid and trials:raise ValueError('trials_with_zero_direction')
        if valid and not trials:raise ValueError('nonzero_direction_without_trial')
        flags=dict(accepted_first_trial=bool(accepted and accepted[0]==0),accepted_after_halving=bool(accepted and accepted[0]>0),
                   all_trials_rejected=bool(trials and not accepted),zero_direction=not valid,
                   has_rejected_trial=any(not t['accepted'] for t in trials),no_update=not bool(accepted),
                   mode_switch=bool(it['mode_switch']),solver_stop=it['stop_reason'] is not None,
                   box_limited_first_trial=bool(trials and trials[0]['actual_bounded_step'] < trials[0]['requested_eta']-1e-12))
        outcome='accepted_first_trial' if flags['accepted_first_trial'] else 'accepted_after_halving' if flags['accepted_after_halving'] else 'all_trials_rejected' if flags['all_trials_rejected'] else 'zero_direction'
        p=None;b=None;den=None;negative=None;tiny=None
        if valid:
            p=np.asarray(d['p_owner'],float)
            if p.shape!=(len(owners),) or not np.isfinite(p).all():raise ValueError('invalid_owner_projection')
            check_close(p.sum(),norm,'projection_total')
            den=float(np.abs(p).sum())
            if den<=0 or not np.isfinite(den):raise ValueError('invalid_projection_denominator')
            b=np.maximum(-p,0)/den;negative=p<0;tiny=np.abs(p)<=1e-10
            if np.any(b<0) or b.sum()>.5+1e-10:raise ValueError('invalid_opposition_fraction')
        elif d['p_owner'] is not None:raise ValueError('projection_on_zero_direction')
        rows.append(dict(t=it['t'],mode=it['mode_before'],eligible=valid,denominator_abs_projection=den,p_owner=p,
                         b_owner=b,opposing=negative,tiny_projection=tiny,outcome=outcome,contexts=flags,
                         protection_effect_status='not_established'))
    def stats(indices,oi):
        eligible=[r for r in indices if r['eligible']];negative=[r for r in eligible if r['opposing'][oi]]
        return dict(rounds=len(indices),eligible_rounds=len(eligible),opposing_rounds=len(negative),
                    eligible_coverage=len(eligible)/len(indices) if indices else None,
                    opposing_rate=len(negative)/len(eligible) if eligible else None,
                    mean_b=float(np.mean([r['b_owner'][oi] for r in eligible])) if eligible else None,
                    mean_b_when_opposing=float(np.mean([r['b_owner'][oi] for r in negative])) if negative else None,
                    tiny_projection_rounds=sum(bool(r['tiny_projection'][oi]) for r in eligible))
    summaries=[]
    for oi,owner in enumerate(owners):
        summaries.append(dict(owner_id=owner,**stats(rows,oi),
                              contexts={name:stats([r for r in rows if r['contexts'][name]],oi) for name in CONTEXTS},
                              protection_effect_status='not_established'))
    return dict(component_version=VERSION,owners=owners,rounds=rows,owner_summary=summaries,
                context_categories_overlap=True,frequency_denominator='eligible direction rounds',
                model_prediction_calls=0,protection_effect_status='not_established')
