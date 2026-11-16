"""Independent confirmation audit/statistics; never called by a prior stage."""
import argparse
import sys
from pathlib import Path

import numpy as np


def sign(x,delta=1e-4):
    x=np.asarray(x);return (x>delta).astype(int)-(x < -delta).astype(int)


def bootstrap_ci(values,seed=20261016):
    v=np.asarray(values,dtype=float)
    if len(v)==0:return None
    rng=np.random.default_rng(seed)
    means=v[rng.integers(0,len(v),size=(10000,len(v)))].mean(axis=1)
    return np.quantile(means,[.00625,.99375]).tolist()


def analyze(root):
    sys.path.insert(0,str(root/'source_snapshot'))
    import torch
    from models import PretrainedModel
    from recorder import features
    from experiment import read,save,sha,csvsave
    import rules
    torch.set_num_threads(1)
    config=read(root/'registration.json');execution=read(root/'confirmation_execution.json')
    model_manifest=read(root/'fitted_models_manifest.json')
    output=root/'confirmation_analysis';output.mkdir(exist_ok=False)
    errors=[];effect_rows=[];costs=[];audit_rows=[];reloaded_predictions=[]
    for name,h in config['source_sha256'].items():
        if sha(root/'source_snapshot'/name)!=h:errors.append('source_hash:'+name)
    for name,h in model_manifest['model_sha256'].items():
        if sha(root/name)!=h:errors.append('model_hash:'+name)
    for scenario in ['clean','controlled_poison']:
        for seed in config['confirmation_seeds']:
            d=root/'confirmation'/scenario/str(seed);key=f'{scenario}/{seed}'
            if not all((d/n).exists() for n in ['preparation.json','predictions.json','scores.json','solves/baseline.json']):
                errors.append('incomplete_scenario:'+key);continue
            state=read(d/'preparation.json');pred=read(d/'predictions.json');base=read(d/'solves/baseline.json')
            if base['status']!='valid_delivery':errors.append('baseline_failed:'+key);continue
            if base['predictions_sha256']!=sha(d/'predictions.json'):errors.append('prediction_lock:'+key)
            if pred['ids']!=state['ids']:errors.append('prediction_ID_order:'+key)
            if pred['evaluator_loaded'] or pred['LOO_loaded'] or base['evaluator_loaded']:errors.append('prediction_information_leakage:'+key)
            if pred['fitted_models_sha256']!=model_manifest['model_sha256']:errors.append('prediction_model_version:'+key)
            for p,h in state['artifact_sha256'].items():
                if not (d/p).exists() or sha(d/p)!=h:errors.append('artifact_hash:'+key+'/'+p)
            z=features(base['trace'],state['candidates'],state['alphas'])
            if not np.allclose(z,pred['features'],rtol=0,atol=1e-12):errors.append('feature_reconstruction:'+key)
            independently_predicted={'zero':np.zeros(len(z)),
                'constant':rules.constant_predict(read(root/'constant_development_model.json'),z[:,5].astype(int))}
            for name in ['full','model_only','candidate_only']:
                independently_predicted[name]=rules.predict(read(root/(name+'_development_model.json')),z)
            for name,values in independently_predicted.items():
                if not np.allclose(values,pred['predictions'][name],rtol=0,atol=1e-12):errors.append('prediction_reconstruction:'+key+'/'+name)
            artifact=d/state['artifact'];mp=torch.load(artifact/'local_models.pt',map_location='cpu',weights_only=True)
            ep=torch.load(artifact/'global_evaluator.pt',map_location='cpu',weights_only=True)
            evaluator=PretrainedModel(2);evaluator.load_state_dict(ep['state_dict']);evaluator.eval()
            local={}
            for i in state['ids']:
                m=PretrainedModel(2);m.load_state_dict(mp['agents'][i]);m.eval();local[i]=m
            scores={r['file']:r for r in read(d/'scores.json')['rows']}
            local_scores={}
            for index in [-1,*range(len(state['ids']))]:
                name='baseline.json' if index==-1 else f'removal_{index}.json';path=d/'solves'/name
                if not path.exists():errors.append('missing_solve:'+key+'/'+name);continue
                solve=read(path)
                if solve['status']!='valid_delivery':errors.append('failed_solve:'+key+'/'+name);continue
                ids=[i for j,i in enumerate(state['ids']) if j!=index]
                if solve.get('remaining_ids')!=ids:errors.append('remaining_IDs:'+key+'/'+name)
                if not np.allclose(solve['alphas'],np.ones(len(ids))/len(ids),atol=1e-12,rtol=0):errors.append('remaining_alphas:'+key+'/'+name)
                if solve['fixed_start']!=state['fixed_start']:errors.append('starting_point:'+key+'/'+name)
                if index>=0 and solve['geometry']['projection_residual']>1e-8:errors.append('removal_geometry:'+key+'/'+name)
                x=np.asarray(solve['result']['solution']);valid=np.isfinite(x).all() and np.max(abs(x))<=1+1e-8
                if not valid:errors.append('infeasible:'+key+'/'+name);continue
                with torch.no_grad():
                    tx=torch.tensor(x,dtype=torch.float32)[None,:]
                    score=abs(float(evaluator(tx).item()))
                    loss=float(np.mean([abs(float(local[i](tx).item())) for i in ids]))
                saved=scores.get(name)
                if saved is None or abs(score-saved['pooled_error'])>1e-7:errors.append('pooled_reload:'+key+'/'+name)
                if abs(loss-solve['result']['loss'])>1e-7:errors.append('internal_reload:'+key+'/'+name)
                audit_rows.append(dict(scenario=scenario,seed=seed,solve=name,
                    pooled_delta=None if saved is None else abs(score-saved['pooled_error']),internal_delta=abs(loss-solve['result']['loss'])))
                local_scores[index]=score
                costs.append(dict(scenario=scenario,seed=seed,index=index,calls=solve['calls'],
                                  input_samples=solve['input_samples'],seconds=solve['seconds']))
            if -1 not in local_scores:continue
            batch=1 if seed in config['confirmation_batches'][0] else 2
            for index,agent_id in enumerate(state['ids']):
                if index not in local_scores:continue
                C=local_scores[index]-local_scores[-1]
                effect_rows.append(dict(scenario=scenario,seed=seed,batch=batch,agent_id=agent_id,C_full=C,
                        **{name:float(values[index]) for name,values in independently_predicted.items()}))
    seed_rows=[];summaries=[];methods=['zero','constant','model_only','candidate_only','full']
    for scenario in ['clean','controlled_poison']:
        effects=[r for r in effect_rows if r['scenario']==scenario]
        for seed in config['confirmation_seeds']:
            rows=[r for r in effects if r['seed']==seed]
            if len(rows)!=(3 if scenario=='clean' else 5):continue
            y=np.array([r['C_full'] for r in rows]);p=np.array([r['full'] for r in rows]);c=np.array([r['constant'] for r in rows])
            seed_rows.append(dict(scenario=scenario,seed=seed,batch=rows[0]['batch'],
                 direction_agreement=float(np.mean(sign(y)==sign(p))),constant_MAE=float(np.mean(abs(y-c))),
                 full_MAE=float(np.mean(abs(y-p))),MAE_improvement=float(np.mean(abs(y-c)-abs(y-p)))))
        per_seed=[r for r in seed_rows if r['scenario']==scenario]
        for name in methods:
            if not effects:continue
            y=np.array([r['C_full'] for r in effects]);p=np.array([r[name] for r in effects])
            tn=sign(y)==-1;pn=sign(p)==-1;tp=int(np.sum(tn&pn))
            summaries.append(dict(scenario=scenario,method=name,n_effects=len(effects),n_seeds=len(per_seed),
                 MAE=float(np.mean(abs(y-p))),direction_agreement=float(np.mean(sign(y)==sign(p))),
                 true_negative=int(tn.sum()),predicted_negative=int(pn.sum()),
                 negative_precision=None if not pn.any() else tp/int(pn.sum()),
                 negative_recall=None if not tn.any() else tp/int(tn.sum())))
    passed=[];details=[]
    for scenario in ['clean','controlled_poison']:
        ss=[r for r in seed_rows if r['scenario']==scenario];es=[r for r in effect_rows if r['scenario']==scenario]
        if not ss:passed.append(False);continue
        a=np.mean([r['direction_agreement'] for r in ss]);aci=bootstrap_ci([r['direction_agreement'] for r in ss])
        improvement=np.mean([r['MAE_improvement'] for r in ss]);ici=bootstrap_ci([r['MAE_improvement'] for r in ss])
        constant=np.mean([r['constant_MAE'] for r in ss]);relative=improvement/constant if constant>0 else None
        met=next(r for r in summaries if r['scenario']==scenario and r['method']=='full')
        negative_seeds=len({r['seed'] for r in es if r['C_full'] < -1e-4})
        batches={str(b):float(np.mean([r['direction_agreement'] for r in ss if r['batch']==b]))
                 for b in [1,2] if any(r['batch']==b for r in ss)}
        ok=(len(ss)==20 and a>=.7 and aci[0]>.5 and relative is not None and relative>=.1 and ici[0]>0
            and met['true_negative']>=10 and negative_seeds>=5 and met['negative_precision'] is not None
            and met['negative_precision']>=.7 and met['negative_recall']>=.7
            and len(batches)==2 and all(v>=.7 for v in batches.values()))
        passed.append(bool(ok));details.append(dict(scenario=scenario,pass_all=bool(ok),direction_mean=float(a),
             direction_CI=aci,MAE_improvement_CI=ici,relative_MAE_improvement=relative,
             negative_reference_seeds=negative_seeds,batch_direction_agreement=batches))
    base_calls=sum(r['calls'] for r in costs if r['index']==-1);all_calls=sum(r['calls'] for r in costs)
    saving=None if not all_calls else 1-base_calls/all_calls
    accepted=execution['status']=='complete' and not errors and len(effect_rows)==160 and len(audit_rows)==200
    cost_pass=read(root/'complete_online_cost_gate.json')['accepted'] and saving is not None and saving>=.5
    conclusion=('limited_contribution_prediction_established' if accepted and all(passed) and cost_pass
                else 'prediction_validity_not_established' if accepted else 'incomplete_or_failed_acceptance')
    csvsave(output/'participant_predictions.csv',effect_rows);csvsave(output/'seed_metrics.csv',seed_rows)
    csvsave(output/'method_comparisons.csv',summaries);csvsave(output/'reload_audit.csv',audit_rows)
    csvsave(output/'search_costs.csv',costs)
    sensitivity=[]
    for delta in [1e-7,1e-3]:
        for scenario in ['clean','controlled_poison']:
            es=[r for r in effect_rows if r['scenario']==scenario]
            if es:sensitivity.append(dict(delta=delta,scenario=scenario,direction_agreement=float(np.mean(sign([r['C_full'] for r in es],delta)==sign([r['full'] for r in es],delta)))))
    csvsave(output/'sensitivity.csv',sensitivity)
    save(output/'acceptance.json',dict(accepted=accepted,errors=errors,n_effects=len(effect_rows),n_reloads=len(audit_rows)))
    save(output/'results.json',dict(status=conclusion,primary=details,cost_pass=cost_pass,
        model_call_saving=saving,baseline_calls=base_calls,reference_total_calls=all_calls,
        independent_confirmation=True,original_P3_results_unchanged=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();analyze(args.root.resolve())
