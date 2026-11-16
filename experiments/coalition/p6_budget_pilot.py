"""Paired P6 budget/eta pilot with one frozen scenario per seed."""
import argparse
import copy
import json
import time
from pathlib import Path


def _source_workspace(source_file):
    """Find the flat workspace or a standalone historical source snapshot."""
    start = Path(source_file).resolve().parent
    for candidate in (start, *start.parents):
        if all((candidate / name).is_file() for name in
               ("host_server.py", "run_experiments.py", "dataset.py")):
            return candidate
    raise FileNotFoundError(f"Cannot locate source workspace from {source_file}")

import torch
from p6_recruitment_experiment import (build_parser, qualified_candidates, evaluate_selection,
    write_csv, provenance)
from alpha_weight_experiments import (
    attach_query_counter,
    build_parser as alpha_parser,
    build_base_server,
)
from recruitment_selection import exact_select, greedy_select, selection_report, residual_dissimilarity
from research_evaluation import save_scenario_artifacts


def legacy_select(candidates, budget, max_size, eta):
    chosen = []
    while len(chosen) < max_size:
        available = [i for i,c in enumerate(candidates) if i not in chosen and
            sum(candidates[j]['bid'] for j in chosen)+c['bid'] <= budget+1e-12]
        if not available: break
        def score(i):
            diversity = min((residual_dissimilarity(candidates[i]['residual'], candidates[j]['residual'])
                             for j in chosen), default=1.)
            return candidates[i]['quality_share']/candidates[i]['bid']*(1+eta*diversity)
        chosen.append(max(available, key=score))
    if not chosen: raise ValueError('No feasible recruitment')
    return selection_report(chosen,candidates,budget,max_size,eta)


def run(output, seeds=(0, 1, 2, 3, 4)):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()): raise FileExistsError(root)
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for source in _source_workspace(__file__).glob('*.py'):
        (snapshot/source.name).write_bytes(source.read_bytes())
    manifest={**provenance(),'status':'running','seeds':list(seeds),
        'budgets':[2.5,3.5,8.0],'etas':[0.,0.5],'constraint_mode':'box_constrained',
        'feasibility_tolerance':1e-8,'qualification_target':0.,'formal_target':0.5,
        'accounting':'base reward realizes the recruitment bid; count it once'}
    (root/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[];candidate_rows=[]
    try:
        for seed in seeds:
            args=build_parser().parse_args(['--output-dir',output])
            base_args=alpha_parser().parse_args([])
            base_args.num_agents=10
            base_args.constraint_mode='box_constrained'
            base_args.settle_payment=False
            seed_started=time.perf_counter()
            server=build_base_server(base_args,seed,2,0.,0.5)
            selection_counter = server.preparation_counter
            selection_counter["stage"] = "selection_precompute"
            precompute_started=time.perf_counter()
            candidates=qualified_candidates(server)
            precompute_seconds=time.perf_counter()-precompute_started
            selection_counter["stage"] = "proposal"
            selection_precompute_requests = selection_counter["requests"][
                "selection_precompute"
            ]
            selection_precompute_input_samples = selection_counter["input_samples"][
                "selection_precompute"
            ]
            if len(candidates)>10: raise ValueError('Expected fixed ten-candidate P6 pool; refusing silent truncation')
            evaluator_started=time.perf_counter()
            evaluator,artifact,metrics=save_scenario_artifacts(root,
                dict(seed=seed,dimension=2,poison_ratio=0.,target=0.5),server.all_agents,server.study_records)
            evaluator_seconds=time.perf_counter()-evaluator_started
            proposals={}
            for c in candidates:
                proposal_started=time.perf_counter()
                torch.manual_seed(seed*100000+c['agent_id'])
                proposals[c['agent_id']]=c['agent'].infer_parameters_D(0.5,steps=500)
                proposal_seconds=time.perf_counter()-proposal_started
                candidate_rows.append(dict(seed=seed,agent_id=c['agent_id'],mse=c['mse'],
                    bid=c['bid'],quality_share=c['quality_share'],residual=json.dumps(c['residual'].tolist()),
                    proposal=json.dumps(proposals[c['agent_id']].tolist()),qualification_target=0.,formal_target=0.5,
                    proposal_seconds=proposal_seconds))
            (root/f'qualification_seed{seed}.json').write_text(json.dumps(server.phase1_report,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x),indent=2))
            for budget in [2.5,3.5,8.]:
                for eta in [0.,0.5]:
                    for name,select in [('exact',exact_select),('objective_greedy',greedy_select),('legacy_greedy',legacy_select)]:
                        report=None
                        started=time.perf_counter()
                        selection_seconds=0.0
                        try:
                            report=select(candidates,budget,3,eta)
                            selection_seconds=time.perf_counter()-started
                            started=time.perf_counter()
                            row=evaluate_selection(name,report,copy.deepcopy(candidates),proposals,evaluator,args,seed)
                            row['delivery_status']='valid' if row['box_feasible'] else 'box_infeasible'
                        except ValueError as exc:
                            elapsed=time.perf_counter()-started
                            no_selection=report is None
                            row=dict(selection_method=name,delivery_status='no_feasible_selection' if no_selection else 'solver_failure',
                                execution_status='completed',
                                failure_stage='selection' if no_selection else 'stage3',failure_reason=str(exc),final_solution='',box_feasible='',
                                budget_feasible='',selection_objective='',selection_seconds=selection_seconds,
                                inversion_seconds=elapsed,external_global_error='',true_error='',stage3_requests=0,
                                stage3_input_samples=0,
                                recruitment_bid_commitment='',accounting_total_committed_cost='',selected_agent_ids='')
                        except Exception as exc:
                            elapsed=time.perf_counter()-started
                            row=dict(selection_method=name,delivery_status='interrupted',execution_status='interrupted',
                                failure_stage='unexpected_exception',failure_reason=f'{type(exc).__name__}: {exc}',
                                final_solution='',box_feasible='',budget_feasible='',selection_objective='',
                                selection_seconds=elapsed if report is None else selection_seconds,
                                inversion_seconds='',external_global_error='',true_error='',stage3_requests='',
                                stage3_input_samples='',recruitment_bid_commitment='',
                                accounting_total_committed_cost='',selected_agent_ids='')
                            row.update(seed=seed,budget=budget,diversity_eta=eta,
                                scenario_artifact_dir=str(artifact.relative_to(root)),formal_target=0.5,
                                qualification_target=0.,payment_status='not_run_by_configuration')
                            rows.append(row)
                            from alpha_weight_experiments import write_csv as union_csv
                            union_csv(root/'selection_results.csv',rows)
                            manifest.update(status='interrupted',error=row['failure_reason'])
                            (root/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
                            raise
                        row.update(seed=seed,budget=budget,diversity_eta=eta,
                            inversion_seconds=row.get('inversion_seconds',time.perf_counter()-started),
                            scenario_artifact_dir=str(artifact.relative_to(root)),
                            formal_target=0.5,qualification_target=0.,
                            selection_precompute_requests=selection_precompute_requests,
                            selection_precompute_input_samples=selection_precompute_input_samples,
                            shared_training_seconds=server.preparation_costs['training_seconds'],
                            shared_qualification_seconds=server.preparation_costs['qualification_seconds'],
                            shared_proposal_seconds=server.preparation_costs['proposal_seconds'],
                            shared_evaluator_seconds=evaluator_seconds,
                            selection_precompute_seconds=precompute_seconds,
                            selection_seconds=(time.perf_counter()-started if report is None else selection_seconds),
                            shared_qualification_requests=server.preparation_costs['qualification_requests'],
                            shared_proposal_requests=server.preparation_costs['proposal_requests'],
                            shared_qualification_input_samples=server.preparation_costs['qualification_input_samples'],
                            shared_proposal_input_samples=server.preparation_costs['proposal_input_samples'],
                            external_score_status='not_applicable' if not row.get('final_solution') else 'scored',
                            payment_status='not_run_by_configuration',
                            seed_shared_total_seconds=time.perf_counter()-seed_started,
                            external_evaluator_test_truth_mse=metrics['test_truth_mse'])
                        rows.append(row)
                        from alpha_weight_experiments import write_csv as union_csv
                        union_csv(root/'selection_results.csv',rows)
            from alpha_weight_experiments import write_csv as union_csv
            union_csv(root/'selection_results.csv',rows)
            write_csv(root/'candidates.csv',candidate_rows)
        manifest['status']='complete'
    except Exception as exc:
        if manifest.get('status') != 'interrupted':
            manifest.update(status='failed',error=str(exc))
        raise
    finally:
        if rows:
            from alpha_weight_experiments import write_csv as union_csv
            union_csv(root/'selection_results.csv',rows)
        if candidate_rows:
            write_csv(root/'candidates.csv',candidate_rows)
        import hashlib
        manifest['artifact_sha256']={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob('*') if path.is_file() and path.name!='run_manifest.json'}
        (root/'run_manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True)
    parser.add_argument('--seeds',default='0,1,2,3,4')
    parsed=parser.parse_args()
    run(parsed.output_dir,[int(value) for value in parsed.seeds.split(',') if value.strip()])
