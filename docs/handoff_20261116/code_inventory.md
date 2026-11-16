> 本文件記錄第一次搬移；目前位置與最新驗證請見 [第二次整理](../reorganization_20261116/README.md)。

# 程式分類與移轉清單

日期：2026-11-16。213支原有Python實體檔案已搬移；另新增主流程封裝、啟動器與小型驗證。原路徑保留相對符號連結，不是重複內容。詳細雜湊見 `migration_manifest.json`。

## 各區數量

| 位置 | 清單內程式數 |
|---|---:|
| `core` | 3 |
| `experiments/alpha/analysis` | 48 |
| `experiments/alpha/methods` | 38 |
| `experiments/alpha/runners` | 37 |
| `experiments/alpha/validation` | 11 |
| `experiments/baselines` | 5 |
| `experiments/coalition` | 9 |
| `experiments/contribution` | 7 |
| `experiments/protocols` | 11 |
| `experiments/shared` | 8 |
| `tests` | 37 |

本表含新增小型驗證1支；不含 `__init__.py`、`core/pipeline.py` 及根啟動器。既有貢獻套件59個另外列於下方，未移動其實體檔案。

## 原根目錄檔案對照

| 舊路徑 | 實體新位置 | 本次內容調整 |
|---|---|---|
| `ablation_main.py` | `experiments/baselines/ablation_main.py` | 內容逐位保留 |
| `adaptive_correlation_alpha.py` | `experiments/alpha/methods/adaptive_correlation_alpha.py` | 內容逐位保留 |
| `adaptive_correlation_pipeline.py` | `experiments/alpha/runners/adaptive_correlation_pipeline.py` | 內容逐位保留 |
| `agent_client.py` | `core/agent_client.py` | 內容逐位保留 |
| `alpha_initialization_selection.py` | `experiments/alpha/methods/alpha_initialization_selection.py` | 內容逐位保留 |
| `alpha_mixture_exploration.py` | `experiments/alpha/methods/alpha_mixture_exploration.py` | 內容逐位保留 |
| `alpha_selection_formal_statistics.py` | `experiments/alpha/methods/alpha_selection_formal_statistics.py` | 內容逐位保留 |
| `alpha_weight_experiments.py` | `experiments/alpha/methods/alpha_weight_experiments.py` | 僅來源路徑定位 |
| `analyze_adaptive_alpha_failure_modes.py` | `experiments/alpha/analysis/analyze_adaptive_alpha_failure_modes.py` | 內容逐位保留 |
| `analyze_adaptive_alpha_replay.py` | `experiments/alpha/analysis/analyze_adaptive_alpha_replay.py` | 內容逐位保留 |
| `analyze_adaptive_correlation.py` | `experiments/alpha/analysis/analyze_adaptive_correlation.py` | 內容逐位保留 |
| `analyze_alpha_cost_frontier.py` | `experiments/alpha/analysis/analyze_alpha_cost_frontier.py` | 內容逐位保留 |
| `analyze_alpha_economic_frontier.py` | `experiments/alpha/analysis/analyze_alpha_economic_frontier.py` | 內容逐位保留 |
| `analyze_alpha_initialization_study.py` | `experiments/alpha/analysis/analyze_alpha_initialization_study.py` | 內容逐位保留 |
| `analyze_alpha_mixture_exploration.py` | `experiments/alpha/analysis/analyze_alpha_mixture_exploration.py` | 內容逐位保留 |
| `analyze_alpha_selection_formal.py` | `experiments/alpha/analysis/analyze_alpha_selection_formal.py` | 內容逐位保留 |
| `analyze_bootstrap_quality_research.py` | `experiments/alpha/analysis/analyze_bootstrap_quality_research.py` | 內容逐位保留 |
| `analyze_conservative_shrinkage.py` | `experiments/alpha/analysis/analyze_conservative_shrinkage.py` | 內容逐位保留 |
| `analyze_correlation_replication.py` | `experiments/alpha/analysis/analyze_correlation_replication.py` | 內容逐位保留 |
| `analyze_ensemble_single_b_frontier.py` | `experiments/alpha/analysis/analyze_ensemble_single_b_frontier.py` | 內容逐位保留 |
| `analyze_ensemble_target_transfer_diagnostics.py` | `experiments/alpha/analysis/analyze_ensemble_target_transfer_diagnostics.py` | 內容逐位保留 |
| `analyze_evaluator_coverage.py` | `experiments/alpha/analysis/analyze_evaluator_coverage.py` | 內容逐位保留 |
| `analyze_four_alpha.py` | `experiments/alpha/analysis/analyze_four_alpha.py` | 內容逐位保留 |
| `analyze_four_way_bayes.py` | `experiments/alpha/analysis/analyze_four_way_bayes.py` | 內容逐位保留 |
| `analyze_marginal_portfolio_diagnostics.py` | `experiments/alpha/analysis/analyze_marginal_portfolio_diagnostics.py` | 內容逐位保留 |
| `analyze_next_alpha_comparison.py` | `experiments/alpha/analysis/analyze_next_alpha_comparison.py` | 內容逐位保留 |
| `analyze_oob_shrinkage.py` | `experiments/alpha/analysis/analyze_oob_shrinkage.py` | 內容逐位保留 |
| `analyze_p2_results.py` | `experiments/protocols/analyze_p2_results.py` | 內容逐位保留 |
| `analyze_p3_external_validity.py` | `experiments/contribution/analyze_p3_external_validity.py` | 僅來源路徑定位 |
| `analyze_p3_results.py` | `experiments/contribution/analyze_p3_results.py` | 內容逐位保留 |
| `analyze_p4_no_progress.py` | `experiments/protocols/analyze_p4_no_progress.py` | 內容逐位保留 |
| `analyze_p5_diagnostics.py` | `experiments/protocols/analyze_p5_diagnostics.py` | 內容逐位保留 |
| `analyze_p5_results.py` | `experiments/protocols/analyze_p5_results.py` | 內容逐位保留 |
| `analyze_p6_enumeration.py` | `experiments/coalition/analyze_p6_enumeration.py` | 內容逐位保留 |
| `analyze_p6_recruitment.py` | `experiments/coalition/analyze_p6_recruitment.py` | 內容逐位保留 |
| `analyze_peer_support.py` | `experiments/alpha/analysis/analyze_peer_support.py` | 內容逐位保留 |
| `analyze_pure_es_extension.py` | `experiments/alpha/analysis/analyze_pure_es_extension.py` | 內容逐位保留 |
| `analyze_repeated_alpha_confirmation.py` | `experiments/alpha/analysis/analyze_repeated_alpha_confirmation.py` | 內容逐位保留 |
| `analyze_robust_alpha_study.py` | `experiments/alpha/analysis/analyze_robust_alpha_study.py` | 內容逐位保留 |
| `analyze_se_qu_extension.py` | `experiments/alpha/analysis/analyze_se_qu_extension.py` | 內容逐位保留 |
| `analyze_selector_information_gap.py` | `experiments/alpha/analysis/analyze_selector_information_gap.py` | 內容逐位保留 |
| `analyze_selector_reliability.py` | `experiments/alpha/analysis/analyze_selector_reliability.py` | 內容逐位保留 |
| `analyze_stacked_extra_candidate.py` | `experiments/alpha/analysis/analyze_stacked_extra_candidate.py` | 內容逐位保留 |
| `analyze_targeted_label_alpha.py` | `experiments/alpha/analysis/analyze_targeted_label_alpha.py` | 內容逐位保留 |
| `analyze_unified_alpha_comparison.py` | `experiments/alpha/analysis/analyze_unified_alpha_comparison.py` | 內容逐位保留 |
| `assess_adaptive_alpha_formal.py` | `experiments/alpha/analysis/assess_adaptive_alpha_formal.py` | 內容逐位保留 |
| `assess_conditional_target.py` | `experiments/alpha/analysis/assess_conditional_target.py` | 內容逐位保留 |
| `assess_economic_multistart.py` | `experiments/alpha/analysis/assess_economic_multistart.py` | 內容逐位保留 |
| `assess_ensemble_target_objective.py` | `experiments/alpha/analysis/assess_ensemble_target_objective.py` | 僅來源路徑定位 |
| `assess_ensemble_target_transfer.py` | `experiments/alpha/analysis/assess_ensemble_target_transfer.py` | 僅來源路徑定位 |
| `assess_expanded_audit_alpha.py` | `experiments/alpha/analysis/assess_expanded_audit_alpha.py` | 內容逐位保留 |
| `assess_local_selector_alpha.py` | `experiments/alpha/analysis/assess_local_selector_alpha.py` | 內容逐位保留 |
| `assess_marginal_alpha.py` | `experiments/alpha/analysis/assess_marginal_alpha.py` | 內容逐位保留 |
| `assess_marginal_cache_formal.py` | `experiments/alpha/analysis/assess_marginal_cache_formal.py` | 內容逐位保留 |
| `assess_marginal_transfer.py` | `experiments/alpha/analysis/assess_marginal_transfer.py` | 內容逐位保留 |
| `assess_portfolio_alpha.py` | `experiments/alpha/analysis/assess_portfolio_alpha.py` | 內容逐位保留 |
| `assess_target_marginal_blend.py` | `experiments/alpha/analysis/assess_target_marginal_blend.py` | 內容逐位保留 |
| `assess_three_candidate_alpha.py` | `experiments/alpha/analysis/assess_three_candidate_alpha.py` | 內容逐位保留 |
| `assess_validation_calibrated_alpha.py` | `experiments/alpha/analysis/assess_validation_calibrated_alpha.py` | 內容逐位保留 |
| `audit_alpha_selection_formal.py` | `experiments/alpha/validation/audit_alpha_selection_formal.py` | 內容逐位保留 |
| `audit_peer_support_completion.py` | `experiments/alpha/validation/audit_peer_support_completion.py` | 內容逐位保留 |
| `audit_repeated_alpha_evidence.py` | `experiments/alpha/validation/audit_repeated_alpha_evidence.py` | 內容逐位保留 |
| `audit_repeated_alpha_final_statistics.py` | `experiments/alpha/validation/audit_repeated_alpha_final_statistics.py` | 內容逐位保留 |
| `audit_stage3_query_reuse.py` | `experiments/alpha/validation/audit_stage3_query_reuse.py` | 內容逐位保留 |
| `bootstrap_quality_research.py` | `experiments/alpha/methods/bootstrap_quality_research.py` | 內容逐位保留 |
| `build_repeated_alpha_report.py` | `experiments/alpha/analysis/build_repeated_alpha_report.py` | 內容逐位保留 |
| `conservative_shrinkage.py` | `experiments/alpha/methods/conservative_shrinkage.py` | 內容逐位保留 |
| `correlation_replication_pipeline.py` | `experiments/alpha/runners/correlation_replication_pipeline.py` | 內容逐位保留 |
| `dataset.py` | `experiments/shared/dataset.py` | 內容逐位保留 |
| `diagnose_repeated_alpha_qualified_cases.py` | `experiments/alpha/analysis/diagnose_repeated_alpha_qualified_cases.py` | 內容逐位保留 |
| `execute_alpha_initialization_selection_pilot.py` | `experiments/alpha/runners/execute_alpha_initialization_selection_pilot.py` | 內容逐位保留 |
| `execute_alpha_selection_formal.py` | `experiments/alpha/runners/execute_alpha_selection_formal.py` | 內容逐位保留 |
| `explain_peer_support_results.py` | `experiments/alpha/analysis/explain_peer_support_results.py` | 內容逐位保留 |
| `explore_calibrated_selection.py` | `experiments/alpha/methods/explore_calibrated_selection.py` | 內容逐位保留 |
| `explore_candidate_alpha_portfolio.py` | `experiments/alpha/methods/explore_candidate_alpha_portfolio.py` | 內容逐位保留 |
| `explore_cv_ab_switch.py` | `experiments/alpha/methods/explore_cv_ab_switch.py` | 內容逐位保留 |
| `explore_ensemble_target_objective.py` | `experiments/alpha/methods/explore_ensemble_target_objective.py` | 內容逐位保留 |
| `explore_expanded_audit_alpha.py` | `experiments/alpha/methods/explore_expanded_audit_alpha.py` | 內容逐位保留 |
| `explore_expanded_audit_selector.py` | `experiments/alpha/methods/explore_expanded_audit_selector.py` | 內容逐位保留 |
| `explore_extended_alpha_selector.py` | `experiments/alpha/methods/explore_extended_alpha_selector.py` | 內容逐位保留 |
| `explore_information_alpha.py` | `experiments/alpha/methods/explore_information_alpha.py` | 內容逐位保留 |
| `explore_initial_alpha.py` | `experiments/alpha/methods/explore_initial_alpha.py` | 內容逐位保留 |
| `explore_jackknife_selector.py` | `experiments/alpha/methods/explore_jackknife_selector.py` | 內容逐位保留 |
| `explore_marginal_stability_gate.py` | `experiments/alpha/methods/explore_marginal_stability_gate.py` | 內容逐位保留 |
| `explore_risk_aware_alpha_selection.py` | `experiments/alpha/methods/explore_risk_aware_alpha_selection.py` | 內容逐位保留 |
| `explore_smooth_bias_alpha_selector.py` | `experiments/alpha/methods/explore_smooth_bias_alpha_selector.py` | 內容逐位保留 |
| `explore_sparse_alpha.py` | `experiments/alpha/methods/explore_sparse_alpha.py` | 內容逐位保留 |
| `explore_stacked_ensemble_alpha.py` | `experiments/alpha/methods/explore_stacked_ensemble_alpha.py` | 內容逐位保留 |
| `explore_target_marginal_blend.py` | `experiments/alpha/methods/explore_target_marginal_blend.py` | 內容逐位保留 |
| `explore_targeted_label_alpha.py` | `experiments/alpha/methods/explore_targeted_label_alpha.py` | 內容逐位保留 |
| `finalize_alpha_mixture_exploration.py` | `experiments/alpha/runners/finalize_alpha_mixture_exploration.py` | 內容逐位保留 |
| `finalize_four_alpha.py` | `experiments/alpha/runners/finalize_four_alpha.py` | 內容逐位保留 |
| `finalize_peer_support_report.py` | `experiments/alpha/runners/finalize_peer_support_report.py` | 內容逐位保留 |
| `fl_comparison.py` | `experiments/baselines/fl_comparison.py` | 內容逐位保留 |
| `fl_comparison_filtered.py` | `experiments/baselines/fl_comparison_filtered.py` | 內容逐位保留 |
| `fl_comparison_numerical.py` | `experiments/baselines/fl_comparison_numerical.py` | 內容逐位保留 |
| `four_alpha_initialization.py` | `experiments/alpha/methods/four_alpha_initialization.py` | 內容逐位保留 |
| `four_way_bayes.py` | `experiments/alpha/methods/four_way_bayes.py` | 內容逐位保留 |
| `host_server.py` | `core/host_server.py` | 內容逐位保留 |
| `initial_alpha_methods.py` | `experiments/alpha/methods/initial_alpha_methods.py` | 內容逐位保留 |
| `main.py` | `experiments/baselines/main.py` | 內容逐位保留 |
| `main_alpha_calibration.py` | `experiments/alpha/runners/main_alpha_calibration.py` | 內容逐位保留 |
| `models.py` | `core/models.py` | 內容逐位保留 |
| `next_alpha_initialization.py` | `experiments/alpha/methods/next_alpha_initialization.py` | 內容逐位保留 |
| `next_alpha_pipeline.py` | `experiments/alpha/runners/next_alpha_pipeline.py` | 內容逐位保留 |
| `oob_shrinkage.py` | `experiments/alpha/methods/oob_shrinkage.py` | 內容逐位保留 |
| `oob_shrinkage_research.py` | `experiments/alpha/methods/oob_shrinkage_research.py` | 內容逐位保留 |
| `p3_contribution_validation.py` | `experiments/contribution/p3_contribution_validation.py` | 僅來源路徑定位 |
| `p4_constrained_no_progress_pilot.py` | `experiments/protocols/p4_constrained_no_progress_pilot.py` | 內容逐位保留 |
| `p5_scenario_suite.py` | `experiments/protocols/p5_scenario_suite.py` | 內容逐位保留 |
| `p6_budget_pilot.py` | `experiments/coalition/p6_budget_pilot.py` | 僅來源路徑定位 |
| `p6_coalition_enumeration.py` | `experiments/coalition/p6_coalition_enumeration.py` | 僅來源路徑定位 |
| `p6_recruitment_experiment.py` | `experiments/coalition/p6_recruitment_experiment.py` | 僅來源路徑定位 |
| `peer_support_alpha.py` | `experiments/alpha/methods/peer_support_alpha.py` | 內容逐位保留 |
| `peer_support_costs.py` | `experiments/alpha/methods/peer_support_costs.py` | 內容逐位保留 |
| `plot_alpha_guard_results.py` | `experiments/alpha/analysis/plot_alpha_guard_results.py` | 僅來源路徑定位 |
| `plot_p2_results.py` | `experiments/protocols/plot_p2_results.py` | 內容逐位保留 |
| `plot_p3_external_validity.py` | `experiments/contribution/plot_p3_external_validity.py` | 內容逐位保留 |
| `plot_p3_results.py` | `experiments/contribution/plot_p3_results.py` | 內容逐位保留 |
| `plot_p4_no_progress.py` | `experiments/protocols/plot_p4_no_progress.py` | 內容逐位保留 |
| `plot_p5_results.py` | `experiments/protocols/plot_p5_results.py` | 內容逐位保留 |
| `plot_p6_recruitment.py` | `experiments/coalition/plot_p6_recruitment.py` | 內容逐位保留 |
| `prepare_alpha_initialization_selection.py` | `experiments/alpha/runners/prepare_alpha_initialization_selection.py` | 內容逐位保留 |
| `prepare_alpha_selection_formal.py` | `experiments/alpha/runners/prepare_alpha_selection_formal.py` | 內容逐位保留 |
| `pure_es_initialization.py` | `experiments/alpha/methods/pure_es_initialization.py` | 內容逐位保留 |
| `reanalyze_legacy_p2.py` | `experiments/protocols/reanalyze_legacy_p2.py` | 內容逐位保留 |
| `recruitment_selection.py` | `experiments/coalition/recruitment_selection.py` | 內容逐位保留 |
| `reevaluate_p3_solutions.py` | `experiments/contribution/reevaluate_p3_solutions.py` | 內容逐位保留 |
| `reevaluate_p6_coalitions.py` | `experiments/coalition/reevaluate_p6_coalitions.py` | 內容逐位保留 |
| `reevaluate_p6_solutions.py` | `experiments/coalition/reevaluate_p6_solutions.py` | 內容逐位保留 |
| `reevaluate_saved_solutions.py` | `experiments/shared/reevaluate_saved_solutions.py` | 內容逐位保留 |
| `repeated_alpha_candidates.py` | `experiments/alpha/methods/repeated_alpha_candidates.py` | 內容逐位保留 |
| `replay_four_alpha.py` | `experiments/alpha/runners/replay_four_alpha.py` | 內容逐位保留 |
| `replay_marginal_stage3_cache.py` | `experiments/alpha/runners/replay_marginal_stage3_cache.py` | 內容逐位保留 |
| `report_four_way_bayes.py` | `experiments/alpha/analysis/report_four_way_bayes.py` | 內容逐位保留 |
| `research_analysis.py` | `experiments/shared/research_analysis.py` | 內容逐位保留 |
| `research_evaluation.py` | `experiments/shared/research_evaluation.py` | 內容逐位保留 |
| `resume_four_alpha_pipeline.py` | `experiments/alpha/runners/resume_four_alpha_pipeline.py` | 內容逐位保留 |
| `run_adaptive_alpha_replay.py` | `experiments/alpha/runners/run_adaptive_alpha_replay.py` | 內容逐位保留 |
| `run_adaptive_correlation_study.py` | `experiments/alpha/runners/run_adaptive_correlation_study.py` | 內容逐位保留 |
| `run_alpha_initialization_selection.py` | `experiments/alpha/runners/run_alpha_initialization_selection.py` | 內容逐位保留 |
| `run_alpha_initialization_study.py` | `experiments/alpha/runners/run_alpha_initialization_study.py` | 內容逐位保留 |
| `run_alpha_selection_formal_batch.py` | `experiments/alpha/runners/run_alpha_selection_formal_batch.py` | 內容逐位保留 |
| `run_conservative_shrinkage_study.py` | `experiments/alpha/runners/run_conservative_shrinkage_study.py` | 內容逐位保留 |
| `run_correlation_replication.py` | `experiments/alpha/runners/run_correlation_replication.py` | 內容逐位保留 |
| `run_experiments.py` | `experiments/shared/run_experiments.py` | 內容逐位保留 |
| `run_four_alpha_batch.py` | `experiments/alpha/runners/run_four_alpha_batch.py` | 內容逐位保留 |
| `run_four_alpha_pipeline.py` | `experiments/alpha/runners/run_four_alpha_pipeline.py` | 內容逐位保留 |
| `run_next_alpha_comparison.py` | `experiments/alpha/runners/run_next_alpha_comparison.py` | 內容逐位保留 |
| `run_peer_support_batch.py` | `experiments/alpha/runners/run_peer_support_batch.py` | 內容逐位保留 |
| `run_peer_support_pipeline.py` | `experiments/alpha/runners/run_peer_support_pipeline.py` | 內容逐位保留 |
| `run_pure_es_batch.py` | `experiments/alpha/runners/run_pure_es_batch.py` | 內容逐位保留 |
| `run_pure_es_extension.py` | `experiments/alpha/runners/run_pure_es_extension.py` | 內容逐位保留 |
| `run_repeated_alpha_batch.py` | `experiments/alpha/runners/run_repeated_alpha_batch.py` | 內容逐位保留 |
| `run_repeated_alpha_pipeline.py` | `experiments/alpha/runners/run_repeated_alpha_pipeline.py` | 內容逐位保留 |
| `run_se_qu_batch.py` | `experiments/alpha/runners/run_se_qu_batch.py` | 內容逐位保留 |
| `run_se_qu_extension.py` | `experiments/alpha/runners/run_se_qu_extension.py` | 內容逐位保留 |
| `run_unified_alpha_comparison.py` | `experiments/alpha/runners/run_unified_alpha_comparison.py` | 內容逐位保留 |
| `run_unified_extra_label_candidates.py` | `experiments/alpha/runners/run_unified_extra_label_candidates.py` | 內容逐位保留 |
| `run_validation_calibrated_alpha.py` | `experiments/alpha/runners/run_validation_calibrated_alpha.py` | 內容逐位保留 |
| `se_qu_initialization.py` | `experiments/alpha/methods/se_qu_initialization.py` | 內容逐位保留 |
| `supplement_four_way_bayes.py` | `experiments/alpha/runners/supplement_four_way_bayes.py` | 內容逐位保留 |
| `support_quality_mix_experiment.py` | `experiments/alpha/methods/support_quality_mix_experiment.py` | 內容逐位保留 |
| `test_adaptive_alpha_replay.py` | `tests/test_adaptive_alpha_replay.py` | 內容逐位保留 |
| `test_adaptive_correlation_alpha.py` | `tests/test_adaptive_correlation_alpha.py` | 內容逐位保留 |
| `test_alpha_initialization_selection.py` | `tests/test_alpha_initialization_selection.py` | 內容逐位保留 |
| `test_alpha_initialization_study.py` | `tests/test_alpha_initialization_study.py` | 內容逐位保留 |
| `test_alpha_mixture_exploration.py` | `tests/test_alpha_mixture_exploration.py` | 內容逐位保留 |
| `test_alpha_selection_failure_flow.py` | `tests/test_alpha_selection_failure_flow.py` | 內容逐位保留 |
| `test_alpha_selection_formal_statistics.py` | `tests/test_alpha_selection_formal_statistics.py` | 內容逐位保留 |
| `test_alpha_weight_experiments.py` | `tests/test_alpha_weight_experiments.py` | 內容逐位保留 |
| `test_assess_ensemble_target_transfer.py` | `tests/test_assess_ensemble_target_transfer.py` | 內容逐位保留 |
| `test_bootstrap_quality_research.py` | `tests/test_bootstrap_quality_research.py` | 內容逐位保留 |
| `test_box_constraints.py` | `tests/test_box_constraints.py` | 內容逐位保留 |
| `test_conservative_shrinkage.py` | `tests/test_conservative_shrinkage.py` | 內容逐位保留 |
| `test_failure_analysis.py` | `tests/test_failure_analysis.py` | 內容逐位保留 |
| `test_four_alpha_analysis_edges.py` | `tests/test_four_alpha_analysis_edges.py` | 內容逐位保留 |
| `test_four_alpha_initialization.py` | `tests/test_four_alpha_initialization.py` | 內容逐位保留 |
| `test_four_way_bayes.py` | `tests/test_four_way_bayes.py` | 內容逐位保留 |
| `test_initial_alpha_methods.py` | `tests/test_initial_alpha_methods.py` | 內容逐位保留 |
| `test_next_alpha_initialization.py` | `tests/test_next_alpha_initialization.py` | 內容逐位保留 |
| `test_oob_shrinkage.py` | `tests/test_oob_shrinkage.py` | 內容逐位保留 |
| `test_p3_external_validity.py` | `tests/test_p3_external_validity.py` | 內容逐位保留 |
| `test_peer_support_alpha.py` | `tests/test_peer_support_alpha.py` | 內容逐位保留 |
| `test_peer_support_analysis.py` | `tests/test_peer_support_analysis.py` | 內容逐位保留 |
| `test_peer_support_completion_audit.py` | `tests/test_peer_support_completion_audit.py` | 內容逐位保留 |
| `test_peer_support_costs.py` | `tests/test_peer_support_costs.py` | 內容逐位保留 |
| `test_peer_support_preparation_edges.py` | `tests/test_peer_support_preparation_edges.py` | 內容逐位保留 |
| `test_pure_es_initialization.py` | `tests/test_pure_es_initialization.py` | 內容逐位保留 |
| `test_repeated_alpha_confirmation.py` | `tests/test_repeated_alpha_confirmation.py` | 內容逐位保留 |
| `test_repeated_alpha_evidence_audit.py` | `tests/test_repeated_alpha_evidence_audit.py` | 內容逐位保留 |
| `test_repeated_alpha_qualification_diagnostic.py` | `tests/test_repeated_alpha_qualification_diagnostic.py` | 內容逐位保留 |
| `test_research_analysis.py` | `tests/test_research_analysis.py` | 內容逐位保留 |
| `test_robust_alpha_study.py` | `tests/test_robust_alpha_study.py` | 內容逐位保留 |
| `test_se_qu_initialization.py` | `tests/test_se_qu_initialization.py` | 內容逐位保留 |
| `test_sparse_alpha.py` | `tests/test_sparse_alpha.py` | 內容逐位保留 |
| `test_support_quality_mix.py` | `tests/test_support_quality_mix.py` | 內容逐位保留 |
| `test_targeted_label_alpha.py` | `tests/test_targeted_label_alpha.py` | 內容逐位保留 |
| `test_unified_alpha_methods.py` | `tests/test_unified_alpha_methods.py` | 內容逐位保留 |
| `test_validation_calibrated_alpha.py` | `tests/test_validation_calibrated_alpha.py` | 內容逐位保留 |
| `unified_alpha_methods.py` | `experiments/alpha/methods/unified_alpha_methods.py` | 內容逐位保留 |
| `unified_alpha_pipeline.py` | `experiments/alpha/runners/unified_alpha_pipeline.py` | 內容逐位保留 |
| `validation_calibrated_alpha.py` | `experiments/alpha/methods/validation_calibrated_alpha.py` | 內容逐位保留 |
| `verify_alpha_initialization_study.py` | `experiments/alpha/validation/verify_alpha_initialization_study.py` | 內容逐位保留 |
| `verify_ensemble_target_assessment.py` | `experiments/alpha/validation/verify_ensemble_target_assessment.py` | 內容逐位保留 |
| `verify_ensemble_target_cache.py` | `experiments/alpha/validation/verify_ensemble_target_cache.py` | 內容逐位保留 |
| `verify_ensemble_target_replay.py` | `experiments/alpha/validation/verify_ensemble_target_replay.py` | 內容逐位保留 |
| `verify_p3_external_validity.py` | `experiments/contribution/verify_p3_external_validity.py` | 內容逐位保留 |
| `verify_p5_replay.py` | `experiments/protocols/verify_p5_replay.py` | 內容逐位保留 |
| `verify_research_pilots.py` | `experiments/shared/verify_research_pilots.py` | 內容逐位保留 |
| `verify_targeted_label_alpha.py` | `experiments/alpha/validation/verify_targeted_label_alpha.py` | 內容逐位保留 |
| `verify_validation_calibrated_alpha.py` | `experiments/alpha/validation/verify_validation_calibrated_alpha.py` | 內容逐位保留 |
| `verify_validity_pilots.py` | `experiments/shared/verify_validity_pilots.py` | 內容逐位保留 |
| `smoke_main_flow.py` | `experiments/shared/smoke_main_flow.py` | 新增小型驗證 |

## 貢獻套件索引

以下入口都是 `experiments/contribution/packages/` 下的連結，目標保留在 `contribution_analysis/`。套件內含多支方法、runner、分析與測試；不代表59個推薦方法。

- `experiments/contribution/packages/adaptive_response_confirmation_v1` → `contribution_analysis/adaptive_response_confirmation_v1`
- `experiments/contribution/packages/adaptive_shared_response_v1` → `contribution_analysis/adaptive_shared_response_v1`
- `experiments/contribution/packages/anchor_ca_model_v_cost_v1` → `contribution_analysis/anchor_ca_model_v_cost_v1`
- `experiments/contribution/packages/anchor_pool_end_to_end_v1` → `contribution_analysis/anchor_pool_end_to_end_v1`
- `experiments/contribution/packages/anchor_pool_shadow_v1` → `contribution_analysis/anchor_pool_shadow_v1`
- `experiments/contribution/packages/anchor_step_validation_v1` → `contribution_analysis/anchor_step_validation_v1`
- `experiments/contribution/packages/combined_contribution_layer1_v1` → `contribution_analysis/combined_contribution_layer1_v1`
- `experiments/contribution/packages/combined_contribution_layer2_v1` → `contribution_analysis/combined_contribution_layer2_v1`
- `experiments/contribution/packages/contribution_diagnostic_v1` → `contribution_analysis/contribution_diagnostic_v1`
- `experiments/contribution/packages/contribution_four_way_v1` → `contribution_analysis/contribution_four_way_v1`
- `experiments/contribution/packages/contribution_search_v1` → `contribution_analysis/contribution_search_v1`
- `experiments/contribution/packages/current_task_residual_v1` → `contribution_analysis/current_task_residual_v1`
- `experiments/contribution/packages/custom_direction_v1` → `contribution_analysis/custom_direction_v1`
- `experiments/contribution/packages/custom_realized_gain_v1` → `contribution_analysis/custom_realized_gain_v1`
- `experiments/contribution/packages/distributed_panel_v1` → `contribution_analysis/distributed_panel_v1`
- `experiments/contribution/packages/dual_role_v2` → `contribution_analysis/dual_role_v2`
- `experiments/contribution/packages/full_removal_budget_v1` → `contribution_analysis/full_removal_budget_v1`
- `experiments/contribution/packages/geometric_replay_v1` → `contribution_analysis/geometric_replay_v1`
- `experiments/contribution/packages/j_local_quality_v1` → `contribution_analysis/j_local_quality_v1`
- `experiments/contribution/packages/j_loo_start_stability_v1` → `contribution_analysis/j_loo_start_stability_v1`
- `experiments/contribution/packages/j_peer_v1` → `contribution_analysis/j_peer_v1`
- `experiments/contribution/packages/j_self_other_v1` → `contribution_analysis/j_self_other_v1`
- `experiments/contribution/packages/local_anchor_step_v1` → `contribution_analysis/local_anchor_step_v1`
- `experiments/contribution/packages/local_response_panel_v1` → `contribution_analysis/local_response_panel_v1`
- `experiments/contribution/packages/low_cost_local_effect_v1` → `contribution_analysis/low_cost_local_effect_v1`
- `experiments/contribution/packages/low_cost_loo_v1` → `contribution_analysis/low_cost_loo_v1`
- `experiments/contribution/packages/method5_cost_parent_review_20261030_final04_v2` → `contribution_analysis/method5_cost_parent_review_20261030_final04_v2`
- `experiments/contribution/packages/method5_cost_v1` → `contribution_analysis/method5_cost_v1`
- `experiments/contribution/packages/model_branch_search_v1` → `contribution_analysis/model_branch_search_v1`
- `experiments/contribution/packages/model_cn_extreme_v1` → `contribution_analysis/model_cn_extreme_v1`
- `experiments/contribution/packages/model_cn_three_way_v1` → `contribution_analysis/model_cn_three_way_v1`
- `experiments/contribution/packages/model_consensus_pool_v1` → `contribution_analysis/model_consensus_pool_v1`
- `experiments/contribution/packages/model_decision_validity_v1` → `contribution_analysis/model_decision_validity_v1`
- `experiments/contribution/packages/model_direction_diagnostic_v1` → `contribution_analysis/model_direction_diagnostic_v1`
- `experiments/contribution/packages/model_direction_pool_v1` → `contribution_analysis/model_direction_pool_v1`
- `experiments/contribution/packages/model_peer_pool_v1` → `contribution_analysis/model_peer_pool_v1`
- `experiments/contribution/packages/model_pool_combinations_v1` → `contribution_analysis/model_pool_combinations_v1`
- `experiments/contribution/packages/model_pool_observed_vd_cost_v1` → `contribution_analysis/model_pool_observed_vd_cost_v1`
- `experiments/contribution/packages/model_pool_v_cm_validation_v1` → `contribution_analysis/model_pool_v_cm_validation_v1`
- `experiments/contribution/packages/model_qv_v1` → `contribution_analysis/model_qv_v1`
- `experiments/contribution/packages/model_raw_c_validation_v1` → `contribution_analysis/model_raw_c_validation_v1`
- `experiments/contribution/packages/model_scale_invariance_v1` → `contribution_analysis/model_scale_invariance_v1`
- `experiments/contribution/packages/model_step_decision_v1` → `contribution_analysis/model_step_decision_v1`
- `experiments/contribution/packages/path_intervention_v1` → `contribution_analysis/path_intervention_v1`
- `experiments/contribution/packages/peer_exclusion_cost_v1` → `contribution_analysis/peer_exclusion_cost_v1`
- `experiments/contribution/packages/process_gradient_v1` → `contribution_analysis/process_gradient_v1`
- `experiments/contribution/packages/rank_consensus_v1` → `contribution_analysis/rank_consensus_v1`
- `experiments/contribution/packages/reliability_risk_v1` → `contribution_analysis/reliability_risk_v1`
- `experiments/contribution/packages/robust_endpoint_v1` → `contribution_analysis/robust_endpoint_v1`
- `experiments/contribution/packages/senior_remaining_v1` → `contribution_analysis/senior_remaining_v1`
- `experiments/contribution/packages/senior_three_packages_v1` → `contribution_analysis/senior_three_packages_v1`
- `experiments/contribution/packages/shared_bank_v1` → `contribution_analysis/shared_bank_v1`
- `experiments/contribution/packages/shared_response_independent_v1` → `contribution_analysis/shared_response_independent_v1`
- `experiments/contribution/packages/shared_response_v1` → `contribution_analysis/shared_response_v1`
- `experiments/contribution/packages/three_pool_t0_t1_v1` → `contribution_analysis/three_pool_t0_t1_v1`
- `experiments/contribution/packages/three_pool_t2_anchor_v1` → `contribution_analysis/three_pool_t2_anchor_v1`
- `experiments/contribution/packages/three_pool_t3_model_v1` → `contribution_analysis/three_pool_t3_model_v1`
- `experiments/contribution/packages/trajectory_calibration_v1` → `contribution_analysis/trajectory_calibration_v1`
- `experiments/contribution/packages/validation_judge_v1` → `contribution_analysis/validation_judge_v1`

## 保留與相容範圍

- 原先未提交修改的 `host_server.py`、`dataset.py`、`run_experiments.py`、`test_alpha_weight_experiments.py` 原內容逐位保留；`alpha_weight_experiments.py` 在原內容上僅調整來源根目錄定位。沒有還原成Git版本。
- 原稿整份移入 `paper/proposal/`，原名保留連結，內容hash逐檔核對。
- 歷史結果、source_snapshot、規約、模型checkpoint及圖表均不搬動。貢獻套件仍可能將程式和證據混放，這次透過索引整理入口。
- 不全面改寫import、硬編碼研究路徑或既有runner；不刪除舊方法。獨立跨平台套件化不在本次已完成範圍。
- `pytest.ini` 明確只預設收集移入 `tests/` 的通用測試；套件內版本測試另行指定。
