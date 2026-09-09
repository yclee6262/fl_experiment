import numpy as np
from scipy.optimize import minimize

class HostServer:
    def __init__(self, target_T, n_features, total_budget=10.0, test_seed=42, n_test=5):
        self.target_T = target_T
        self.n_features = n_features
        self.total_budget = float(total_budget)
        self.trusted_agents = []
        self.alphas = []
        self.I_list = []
        self.phase1_report = []
        self.phase4_report = []
        self.n_test = int(n_test)

        np.random.seed(test_seed)
        self.test_X = np.random.uniform(-1, 1, (self.n_test, self.n_features)) 
        
        # ⭐ 3. 同步生成對應的標準答案 test_y
        self.test_y = np.sum(self.test_X, axis=1)
        if self.n_features > 1:
            self.test_y += np.sum(self.test_X[:, :-1] * self.test_X[:, 1:], axis=1)

    def _residual_diversity(self, residual, selected_reports, eps=1e-8):
        """Return D(i, C): residual cosine dissimilarity to the selected coalition."""
        if not selected_reports:
            return 1.0

        residual_norm = np.linalg.norm(residual)
        max_similarity = 0.0
        for report in selected_reports:
            selected_residual = report["residual"]
            denom = residual_norm * np.linalg.norm(selected_residual) + eps
            similarity = abs(float(np.dot(residual, selected_residual) / denom))
            max_similarity = max(max_similarity, similarity)
        return max(0.0, 1.0 - max_similarity)

    def phase1_filter_agents(
        self,
        all_agents,
        mse_threshold=0.1,
        budget_fraction=0.8,
        diversity_eta=0.5,
        min_selection_score=0.0,
        k_api=None,
        k_red=None,
        enable_inverse_check=True,
        inverse_target=0.0,
        inverse_loss_threshold=0.1,
        inverse_steps=500,
        feasible_lower=-1.0,
        feasible_upper=1.0,
    ):
        """Stage 0 qualification plus Stage 1 bid-aware coalition selection."""
        print("\n--- Phase 1: 盲測信任評估與投標式節點選擇 ---")
        self.trusted_agents = []
        self.alphas = []
        self.I_list = []
        self.phase1_report = []
        
        qualified_reports = []
        for agent in all_agents:
            pred_y = agent.api_predict(self.test_X)
            residual = pred_y - self.test_y
            mse = np.mean(residual**2)
            bid = agent.get_minimum_bid() if hasattr(agent, "get_minimum_bid") else 1.0
            
            if mse <= mse_threshold:
                raw_score = 1.0 / (mse + 1e-5)
                report = {
                    "agent": agent,
                    "agent_id": agent.agent_id,
                    "mse": float(mse),
                    "bid": float(bid),
                    "raw_score": float(raw_score),
                    "residual": residual,
                    "alpha_pre": 0.0,
                    "cost_performance": 0.0,
                    "diversity": 0.0,
                    "selection_score": 0.0,
                    "selected": False,
                    "inverse_loss": "",
                    "inverse_feasible": "",
                }
                qualified_reports.append(report)
                print(f"Agent {agent.agent_id} 通過盲測 (MSE: {mse:.4f}, bid: {bid:.3f})")
            else:
                self.phase1_report.append({
                    "agent_id": agent.agent_id,
                    "mse": float(mse),
                    "bid": float(bid),
                    "inverse_loss": "",
                    "inverse_feasible": "",
                    "selected": False,
                    "reason": "failed_mse_threshold",
                })
                print(f"Agent {agent.agent_id} 被剔除 (MSE: {mse:.4f}, bid: {bid:.3f})")

        if not qualified_reports:
            raise ValueError("No agents passed Stage 1 blind-test filtering.")

        blind_test_passed_count = len(qualified_reports)
        if enable_inverse_check:
            inverse_qualified_reports = []
            for report in qualified_reports:
                agent = report["agent"]
                candidate = agent.infer_parameters_D(inverse_target, steps=inverse_steps)
                candidate = np.asarray(candidate, dtype=float)
                feasible = bool(
                    np.all(candidate >= feasible_lower)
                    and np.all(candidate <= feasible_upper)
                )
                pred = agent.api_predict(candidate)[0]
                inverse_loss = float(abs(pred - inverse_target))

                report["inverse_loss"] = inverse_loss
                report["inverse_feasible"] = feasible

                if feasible and inverse_loss <= inverse_loss_threshold:
                    inverse_qualified_reports.append(report)
                    print(
                        f"Agent {agent.agent_id} 通過反推能力檢查 "
                        f"(T_qual={inverse_target:.4f}, inv_loss={inverse_loss:.4f})"
                    )
                else:
                    self.phase1_report.append({
                        "agent_id": report["agent_id"],
                        "mse": report["mse"],
                        "bid": report["bid"],
                        "inverse_loss": inverse_loss,
                        "inverse_feasible": feasible,
                        "selected": False,
                        "reason": "failed_inverse_feasibility",
                    })
                    print(
                        f"Agent {agent.agent_id} 被剔除：反推能力檢查未通過 "
                        f"(feasible={feasible}, inv_loss={inverse_loss:.4f})"
                    )

            qualified_reports = inverse_qualified_reports
            if not qualified_reports:
                raise ValueError("No agents passed Stage 0 inverse-feasibility checking.")

        total_raw_score = sum(report["raw_score"] for report in qualified_reports)
        for report in qualified_reports:
            report["alpha_pre"] = report["raw_score"] / total_raw_score
            report["cost_performance"] = report["alpha_pre"] / max(report["bid"], 1e-8)

        budget_select = budget_fraction * self.total_budget
        k_red = min(3, int(np.ceil(0.3 * self.n_features))) if k_red is None else k_red
        k_api = len(qualified_reports) if k_api is None else k_api
        k_max = min(len(qualified_reports), k_api, self.n_features + k_red)
        print(
            f"  選擇設定：B_select={budget_select:.3f}, "
            f"K_API={k_api}, K_red={k_red}, K_max={k_max}"
        )

        selected_reports = []
        spent_budget = 0.0
        while len(selected_reports) < k_max:
            best_report = None
            best_score = -np.inf
            best_diversity = 0.0

            for report in qualified_reports:
                if report["selected"]:
                    continue
                if spent_budget + report["bid"] > budget_select:
                    continue

                diversity = self._residual_diversity(report["residual"], selected_reports)
                selection_score = report["cost_performance"] * (1.0 + diversity_eta * diversity)
                if selection_score > best_score:
                    best_score = selection_score
                    best_report = report
                    best_diversity = diversity

            if best_report is None or best_score < min_selection_score:
                break

            best_report["selected"] = True
            best_report["diversity"] = float(best_diversity)
            best_report["selection_score"] = float(best_score)
            selected_reports.append(best_report)
            spent_budget += best_report["bid"]
            print(
                f"  選入 Agent {best_report['agent_id']} | "
                f"R={best_report['cost_performance']:.4f}, "
                f"D={best_report['diversity']:.4f}, "
                f"G={best_report['selection_score']:.4f}, "
                f"累計 bid={spent_budget:.3f}/{budget_select:.3f}"
            )

        if not selected_reports:
            raise ValueError("No agents were selected under the Stage 1 budget/quality constraints.")

        selected_score_sum = sum(report["raw_score"] for report in selected_reports)
        for report in selected_reports:
            self.trusted_agents.append(report["agent"])
            self.alphas.append(report["raw_score"] / selected_score_sum)

        for report in qualified_reports:
            self.phase1_report.append({
                "agent_id": report["agent_id"],
                "mse": report["mse"],
                "bid": report["bid"],
                "inverse_loss": report["inverse_loss"],
                "inverse_feasible": report["inverse_feasible"],
                "alpha_pre": report["alpha_pre"],
                "cost_performance": report["cost_performance"],
                "diversity": report["diversity"],
                "selection_score": report["selection_score"],
                "selected": report["selected"],
                "reason": "selected" if report["selected"] else "not_selected_by_coalition_rule",
            })

        print(
            f"Stage 1 完成：從 {len(all_agents)} 個 Agent 中，"
            f"{blind_test_passed_count} 個通過盲測，"
            f"{len(qualified_reports)} 個通過完整 Stage 0，"
            f"最終選入 K={len(self.trusted_agents)} 個。"
        )

    def _agent_bid(self, agent):
        return agent.get_minimum_bid() if hasattr(agent, "get_minimum_bid") else 1.0

    def _weighted_consensus_loss(self, S_array, agents=None, alphas=None):
        """Evaluate the Stage 3/4 consensus loss at a candidate decision vector."""
        return self._consensus_loss(
            S_array,
            agents=agents,
            mode="fixed" if alphas is not None else "optimization",
            weights=alphas,
        )

    def _consensus_loss(
        self,
        S_array,
        agents=None,
        mode="optimization",
        weights=None,
        trim_fraction=0.2,
    ):
        """Evaluate a consensus loss without conflating optimization and evaluation weights.

        ``optimization`` preserves the existing behavior and uses ``self.alphas``.
        ``fixed`` uses explicitly supplied weights. ``uniform``, ``median``, and
        ``trimmed`` are alpha-independent evaluators for contribution experiments.
        """
        agents = self.trusted_agents if agents is None else agents
        if not agents:
            raise ValueError("Consensus loss requires at least one agent.")

        losses = np.asarray(
            [abs(agent.api_predict(S_array)[0] - self.target_T) for agent in agents],
            dtype=float,
        )

        if mode in {"optimization", "fixed"}:
            selected_weights = self.alphas if mode == "optimization" else weights
            if selected_weights is None or len(selected_weights) != len(agents):
                raise ValueError("Consensus weights must match the number of agents.")
            selected_weights = np.asarray(selected_weights, dtype=float)
            total = float(np.sum(selected_weights))
            if total <= 0 or not np.isfinite(total):
                raise ValueError("Consensus weights must have a finite positive sum.")
            return float(np.dot(selected_weights / total, losses))

        if mode == "uniform":
            return float(np.mean(losses))
        if mode == "median":
            return float(np.median(losses))
        if mode == "trimmed":
            if not 0.0 <= trim_fraction < 0.5:
                raise ValueError("trim_fraction must be in [0, 0.5).")
            trim_count = int(np.floor(len(losses) * trim_fraction))
            ordered = np.sort(losses)
            if trim_count:
                ordered = ordered[trim_count:-trim_count]
            return float(np.mean(ordered))

        raise ValueError(
            "mode must be one of: optimization, fixed, uniform, median, trimmed."
        )

    def phase2_collect_proposals(self):
        """請合格 Agent 利用虛設層反推初步參數"""
        print("\n--- Phase 2: 收集初步提議參數 (I_i) ---")
        for agent in self.trusted_agents:
            I_i = agent.infer_parameters_D(self.target_T)
            self.I_list.append(I_i)
            print(f"Agent {agent.agent_id} 提議參數: {I_i}")

    def _true_target_error(self, S_array):
        """Synthetic ground-truth metric used only for experiment histories."""
        y_value = np.sum(S_array)
        if len(S_array) > 1:
            y_value += np.sum(S_array[:-1] * S_array[1:])
        return float(abs(y_value - self.target_T))

    def _solve_subspace_bfgs(self, candidates, objective, history_metric=None):
        candidates = np.asarray(candidates, dtype=float)
        history = []

        def beta_objective(betas):
            return objective(np.dot(betas, candidates))

        def callback(betas):
            solution = np.dot(betas, candidates)
            history.append(float((history_metric or objective)(solution)))

        initial_betas = np.ones(len(candidates), dtype=float) / len(candidates)
        result = minimize(beta_objective, initial_betas, method="BFGS", callback=callback)
        solution = np.dot(result.x, candidates)
        loss = float(objective(solution))
        if not history:
            history.append(float((history_metric or objective)(solution)))
        return {
            "solution": solution,
            "loss": loss,
            "history": history,
            "states": ["bfgs"] * len(history),
            "success": bool(result.success),
            "message": str(result.message),
        }

    def _solve_subspace_custom(
        self,
        candidates,
        objective,
        num_iterations=50,
        use_annealing=True,
        allow_tangent=True,
        direction_weights=None,
        history_metric=None,
        verbose=False,
    ):
        """Run the secant/tangent engine against an arbitrary subspace objective."""
        candidates = np.asarray(candidates, dtype=float)
        n_candidates = len(candidates)
        S_current = np.mean(candidates, axis=0)
        if direction_weights is None:
            direction_weights = np.ones(n_candidates, dtype=float) / n_candidates
        else:
            direction_weights = np.asarray(direction_weights, dtype=float)
            direction_weights = direction_weights / np.sum(direction_weights)

        metric = history_metric or objective
        history = [float(metric(S_current))]
        states = ["Start"]
        anchor_losses = [float(objective(candidate)) for candidate in candidates]
        best_loss = float(objective(S_current))
        global_best_S = S_current.copy()
        global_best_loss = best_loss
        eta = 0.1
        current_method = "secant"
        delta = 0.0001

        for iteration in range(num_iterations):
            gradient = np.zeros_like(S_current)
            for index, anchor in enumerate(candidates):
                direction = anchor - S_current
                distance = np.linalg.norm(direction)
                if distance < 1e-8:
                    continue
                unit_direction = direction / distance
                if current_method == "secant":
                    derivative = (anchor_losses[index] - best_loss) / distance
                else:
                    perturbed_loss = objective(S_current + delta * unit_direction)
                    derivative = (perturbed_loss - best_loss) / delta
                gradient += direction_weights[index] * derivative * unit_direction

            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm <= 1e-8:
                break
            gradient = gradient / gradient_norm
            current_eta = eta
            success = False
            state = current_method

            if use_annealing:
                for attempt in range(10):
                    candidate_S = S_current - current_eta * gradient
                    candidate_loss = float(objective(candidate_S))
                    if candidate_loss < best_loss:
                        if attempt > 0:
                            state = "Annealing Triggered"
                        S_current = candidate_S
                        best_loss = candidate_loss
                        if best_loss < global_best_loss:
                            global_best_loss = best_loss
                            global_best_S = S_current.copy()
                        eta = min(0.5, current_eta * 1.5)
                        success = True
                        break
                    current_eta /= 2.0
            else:
                candidate_S = S_current - current_eta * gradient
                candidate_loss = float(objective(candidate_S))
                if candidate_loss < best_loss:
                    S_current = candidate_S
                    best_loss = candidate_loss
                    if best_loss < global_best_loss:
                        global_best_loss = best_loss
                        global_best_S = S_current.copy()
                    success = True

            if not success:
                if current_method == "secant" and allow_tangent:
                    current_method = "dynamic"
                    eta = 0.5
                else:
                    break

            history.append(float(metric(S_current)))
            states.append(state)
            if verbose:
                print(
                    f"  [Iter {iteration + 1} - {current_method}] "
                    f"objective={best_loss:.6f}"
                )

        return {
            "solution": global_best_S,
            "loss": float(objective(global_best_S)),
            "history": history,
            "states": states,
            "success": True,
            "message": "custom search completed",
        }

    def optimize_candidate_subspace(
        self,
        candidates=None,
        mode="optimization",
        weights=None,
        trim_fraction=0.2,
        strategy="best_of",
        custom_iterations=30,
        use_annealing=True,
        allow_tangent=True,
        history_metric=None,
        verbose=False,
    ):
        """Optimize one candidate subspace and select engines by one objective."""
        candidates = np.asarray(self.I_list if candidates is None else candidates, dtype=float)
        if candidates.ndim != 2 or not len(candidates):
            raise ValueError("Candidate subspace must contain at least one vector.")

        def objective(S_array):
            return self._consensus_loss(
                S_array,
                mode=mode,
                weights=weights,
                trim_fraction=trim_fraction,
            )

        results = {}
        if strategy in {"bfgs", "best_of"}:
            results["bfgs"] = self._solve_subspace_bfgs(
                candidates, objective, history_metric=history_metric
            )
        if strategy in {"custom", "best_of"}:
            direction_weights = (
                self.alphas
                if mode == "optimization" and len(candidates) == len(self.alphas)
                else None
            )
            results["custom"] = self._solve_subspace_custom(
                candidates,
                objective,
                num_iterations=custom_iterations,
                use_annealing=use_annealing,
                allow_tangent=allow_tangent,
                direction_weights=direction_weights,
                history_metric=history_metric,
                verbose=verbose,
            )
        if not results:
            raise ValueError("strategy must be one of: custom, bfgs, best_of.")

        finite_results = {
            name: result for name, result in results.items()
            if np.isfinite(result["loss"])
        }
        if not finite_results:
            raise ValueError("Every subspace optimizer returned a non-finite loss.")
        chosen_engine, chosen = min(
            finite_results.items(), key=lambda item: item[1]["loss"]
        )
        output = dict(chosen)
        output["chosen_engine"] = chosen_engine
        output["engine_losses"] = {
            name: float(result["loss"]) for name, result in results.items()
        }
        self.last_subspace_optimization = output
        return output

    def phase3_global_optimization(self):
        """使用 SciPy BFGS 計算最佳混合比例 (Betas)"""
        print("\n--- Phase 3: 全域最佳化 (BFGS 演算法) ---")
        result = self.optimize_candidate_subspace(
            strategy="bfgs", history_metric=self._true_target_error
        )
        return result["solution"], result["history"]

    def phase3_best_of_optimization(self, custom_iterations=30):
        """Run both Stage 3 engines and keep the lower-L_opt solution."""
        print("\n--- Phase 3: 全域最佳化 (best_of: BFGS vs custom) ---")
        result = self.optimize_candidate_subspace(
            strategy="best_of",
            custom_iterations=custom_iterations,
            history_metric=self._true_target_error,
        )
        print(
            f"best_of 選擇 {result['chosen_engine']}，"
            f"L_opt={result['loss']:.6f}，engine losses={result['engine_losses']}"
        )
        return result["solution"], result["history"], result["states"]

    def _compute_exclusion_reports(
        self,
        final_S,
        verbose=True,
        evaluation_mode="trimmed",
        evaluation_weights=None,
        trim_fraction=0.2,
        optimizer="best_of",
        custom_iterations=30,
        return_summary=False,
    ):
        """Score operational full/LOO Stage 3 solutions with a fixed L_eval."""
        I_matrix = np.array(self.I_list)
        n_agents = len(self.trusted_agents)
        base_loss = self._consensus_loss(
            final_S,
            mode=evaluation_mode,
            weights=evaluation_weights,
            trim_fraction=trim_fraction,
        )
        exclusion_reports = []

        for excluded_idx, excluded_agent in enumerate(self.trusted_agents):
            remaining_indices = [idx for idx in range(n_agents) if idx != excluded_idx]

            if not remaining_indices:
                restricted_loss = float("inf")
                restricted_S = None
                restricted_engine = None
                restricted_engine_losses = {}
                restricted_optimization_loss = float("inf")
            else:
                restricted_I = I_matrix[remaining_indices]
                restricted_result = self.optimize_candidate_subspace(
                    restricted_I,
                    # Re-run the operational Stage 3 objective with the current
                    # alpha. L_eval is a fixed judge, not the search objective.
                    mode="optimization",
                    strategy=optimizer,
                    custom_iterations=custom_iterations,
                )
                restricted_S = restricted_result["solution"]
                restricted_optimization_loss = restricted_result["loss"]
                restricted_loss = self._consensus_loss(
                    restricted_S,
                    mode=evaluation_mode,
                    weights=evaluation_weights,
                    trim_fraction=trim_fraction,
                )
                restricted_engine = restricted_result["chosen_engine"]
                restricted_engine_losses = restricted_result["engine_losses"]

            marginal_contribution = restricted_loss - base_loss
            positive_contribution = max(float(marginal_contribution), 0.0)
            report = {
                "index": excluded_idx,
                "agent": excluded_agent,
                "agent_id": excluded_agent.agent_id,
                "alpha": float(self.alphas[excluded_idx]),
                "bid": float(self._agent_bid(excluded_agent)),
                "loss_without_agent": float(restricted_loss),
                "restricted_optimization_loss": float(restricted_optimization_loss),
                "marginal_contribution": float(marginal_contribution),
                "positive_contribution": positive_contribution,
                "restricted_solution": restricted_S,
                "restricted_engine": restricted_engine,
                "restricted_engine_losses": restricted_engine_losses,
            }
            exclusion_reports.append(report)

            if verbose:
                print(
                    f"Agent {excluded_agent.agent_id}: "
                    f"L(-i)={restricted_loss:.6f}, "
                    f"C_i={marginal_contribution:.6f}, "
                    f"C_i+={positive_contribution:.6f}"
                )

        summary = {
            "delivered_solution": np.asarray(final_S, dtype=float),
            "delivered_solution_eval_loss": float(base_loss),
            "full_stage3_eval_loss": float(base_loss),
            "full_solution_source": "stage3_l_opt",
            "loo_search_objective": "l_opt_current_alpha",
            "evaluation_mode": evaluation_mode,
            "loo_optimizer": optimizer,
        }
        self.last_exclusion_summary = summary
        if verbose:
            print(
                f"L_eval(Stage 3 delivered solution)={base_loss:.6f}; "
                f"LOO search objective=L_opt(current alpha), optimizer={optimizer}"
            )
        if return_summary:
            return base_loss, exclusion_reports, summary
        return base_loss, exclusion_reports

    def _remove_trusted_agent_at(self, index):
        removed_agent = self.trusted_agents.pop(index)
        removed_proposal = self.I_list.pop(index)
        removed_alpha = self.alphas.pop(index)

        alpha_sum = sum(self.alphas)
        if alpha_sum > 0:
            self.alphas = [alpha / alpha_sum for alpha in self.alphas]

        return {
            "agent_id": removed_agent.agent_id,
            "alpha": float(removed_alpha),
            "proposal": removed_proposal,
        }

    def _rerun_stage3_optimizer(self, optimizer, custom_iterations):
        if optimizer == "bfgs":
            final_S, history = self.phase3_global_optimization()
            return final_S, history, []
        if optimizer == "custom":
            final_S, history, states = self.phase3_custom_secant_optimization(
                num_iterations=custom_iterations
            )
            return final_S, history, states
        if optimizer == "best_of":
            return self.phase3_best_of_optimization(
                custom_iterations=custom_iterations
            )
        raise ValueError("optimizer must be one of: custom, bfgs, best_of.")

    def prune_negative_contributors(
        self,
        final_S,
        epsilon=1e-6,
        max_pruning_rounds=None,
        optimizer="best_of",
        custom_iterations=30,
    ):
        """Iteratively remove the most negative-contribution agent and re-optimize."""
        print("\n--- Negative Contribution Pruning Experiment ---")
        if not self.trusted_agents or not self.I_list:
            raise ValueError("Pruning requires trusted agents and Phase 2 proposals.")

        max_pruning_rounds = (
            max(0, len(self.trusted_agents) - 1)
            if max_pruning_rounds is None else max_pruning_rounds
        )

        pruning_log = []
        final_history = []
        final_states = []
        current_S = final_S

        for pruning_round in range(max_pruning_rounds + 1):
            coalition_ids = [agent.agent_id for agent in self.trusted_agents]
            print(f"\n[Pruning Round {pruning_round}] coalition={coalition_ids}")
            base_loss, exclusion_reports = self._compute_exclusion_reports(current_S)
            min_report = min(exclusion_reports, key=lambda row: row["marginal_contribution"])

            round_log = {
                "round": pruning_round,
                "coalition_ids": coalition_ids,
                "base_loss": base_loss,
                "reports": [
                    {key: value for key, value in report.items() if key not in {"agent", "restricted_solution"}}
                    for report in exclusion_reports
                ],
                "removed_agent_id": None,
                "status": "stable",
            }

            if min_report["marginal_contribution"] >= -epsilon:
                print(
                    f"停止 pruning：最小 C_i={min_report['marginal_contribution']:.6f} "
                    f">= -epsilon ({-epsilon:.6f})"
                )
                pruning_log.append(round_log)
                break

            if len(self.trusted_agents) <= 1:
                print("停止 pruning：coalition 只剩 1 個 agent。")
                round_log["status"] = "stopped_single_agent"
                pruning_log.append(round_log)
                break

            removed = self._remove_trusted_agent_at(min_report["index"])
            round_log["removed_agent_id"] = removed["agent_id"]
            round_log["status"] = "removed_negative_contributor"
            pruning_log.append(round_log)
            print(
                f"移除 Agent {removed['agent_id']} "
                f"(C_i={min_report['marginal_contribution']:.6f})，重新執行 Stage 3。"
            )

            current_S, final_history, final_states = self._rerun_stage3_optimizer(
                optimizer=optimizer,
                custom_iterations=custom_iterations,
            )

        return {
            "final_solution": current_S,
            "final_history": final_history,
            "final_states": final_states,
            "final_coalition_ids": [agent.agent_id for agent in self.trusted_agents],
            "pruning_log": pruning_log,
        }

    def phase4_profit_sharing(
        self,
        final_S,
        total_budget=None,
        omega_trust=0.4,
        omega_contribution=0.6,
        min_positive_contribution=1e-8,
    ):
        """Stage 4: Subspace Exclusion Method and budget-feasible profit sharing."""
        print("\n--- Phase 4: 子空間排除法貢獻度分潤 ---")
        if not self.trusted_agents or not self.I_list:
            raise ValueError("Phase 4 requires trusted agents and Phase 2 proposals.")

        total_budget = self.total_budget if total_budget is None else float(total_budget)
        base_loss, exclusion_reports = self._compute_exclusion_reports(final_S)

        active_reports = [
            report for report in exclusion_reports
            if report["positive_contribution"] > min_positive_contribution
        ]

        if not active_reports:
            self.phase4_report = {
                "base_loss": base_loss,
                "total_budget": total_budget,
                "active_agent_ids": [],
                "payments": [],
                "status": "rejected_no_positive_contributors",
            }
            print("沒有正邊際貢獻者，分潤流程拒絕此 coalition。")
            return self.phase4_report

        bid_sum = sum(report["bid"] for report in active_reports)
        if bid_sum > total_budget:
            self.phase4_report = {
                "base_loss": base_loss,
                "total_budget": total_budget,
                "active_agent_ids": [report["agent_id"] for report in active_reports],
                "minimum_bid_sum": bid_sum,
                "payments": [],
                "status": "infeasible_minimum_bids_exceed_budget",
            }
            raise ValueError(
                f"Stage 4 budget infeasible: positive contributors require bids "
                f"{bid_sum:.4f}, exceeding total budget {total_budget:.4f}."
            )

        weight_sum = omega_trust + omega_contribution
        omega_trust = omega_trust / weight_sum
        omega_contribution = omega_contribution / weight_sum

        alpha_sum = sum(report["alpha"] for report in active_reports)
        contribution_sum = sum(report["positive_contribution"] for report in active_reports)
        surplus = total_budget - bid_sum

        payment_reports = []
        for report in active_reports:
            alpha_share = report["alpha"] / alpha_sum if alpha_sum > 0 else 0.0
            if len(active_reports) == 1:
                contribution_share = 1.0
            elif np.isfinite(contribution_sum) and contribution_sum > 0:
                contribution_share = report["positive_contribution"] / contribution_sum
            else:
                contribution_share = 0.0
            profit_share = omega_trust * alpha_share + omega_contribution * contribution_share
            payment = report["bid"] + profit_share * surplus

            payment_report = {
                "agent_id": report["agent_id"],
                "alpha": report["alpha"],
                "bid": report["bid"],
                "marginal_contribution": report["marginal_contribution"],
                "positive_contribution": report["positive_contribution"],
                "alpha_share": alpha_share,
                "contribution_share": contribution_share,
                "profit_share": profit_share,
                "payment": payment,
            }
            payment_reports.append(payment_report)
            print(
                f"Payment Agent {report['agent_id']}: "
                f"bid={report['bid']:.3f}, "
                f"alpha_share={alpha_share:.3f}, "
                f"contrib_share={contribution_share:.3f}, "
                f"payment={payment:.3f}"
            )

        paid_total = sum(report["payment"] for report in payment_reports)
        self.phase4_report = {
            "base_loss": base_loss,
            "total_budget": total_budget,
            "minimum_bid_sum": bid_sum,
            "surplus": surplus,
            "paid_total": paid_total,
            "active_agent_ids": [report["agent_id"] for report in active_reports],
            "exclusion_reports": [
                {key: value for key, value in report.items() if key not in {"agent", "restricted_solution"}}
                for report in exclusion_reports
            ],
            "payments": payment_reports,
            "status": "ok",
        }
        print(f"Phase 4 完成：總支付 {paid_total:.3f}/{total_budget:.3f}")
        return self.phase4_report
    
    def phase3_custom_secant_optimization(self, num_iterations=50, use_annealing=True, allow_tangent=True):
        """Phase 3 (Alternative): 使用原創的割線/切線法進行子空間尋路 (支援消融實驗)"""
        print("\n--- Phase 3: 全域最佳化 (啟動割線/切線退火引擎) ---")
        result = self.optimize_candidate_subspace(
            strategy="custom",
            custom_iterations=num_iterations,
            use_annealing=use_annealing,
            allow_tangent=allow_tangent,
            history_metric=self._true_target_error,
            verbose=True,
        )
        print(f"✅ 法二法三引擎尋路完成！最終決策變數 S = {result['solution']}")
        return result["solution"], result["history"], result["states"]
