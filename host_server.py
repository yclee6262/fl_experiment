import numpy as np
from scipy.optimize import minimize

class HostServer:
    def __init__(self, target_T, n_features, total_budget=10.0):
        self.target_T = target_T
        self.n_features = n_features
        self.total_budget = float(total_budget)
        self.trusted_agents = []
        self.alphas = []
        self.I_list = []
        self.phase1_report = []
        self.phase4_report = []

        np.random.seed(42)
        self.test_X = np.random.uniform(-1, 1, (5, self.n_features)) 
        
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
    ):
        """Stage 1: blind-test, bid-aware, diversity-aware coalition selection."""
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
                }
                qualified_reports.append(report)
                print(f"Agent {agent.agent_id} 通過盲測 (MSE: {mse:.4f}, bid: {bid:.3f})")
            else:
                self.phase1_report.append({
                    "agent_id": agent.agent_id,
                    "mse": float(mse),
                    "bid": float(bid),
                    "selected": False,
                    "reason": "failed_mse_threshold",
                })
                print(f"Agent {agent.agent_id} 被剔除 (MSE: {mse:.4f}, bid: {bid:.3f})")

        if not qualified_reports:
            raise ValueError("No agents passed Stage 1 blind-test filtering.")

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
                "alpha_pre": report["alpha_pre"],
                "cost_performance": report["cost_performance"],
                "diversity": report["diversity"],
                "selection_score": report["selection_score"],
                "selected": report["selected"],
                "reason": "selected" if report["selected"] else "not_selected_by_coalition_rule",
            })

        print(
            f"Stage 1 完成：從 {len(all_agents)} 個 Agent 中，"
            f"{len(qualified_reports)} 個通過盲測，最終選入 K={len(self.trusted_agents)} 個。"
        )

    def _agent_bid(self, agent):
        return agent.get_minimum_bid() if hasattr(agent, "get_minimum_bid") else 1.0

    def _weighted_consensus_loss(self, S_array, agents=None, alphas=None):
        """Evaluate the Stage 3/4 consensus loss at a candidate decision vector."""
        agents = self.trusted_agents if agents is None else agents
        alphas = self.alphas if alphas is None else alphas
        total_loss = 0.0
        for agent, alpha in zip(agents, alphas):
            pred = agent.api_predict(S_array)[0]
            total_loss += alpha * abs(pred - self.target_T)
        return float(total_loss)

    def phase2_collect_proposals(self):
        """請合格 Agent 利用虛設層反推初步參數"""
        print("\n--- Phase 2: 收集初步提議參數 (I_i) ---")
        for agent in self.trusted_agents:
            I_i = agent.infer_parameters_D(self.target_T)
            self.I_list.append(I_i)
            print(f"Agent {agent.agent_id} 提議參數: {I_i}")

    def phase3_global_optimization(self):
        """使用 SciPy BFGS 計算最佳混合比例 (Betas)"""
        print("\n--- Phase 3: 全域最佳化 (BFGS 演算法) ---")
        I_matrix = np.array(self.I_list)
        error_history = []
        
        def total_loss_function(betas):
            S_current = np.dot(betas, I_matrix)
            total_loss = 0.0

            # 呼叫每個 Agent 的 API 算預測值
            for i, agent in enumerate(self.trusted_agents):
                pred_i = agent.api_predict(S_current)[0]
                # 加權誤差: alpha * |f(S) - T|
                total_loss += self.alphas[i] * abs(pred_i - self.target_T)
            return total_loss

        def callback(betas):
            S_current = np.dot(betas, I_matrix)
            # 代入真實公式 (這裡假設在 Host 裡也能算真實 y 來當作評估指標)
            y_val = np.sum(S_current)
            if len(S_current) > 1: 
                y_val += np.sum(S_current[:-1] * S_current[1:])
            error_history.append(abs(y_val - self.target_T))
        

        # 初始猜測：平均分配
        initial_betas = np.ones(len(self.trusted_agents)) / len(self.trusted_agents)
        
        result = minimize(total_loss_function, initial_betas, method='BFGS', callback=callback)
        best_betas = result.x
        final_S = np.dot(best_betas, I_matrix)
        
        return final_S, error_history

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
        I_matrix = np.array(self.I_list)
        n_agents = len(self.trusted_agents)
        base_loss = self._weighted_consensus_loss(final_S)

        exclusion_reports = []
        for excluded_idx, excluded_agent in enumerate(self.trusted_agents):
            remaining_indices = [idx for idx in range(n_agents) if idx != excluded_idx]

            if not remaining_indices:
                restricted_loss = float("inf")
                restricted_S = None
            else:
                restricted_I = I_matrix[remaining_indices]

                def restricted_loss_function(betas):
                    S_candidate = np.dot(betas, restricted_I)
                    # Keep evaluation on the full selected coalition so every C_i
                    # is compared against the same global consensus objective.
                    return self._weighted_consensus_loss(S_candidate)

                initial_betas = np.ones(len(remaining_indices)) / len(remaining_indices)
                result = minimize(restricted_loss_function, initial_betas, method="BFGS")
                restricted_S = np.dot(result.x, restricted_I)
                restricted_loss = self._weighted_consensus_loss(restricted_S)

            marginal_contribution = restricted_loss - base_loss
            positive_contribution = max(float(marginal_contribution), 0.0)
            exclusion_reports.append({
                "agent": excluded_agent,
                "agent_id": excluded_agent.agent_id,
                "alpha": float(self.alphas[excluded_idx]),
                "bid": float(self._agent_bid(excluded_agent)),
                "loss_without_agent": float(restricted_loss),
                "marginal_contribution": float(marginal_contribution),
                "positive_contribution": positive_contribution,
                "restricted_solution": restricted_S,
            })
            print(
                f"Agent {excluded_agent.agent_id}: "
                f"L(-i)={restricted_loss:.6f}, "
                f"C_i={marginal_contribution:.6f}, "
                f"C_i+={positive_contribution:.6f}"
            )

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
            contribution_share = (
                report["positive_contribution"] / contribution_sum
                if contribution_sum > 0 else 0.0
            )
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
        
        # 1. 將 Agent 提議的參數轉為矩陣，並計算起點 (平均值 M^0)
        I_matrix = np.array(self.I_list)
        n_agents = len(self.trusted_agents)
        S_current = np.mean(I_matrix, axis=0) # 從平均點出發
        
        error_history = []
        states_history = [] # 新增：紀錄每次迭代的演算法狀態
        
        # 內部評估函數：呼叫 API 並計算總誤差 (Loss)
        def evaluate_S(S_array):
            total_loss = 0.0
            for i, agent in enumerate(self.trusted_agents):
                pred_i = agent.api_predict(S_array)[0]
                total_loss += self.alphas[i] * abs(pred_i - self.target_T)
            return total_loss

        # 2. 預先計算各個錨點 (I_i) 的 Loss，給割線法當作斜率參考
        loss_anchors = [evaluate_S(I_i) for I_i in self.I_list]
        
        best_loss = evaluate_S(S_current)
        eta = 0.1
        current_method = "secant"
        delta = 0.0001 # 切線法的微小偏移量

        # 紀錄歷史最佳解
        global_best_S = S_current.copy()
        global_best_loss = best_loss
        
        # --- 紀錄起點 (Iter 0) 的真實誤差與狀態 ---
        y_val_start = np.sum(S_current)
        if len(S_current) > 1: 
            y_val_start += np.sum(S_current[:-1] * S_current[1:])
        error_history.append(abs(y_val_start - self.target_T))
        states_history.append("Start")
        
        # 3. 開始手動尋路迴圈
        for k in range(num_iterations):
            grad_S = np.zeros_like(S_current)
            
            # --- 步驟 A：計算合成梯度 ---
            for i in range(n_agents):
                direction = self.I_list[i] - S_current
                dist = np.linalg.norm(direction)
                if dist < 1e-8: continue
                unit_dir = direction / dist
                
                if current_method == "secant":
                    # 割線法：用端點 Loss 與目前 Loss 的高低差當作斜率
                    deriv = (loss_anchors[i] - best_loss) / dist
                else:
                    # 切線法 (動態方向)：往前踩一小步 delta 測試真實斜率
                    S_perturb = S_current + delta * unit_dir
                    loss_p = evaluate_S(S_perturb)
                    deriv = (loss_p - best_loss) / delta
                    
                # 累加各個方向的梯度 (乘上信任權重 alpha)
                grad_S += self.alphas[i] * deriv * unit_dir
                
            grad_norm = np.linalg.norm(grad_S)
            if grad_norm > 1e-8:
                grad_S = grad_S / grad_norm 
            else:
                print(f"  [Iter {k+1}] 梯度趨近於零，提早收斂。")
                break

            # --- 步驟 B：退火與步長更新機制 (加入消融開關) ---
            current_eta = eta
            success = False
            state_this_iter = current_method # 預設狀態為當前的引擎
            
            if use_annealing:
                for attempt in range(10): # 最多嘗試退火 10 次
                    S_try = S_current - current_eta * grad_S
                    try_loss = evaluate_S(S_try)
                    
                    if try_loss < best_loss:
                        if attempt > 0:
                            state_this_iter = "Annealing Triggered" # 標記成功觸發退火
                        print(f"  [Iter {k+1} - {current_method}] ✅ 步長 {current_eta:.4f} -> Loss: {try_loss:.4f}")
                        S_current = S_try
                        best_loss = try_loss

                        if best_loss < global_best_loss:
                            global_best_loss = best_loss
                            global_best_S = S_current.copy()

                        eta = min(0.5, current_eta * 1.5) # 樂觀加速
                        success = True
                        break
                    else:
                        current_eta /= 2.0 # 退火減半
            else:
                # 關閉退火：直接往前走一步，不測試縮減步長
                S_try = S_current - current_eta * grad_S
                try_loss = evaluate_S(S_try)
                if try_loss < best_loss:
                    print(f"  [Iter {k+1} - {current_method}] ✅ 無退火步長 {current_eta:.4f} -> Loss: {try_loss:.4f}")
                    S_current = S_try
                    best_loss = try_loss
                    success = True
                    
            # --- 步驟 C：引擎切換機制 (加入消融開關) ---
            if not success:
                if current_method == "secant" and allow_tangent:
                    print(f"  [Iter {k+1}] 割線法失真，切換至高精度切線法！")
                    current_method = "dynamic"
                    eta = 0.5 
                else:
                    print(f"  [Iter {k+1}] 高精度引擎亦達極限 (或不允許切換)，演算法收斂。")
                    break

            # --- 計算真實誤差並紀錄 ---
            y_val = np.sum(S_current)
            if len(S_current) > 1: 
                y_val += np.sum(S_current[:-1] * S_current[1:])
            error_history.append(abs(y_val - self.target_T))
            states_history.append(state_this_iter)

        print(f"✅ 法二法三引擎尋路完成！最終決策變數 S = {S_current}")
        return global_best_S, error_history, states_history
