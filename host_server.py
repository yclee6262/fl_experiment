import numpy as np
from scipy.optimize import minimize

class HostServer:
    def __init__(self, target_T, n_features):
        self.target_T = target_T
        self.n_features = n_features
        self.trusted_agents = []
        self.alphas = []
        self.I_list = []

        np.random.seed(42)
        self.test_X = np.random.uniform(-1, 1, (5, self.n_features)) 
        
        # ⭐ 3. 同步生成對應的標準答案 test_y
        self.test_y = np.sum(self.test_X, axis=1)
        if self.n_features > 1:
            self.test_y += np.sum(self.test_X[:, :-1] * self.test_X[:, 1:], axis=1)

    def phase1_filter_agents(self, all_agents):
        """發送測試題，過濾掉誤差太大的惡意節點"""
        print("\n--- Phase 1: 節點信任度測驗 ---")
        
        scores = []
        for agent in all_agents:
            pred_y = agent.api_predict(self.test_X)
            mse = np.mean((pred_y - self.test_y)**2)
            
            if mse < 0.1: # 門檻值：過濾掉誤差極大的惡意節點
                scores.append((agent, 1.0 / (mse + 1e-5))) # 誤差越小分數越高
                print(f"Agent {agent.agent_id} 通過測驗 (MSE: {mse:.4f})")
            else:
                print(f"Agent {agent.agent_id} 被剔除 (MSE: {mse:.4f})")
                
        # 正規化分數變成 alphas (加總為 1)
        total_score = sum([s[1] for s in scores])
        for agent, score in scores:
            self.trusted_agents.append(agent)
            self.alphas.append(score / total_score)

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