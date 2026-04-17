import numpy as np
from scipy.optimize import minimize

class HostServer:
    def __init__(self, target_T):
        self.target_T = target_T
        self.trusted_agents = []
        self.alphas = []
        self.I_list = [] # 收集大家的初步建議參數

    def phase1_filter_agents(self, all_agents):
        """發送測試題，過濾掉誤差太大的惡意節點"""
        print("\n--- Phase 1: 節點信任度測驗 ---")
        test_X = np.array([[0.5, 0.5], [-0.2, 0.8]])
        # 真正的 y = a + ab + b
        true_y = np.array([0.5 + 0.25 + 0.5, -0.2 - 0.16 + 0.8]) 
        
        scores = []
        for agent in all_agents:
            pred_y = agent.api_predict(test_X)
            mse = np.mean((pred_y - true_y)**2)
            
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
        
        def total_loss_function(betas):
            S_current = np.dot(betas, I_matrix)
            total_loss = 0.0
            
            # 呼叫每個 Agent 的 API 算預測值
            for i, agent in enumerate(self.trusted_agents):
                pred_i = agent.api_predict(S_current)[0]
                # 加權誤差: alpha * |f(S) - T|
                total_loss += self.alphas[i] * abs(pred_i - self.target_T)
            return total_loss

        # 初始猜測：平均分配
        initial_betas = np.ones(len(self.trusted_agents)) / len(self.trusted_agents)
        
        result = minimize(total_loss_function, initial_betas, method='BFGS')
        best_betas = result.x
        final_S = np.dot(best_betas, I_matrix)
        
        return final_S
    
    def phase3_custom_secant_optimization(self, num_iterations=50):
        """Phase 3 (Alternative): 使用原創的割線/切線法進行子空間尋路"""
        print("\n--- Phase 3: 全域最佳化 (啟動割線/切線退火引擎) ---")
        
        # 1. 將 Agent 提議的參數轉為矩陣，並計算起點 (平均值 M^0)
        I_matrix = np.array(self.I_list)
        n_agents = len(self.trusted_agents)
        S_current = np.mean(I_matrix, axis=0) # 從平均點出發
        
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
        eta = 1.0
        current_method = "secant"
        delta = 0.5 # 切線法的微小偏移量
        
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
                
            # --- 步驟 B：退火與步長更新機制 ---
            current_eta = eta
            success = False
            for attempt in range(4): # 最多嘗試退火 4 次
                S_try = S_current - current_eta * grad_S
                try_loss = evaluate_S(S_try)
                
                if try_loss < best_loss:
                    print(f"  [Iter {k+1} - {current_method}] ✅ 步長 {current_eta:.4f} -> Loss: {try_loss:.4f}")
                    S_current = S_try
                    best_loss = try_loss
                    eta = min(2.0, current_eta * 1.2) # 樂觀加速
                    success = True
                    break
                else:
                    current_eta /= 2.0 # 退火減半
                    
            # --- 步驟 C：引擎切換機制 ---
            if not success:
                if current_method == "secant":
                    print(f"  [Iter {k+1}] 割線法失真，切換至高精度切線法！")
                    current_method = "dynamic"
                    eta = 0.5 
                else:
                    print(f"  [Iter {k+1}] 高精度引擎亦達極限，演算法收斂。")
                    break

        print(f"✅ 法二法三引擎尋路完成！最終決策變數 S = {S_current}")
        return S_current