import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import matplotlib.pyplot as plt

# 引用您專案內的核心模組
from models import PretrainedModel
from dataset import generate_agent_dataloaders
from agent_client import AgentNode
from host_server import HostServer

# ==========================================
# 實驗超參數設定
# ==========================================
TARGET_T = 1.5
NUM_ROUNDS = 50          # 傳統 FL 的通訊回合數
N_AGENTS = 20            # 參與節點總數
POISON_RATIO = 0.4       # 毒化節點比例 (40% 異質節點)
NUM_POISONED = int(N_AGENTS * POISON_RATIO)
N_FEATURES = 5           # 決策變數維度 (a, b)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 輔助函數：計算真實物理誤差
# ==========================================
def evaluate_S_with_ground_truth(S):
    """代入真實物理公式 y = a + ab + b 計算絕對誤差"""
    a, b = S[0], S[1]
    y_true = a + (a * b) + b
    return abs(y_true - TARGET_T)

# ==========================================
# 傳統 FL 核心機制：聚合與伺服器端反推
# ==========================================
def compute_fedavg(local_weights_list):
    """傳統 FedAvg 權重平均"""
    w_avg = copy.deepcopy(local_weights_list[0])
    for k in w_avg.keys():
        for i in range(1, len(local_weights_list)):
            w_avg[k] += local_weights_list[i][k]
        w_avg[k] = torch.div(w_avg[k], len(local_weights_list))
    return w_avg

def compute_krum(local_weights_list, num_malicious):
    """Krum 聚合演算法 (具備防禦毒化節點能力)"""
    n = len(local_weights_list)
    flat_weights = []
    
    # 將每家權重攤平成一維向量以便計算距離
    for w in local_weights_list:
        flat_w = torch.cat([v.flatten() for v in w.values()])
        flat_weights.append(flat_w)
        
    distances = torch.zeros(n, n)
    for i in range(n):
        for j in range(i+1, n):
            d = torch.norm(flat_weights[i] - flat_weights[j])**2
            distances[i, j] = distances[j, i] = d
            
    scores = []
    k = n - num_malicious - 2 # Krum 的核心參數
    for i in range(n):
        dists = distances[i].clone()
        dists[i] = float('inf') # 忽略自己
        sorted_dists, _ = torch.sort(dists)
        scores.append(torch.sum(sorted_dists[:k]).item())
        
    # 選出距離大家最近 (最正常) 的那個模型權重
    best_idx = np.argmin(scores)
    return local_weights_list[best_idx]

def server_side_method_d_inversion(global_model, target_T, steps=500):
    """伺服器端反推：傳統 FL 聚合出全域模型後，用方法 D 找答案"""
    global_model.eval()
    for param in global_model.parameters():
        param.requires_grad = False
        
    input_tensor = torch.randn(1, N_FEATURES, device=DEVICE, requires_grad=True)
    optimizer = optim.Adam([input_tensor], lr=0.01)
    criterion = nn.MSELoss()
    target_tensor = torch.tensor([[target_T]], dtype=torch.float32).to(DEVICE)

    for _ in range(steps):
        optimizer.zero_grad()
        output = global_model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            input_tensor.clamp_(-1.0, 1.0) # 物理邊界限制
            
    return input_tensor.detach().cpu().numpy().flatten()

# ==========================================
# 模擬執行傳統 FL (FedAvg / Krum)
# ==========================================
def run_traditional_fl_baseline(dataloaders, aggregator="fedavg"):
    print(f"\n🚀 啟動傳統 FL 訓練基準線 [{aggregator.upper()}] ...")
    
    # 建立 20 個未訓練的空模型
    local_models = [PretrainedModel(in_features=N_FEATURES).to(DEVICE) for _ in range(N_AGENTS)]
    global_model = PretrainedModel(in_features=N_FEATURES).to(DEVICE)
    global_weights = global_model.state_dict()
    
    error_history = []
    
    for round_idx in range(NUM_ROUNDS):
        local_weights_list = []
        
        # 1. 各節點下載 Global Model 並在本地訓練 1 個 Epoch
        for i, loader in enumerate(dataloaders):
            model = local_models[i]
            model.load_state_dict(global_weights)
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.MSELoss()
            
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.float().to(DEVICE), y_batch.float().to(DEVICE)
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()
                
            local_weights_list.append(model.state_dict())
            
        # 2. Server 聚合權重
        if aggregator == "fedavg":
            global_weights = compute_fedavg(local_weights_list)
        elif aggregator == "krum":
            global_weights = compute_krum(local_weights_list, NUM_POISONED)
            
        global_model.load_state_dict(global_weights)
        
        # 3. Server 端利用全域模型進行反推，並計算當前回合的誤差
        S_pred = server_side_method_d_inversion(global_model, TARGET_T)
        err = evaluate_S_with_ground_truth(S_pred)
        error_history.append(err)
        
        if (round_idx + 1) % 10 == 0:
            print(f"  [{aggregator.upper()}] Round {round_idx+1}/{NUM_ROUNDS} | Absolute Error: {err:.4f}")
            
    return error_history

# ==========================================
# 主程式：大擂台對決與繪圖
# ==========================================
def main():
    print("=== 🏆 聯邦學習黑箱尋路終極擂台賽 ===")
    
    # 1. 準備資料
    dataloaders = generate_agent_dataloaders(num_agents=N_AGENTS, poison_ratio=POISON_RATIO, n_features=N_FEATURES)
    
    # 2. 執行傳統 FL (Baseline)
    fedavg_errors = run_traditional_fl_baseline(dataloaders, aggregator="fedavg")
    krum_errors = run_traditional_fl_baseline(dataloaders, aggregator="krum")
    
    # 3. 執行您的法五 (Ours: Subspace + BFGS)
    print("\n🚀 啟動原創演算法 [Ours: Federated Subspace + Black-Box Optimization] ...")
    
    # (此處為模擬：在實務中，您的 Agent 已經各自 pre-train 好了)
    # 我們讓 Agent 訓練 5 個 epoch 達到穩定狀態
    agents = []
    for i in range(N_AGENTS):
        agent = AgentNode(agent_id=i)
        agent.model.train()
        optimizer = optim.Adam(agent.model.parameters(), lr=0.01)
        for _ in range(5): 
            for X_batch, y_batch in dataloaders[i]:
                optimizer.zero_grad()
                loss = nn.MSELoss()(agent.model(X_batch.float().to(DEVICE)), y_batch.float().to(DEVICE).unsqueeze(1))
                loss.backward()
                optimizer.step()
        agents.append(agent)
        
    server = HostServer(target_T=TARGET_T)
    
    # Phase 1, 2, 3
    server.phase1_filter_agents(agents)
    server.phase2_collect_proposals()
    final_S_ours = server.phase3_global_optimization() # 使用 BFGS 引擎
    
    ours_error = evaluate_S_with_ground_truth(final_S_ours)
    print(f"\n🎯 [Ours] 僅需 1 次通訊回合，最終 Absolute Error: {ours_error:.4f}")
    
    # 因為 Ours 只需要 1 回合，為了畫在同一張圖上，我們把它延伸成一條水平線
    ours_errors_line = [ours_error] * NUM_ROUNDS

    # ==========================================
    # 繪製終極對比圖表
    # ==========================================
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    rounds_x = list(range(1, NUM_ROUNDS + 1))
    
    plt.plot(rounds_x, fedavg_errors, label='Baseline: FedAvg (Server Inversion)', color='#e74c3c', linestyle='-', linewidth=2)
    plt.plot(rounds_x, krum_errors, label='Baseline: Krum (Server Inversion)', color='#f39c12', linestyle='-', linewidth=2)
    
    # 畫出我們的線，並在 Round 1 標記一顆大星星
    plt.plot(rounds_x, ours_errors_line, label='Ours: Federated Subspace + API Opt.', color='#2ecc71', linestyle='--', linewidth=2.5)
    plt.plot(1, ours_error, marker='*', markersize=18, color='#2ecc71')
    
    plt.title(f"Performance Comparison (Target T = {TARGET_T}, 40% Poisoned Agents)", fontsize=16, fontweight='bold')
    plt.xlabel("Communication Rounds (Network Overhead)", fontsize=12)
    plt.ylabel("Absolute Error of Derived Variables $S$", fontsize=12)
    plt.legend(fontsize=11)
    
    # 設定 Y 軸對數刻度可以更清楚看見微小誤差的差距
    # plt.yscale('log') 
    
    plt.tight_layout()
    plt.savefig('fl_ultimate_comparison.png', dpi=300)
    print("\n✅ 實驗圖表已儲存為 fl_ultimate_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()