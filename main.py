import copy
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

from dataset import generate_agent_dataloaders
from agent_client import AgentNode
from host_server import HostServer
from models import PretrainedModel

def run_fedavg_baseline(agents, n_features, target_T, global_rounds=15, local_epochs=5):
    """執行傳統 FedAvg 並記錄每個 Round 的反推誤差"""
    print("\n" + "="*50)
    print("啟動傳統 baseline：FedAvg (Federated Averaging)")
    print("="*50)
    device = agents[0].device
    global_model = PretrainedModel(in_features=n_features).to(device)
    
    error_history = []
    
    for round_num in range(global_rounds):
        local_weights = []
        global_state_dict = global_model.state_dict()
        
        for agent in agents:
            agent.model.load_state_dict(copy.deepcopy(global_state_dict))
            agent.model.train()
            for param in agent.model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(agent.model.parameters(), lr=0.01)
            criterion = nn.MSELoss()
            for _ in range(local_epochs):
                for X_batch, y_batch in agent.dataloader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
                    optimizer.zero_grad()
                    loss = criterion(agent.model(X_batch), y_batch)
                    loss.backward()
                    optimizer.step()
            local_weights.append(copy.deepcopy(agent.model.state_dict()))
            
        # FedAvg: 權重平均聚合
        avg_state_dict = copy.deepcopy(local_weights[0])
        for key in avg_state_dict.keys():
            for i in range(1, len(local_weights)):
                avg_state_dict[key] += local_weights[i][key]
            avg_state_dict[key] = torch.div(avg_state_dict[key], len(local_weights))
        global_model.load_state_dict(avg_state_dict)
        
        # --- 快速反推記錄當下誤差 ---
        global_model.eval()
        input_tensor = torch.randn(1, n_features, device=device, requires_grad=True)
        opt_inv = optim.Adam([input_tensor], lr=0.05)
        target_tensor = torch.tensor([[target_T]], dtype=torch.float32).to(device)
        for _ in range(200): 
            opt_inv.zero_grad()
            output = global_model(input_tensor)
            loss_inv = nn.MSELoss()(output, target_tensor)
            loss_inv.backward()
            opt_inv.step()
            with torch.no_grad(): input_tensor.clamp_(-1.0, 1.0)
            
        current_S = input_tensor.detach().cpu().numpy().flatten()
        y_val = np.sum(current_S)
        if len(current_S) > 1: y_val += np.sum(current_S[:-1] * current_S[1:])
        current_error = abs(y_val - target_T)
        error_history.append(current_error)
        print(f"  [FedAvg] Round {round_num + 1}/{global_rounds} 誤差: {current_error:.4f}")

    return current_S, error_history

def main():
    TARGET_T = 0
    NUM_AGENTS = 20      # 擴大參與人數
    POISON_RATIO = 0.3   # 設定 40% 的人是異質節點
    N = 10                # 設定是幾元多項式
    
    print(f"=== Phase 0: 準備資料與訓練本地神經網路 (共 {NUM_AGENTS} 個 Agent) ===")
    # 傳入設定的參數
    loaders = generate_agent_dataloaders(num_agents=NUM_AGENTS, poison_ratio=POISON_RATIO, n_features=N)
    
    all_agents = []
    for i in range(NUM_AGENTS):
        agent = AgentNode(agent_id=i+1, dataloader=loaders[i], n_features=N)
        print(f"訓練 Agent {i+1} 中...")
        agent.train_local_model(epochs=30)
        all_agents.append(agent)
        
    print("\n=== 開始反推演算法 ===")
    server = HostServer(target_T=TARGET_T, n_features=N)
    
    # Phase 1: 考試過濾
    server.phase1_filter_agents(all_agents)
    
    # Phase 2: 反推
    server.phase2_collect_proposals()
    
    # Phase 3: 黑箱最佳化 (雙引擎對決！)
    print("\n" + "="*50)
    print("子空間法開始，目標 T = {}".format(TARGET_T))
    print("="*50)
    
    # 引擎 1：工業級 SciPy BFGS
    final_S_bfgs, hist_bfgs = server.phase3_global_optimization()
    
    # 引擎 2：割線/切線法
    final_S_custom, hist_custom, states_custom = server.phase3_custom_secant_optimization(num_iterations=30)

    # Phase 3.5: 移除負邊際貢獻者並重新最佳化
    pruning_report = server.prune_negative_contributors(
        final_S_custom,
        epsilon=1e-6,
        optimizer="custom",
        custom_iterations=30,
    )
    final_S_custom = pruning_report["final_solution"]
    if pruning_report["final_history"]:
        hist_custom = pruning_report["final_history"]
        states_custom = pruning_report["final_states"]

    # Phase 4: 對 pruning 後的穩定 coalition 做子空間排除法分潤
    profit_report = server.phase4_profit_sharing(final_S_custom)

    final_S_fedavg, hist_fedavg = run_fedavg_baseline(all_agents, N, TARGET_T, global_rounds=15)
    
    # === 驗證與比較結果 ===
    print("\n=== 最終對比結果 ===")
    def get_formula_string(n):
        linear_terms = [f"x{i}" for i in range(n)]
        cross_terms = [f"(x{i}*x{i+1})" for i in range(n-1)]
        return "y = " + " + ".join(linear_terms + cross_terms)
        
    formula_str = get_formula_string(N)

    def calculate_true_y(S):
        y_val = np.sum(S)
        if len(S) > 1:
            y_val += np.sum(S[:-1] * S[1:])
        return y_val
    
    # 計算 BFGS 的真實 y
    y_bfgs = calculate_true_y(final_S_bfgs)
    
    # 計算自創引擎的真實 y
    y_custom = calculate_true_y(final_S_custom)
    
    print(f"目標 T: {TARGET_T}")
    print(f"目標公式: {formula_str}")
    # 為了版面整潔，將高維度變數陣列四捨五入印出
    S_bfgs_str = np.array2string(final_S_bfgs, formatter={'float_kind':lambda x: "%.4f" % x})
    S_custom_str = np.array2string(final_S_custom, formatter={'float_kind':lambda x: "%.4f" % x})
    
    print(f"[SciPy BFGS 引擎] 求得變數: {S_bfgs_str} | 代入目標公式 y={y_bfgs:.4f}")
    print(f"[法二法三引擎] 求得變數: {S_custom_str} | 代入目標公式 y={y_custom:.4f}")

    print("\n=== 負貢獻 Pruning 結果 ===")
    print(f"最終 coalition: {pruning_report['final_coalition_ids']}")
    for row in pruning_report["pruning_log"]:
        if row["removed_agent_id"] is not None:
            print(
                f"Round {row['round']}: 移除 Agent {row['removed_agent_id']} "
                f"(base_loss={row['base_loss']:.6f})"
            )
        else:
            print(f"Round {row['round']}: {row['status']} (base_loss={row['base_loss']:.6f})")

    print("\n=== 分潤結果 (Stage 4) ===")
    if profit_report["status"] == "ok":
        for row in profit_report["payments"]:
            print(
                f"Agent {row['agent_id']}: "
                f"C_i+={row['positive_contribution']:.6f}, "
                f"profit_share={row['profit_share']:.4f}, "
                f"payment={row['payment']:.4f}"
            )
    else:
        print(f"分潤狀態: {profit_report['status']}")

    # === 🎨 終極大合併畫圖 ===
    print("\n正在繪製實驗對比圖...")
    os.makedirs("results_plots", exist_ok=True)
    plt.figure(figsize=(12, 7))
    
    # 由於 BFGS 收斂的迭代次數不固定，我們分別產生各自的 X 軸
    x_fedavg = range(1, len(hist_fedavg) + 1)
    x_bfgs = range(1, len(hist_bfgs) + 1)
    x_custom = range(1, len(hist_custom) + 1)
    
    # 畫出三條線 (使用您熟悉的配色與標記)
    plt.plot(x_fedavg, hist_fedavg, label='Baseline: Traditional FedAvg', 
             color='red', marker='x', linestyle=':', linewidth=2)
             
    plt.plot(x_bfgs, hist_bfgs, label='Ours: Subspace (BFGS)', 
             color='orange', marker='s', linestyle='--', alpha=0.8, linewidth=2.5)
             
    plt.plot(x_custom, hist_custom, label='Ours: Subspace (Secant/Tangent Engine)', 
             color='green', marker='*', linewidth=3.5, markersize=12)
    
    # 標題與標籤設定
    plt.title(f'Performance Comparison: FedAvg vs. Subspace Methods (T={TARGET_T}, N={N})', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Communication Rounds (FedAvg) / API Iterations (Ours)', fontsize=14)
    plt.ylabel('Absolute Error |y_pred - T|', fontsize=14)
    
    # 讓 Y 軸使用對數尺度 (Log Scale)，因為法五的誤差會降得非常低，用 Log 才看得出差距！
    plt.yscale('log') 
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
    
    # 儲存圖片
    save_path = f"results_plots/fl_inversion_test_comparison_N{N}_T{TARGET_T}_rho{POISON_RATIO}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"=== 🎉 實驗對比圖已儲存至：{save_path} ===")
    plt.show()
    
if __name__ == "__main__":
    main()
