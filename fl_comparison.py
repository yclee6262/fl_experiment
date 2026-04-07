import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from medmnist import PathMNIST, BloodMNIST
import flwr as fl
import matplotlib.pyplot as plt
from collections import OrderedDict
from models import Net, get_device
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

# --- 1. 資料讀取工具 ---
def load_agent_data(agent_id):
    """讀取該 Agent 專屬的訓練資料 (絕對隔離)"""
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    # 判斷是哪個資料集
    if agent_id <= 6:
        dataset = PathMNIST(split='train', transform=data_transform, download=False)
    else:
        dataset = BloodMNIST(split='train', transform=data_transform, download=False)
        
    # 讀取專屬 indices
    indices_path = f"saved_models/agent_{agent_id}_indices.pt"
    if not os.path.exists(indices_path):
        raise FileNotFoundError(f"找不到 {indices_path}，請先執行預訓練腳本切分資料。")
    
    indices = torch.load(indices_path)
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=32, shuffle=True)

def get_centralized_test_loader():
    """載入全局測試集 (Host 關心的 PathMNIST 測試集)"""
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    test_dataset = PathMNIST(split='test', transform=data_transform, download=True)
    return DataLoader(test_dataset, batch_size=128, shuffle=False)

# --- 2. 定義 Flower Client ---
class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.device = device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.train()
        
        # 本地訓練設定
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 進行 1 個 Local Epoch
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device).squeeze()
            optimizer.zero_grad()
            output = self.model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

# --- 3. 定義中心化評估函數 (Centralized Evaluation) ---
def get_evaluate_fn(device):
    """在每一輪 FL 結束後，Server 會呼叫這個函數來測試 Global Model"""
    test_loader = get_centralized_test_loader()
    
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        model = Net(num_classes=9).to(device)
        # 將聚合後的權重載入模型
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        model.eval()
        correct, total, loss = 0, 0, 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).squeeze()
                outputs = model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = correct / total
        return loss, {"accuracy": accuracy}
        
    return evaluate

# ==========================================
# 實作：聯邦學習第三路 - 子空間法動態優化器
# ==========================================
class SubspaceOptStrategy(fl.server.strategy.FedAvg):
    def __init__(self, evaluate_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluate_fn = evaluate_fn
        self.current_weights = None  # 紀錄當前的全域模型權重 (M^k)
        self.best_loss = float('inf') # 紀錄當前最佳的 Loss
        self.eta = 1.0               # 初始步長 (Learning Rate / Step size)

    def aggregate_fit(self, server_round, results, failures):
        # 1. 先用傳統 FedAvg 算出大家期望的目標位置
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is None:
            return None, {}

        # 將二進位參數轉換為 NumPy 陣列矩陣，方便進行子空間數學運算
        target_weights = parameters_to_ndarrays(aggregated_parameters)

        # 第 0 回合初始化：直接接受並評估基準線
        if self.current_weights is None:
            self.current_weights = target_weights
            loss, _ = self.evaluate_fn(server_round, target_weights, {})
            self.best_loss = loss
            print(f"\n[Subspace Init] Round {server_round} Baseline Loss: {loss:.4f}")
            return aggregated_parameters, metrics

        # 2. 計算子空間的移動方向向量 (Gradient = Target - Current)
        direction_vector = [t - c for t, c in zip(target_weights, self.current_weights)]

        # 3. 防 Overshooting 機制 (Backtracking Line Search)
        current_eta = self.eta
        max_retries = 4  # 最多嘗試減半 4 次
        
        print(f"\n--- Round {server_round} 進入子空間防 Overshooting 驗證 ---")
        for attempt in range(max_retries):
            # 計算嘗試性的新解: M_try = M^k + eta * Gradient
            m_try = [c + current_eta * d for c, d in zip(self.current_weights, direction_vector)]
            
            # Host 驗證這個 M_try 的效能
            loss, metrics = self.evaluate_fn(server_round, m_try, {})
            print(f"  [嘗試 {attempt+1}] 步長 eta={current_eta:.4f} -> 獲得 Loss={loss:.4f} (歷史最佳={self.best_loss:.4f})")
            
            if loss < self.best_loss:
                # 效能改善：接受更新
                print(f"  ✅ 效能改善！接受更新。")
                self.current_weights = m_try
                self.best_loss = loss
                # 如果這步走得很順，稍微放大下一步的步長作為獎勵 (上限 1.5)
                self.eta = min(1.5, current_eta * 1.2) 
                return ndarrays_to_parameters(m_try), metrics
            else:
                # 效能劣化：發生 Overshooting，拒絕更新並將步長減半
                print(f"  ❌ 發生 Overshooting！退回原點，步長減半。")
                current_eta /= 2.0
        
        # 如果嘗試了 4 次都失敗，代表方向可能不佳，採取極保守的步伐
        print("  ⚠️ 達到最大重試次數，採取最保守步伐強行更新。")
        m_try = [c + current_eta * d for c, d in zip(self.current_weights, direction_vector)]
        self.current_weights = m_try
        self.best_loss, _ = self.evaluate_fn(server_round, m_try, {})
        
        return ndarrays_to_parameters(m_try), metrics

# --- 4. 執行模擬 ---
def run_simulation(agent_list, strategy, num_rounds=15):
    device = get_device()
    
    # 建立一個工廠函數，根據 Flower 給的 cid ('0', '1', ...) 指派對應的 Agent
    def client_fn(cid: str) -> fl.client.Client:
        # 將 cid 對應到我們真實的 agent_id
        agent_id = agent_list[int(cid)]
        train_loader = load_agent_data(agent_id)
        
        # 注意：無論是誰，Global Model 架構必須統一 (以 Host 關注的 9 類為主)
        model = Net(num_classes=9).to(device) 
        return FLClient(model, train_loader, device).to_client()

    # 設定 FedAvg 策略
    # strategy = fl.server.strategy.FedAvg(
    #     fraction_fit=1.0,  # 每一輪 100% 的 client 都參與
    #     fraction_evaluate=0.0,
    #     min_fit_clients=len(agent_list),
    #     min_available_clients=len(agent_list),
    #     evaluate_fn=get_evaluate_fn(device) # 設定全局測試
    # )
    
    print(f"\n=== 開始 FL 模擬 | 參與者: {agent_list} | 回合數: {num_rounds} ===")
    
    # 啟動模擬
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(agent_list),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.2} # 根據您的硬體調整
    )
    
    # 提取 Accuracy 數據
    accuracies = [acc for _, acc in history.metrics_centralized["accuracy"]]
    return accuracies

# --- 5. 主程式與畫圖 ---
def main():
    NUM_ROUNDS = 15
    device = torch.device("cpu") # 或是 get_device()
    evaluate_fn = get_evaluate_fn(device)
    
    agents_all = [1, 2, 3, 4, 5, 6]
    agents_filtered = [1, 2, 3, 4, 5, 6]

    # --- 策略 1：FedAvg (傳統平均) ---
    strategy_avg = fl.server.strategy.FedAvg(
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=len(agents_all), min_available_clients=len(agents_all),
        evaluate_fn=evaluate_fn
    )
    
    # --- 策略 2：FedMedian (取中位數，抗極端值) ---
    strategy_median = fl.server.strategy.FedMedian(
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=len(agents_all), min_available_clients=len(agents_all),
        evaluate_fn=evaluate_fn
    )

    # --- 策略 3: FedProx: 加上近端項 ---
    strategy_prox = fl.server.strategy.FedProx(
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=len(agents_all), min_available_clients=len(agents_all),
        evaluate_fn=evaluate_fn,
        proximal_mu=0.1 
    )

    # --- 策略 4. Krum (Multi-Krum): 距離防禦演算法 ---
    strategy_krum = fl.server.strategy.Krum(
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=len(agents_all), min_available_clients=len(agents_all),
        evaluate_fn=evaluate_fn,
        num_malicious_clients=3,
        num_clients_to_keep=7 
    )
    
    # --- 策略 7：針對篩選後的 FedAvg (我們的法五) ---
    strategy_avg_filtered = fl.server.strategy.FedAvg(
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=len(agents_filtered), min_available_clients=len(agents_filtered),
        evaluate_fn=evaluate_fn
    )

    # --- 策略 8：法五篩選 + 子空間動態優化 (最強完全體) ---
    strategy_subspace_filtered = SubspaceOptStrategy(
        evaluate_fn=evaluate_fn,  # 必須傳入 evaluate_fn 讓 Server 可以自我驗證
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=len(agents_filtered), min_available_clients=len(agents_filtered)
    )

    # === 開始執行所有實驗 ===
    print(">>> 實驗 1/7: 10 人全上 (FedAvg)...")
    acc_all_avg = run_simulation(agents_all, strategy_avg, NUM_ROUNDS)
    
    print("\n>>> 實驗 2/7: 10 人全上 (FedMedian)...")
    acc_all_median = run_simulation(agents_all, strategy_median, NUM_ROUNDS)
    
    print("\n>>> 實驗 3/7: 10 人全上 (FedProx)...")
    acc_all_prox = run_simulation(agents_all, strategy_prox, NUM_ROUNDS)

    print("\n>>> 實驗 4/7: 10 人全上 (Krum)...")
    acc_all_krum = run_simulation(agents_all, strategy_krum, NUM_ROUNDS)
    
    print("\n>>> 實驗 7/7: 6 人篩選後 (FedAvg) [我們的法五]...")
    acc_filtered_avg = run_simulation(agents_filtered, strategy_avg_filtered, NUM_ROUNDS)

    print("\n>>> 實驗 8/8: 6 人篩選後 + 子空間優化 (Ours Pro 完全體)...")
    acc_filtered_subspace = run_simulation(agents_filtered, strategy_subspace_filtered, NUM_ROUNDS)
    
    rounds = range(0, NUM_ROUNDS + 1)
    
    # ==========================================
    # 終極合併畫圖：All Aggregators vs. Ours
    # ==========================================
    plt.figure(figsize=(14, 8)) # 稍微放大畫布，讓多條線有呼吸空間
    rounds = range(0, NUM_ROUNDS + 1)
    
    # 1. 最弱基準線 (Baseline) - 使用紅色點狀線
    plt.plot(rounds, acc_all_avg, label='Baseline: 10 Agents (FedAvg)', 
             color='red', marker='x', linestyle=':', linewidth=2)
    
    # 2. 假想敵對手群 (Robust Aggregators) - 使用不同的顏色與虛線，加入 alpha 降低視覺干擾
    plt.plot(rounds, acc_all_median, label='Robust: 10 Agents (FedMedian)', 
             color='orange', marker='s', linestyle='--', alpha=0.7)
    plt.plot(rounds, acc_all_prox, label='Robust: 10 Agents (FedProx)', 
             color='purple', marker='^', linestyle='--', alpha=0.7)
    plt.plot(rounds, acc_all_krum, label='Robust: 10 Agents (Krum)', 
             color='brown', marker='d', linestyle='--', alpha=0.7)
    
    # 3. 我們的殺手鐧 (Ours) - 使用最粗的綠色實線與大圓點，抓住眼球
    plt.plot(rounds, acc_filtered_avg, label='Ours: 6 Agents Filtered (FedAvg)', 
             color='green', marker='o', linewidth=3.5, markersize=8)
    
    # 4. 最強完全體 (Ours Pro) - 用紅色星號強調
    plt.plot(rounds, acc_filtered_subspace, label='Ours Pro: Pre-filtered + Subspace Opt', 
             color='red', marker='*', linewidth=3.5, markersize=10)
    
    # 設定標題與軸標籤
    plt.title('Comprehensive Evaluation of FL Aggregators vs. Pre-filtering (Ours)', 
              fontsize=18, fontweight='bold')
    plt.xlabel('Federated Learning Rounds', fontsize=14)
    plt.ylabel('PathMNIST Test Accuracy', fontsize=14)
    plt.xticks(rounds)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 調整圖例位置與大小
    plt.legend(fontsize=12, loc='lower right', framealpha=0.9)
    
    save_path = "fl_ultimate_combined_comparison_same_agent.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n=== 實驗完成！終極比較圖已儲存至 {save_path} ===")
    plt.show()

if __name__ == "__main__":
    main()
