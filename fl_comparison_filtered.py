import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from medmnist import PathMNIST, BloodMNIST
import flwr as fl
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np

# 如果您有自定義 models.py，請保留這行；這裡為了確保能跑，我假設您有 Net 和 get_device
from models import Net, get_device 

# ==========================================
# 工具函數：扁平化權重 (幫助子空間做向量數學運算)
# ==========================================
def flatten_state_dict(state_dict):
    """將 state_dict 轉為 1D Tensor"""
    return torch.cat([v.flatten() for v in state_dict.values()])

def unflatten_state_dict(flat_tensor, ref_state_dict):
    """將 1D Tensor 轉回 state_dict 格式"""
    new_dict = OrderedDict()
    offset = 0
    for k, v in ref_state_dict.items():
        numel = v.numel()
        new_dict[k] = flat_tensor[offset:offset+numel].view_as(v)
        offset += numel
    return new_dict

# --- 1. 資料讀取工具 ---
def load_agent_data(agent_id):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    if agent_id <= 6:
        dataset = PathMNIST(split='train', transform=data_transform, download=False)
    else:
        dataset = BloodMNIST(split='train', transform=data_transform, download=False)
        
    indices_path = f"saved_models/agent_{agent_id}_indices.pt"
    if not os.path.exists(indices_path):
        raise FileNotFoundError(f"找不到 {indices_path}，請先執行切分資料腳本。")
    
    indices = torch.load(indices_path)
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=32, shuffle=True)

def get_centralized_test_loader():
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    test_dataset = PathMNIST(split='test', transform=data_transform, download=True)
    return DataLoader(test_dataset, batch_size=128, shuffle=False)

# --- 2. 評估函數 (Server 驗證用) ---
def get_evaluate_fn(device):
    test_loader = get_centralized_test_loader()
    
    # 用於 Flower 框架的評估
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        model = Net(num_classes=9).to(device)
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

# 給子空間優化器使用的純 PyTorch 評估函數
def evaluate_pytorch_model(model_state_dict, device):
    test_loader = get_centralized_test_loader()
    model = Net(num_classes=9).to(device)
    model.load_state_dict(model_state_dict)
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
    return loss, correct / total

# --- 3. 定義 Flower Client 與傳統 FL 模擬函數 ---
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device).squeeze()
            optimizer.zero_grad()
            output = self.model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

def run_simulation(agent_list, strategy, num_rounds=15):
    device = torch.device("cpu") # 若有 GPU 請自行更改
    def client_fn(cid: str) -> fl.client.Client:
        agent_id = agent_list[int(cid)]
        train_loader = load_agent_data(agent_id)
        model = Net(num_classes=9).to(device) 
        return FLClient(model, train_loader, device).to_client()

    print(f"\n=== 開始 FL 模擬 (通訊耗時) | 策略: {strategy.__class__.__name__} ===")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(agent_list),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.25} # 依硬體調整
    )
    return [acc for _, acc in history.metrics_centralized["accuracy"]]

## 實驗:畫剖面圖
def run_displacement_profiling(agent_list, device):
    """
    實驗一：偏移量(Delta)的窮舉剖面分析
    目的：畫出往特定 Agent 方向移動時，Loss 的真實地形圖。
    """
    print(f"\n{'='*50}\n🔬 實驗一啟動：偏移量 (Delta) 剖面地形掃描\n{'='*50}")
    
    # 1. 準備起點 (Global Init) 與終點 (某一個錨點 m_1^*)
    init_path = "saved_models/global_init.pt"
    ref_state_dict = torch.load(init_path, map_location=device)
    M_current_flat = flatten_state_dict(ref_state_dict) # 起點：嬰兒模型
    
    # 我們隨便挑選 Agent 1 作為觀察方向
    target_agent = agent_list[0]
    path = f"saved_models/m_star_agent_{target_agent}.pt"
    m_star_dict = torch.load(path, map_location=device)
    m_star_flat = flatten_state_dict(m_star_dict)       # 終點：訓練好的錨點
    
    # 2. 計算單位方向向量 (Unit Direction Vector)
    direction = m_star_flat - M_current_flat
    dist = torch.norm(direction)
    unit_dir = direction / dist
    
    # 先計算起點的 Loss (Delta = 0 的截距)
    base_loss, _ = evaluate_pytorch_model(ref_state_dict, device)
    print(f"📍 起點 (Delta=0) 的基準 Loss: {base_loss:.4f}")
    
    # 3. 準備要窮舉的 Delta 候選名單
    # 我們使用對數刻度 (Log space)，從 10^-6 掃描到 10^1，共產生 30 個點
    deltas = np.logspace(-6, 1, 30)
    losses = []
    
    # 4. 開始不計代價地窮舉描點
    print("⏳ 正在沿著方向發射探測波，請稍候...")
    for delta in deltas:
        # 計算位移後的新權重： M' = M + delta * U
        M_perturbed_flat = M_current_flat + delta * unit_dir
        
        # 評估這個新地點的 Loss
        loss_p, _ = evaluate_pytorch_model(unflatten_state_dict(M_perturbed_flat, ref_state_dict), device)
        losses.append(loss_p)
        print(f"  探測 Delta = {delta:.1e} --> Loss = {loss_p:.4f}")

    # 5. 繪製橫切剖面圖 (Displacement Profile)
    plt.figure(figsize=(10, 6))
    plt.plot(deltas, losses, marker='o', linestyle='-', color='b')
    
    # 畫一條紅色的水平虛線，代表基準 Loss
    plt.axhline(y=base_loss, color='r', linestyle='--', label='Baseline Loss (Delta=0)')
    
    plt.xscale('log') # 橫軸使用對數刻度，方便觀察極小的 Delta
    plt.title('Displacement Profiling: Loss vs. Delta', fontsize=16)
    plt.xlabel('Displacement Delta ($\Delta$) (Log Scale)', fontsize=14)
    plt.ylabel('Test Loss', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    save_path = "results_plots/delta_profiling.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n🎉 剖面掃描完成！圖表已存至：{save_path}")
    plt.show()

# =====================================================================
# ★ 聯邦學習第三路核心功能 ★
# =====================================================================

def generate_initial_m_stars(agent_list, device, local_epochs=30):
    """階段二：讓過關的 Agent 在本地訓練到底，獲取錨點 m_i^*"""
    print(f"\n{'='*50}\n🚀 階段二：產生各 Agent 的局部最佳解 (m_i^*)\n{'='*50}")
    os.makedirs("saved_models", exist_ok=True)
    
    # ==========================================
    # ⭐️ 核心修正：強制所有 Agent 擁有相同的初始起點
    # ==========================================
    init_path = "saved_models/global_init.pt"
    if not os.path.exists(init_path):
        init_model = Net(num_classes=9).to(device)
        torch.save(init_model.state_dict(), init_path)
        print("✅ 建立統一的初始全域模型 (Global Init) 成功！")
    
    for agent_id in agent_list:
        save_path = f"saved_models/m_star_agent_{agent_id}.pt"
        if os.path.exists(save_path):
            print(f"✅ Agent {agent_id} 的 m_i^* 已存在，跳過訓練。")
            continue
            
        print(f"正在訓練 Agent {agent_id} (Epochs: {local_epochs})...")
        train_loader = load_agent_data(agent_id)
        model = Net(num_classes=9).to(device)
        
        # ⭐️ 每個 Agent 訓練前，先載入相同的初始權重
        model.load_state_dict(torch.load(init_path, map_location=device))
        
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(local_epochs):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device).squeeze()
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
        torch.save(model.state_dict(), save_path)
        print(f"✅ Agent {agent_id} 訓練完成！已儲存至 {save_path}")

def run_server_subspace_optimization(agent_list, device, num_iterations=15):
    """階段三：純伺服器端的子空間動態優化 (零通訊成本)"""
    print(f"\n{'='*50}\n🚀 階段三：啟動伺服器端子空間動態優化 (Ours Pro)\n{'='*50}")
    
    m_stars_flat = []
    ref_state_dict = None
    for agent_id in agent_list:
        path = f"saved_models/m_star_agent_{agent_id}.pt"
        state_dict = torch.load(path, map_location=device)
        if ref_state_dict is None: ref_state_dict = state_dict
        m_stars_flat.append(flatten_state_dict(state_dict))
    
    n_agents = len(m_stars_flat)
    alphas = [1.0 / n_agents] * n_agents
    
    print("⏳ 正在預先評估各錨點的目標函數值...")
    loss_m_stars = []
    for i in range(n_agents):
        loss_i, _ = evaluate_pytorch_model(unflatten_state_dict(m_stars_flat[i], ref_state_dict), device)
        loss_m_stars.append(loss_i)
        
    M_current_flat = torch.mean(torch.stack(m_stars_flat), dim=0)
    current_loss, current_acc = evaluate_pytorch_model(unflatten_state_dict(M_current_flat, ref_state_dict), device)
    
    accuracies = [current_acc]
    best_loss = current_loss
    
    eta = 1.0
    current_method = "secant"
    delta = 1e-4
    
    for k in range(num_iterations):
        print(f"--- Server Iteration {k+1}/{num_iterations} | 引擎: {current_method} ---")
        
        grad_flat = torch.zeros_like(M_current_flat)
        for i in range(n_agents):
            direction = m_stars_flat[i] - M_current_flat
            dist = torch.norm(direction)
            if dist < 1e-8: continue
            unit_dir = direction / dist
            
            if current_method == "secant":
                deriv = (loss_m_stars[i] - best_loss) / dist
            else: 
                M_perturb = M_current_flat + delta * unit_dir
                loss_p, _ = evaluate_pytorch_model(unflatten_state_dict(M_perturb, ref_state_dict), device)
                deriv = (loss_p - best_loss) / delta
                
            grad_flat += alphas[i] * deriv * unit_dir
            
        current_eta = eta
        max_retries = 4
        success = False
        
        for attempt in range(max_retries):
            M_try_flat = M_current_flat - current_eta * grad_flat
            try_loss, try_acc = evaluate_pytorch_model(unflatten_state_dict(M_try_flat, ref_state_dict), device)
            
            if try_loss < best_loss:
                print(f"  ✅ [成功] 步長 {current_eta:.4f} -> Loss: {try_loss:.4f}, Acc: {try_acc:.4f}")
                M_current_flat = M_try_flat
                best_loss = try_loss
                accuracies.append(try_acc)
                eta = min(2.0, current_eta * 1.2)
                success = True
                break
            else:
                print(f"  ❌ [Overshoot] 步長 {current_eta:.4f} 導致 Loss 上升 ({try_loss:.4f})，步長減半！")
                current_eta /= 2.0
                
        if not success:
            if current_method == "secant":
                print("  🔄 [切換引擎] 割線法失真，下一回合啟動高精度『動態方向法 (Dynamic)』！")
                current_method = "dynamic"
                eta = 0.5 
                accuracies.append(accuracies[-1]) 
            else:
                print("  🛑 高精度引擎亦達極限，模型已收斂。")
                accuracies.append(accuracies[-1])
                
    return accuracies

def main():
    NUM_ROUNDS = 15
    device = torch.device("cpu") # 或是 get_device()
    evaluate_fn = get_evaluate_fn(device)
    
    # 建立圖表資料夾
    plot_dir = "results_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # 這次我們只關注篩選後的這 6 個人
    agents_filtered = [1, 2, 3, 4, 5, 6]

    print(f"\n{'='*50}\n🔬 消融實驗：針對 6 名純淨 Agent 的 Aggregator 測試\n{'='*50}")

    # --- 策略 1：Filtered + FedAvg (我們的基準) ---
    strategy_avg = fl.server.strategy.FedAvg(
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=len(agents_filtered), min_available_clients=len(agents_filtered),
        evaluate_fn=evaluate_fn
    )
    
    # --- 策略 2：Filtered + FedMedian ---
    strategy_median = fl.server.strategy.FedMedian(
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=len(agents_filtered), min_available_clients=len(agents_filtered),
        evaluate_fn=evaluate_fn
    )
    
    # --- 策略 3：Filtered + FedProx ---
    strategy_prox = fl.server.strategy.FedProx(
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=len(agents_filtered), min_available_clients=len(agents_filtered),
        evaluate_fn=evaluate_fn, proximal_mu=0.1
    )
    
    # --- 策略 4：Filtered + Krum ---
    # 假設 Server 有被害妄想症，即便只有 6 個好人，還是設定丟掉 1 個最遠的
    strategy_krum = fl.server.strategy.Krum(
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=len(agents_filtered), min_available_clients=len(agents_filtered),
        evaluate_fn=evaluate_fn, 
        num_malicious_clients=1,  # 假定 1 個是極端值
        num_clients_to_keep=5     # 只留 5 個平均
    )

    # === 開始執行實驗 ===
    print(">>> [1/4] 執行 6 人 (FedAvg)...")
    acc_avg = run_simulation(agents_filtered, strategy_avg, NUM_ROUNDS)
    
    print("\n>>> [2/4] 執行 6 人 (FedMedian)...")
    acc_median = run_simulation(agents_filtered, strategy_median, NUM_ROUNDS)
    
    print("\n>>> [3/4] 執行 6 人 (FedProx)...")
    acc_prox = run_simulation(agents_filtered, strategy_prox, NUM_ROUNDS)
    
    print("\n>>> [4/4] 執行 6 人 (Krum)...")
    acc_krum = run_simulation(agents_filtered, strategy_krum, NUM_ROUNDS)

    # === 畫圖 ===
    plt.figure(figsize=(12, 7))
    rounds = range(0, NUM_ROUNDS + 1)
    
    # 畫出四條線
    plt.plot(rounds, acc_avg, label='Ours: 6 Agents (FedAvg)', color='green', marker='o', linewidth=3, markersize=9)
    plt.plot(rounds, acc_median, label='Ablation: 6 Agents (FedMedian)', color='orange', marker='s', linestyle='--', alpha=0.8)
    plt.plot(rounds, acc_prox, label='Ablation: 6 Agents (FedProx)', color='purple', marker='^', linestyle='--', alpha=0.8)
    plt.plot(rounds, acc_krum, label='Ablation: 6 Agents (Krum)', color='brown', marker='d', linestyle='--', alpha=0.8)
    
    # 圖表美化
    plt.title('Ablation Study: Performance of Different Aggregators on Filtered Agents', fontsize=16, fontweight='bold')
    plt.xlabel('Federated Learning Rounds', fontsize=14)
    plt.ylabel('PathMNIST Test Accuracy', fontsize=14)
    plt.xticks(rounds)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12, loc='lower right')
    
    save_path = os.path.join(plot_dir, "fl_ablation_6_agents.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"\n=== 🎉 實驗大功告成！消融實驗比較圖已存至：{save_path} ===")
    plt.show()

def exp():
    device = torch.device("cpu") # 或是 get_device()
    
    # 建立圖表資料夾
    plot_dir = "results_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # 這次我們只關注篩選後的這 6 個人
    agents_filtered = [1, 2, 3, 4, 5, 6]
    run_displacement_profiling(agents_filtered, device)

if __name__ == "__main__":
    # main()
    exp()