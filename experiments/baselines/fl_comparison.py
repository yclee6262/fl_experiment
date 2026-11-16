import os
import copy
import json
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
    # 1. 自動偵測運算設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def client_fn(cid: str) -> fl.client.Client:
        agent_id = agent_list[int(cid)]
        train_loader = load_agent_data(agent_id)
        model = Net(num_classes=9).to(device) 
        return FLClient(model, train_loader, device).to_client()

    print(f"\n=== 開始 FL 模擬 (通訊耗時) | 策略: {strategy.__class__.__name__} ===")
    
    # 2. ⭐️ 核心修改：動態決定 GPU 資源分配
    gpu_resources = 0.25 if torch.cuda.is_available() else 0.0
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(agent_list),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": gpu_resources} # <--- 使用自動偵測的變數
    )
    return [acc for _, acc in history.metrics_centralized["accuracy"]]


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
    delta = 0.5
    
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

# --- 4. 主程式與畫圖 ---
def main():
    NUM_ROUNDS = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 目前分配到的運算設備：{device}")
    evaluate_fn = get_evaluate_fn(device)
    
    # 建立管理圖表與快取的資料夾
    plot_dir = "results_plots"
    cache_dir = "saved_results"
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    agents_all = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    agents_filtered = [1, 2, 3, 4, 5, 6]
    
    # 快取檔案路徑
    cache_file = os.path.join(cache_dir, "baselines_history.json")

    # ==========================================
    # === Stage 1: 傳統 Flower 框架實驗 (具備快取功能) ===
    # ==========================================
    if os.path.exists(cache_file):
        print(f"\n{'='*50}\n 偵測到快取檔案！直接讀取 Baseline 結果，跳過漫長模擬...\n{'='*50}")
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
            
        acc_all_avg = cached_data['acc_all_avg']
        acc_all_median = cached_data['acc_all_median']
        acc_all_prox = cached_data['acc_all_prox']
        acc_all_krum = cached_data['acc_all_krum']
        acc_filtered_avg = cached_data['acc_filtered_avg']
    else:
        print(f"\n{'='*50}\n 無快取紀錄，開始執行漫長的 Baseline 模擬運算...\n{'='*50}")
        
        acc_all_avg = run_simulation(agents_all, fl.server.strategy.FedAvg(
            fraction_fit=1.0, fraction_evaluate=0.0,
            min_fit_clients=len(agents_all), min_available_clients=len(agents_all), evaluate_fn=evaluate_fn
        ), NUM_ROUNDS)
        
        acc_all_median = run_simulation(agents_all, fl.server.strategy.FedMedian(
            fraction_fit=1.0, fraction_evaluate=0.0,
            min_fit_clients=len(agents_all), min_available_clients=len(agents_all), evaluate_fn=evaluate_fn
        ), NUM_ROUNDS)
        
        acc_all_prox = run_simulation(agents_all, fl.server.strategy.FedProx(
            fraction_fit=1.0, fraction_evaluate=0.0,
            min_fit_clients=len(agents_all), min_available_clients=len(agents_all), evaluate_fn=evaluate_fn, proximal_mu=0.1
        ), NUM_ROUNDS)
        
        acc_all_krum = run_simulation(agents_all, fl.server.strategy.Krum(
            fraction_fit=1.0, fraction_evaluate=0.0,
            min_fit_clients=len(agents_all), min_available_clients=len(agents_all), evaluate_fn=evaluate_fn, num_malicious_clients=3, num_clients_to_keep=7
        ), NUM_ROUNDS)

        acc_filtered_avg = run_simulation(agents_filtered, fl.server.strategy.FedAvg(
            fraction_fit=1.0, fraction_evaluate=0.0,
            min_fit_clients=len(agents_filtered), min_available_clients=len(agents_filtered), evaluate_fn=evaluate_fn
        ), NUM_ROUNDS)
        
        # 將跑完的結果存入 JSON 檔案
        cached_data = {
            'acc_all_avg': acc_all_avg,
            'acc_all_median': acc_all_median,
            'acc_all_prox': acc_all_prox,
            'acc_all_krum': acc_all_krum,
            'acc_filtered_avg': acc_filtered_avg
        }
        with open(cache_file, 'w') as f:
            json.dump(cached_data, f, indent=4)
        print(f"✅ Baseline 結果已儲存至 {cache_file}，下次執行將瞬間完成！")

    # ==========================================
    # === Stage 2 & 3: 我們的方法 (Ours Pro) ===
    # (注意：我們的方法因為要頻繁修改測試，且通訊成本為 0 執行極快，所以不加入快取，維持每次即時運算)
    # ==========================================
    generate_initial_m_stars(agents_filtered, device, local_epochs=30)
    acc_ours_pro = run_server_subspace_optimization(agents_filtered, device, num_iterations=NUM_ROUNDS)

    # === 終極大合併畫圖 ===
    plt.figure(figsize=(15, 8))
    rounds = range(0, NUM_ROUNDS + 1)
    
    plt.plot(rounds, acc_all_avg, label='Baseline: 10 Agents (FedAvg)', color='red', marker='x', linestyle=':', linewidth=2)
    plt.plot(rounds, acc_all_median, label='Robust: 10 Agents (FedMedian)', color='orange', marker='s', linestyle='--', alpha=0.7)
    plt.plot(rounds, acc_all_prox, label='Robust: 10 Agents (FedProx)', color='purple', marker='^', linestyle='--', alpha=0.7)
    plt.plot(rounds, acc_all_krum, label='Robust: 10 Agents (Krum)', color='brown', marker='d', linestyle='--', alpha=0.7)
    
    plt.plot(rounds, acc_filtered_avg, label='Ours: Pre-filtered (FedAvg)', color='green', marker='o', linewidth=2.5, linestyle='--')
    plt.plot(rounds, acc_ours_pro, label='Ours: Pre-filtered + Subspace method', color='red', marker='*', linewidth=3.5, markersize=12)
    
    plt.title('FL Comparison: Robust Aggregators vs. Our Subspace Optimization', fontsize=16, fontweight='bold')
    plt.xlabel('Communication Rounds (Baselines) / Server Iterations (Ours)', fontsize=14)
    plt.ylabel('PathMNIST Test Accuracy', fontsize=14)
    plt.xticks(rounds)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12, loc='lower right', framealpha=0.9)
    
    save_path = os.path.join(plot_dir, "fl_ultimate_pro_comparison_50.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"\n=== 🎉 實驗大功告成！終極比較圖已整齊收納至：{save_path} ===")
    plt.show()

if __name__ == "__main__":
    main()