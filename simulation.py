import flwr as fl
import torch
import json
from datetime import datetime
import random
import os
import matplotlib.pyplot as plt  # [新增] 繪圖庫
from medmnist import PathMNIST
import torchvision.transforms as transforms
from client import Method5Client
from strategy import Method5Strategy
from models import get_device

# --- 設定 ---
NUM_CLIENTS = 5
TARGET_CLASS = 0
NUM_ROUNDS = 100
LR = 0.01
# -----------

def plot_loss_curve(history, save_dir):
    """
    從 Flower 的 History 物件中提取數據並繪製 Loss Curve
    """
    # 1. 提取數據
    # 在你的 strategy 中，loss 是透過 metrics 回傳的，所以存在 metrics_distributed_fit 裡
    # 格式: [(round, value), (round, value), ...]
    try:
        loss_data = history.metrics_distributed_fit["avg_loss"]
        rounds, losses = zip(*loss_data)
        
        # 2. 繪圖
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, losses, marker='o', linestyle='-', color='b', label='Training Loss')
        
        plt.title(f"Method 5 Convergence (Target: {TARGET_CLASS})")
        plt.xlabel("Communication Rounds")
        plt.ylabel("Loss (Avg)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 3. 存檔
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "loss_curve.png")
        plt.savefig(save_path, dpi=300)
        print(f"--- [System] Loss Curve 已儲存至 {save_path} ---")
        
        # 額外畫一張 Log Scale (對數座標) 的圖，方便觀察後期的收斂細節
        plt.yscale('log')
        save_path_log = os.path.join(save_dir, "loss_curve_log.png")
        plt.savefig(save_path_log, dpi=300)
        # print(f"--- [System] Log Scale Curve 已儲存至 {save_path_log} ---")
        
        plt.close()
        
    except KeyError:
        print("警告：無法在 History 中找到 'avg_loss'，請檢查 Strategy 回傳的 metrics key 是否正確。")

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiment_results/exp_{timestamp}_N{NUM_CLIENTS}_T{TARGET_CLASS}"
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"=== 本次實驗資料夾: {experiment_dir} ===")

    # 1. 準備資料集
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    test_dataset = PathMNIST(split='test', transform=data_transform, download=True)
    
    # 2. 隨機抽取 N 個不重複的索引
    total_samples = len(test_dataset)
    anchor_indices = random.sample(range(total_samples), NUM_CLIENTS)
    
    print(f"本次實驗隨機抽取的 Anchor 索引: {anchor_indices}")

    # 3. 存檔設定
    config_data = {
        "anchor_indices": anchor_indices,
        "target_class": TARGET_CLASS,
        "num_clients": NUM_CLIENTS,
        "num_rounds": NUM_ROUNDS
    }
    config_path = os.path.join(experiment_dir, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(config_data, f)
    print(f"--- [System] 設定檔已存至 {config_path} ---")

    # 4. 取出影像
    anchors = []
    for idx in anchor_indices:
        img, label = test_dataset[idx]
        anchors.append(img.unsqueeze(0))
        print(f"  Anchor ID {idx}: Class {label.item()}")

    # 5. Client 生成函數
    def client_fn(cid: str) -> fl.client.Client:
        agent_id = int(cid) + 1
        model_path = f"saved_models/agent_{agent_id}.pth"
        return Method5Client(cid, model_path, anchors, TARGET_CLASS).to_client()

    # 6. 設定策略
    strategy = Method5Strategy(
        num_clients=NUM_CLIENTS,
        learning_rate=LR, # learning rate
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        log_dir=experiment_dir
    )

    print(f"開始 Flower 模擬 (N={NUM_CLIENTS})... Target Class: {TARGET_CLASS}")
    
    # 接收回傳的 history 物件
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1}
    )
    
    # 7. 繪製 Loss Curve
    plot_loss_curve(hist, experiment_dir)

if __name__ == "__main__":
    main()