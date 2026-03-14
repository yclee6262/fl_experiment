import torch
import torch.nn as nn
import torch.optim as optim
from medmnist import PathMNIST, BloodMNIST
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random
import argparse  # [新增] 參數解析庫
from datetime import datetime
from models import Net, get_device
from torch.utils.data import Subset

# --- 預設設定 (如果沒給參數就用這些) ---
DEFAULT_HOST_ID = 1
DEFAULT_TARGET_CLASS = 0
NUM_AGENTS = 10
# ------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Run Collaborative Selection Simulation")
    parser.add_argument('--host', type=int, default=DEFAULT_HOST_ID, help='指定 Host Agent ID (1-10)')
    parser.add_argument('--target', type=int, default=DEFAULT_TARGET_CLASS, help='指定目標類別 Target Class (e.g., 0)')
    return parser.parse_args()

def inspect_model_classes(model_path):
    """
    [快篩核心] 讀取模型權重檔，偵測輸出層維度
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型檔案: {model_path}")
        
    state_dict = torch.load(model_path, map_location='cpu')
    key_name = 'classifier.2.weight'
    
    if key_name in state_dict:
        return state_dict[key_name].shape[0]
    else:
        keys = [k for k in state_dict.keys() if 'classifier' in k and 'weight' in k]
        if keys: return state_dict[keys[-1]].shape[0]
        else: raise ValueError(f"無法偵測類別數: {model_path}")

def get_tv_loss(img):
    """
    Total Variation Loss: 計算相鄰像素的差異
    這會強迫圖片變得平滑，消除高頻雜訊點
    """
    b, c, h, w = img.size()
    h_tv = torch.pow((img[:, :, 1:, :] - img[:, :, :h-1, :]), 2).sum()
    w_tv = torch.pow((img[:, :, :, 1:] - img[:, :, :, :w-1]), 2).sum()
    return h_tv + w_tv

def generate_proof(agent_model, target_class, device, steps=200): # 增加步數到 200
    agent_model.eval()
    
    # [技巧 1] 不要從純雜訊開始，從「平均灰度」加上一點點雜訊開始
    # 這樣比較容易長出結構
    proof_img = torch.zeros(1, 3, 28, 28, device=device) + 0.1 * torch.randn(1, 3, 28, 28, device=device)
    proof_img.requires_grad_(True)
    
    # [技巧 2] 使用 Adam 優化器，通常比 SGD 收斂得更好
    optimizer = optim.Adam([proof_img], lr=0.05)
    
    criterion = nn.CrossEntropyLoss()
    target = torch.tensor([target_class], device=device).long()

    # 這會強迫模型畫出具備 "平移不變性" 的特徵 (即：真正的結構)
    jitter = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # 上下左右隨機動 10%
    ])
    
    # 權重設定 (需要微調)
    lambda_tv = 0.00005  # TV Loss 權重 (控制平滑度)
    lambda_l2 = 0.03   # L2 Regularization (控制像素值不要爆掉)
    
    for i in range(steps):
        optimizer.zero_grad()

        jittered_img = jitter(proof_img)
        output = agent_model(jittered_img)
        
        # 1. 分類 Loss (原本的目標)
        class_loss = criterion(output, target)
        
        # 2. TV Loss (平滑化)
        tv_loss = get_tv_loss(proof_img)
        
        # 3. L2 Loss (防止像素值過大或過小)
        l2_loss = torch.norm(proof_img, 2)
        
        # 總 Loss
        total_loss = class_loss + lambda_tv * tv_loss + lambda_l2 * l2_loss
        
        total_loss.backward()
        optimizer.step()
        
        # [技巧 3] 偶爾將圖片數值 clip 回合理範圍 (-1~1 或 0~1)
        # 這裡我們不做硬性 clip，讓 L2 去控制
        # [保護] 每一輪都把數值夾回合理範圍 (例如 -1 ~ 1 之間)
        with torch.no_grad():
            proof_img.data.clamp_(-1.5, 1.5) # 放寬一點範圍防止飽和
            
    return proof_img.detach()

def generate_proof_dummy_layer(agent_model, anchors, target_class, device, steps=200):
    """
    [Candidate 行為 - 虛設層法 (Method 5)]
    利用 Agent 私有的 K 張圖片 (anchors)，學習一組權重 (coeffs) 來合成證明圖。
    """
    agent_model.eval()
    K = anchors.size(0) # 假設 Agent 有 K 張私有圖片
    
    # 1. 初始化虛設層權重 (Coefficients)
    # 初始值設為 1/K，代表一開始是所有圖片的平均混合
    coeffs = torch.ones(K, device=device, requires_grad=True)
    with torch.no_grad():
        coeffs.data = coeffs.data / K
        
    # 2. 優化器 [關鍵]：我們只優化這 K 個數字，不優化整張圖片的像素
    optimizer = optim.Adam([coeffs], lr=0.05)
    criterion = nn.CrossEntropyLoss()
    target = torch.tensor([target_class], device=device).long()
    
    for _ in range(steps):
        optimizer.zero_grad()
        
        # 3. 虛設層的前向傳播 (Forward Pass)
        # 將權重與圖片進行線性組合 (Linear Combination)
        # coeffs shape: [K] -> [K, 1, 1, 1] 為了與圖片 [K, 3, 28, 28] 進行廣播相乘
        w = coeffs.view(K, 1, 1, 1)
        proof_img = torch.sum(w * anchors, dim=0, keepdim=True) # 輸出 [1, 3, 28, 28]
        
        # 丟進模型預測
        output = agent_model(proof_img)
        
        # 4. 計算誤差
        class_loss = criterion(output, target)
        
        # 加入一點 L2 正則化，防止某個權重無限飆高 (避免畫面過度曝光)
        l2_reg = 0.01 * torch.norm(coeffs, 2)
        total_loss = class_loss + l2_reg
        
        # 5. 反向傳播更新虛設層權重
        total_loss.backward()
        optimizer.step()
        
    # 6. 迴圈結束，產出最終的合成圖
    with torch.no_grad():
        w = coeffs.view(K, 1, 1, 1)
        final_proof = torch.sum(w * anchors, dim=0, keepdim=True)
        final_proof.clamp_(-1, 1) # 確保像素值在合理範圍 (假設經過 Normalize)
        
    return final_proof.detach()

def get_agent_anchors_random(detected_classes, device, K=5):
    """
    [真實模擬] 每個 Agent 隨機從自己的本地資料庫中抽出 K 張不重複的圖片當作 Anchor
    """
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]) # 必須與預訓練時的正規化一致
    ])
    
    # 1. 根據維度判斷該讀取哪個本地資料庫
    if detected_classes == 9:
        dataset = PathMNIST(split='train', transform=data_transform, download=False)
    else:
        dataset = BloodMNIST(split='train', transform=data_transform, download=False)
        
    dataset_size = len(dataset)
    
    # 2. 從 0 到 dataset_size-1 之間，隨機抽出 K 個「不重複」的 index
    random_indices = random.sample(range(dataset_size), K)
    
    anchors = []
    # 3. 根據抽出的 index 取出圖片
    for idx in random_indices:
        img, _ = dataset[idx] # 我們不需要知道這張圖的真實 label，只要圖就好
        anchors.append(img)
        
    return torch.stack(anchors).to(device) # 回傳 shape: [K, 3, 28, 28]

def get_agent_best_anchors(agent_id, detected_classes, target_class, device, K=10):
    """
    [絕對隔離版] Agent 只能從自己預訓練時分配到的專屬資料(Subset)中挑選圖片。
    """
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    # 1. 載入原始巨型資料集 (先不載入記憶體，只是取得物件)
    if detected_classes == 9:
        full_dataset = PathMNIST(split='train', transform=data_transform, download=False)
    else:
        full_dataset = BloodMNIST(split='train', transform=data_transform, download=False)
        
    # 2. 讀取該 Agent 的專屬「地契 (Indices)」
    indices_path = f"saved_models/agent_{agent_id}_indices.pt"
    if not os.path.exists(indices_path):
        raise FileNotFoundError(f"找不到 Agent {agent_id} 的專屬資料索引檔！請重新執行 pretrain_multiverse.py")
        
    agent_indices = torch.load(indices_path)
    
    # 3. 建立專屬於該 Agent 的本地資料集 (Local Dataset)
    # 從此刻起，它絕對無法存取別人的資料
    local_dataset = Subset(full_dataset, agent_indices)
    
    anchors = []
    
    # 4. 在「自己的」資料庫中尋找符合 target_class 的精華圖片
    # 打亂順序確保隨機性
    local_indices = list(range(len(local_dataset)))
    random.shuffle(local_indices)
    
    for idx in local_indices:
        img, label = local_dataset[idx]
        if int(label[0]) == target_class:
            anchors.append(img)
            if len(anchors) == K:
                break
                
    # 保護機制：如果它自己的小資料庫裡，剛好連 K 張 Target Class 的圖都湊不齊？
    if len(anchors) < K:
        print(f"\n      [警告] Agent {agent_id} 本地庫僅有 {len(anchors)} 張目標類別，使用雜訊填補...", end="")
        while len(anchors) < K:
            anchors.append(torch.zeros(3, 28, 28) + 0.1 * torch.randn(3, 28, 28))

    return torch.stack(anchors).to(device)

def host_evaluate(host_model, proof_img):
    host_model.eval()
    with torch.no_grad():
        output = host_model(proof_img)
        probs = torch.softmax(output, dim=1)
        conf, pred = probs.max(dim=1)
    return pred.item(), conf.item()

def main():
    # 1. 解析參數
    args = get_args()
    HOST_ID = args.host
    TARGET_CLASS = args.target
    
    device = get_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 資料夾名稱加入 Host 資訊，方便區分
    save_dir = f"selection_results/exp_{timestamp}_Host{HOST_ID}_Target{TARGET_CLASS}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"=== 實驗開始: Host=Agent {HOST_ID}, Target=Class {TARGET_CLASS} ===")

    # 2. Host 載入自己的模型
    host_path = f"saved_models/agent_{HOST_ID}.pth"
    try:
        host_classes = inspect_model_classes(host_path)
    except Exception as e:
        print(f"錯誤: Host 模型無法讀取 ({e})")
        return

    # [Host 自檢] 檢查 Host 自己有沒有能力辨識 Target Class
    if TARGET_CLASS >= host_classes:
        print(f"錯誤: Host (Capacity {host_classes}) 無法辨識 Target Class {TARGET_CLASS}。實驗終止。")
        return

    host_model = Net(num_classes=host_classes).to(device)
    host_model.load_state_dict(torch.load(host_path, map_location=device))
    print(f"Host 模型載入成功 (Capacity: {host_classes} classes)")
    
    results = []
    
    # 3. 遍歷所有候選人
    for i in range(1, NUM_AGENTS + 1):
        model_path = f"saved_models/agent_{i}.pth"
        
        # [A] 動態偵測
        try:
            detected_classes = inspect_model_classes(model_path)
        except Exception as e:
            print(f"  Agent {i}: 模型結構錯誤 ({e}) -> 跳過")
            continue

        print(f"  Agent {i} (Cap: {detected_classes})...", end=" ")

        # [B] 快篩機制 (Fast Reject)
        if TARGET_CLASS >= detected_classes:
            print(f"-> [淘汰] 認知範圍不足")
            results.append({
                "id": i,
                "type": "Dim Mismatch",
                "img": torch.zeros(1, 3, 28, 28),
                "pred": -1,
                "conf": 0.0,
                "status": "REJECT (DIM)",
                "n_classes": detected_classes
            })
            continue 

        # [C] 生成證明
        candidate_model = Net(num_classes=detected_classes).to(device)
        candidate_model.load_state_dict(torch.load(model_path, map_location=device))
        
        print(f"  Agent {i} (Cap: {detected_classes})...", end=" ")

        # [修改] 使用隨機抽樣函數，抽出 K=5 張圖 (可根據算力調整 K，例如 K=10 組合能力更強)
        anchors = get_agent_best_anchors(i, detected_classes, TARGET_CLASS, device, K=10)

        # 呼叫虛設層法 (Method 5) 來生成證明圖
        # 將這 K 張隨機圖片交給演算法，讓它自己找出一組最佳混合係數
        proof_img = generate_proof_dummy_layer(candidate_model, anchors, TARGET_CLASS, device)
        
        # Host 驗證
        pred, conf = host_evaluate(host_model, proof_img)
        
        # 判定
        is_accepted = (pred == TARGET_CLASS) and (conf > 0.8)
        status = "ACCEPT" if is_accepted else "REJECT"
        print(f"-> Host: C{pred} ({conf:.2f}) -> {status}")
        
        # 標記類型
        agent_type = "PathMNIST" if detected_classes == 9 else "BloodMNIST"
        if i == HOST_ID: agent_type = "Host (Self)" # [修改] 動態標記 Host 自己
        
        results.append({
            "id": i,
            "type": agent_type,
            "img": proof_img.cpu(),
            "pred": pred,
            "conf": conf,
            "status": status,
            "n_classes": detected_classes
        })

    # 4. 視覺化結果
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, res in enumerate(results):
        ax = axes[idx]
        
        # 處理圖片顯示
        img = res['img'].squeeze().permute(1, 2, 0)
        if "DIM" in res['status']: # 維度不合顯示全黑
            img = torch.zeros_like(img)
            cmap = 'gray'
        else:
            img = torch.clamp(img * 0.5 + 0.5, 0, 1)
            cmap = None
        
        ax.imshow(img, cmap=cmap)
        
        # 設定顏色
        if res['status'] == "ACCEPT": color = 'green'
        elif "DIM" in res['status']: color = 'gray'
        else: color = 'red'
        
        # 標題
        title = f"Agent {res['id']} [{res['n_classes']}c]\n"
        if res['type'] == "Host (Self)":
            title += "[ HOST ]\n" # 特別標註
        elif res['pred'] != -1:
            title += f"Host: C{res['pred']} ({res['conf']:.2f})\n"
        else:
            title += "Dim Mismatch\n"
            
        title += f"{res['status']}"
        
        ax.set_title(title, fontsize=9, color=color, fontweight='bold')
        ax.axis('off')
        
    plt.tight_layout()
    plt.suptitle(f"Selection by Host {HOST_ID} (Target Class {TARGET_CLASS})", fontsize=16, y=1.02)
    
    save_path = f"{save_dir}/dashboard_H{HOST_ID}_T{TARGET_CLASS}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"=== 實驗完成！結果已存至 {save_path} ===")
    plt.show()

if __name__ == "__main__":
    main()