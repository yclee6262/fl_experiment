import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from medmnist import PathMNIST, BloodMNIST
from models import Net, get_device

# --- 設定 ---
# Agent 1-6: PathMNIST (同源)
# Agent 7-10: BloodMNIST (異源/雜訊)
# -----------

def train_agent(agent_id, dataset, num_classes, epochs=5):
    device = get_device()
    model = Net(num_classes=num_classes).to(device)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    print(f"--- [Agent {agent_id}] Training on {type(dataset).__name__} ---")
    model.train()
    for epoch in range(epochs):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device).long().squeeze()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
    save_path = f"saved_models/agent_{agent_id}.pth"
    torch.save(model.state_dict(), save_path)

def main():
    os.makedirs("saved_models", exist_ok=True)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    # 1. 準備兩種資料
    print("下載資料集中...")
    path_data = PathMNIST(split='train', transform=data_transform, download=True)
    blood_data = BloodMNIST(split='train', transform=data_transform, download=True)
    
    # 設定亂數種子，確保每次切分結果固定
    generator = torch.Generator().manual_seed(42)
    
    # 將 PathMNIST (89,996 張) 切成 6 份
    path_len = len(path_data)
    path_lengths = [path_len // 6] * 6
    path_lengths[-1] += path_len % 6 # 把除不盡的餘數加給最後一個
    path_subsets = random_split(path_data, path_lengths, generator=generator)
    
    # 將 BloodMNIST (11,959 張) 切成 4 份
    blood_len = len(blood_data)
    blood_lengths = [blood_len // 4] * 4
    blood_lengths[-1] += blood_len % 4
    blood_subsets = random_split(blood_data, blood_lengths, generator=generator)

    # --- [關鍵修改 2]：訓練並儲存專屬的 Index 檔案 ---
    # 訓練 Group A (PathMNIST, Agent 1~6)
    for i in range(6):
        agent_id = i + 1
        subset = path_subsets[i]
        
        # 儲存這個 Agent 專屬的資料索引！
        torch.save(subset.indices, f"saved_models/agent_{agent_id}_indices.pt")
        
        # 注意：現在 train_agent 吃到的是 subset，他只會用這 1/6 的資料訓練
        train_agent(agent_id, subset, num_classes=9) 
        
    # 訓練 Group B (BloodMNIST, Agent 7~10)
    for i in range(4):
        agent_id = i + 7
        subset = blood_subsets[i]
        
        # 儲存專屬索引
        torch.save(subset.indices, f"saved_models/agent_{agent_id}_indices.pt")
        
        # 用他專屬的 1/4 資料訓練
        train_agent(agent_id, subset, num_classes=8)

if __name__ == "__main__":
    main()