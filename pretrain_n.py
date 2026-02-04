# filename: pretrain_n.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from medmnist import PathMNIST
from models import Net, get_device

# --- 設定擴展規模 ---
NUM_CLIENTS = 10
# ------------------

def train_model(agent_id, train_loader):
    print(f"[Agent {agent_id}] 開始訓練 Local Engine...")
    device = get_device()
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3): # 保持輕量訓練
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.squeeze().long().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    save_path = f"saved_models/agent_{agent_id}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"  -> 模型已儲存: {save_path}")

def main():
    os.makedirs("saved_models", exist_ok=True)
    device = get_device()
    
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    train_dataset = PathMNIST(split='train', transform=data_transform, download=True)
    
    # 1. 動態切分資料集
    total_len = len(train_dataset)
    lengths = [total_len // NUM_CLIENTS] * NUM_CLIENTS
    # 處理餘數，加到最後一份
    lengths[-1] += total_len - sum(lengths)
    
    datasets = random_split(train_dataset, lengths)
    
    # 2. 迴圈訓練 N 個模型
    for i in range(NUM_CLIENTS):
        loader = DataLoader(datasets[i], batch_size=32, shuffle=True)
        # Agent ID 從 1 開始編號
        train_model(i + 1, loader)

if __name__ == "__main__":
    main()