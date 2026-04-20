import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def generate_agent_dataloaders(num_agents=20, n_samples_per_agent=500, poison_ratio=0.4, n_features=2):
    """
    生成 Agent 訓練用的模擬數據
    ⭐ 加入 n_features 參數，並動態切換物理公式
    """
    np.random.seed(42)
    loaders = []
    num_poisoned = int(num_agents * poison_ratio)
    
    print(f"\n[環境設定] 決策變數維度: {n_features}D")
    print(f"[真實函數] y = sum(x_i) + sum(x_i * x_i+1)")
    print("-" * 50)
    
    for i in range(num_agents):
        # 1. 隨機生成 N 維的特徵陣列 X
        X = np.random.uniform(-1, 1, (n_samples_per_agent, n_features))
        
        # 2. 泛化 N 維物理公式: 所有特徵相加 + 相鄰特徵相乘
        y = np.sum(X, axis=1) 
        if n_features > 1:
            y += np.sum(X[:, :-1] * X[:, 1:], axis=1)
            
        noise = np.random.normal(0, 0.05, n_samples_per_agent)
        y = y + noise
        
        # 3. 模擬惡意節點 (毒化節點公式也做泛化)
        if i >= num_agents - num_poisoned:
            y = -3 * np.sum(X, axis=1)
            if n_features > 1:
                y += 2 * np.sum(X[:, :-1] ** 2, axis=1)
            y += noise
                
        dataset = TabularDataset(X, y)
        loaders.append(DataLoader(dataset, batch_size=32, shuffle=True))
        
    return loaders