import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def generate_agent_dataloaders(num_agents=20, n_samples_per_agent=500, poison_ratio=0.4):
    """
    為每個 Agent 生成獨立的訓練資料。
    poison_ratio: 惡意/異質節點的比例 (預設 0.4，即 40%)
    """
    np.random.seed(42)
    loaders = []
    
    # 計算有多少個 Agent 要扮演「雷隊友」
    num_poisoned = int(num_agents * poison_ratio)
    
    for i in range(num_agents):
        # 模擬 Agent 手上的決策變數 a, b
        a = np.random.uniform(-1, 1, n_samples_per_agent)
        b = np.random.uniform(-1, 1, n_samples_per_agent)
        
        # 標籤 y (正常公式加上微小雜訊)
        noise = np.random.normal(0, 0.05, n_samples_per_agent)
        y = a + a * b + b + noise
        
        # 模擬異質/惡意節點：讓最後幾名的 Agent 資料帶有嚴重偏差
        if i >= num_agents - num_poisoned:
            # 這是錯誤的公式，用來測試 Host 的防禦力
            y = a - b * 5 + noise 
            
        X = np.column_stack((a, b))
        dataset = TabularDataset(X, y)
        loaders.append(DataLoader(dataset, batch_size=32, shuffle=True))
        
    return loaders