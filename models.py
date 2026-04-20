import torch
import torch.nn as nn

# --- 1. 本地預訓練模型 (取代原本的 CNN) ---
class PretrainedModel(nn.Module):
    def __init__(self, in_features=2):
        super(PretrainedModel, self).__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1) # 輸出目標 y

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- 2. 虛設層反推模型 ---
class AgentReverseModel(nn.Module):
    def __init__(self, pretrained_model, init_guess):
        super(AgentReverseModel, self).__init__()
        self.dummy = nn.Identity()
        # arc 就是我們要反推的決策變數 (例如 a, b)
        self.arc = nn.Parameter(init_guess)
        self.pretrained = pretrained_model

        # 凍結預訓練模型的權重，保證隱私且只優化 arc
        for param in self.pretrained.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.dummy(x)
        x = x * self.arc  # 全 1 向量乘上待優化的參數
        x = self.pretrained(x)
        return x

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")