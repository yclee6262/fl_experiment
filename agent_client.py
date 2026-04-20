import torch
import torch.nn as nn
import torch.optim as optim
from models import PretrainedModel, AgentReverseModel, get_device

class AgentNode:
    def __init__(self, agent_id, dataloader, n_features=2):
        self.agent_id = agent_id
        self.device = get_device()
        self.n_feature = n_features
        self.model = PretrainedModel(in_features=n_features).to(self.device)
        self.dataloader = dataloader

    def train_local_model(self, epochs=50):
        """Phase 0: 訓練本地黑箱模型"""
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            for X_batch, y_batch in self.dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X_batch), y_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()
        self.model.eval()

    def api_predict(self, X_array):
        """Phase 1 & 3: 開放給 Host 呼叫的黑箱預測 API"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_array, dtype=torch.float32).to(self.device)
            # 支援 batch 預測或單筆預測
            if X_tensor.dim() == 1:
                X_tensor = X_tensor.unsqueeze(0)
            return self.model(X_tensor).cpu().numpy().flatten()

    def infer_parameters_C(self, target_T, steps=500, n_features=2):
        """Phase 2: 虛設層反推 (Method C)"""
        # 1. 根據 n_features 動態初始化
        init_guess = torch.randn(self.n_feature).to(self.device) 
        reverse_model = AgentReverseModel(self.model, init_guess).to(self.device)
        
        optimizer = optim.Adam(reverse_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # 2. 動態長度的全 1 向量
        raw_input = torch.ones(self.n_feature).to(self.device) 
        target_tensor = torch.tensor([target_T], dtype=torch.float32).to(self.device)

        for _ in range(steps):
            optimizer.zero_grad()
            output = reverse_model(raw_input)
            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                reverse_model.arc.clamp_(-1.0, 1.0)
                
        return reverse_model.arc.detach().cpu().numpy()
    
    def infer_parameters_D(self, target_T, steps=500, n_features=2):
        """Phase 2: 方法 D - 直接輸入梯度反推"""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 根據 n_features 動態生成 input_tensor
        input_tensor = torch.randn(1, self.n_feature, device=self.device, requires_grad=True)
        
        # 建立優化器，直接把 input_tensor 交給優化器去更新！
        optimizer = optim.Adam([input_tensor], lr=0.01)
        criterion = nn.MSELoss()
        target_tensor = torch.tensor([[target_T]], dtype=torch.float32).to(self.device)

        # 梯度下降優化迴圈
        for _ in range(steps):
            optimizer.zero_grad()
            
            # 前向傳播：把可微的輸入丟進黑箱模型
            output = self.model(input_tensor)
            
            # 計算誤差
            loss = criterion(output, target_tensor)
            
            # 反向傳播：計算 Loss 對 input_tensor 的梯度
            loss.backward()
            
            # 更新 input_tensor 本身的數值
            optimizer.step()
            
            # [工程防護技巧] 確保參數在合理範圍 (例如 -1 到 1)
            # 因為直接優化輸入很容易衝出物理邊界，我們在不影響梯度的情況下把它拉回來
            with torch.no_grad():
                input_tensor.clamp_(-1.0, 1.0)
                
        # 回傳優化完成的決策變數 (I_i)
        return input_tensor.detach().cpu().numpy().flatten()