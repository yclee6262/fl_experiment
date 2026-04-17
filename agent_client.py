import torch
import torch.nn as nn
import torch.optim as optim
from models import PretrainedModel, AgentReverseModel, get_device

class AgentNode:
    def __init__(self, agent_id, dataloader):
        self.agent_id = agent_id
        self.device = get_device()
        self.model = PretrainedModel().to(self.device)
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
                loss = criterion(self.model(X_batch), y_batch)
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

    def infer_parameters_C(self, target_T, steps=500):
        """Phase 2: 虛設層反推 (Method C)"""
        init_guess = torch.randn(2).to(self.device) # 假設 2 個特徵
        reverse_model = AgentReverseModel(self.model, init_guess).to(self.device)
        
        optimizer = optim.Adam(reverse_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        raw_input = torch.tensor([1.0, 1.0]).to(self.device) # 永遠全 1
        target_tensor = torch.tensor([target_T], dtype=torch.float32).to(self.device)

        for _ in range(steps):
            optimizer.zero_grad()
            output = reverse_model(raw_input)
            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()
            
        return reverse_model.arc.detach().cpu().numpy()
    
    def infer_parameters_D(self, target_T, steps=500):
        """Phase 2: 方法 D - 直接輸入梯度反推 (Input Gradient Optimization)"""
        
        # 1. 確保本地預訓練模型被完全凍結，且處於 eval 模式
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 2. 宣告『輸入變數 (Input Tensor)』，並開啟 requires_grad=True
        # 假設有 2 個決策變數 (a, b)，我們給一個隨機的初始起點
        # 注意：形狀要是 [1, 2] 以符合神經網路 batch 輸入的要求
        input_tensor = torch.randn(1, 2, device=self.device, requires_grad=True)
        
        # 3. 建立優化器，⭐ 關鍵：直接把 input_tensor 交給優化器去更新！
        optimizer = optim.Adam([input_tensor], lr=0.01)
        criterion = nn.MSELoss()
        target_tensor = torch.tensor([[target_T]], dtype=torch.float32).to(self.device)

        # 4. 梯度下降優化迴圈
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
                
        # 5. 回傳優化完成的決策變數 (I_i)
        return input_tensor.detach().cpu().numpy().flatten()