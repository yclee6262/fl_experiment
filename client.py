import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from models import Net, get_device

class Method5Client(fl.client.NumPyClient):
    def __init__(self, cid, model_path, anchors, target_class):
        """
        anchors: 現在這是一個 List of Tensors [I1, I2, ..., In]
        """
        self.cid = cid
        self.device = get_device()
        
        self.model = Net().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # 將所有 Anchors 轉為 Tensor 並堆疊起來，方便向量化運算
        # anchors_tensor shape: [N, 1, 3, 28, 28] (假設 batch size=1)
        # 我們需要 squeeze 掉 batch 維度來做運算，變成 [N, 3, 28, 28]
        self.anchors = [a.to(self.device).float() for a in anchors]
        self.target = torch.tensor([target_class], device=self.device).long()
        
    def get_parameters(self, config):
        return []

    def fit(self, parameters, config):
        # parameters 是一個 list，長度為 N (每個 Agent 對應一個係數)
        num_agents = len(self.anchors)
        
        # 1. 處理輸入係數
        if len(parameters) == 0:
            # 初始狀態：全部設為 1/N
            coeffs_np = [1.0 / num_agents] * num_agents
        else:
            coeffs_np = [p.item() for p in parameters]
        
        # 2. 建立可微分的 Tensor 列表
        coeffs_tensor = []
        for c in coeffs_np:
            # 關鍵：dtype=torch.float32 避免報錯
            ct = torch.tensor([c], device=self.device, requires_grad=True, dtype=torch.float32)
            coeffs_tensor.append(ct)
            
        # 3. 計算合成輸入 S' = Sum(w_i * I_i)
        # 這裡用迴圈累加
        synthetic_input = torch.zeros_like(self.anchors[0])
        for i in range(num_agents):
            synthetic_input += coeffs_tensor[i] * self.anchors[i]
            
        # 4. Forward & Backward
        # ---------------------------------------------------------
        # [Loss Function 實作區域]
        # ---------------------------------------------------------
        
        # 1. 取得模型對合成影像的預測 (Forward Pass)
        # 對應法5筆記：計算 f_i(S')
        output = self.model(synthetic_input)
        
        # 2. 定義 Loss Function
        # 這裡我們使用 CrossEntropyLoss，它等同於筆記中衡量「預測與目標差距」的數學實作
        criterion = nn.CrossEntropyLoss()
        
        # 3. 計算 Loss
        # 對應法5筆記：Loss = | f_i(S') - Target |
        # self.target 就是我們設定的 Target Class (例如 0)
        loss = criterion(output, self.target)
        
        # 4. 反向傳播算出梯度 (Backward Pass)
        # 對應法5筆記：計算 dL/d(beta), dL/d(gamma)...
        loss.backward()
        
        # 5. 收集所有係數的梯度
        grads_np = []
        for ct in coeffs_tensor:
            grads_np.append(np.array([ct.grad.item()], dtype=np.float32))
            
        return grads_np, 1, {"loss": loss.item()}

    def evaluate(self, parameters, config):
        num_agents = len(self.anchors)
        if len(parameters) == 0:
            coeffs_np = [1.0 / num_agents] * num_agents
        else:
            coeffs_np = [p.item() for p in parameters]

        synthetic_input = torch.zeros_like(self.anchors[0])
        for i in range(num_agents):
            # Evaluate 不需要 gradient
            c_val = torch.tensor([coeffs_np[i]], device=self.device, dtype=torch.float32)
            synthetic_input += c_val * self.anchors[i]
            
        with torch.no_grad():
            output = self.model(synthetic_input)
            pred = output.argmax(dim=1).item()
            is_correct = (pred == self.target.item())
            
        return float(0.0), 1, {"is_target": int(is_correct), "pred": pred}