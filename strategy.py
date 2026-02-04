import flwr as fl
import json
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np
import os

class Method5Strategy(fl.server.strategy.FedAvg):
    def __init__(self, num_clients, learning_rate=0.1, log_dir=".", **kwargs):
        super().__init__(**kwargs)
        self.num_clients = num_clients
        self.lr = learning_rate
        # 初始化 N 個係數，平均分配
        self.current_coeffs = [1.0 / num_clients] * num_clients
        self.log_dir = log_dir

    def initialize_parameters(self, client_manager):
        print(f"--- [Server] 初始化 {self.num_clients} 個係數 ---")
        return ndarrays_to_parameters([np.array([c]) for c in self.current_coeffs])

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # results[0].parameters 裡的長度應該等於 num_clients
        # 我們要把每個 client 回傳的 "N個梯度" 分別加總起來
        
        # 初始化梯度累加器
        sum_grads = [0.0] * self.num_clients
        num_participants = len(results)

        for _, fit_res in results:
            # client_grads 是一個 list of arrays, e.g., [[g1], [g2], ..., [gn]]
            client_grads = parameters_to_ndarrays(fit_res.parameters)
            
            for i in range(self.num_clients):
                sum_grads[i] += client_grads[i][0]

        # 計算平均梯度並更新
        print(f"--- [Server] Round {server_round} 更新 ---")
        log_str = "Coeffs: "
        
        for i in range(self.num_clients):
            avg_grad = sum_grads[i] / num_participants
            self.current_coeffs[i] -= self.lr * avg_grad
            log_str += f"{self.current_coeffs[i]:.2f} "
            
        print(log_str)

        # 打包回傳
        new_params = ndarrays_to_parameters([np.array([c]) for c in self.current_coeffs])
        
        losses = [r.metrics["loss"] for _, r in results]
        avg_loss = sum(losses) / len(losses)

        if server_round == 100: # 視你的總回合數而定
            # 確保資料夾存在
            os.makedirs(self.log_dir, exist_ok=True)
            save_path = os.path.join(self.log_dir, "last_run_coeffs.json")
            
            coeffs_to_save = [float(c) for c in self.current_coeffs]
            with open(save_path, "w") as f:
                json.dump(coeffs_to_save, f)
            print(f"--- [Server] 係數已儲存至 {save_path} ---")
        
        return new_params, {"avg_loss": avg_loss}