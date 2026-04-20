from dataset import generate_agent_dataloaders
from agent_client import AgentNode
from host_server import HostServer
import numpy as np

def main():
    TARGET_T = 1.5
    NUM_AGENTS = 40      # 擴大參與人數！(可自由更改為 30, 50 等)
    POISON_RATIO = 0.1   # 設定 40% 的人是異質節點
    N = 5                # 設定是幾元多項式
    
    print(f"=== Phase 0: 準備資料與訓練本地神經網路 (共 {NUM_AGENTS} 個 Agent) ===")
    # 傳入設定的參數
    loaders = generate_agent_dataloaders(num_agents=NUM_AGENTS, poison_ratio=POISON_RATIO, n_features=N)
    
    all_agents = []
    for i in range(NUM_AGENTS):
        agent = AgentNode(agent_id=i+1, dataloader=loaders[i], n_features=N)
        print(f"訓練 Agent {i+1} 中...")
        agent.train_local_model(epochs=30)
        all_agents.append(agent)
        
    print("\n=== 開始去中心化反推演算法 ===")
    server = HostServer(target_T=TARGET_T, n_features=N)
    
    # Phase 1: 考試過濾
    server.phase1_filter_agents(all_agents)
    
    # Phase 2: 虛設層反推
    server.phase2_collect_proposals()
    
    # Phase 3: 黑箱最佳化 (雙引擎對決！)
    print("\n" + "="*50)
    print("🏆 尋路引擎對決開始！目標 T = {}".format(TARGET_T))
    print("="*50)
    
    # 引擎 1：工業級 SciPy BFGS
    final_S_bfgs = server.phase3_global_optimization()
    
    # 引擎 2：割線/切線法
    final_S_custom = server.phase3_custom_secant_optimization(num_iterations=30)
    
    # === 驗證與比較結果 ===
    print("\n=== 最終對比結果 ===")
    def get_formula_string(n):
        linear_terms = [f"x{i}" for i in range(n)]
        cross_terms = [f"(x{i}*x{i+1})" for i in range(n-1)]
        return "y = " + " + ".join(linear_terms + cross_terms)
        
    formula_str = get_formula_string(N)

    def calculate_true_y(S):
        y_val = np.sum(S)
        if len(S) > 1:
            y_val += np.sum(S[:-1] * S[1:])
        return y_val
    
    # 計算 BFGS 的真實 y
    y_bfgs = calculate_true_y(final_S_bfgs)
    
    # 計算自創引擎的真實 y
    y_custom = calculate_true_y(final_S_custom)
    
    print(f"目標 T: {TARGET_T}")
    print(f"真實公式: {formula_str}")
    # 為了版面整潔，將高維度變數陣列四捨五入印出
    S_bfgs_str = np.array2string(final_S_bfgs, formatter={'float_kind':lambda x: "%.4f" % x})
    S_custom_str = np.array2string(final_S_custom, formatter={'float_kind':lambda x: "%.4f" % x})
    
    print(f"[SciPy BFGS 引擎] 求得變數: {S_bfgs_str} | 代入真實公式 y={y_bfgs:.4f}")
    print(f"[法二法三引擎] 求得變數: {S_custom_str} | 代入真實公式 y={y_custom:.4f}")
    
if __name__ == "__main__":
    main()