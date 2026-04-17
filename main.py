from dataset import generate_agent_dataloaders
from agent_client import AgentNode
from host_server import HostServer

def main():
    TARGET_T = 2.4
    NUM_AGENTS = 20      # 擴大參與人數！(可自由更改為 30, 50 等)
    POISON_RATIO = 0.4   # 設定 40% 的人是異質節點
    
    print(f"=== Phase 0: 準備資料與訓練本地神經網路 (共 {NUM_AGENTS} 個 Agent) ===")
    # 傳入設定的參數
    loaders = generate_agent_dataloaders(num_agents=NUM_AGENTS, poison_ratio=POISON_RATIO)
    
    all_agents = []
    for i in range(NUM_AGENTS):
        agent = AgentNode(agent_id=i+1, dataloader=loaders[i])
        print(f"訓練 Agent {i+1} 中...")
        agent.train_local_model(epochs=30)
        all_agents.append(agent)
        
    print("\n=== 開始去中心化反推演算法 ===")
    server = HostServer(target_T=TARGET_T)
    
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
    
    # 計算 BFGS 的真實 y
    a1, b1 = final_S_bfgs[0], final_S_bfgs[1]
    y_bfgs = a1 + a1*b1 + b1
    
    # 計算自創引擎的真實 y
    a2, b2 = final_S_custom[0], final_S_custom[1]
    y_custom = a2 + a2*b2 + b2
    
    print(f"目標 T: {TARGET_T}")
    print(f"[SciPy BFGS 引擎] 求得變數: a={a1:.4f}, b={b1:.4f} | 代入真實公式 y={y_bfgs:.4f}")
    print(f"[法二法三引擎] 求得變數: a={a2:.4f}, b={b2:.4f} | 代入真實公式 y={y_custom:.4f}")
    
if __name__ == "__main__":
    main()