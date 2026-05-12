from dataset import generate_agent_dataloaders
from agent_client import AgentNode
from host_server import HostServer
import matplotlib.pyplot as plt

def run_ablation_study():
    TARGET_T = 1.5
    NUM_AGENTS = 20      
    POISON_RATIO = 0.4   # 40% 高毒化率，逼出演算法極限
    N = 10               # 高維度地貌
    
    print("=== 準備高難度測試環境 ===")
    loaders = generate_agent_dataloaders(num_agents=NUM_AGENTS, poison_ratio=POISON_RATIO, n_features=N)
    
    all_agents = []
    for i in range(NUM_AGENTS):
        agent = AgentNode(agent_id=i+1, dataloader=loaders[i], n_features=N)
        agent.train_local_model(epochs=30)
        all_agents.append(agent)
        
    server = HostServer(target_T=TARGET_T, n_features=N)
    server.phase1_filter_agents(all_agents)
    server.phase2_collect_proposals()
    
    print("\n=== 開始執行消融實驗 (Ablation Study) ===")
    results = {}
    
    # Config A: 純 BFGS
    # (因為你的寫法直接回傳純數字 list，所以直接賦值即可)
    _, hist_A = server.phase3_global_optimization()
    results['A (BFGS)'] = hist_A
    
    # Config B: 純割線 (無退火、無切線)
    # (注意：這裡要用三個變數接收，因為現在回傳 S_current, error, states)
    _, hist_B, _ = server.phase3_custom_secant_optimization(num_iterations=30, use_annealing=False, allow_tangent=False)
    results['B (Secant Only)'] = hist_B
    
    # Config C: 割線 + 退火 (無切線)
    _, hist_C, _ = server.phase3_custom_secant_optimization(num_iterations=30, use_annealing=True, allow_tangent=False)
    results['C (Secant+Annealing)'] = hist_C
    
    # Config D: 割線 + 切線 (無退火)
    _, hist_D, _ = server.phase3_custom_secant_optimization(num_iterations=30, use_annealing=False, allow_tangent=True)
    results['D (Secant+Tangent)'] = hist_D
    
    # Config E: 完整自適應引擎二 (Ours)
    # (把 states_E 接起來，等一下畫狀態轉移圖會用到！)
    _, hist_E, states_E = server.phase3_custom_secant_optimization(num_iterations=30, use_annealing=True, allow_tangent=True)
    results['E (Adaptive Full Engine)'] = hist_E

    # --- 畫圖一：所有 Config 的軌跡對比 ---
    import os
    os.makedirs("results_plots", exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    for label, losses in results.items():
        plt.plot(range(1, len(losses) + 1), losses, marker='x', linestyle='--', label=label)
    
    plt.yscale('log')
    plt.xlabel('API Iterations / BFGS Steps')
    plt.ylabel('Absolute Error |y_pred - T| (Log Scale)')
    plt.title('Ablation Study: Subspace Optimization Trajectories')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('results_plots/plot_ablation_comparison.png', dpi=300)
    plt.show()

    # --- 畫圖二：引擎二的狀態轉移解剖圖 ---
    plt.figure(figsize=(10, 6))
    
    # 現在 hist_E 是誤差串列，states_E 是狀態串列
    iters = list(range(1, len(hist_E) + 1))
    losses = hist_E
    states = states_E
    
    # 畫底線
    plt.plot(iters, losses, color='gray', linestyle='-', alpha=0.5, zorder=1)
    
    # 根據狀態打上不同的 marker
    for i in range(len(iters)):
        if states[i] == "secant" or states[i] == "Start":
            plt.scatter(iters[i], losses[i], c='blue', marker='o', s=80, zorder=2, label='Secant Mode' if 'Secant Mode' not in plt.gca().get_legend_handles_labels()[1] else "")
        elif states[i] == "Annealing Triggered":
            plt.scatter(iters[i], losses[i], c='orange', marker='^', s=100, zorder=2, label='Annealing Triggered' if 'Annealing Triggered' not in plt.gca().get_legend_handles_labels()[1] else "")
        elif states[i] == "dynamic":
            plt.scatter(iters[i], losses[i], c='red', marker='*', s=150, zorder=2, label='Tangent Precision Mode' if 'Tangent Precision Mode' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.yscale('log')
    plt.xlabel('API Iterations')
    plt.ylabel('Absolute Error |y_pred - T| (Log Scale)')
    plt.title('State Transition Anatomy of Adaptive Dual-Engine')
    # 解決重複標籤的問題
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('results_plots/plot_engine_anatomy.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_ablation_study()