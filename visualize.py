import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import os
import json
import math
import glob
from datetime import datetime
from medmnist import PathMNIST
from models import Net, get_device

# ==========================================
# [設定區]
# ==========================================
# CONFIG_FILE = "experiment_config_5.json"
# COEFFS_FILE = "last_run_coeffs.json"
# SAVE_DIR = "experiment_results_5"
# ==========================================

RESULTS_ROOT = "experiment_results"

def get_latest_experiment_dir():
    """自動尋找最新的實驗資料夾"""
    # 搜尋所有 exp_ 開頭的資料夾
    all_dirs = glob.glob(os.path.join(RESULTS_ROOT, "exp_*"))
    if not all_dirs:
        return None
    # 按照建立時間排序 (最新的在最後)
    latest_dir = max(all_dirs, key=os.path.getmtime)
    return latest_dir

def save_experiment_data(save_path, data_dict):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)
    print(f"  -> [數據存檔] {save_path}")

def save_prediction_table(validation_results, num_clients, save_path):
    """
    專門繪製並儲存預測結果表格圖片
    """
    # 準備表格數據
    columns = ["Agent ID", "Prediction", "Confidence", "Status"]
    cell_text = []
    colors = []
    
    # 依序讀取 Agent 1 ~ N
    for i in range(1, num_clients + 1):
        key = f"agent_{i}"
        res = validation_results[key]
        
        pred = res['prediction']
        conf = res['confidence']
        status = res['status']
        
        row = [f"Agent {i}", f"Class {pred}", f"{conf:.4f}", status]
        cell_text.append(row)
        
        # 設定顏色：失敗顯示紅色背景，成功顯示淺綠色
        if status == "Fail":
            colors.append(["#ffcccc", "#ffcccc", "#ffcccc", "#ffcccc"]) # 淺紅
        else:
            colors.append(["#ccffcc", "#ccffcc", "#ccffcc", "#ccffcc"]) # 淺綠

    # 繪圖
    fig, ax = plt.subplots(figsize=(8, num_clients * 0.6 + 1)) # 高度隨 N 自動調整
    ax.axis('tight')
    ax.axis('off')
    
    # 建立表格
    table = ax.table(cellText=cell_text,
                     colLabels=columns,
                     cellColours=colors,
                     loc='center',
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8) # 調整格子高度

    plt.title(f"Prediction Results (N={num_clients})", fontweight="bold", y=1)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> [表格存檔] {save_path}")

def main():
    # 1. 讀取設定
    exp_dir = get_latest_experiment_dir()
    if not exp_dir:
        print(f"錯誤：找不到任何實驗資料夾在 {RESULTS_ROOT}")
        return
    
    print(f"=== 正在分析最新的實驗: {exp_dir} ===")

    # 設定檔案路徑
    config_file = os.path.join(exp_dir, "experiment_config.json")
    coeffs_file = os.path.join(exp_dir, "last_run_coeffs.json")

    if not os.path.exists(config_file) or not os.path.exists(coeffs_file):
        print(f"錯誤：在 {exp_dir} 中找不到 config 或 coeffs json 檔")
        return

    with open(config_file, "r") as f:
        config = json.load(f)
        anchor_indices = config["anchor_indices"]
        target_class = config["target_class"]
        num_clients = config["num_clients"]
    
    with open(coeffs_file, "r") as f:
        final_coeffs = json.load(f)

    print(f"=== 視覺化 (N={num_clients}) ===")
    
    # 2. 載入資料
    device = get_device()
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    test_dataset = PathMNIST(split='test', transform=data_transform, download=True)
    
    anchors = []
    for idx in anchor_indices:
        img, label = test_dataset[idx]
        anchors.append(img.unsqueeze(0).to(device).float())

    # 3. 合成影像
    synthetic_input = torch.zeros_like(anchors[0])
    for i, coeff in enumerate(final_coeffs):
        synthetic_input += torch.tensor([coeff], device=device) * anchors[i]

    # 4. 模型驗證 (蒐集數據)
    print("\n[模型驗證]")
    validation_results = {}
    for i in range(1, num_clients + 1):
        model = Net().to(device)
        model.load_state_dict(torch.load(f"saved_models/agent_{i}.pth", map_location=device))
        model.eval()
        
        with torch.no_grad():
            output = model(synthetic_input)
            pred = output.argmax(dim=1).item()
            probs = torch.softmax(output, dim=1)
            conf = probs.max().item()
            target_prob = float(probs[0][target_class].item())

        status = "Success" if pred == target_class else "Fail"
        print(f"  Agent {i}: Pred {pred} (Conf {conf:.4f}) -> {status}")

        validation_results[f"agent_{i}"] = {
            "prediction": pred,
            "confidence": conf,
            "target_class_prob": target_prob,
            "status": status
        }

    save_experiment_data(os.path.join(exp_dir, "final_report.json"), validation_results)

    # (B) 儲存預測表格圖片
    save_prediction_table(validation_results, num_clients, os.path.join(exp_dir, "prediction_table.png"))

    # 6. 繪製 Dashboard (維持 Grid Layout)
    ANCHOR_COLS = 5
    anchor_rows = math.ceil(num_clients / ANCHOR_COLS)
    fig_height = 4 + 3 * anchor_rows
    fig = plt.figure(figsize=(12, fig_height))
    gs = fig.add_gridspec(2 + anchor_rows, ANCHOR_COLS + 2) 

    # 合成圖
    ax_syn = fig.add_subplot(gs[0:2, 0:2])
    img_show = synthetic_input.detach().cpu().squeeze().permute(1, 2, 0)
    img_show = torch.clamp(img_show * 0.5 + 0.5, 0, 1)
    ax_syn.imshow(img_show)
    ax_syn.set_title(f"Synthetic Input\n(Target: {target_class})", fontweight='bold', color='green')
    ax_syn.axis('off')

    # 係數圖
    ax_bar = fig.add_subplot(gs[0:2, 2:])
    colors = ['red' if c < 0 else 'blue' for c in final_coeffs]
    bars = ax_bar.bar(range(1, num_clients + 1), final_coeffs, color=colors)
    ax_bar.axhline(0, color='black', linewidth=0.8)
    ax_bar.set_title("Contribution Coefficients")
    ax_bar.set_xlabel("Agent ID")
    ax_bar.set_xticks(range(1, num_clients + 1))
    if num_clients > 15:
        plt.setp(ax_bar.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    # Anchors
    for i, idx in enumerate(anchor_indices):
        row = 2 + (i // ANCHOR_COLS)
        col = (i % ANCHOR_COLS) + 1
        ax_anch = fig.add_subplot(gs[row, col if col < ANCHOR_COLS+2 else -1])
        
        raw_img = anchors[i].detach().cpu().squeeze().permute(1, 2, 0)
        raw_img = torch.clamp(raw_img * 0.5 + 0.5, 0, 1)
        ax_anch.imshow(raw_img)
        
        true_label = test_dataset[idx][1].item()
        title_color = 'red' if true_label != target_class else 'blue'
        ax_anch.set_title(f"A{i+1}\nTrue: {true_label}", fontsize=9, color=title_color)
        ax_anch.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "dashboard.png"), dpi=300, bbox_inches='tight')
    print(f"=== 分析完成！所有結果已存於: {exp_dir} ===")
    
    plt.show()

if __name__ == "__main__":
    main()