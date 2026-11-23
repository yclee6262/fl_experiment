# 結算政策重算

以已驗收的 600 筆起點配置比較三本帳。原 peer 影子帳本以兩角色負分取消基本費。修訂的影子帳本按服務履行給付基本費，沿用 peer 錨點及模型獎金分配。主政策在這批資料中給付已履行服務的費用，並保留兩個獎金池，因為資料沒有獨立的交付驗收及貢獻規則啟用證據。

```sh
python run.py settlement-replay \
  --source outputs/contribution_start_stability/task11_formal_v1 \
  --output-dir outputs/settlement_policy/new-run
```

[規約](protocol.json)中的 100／500／500 為沿用影子實驗的模擬單位，每位入選者在保存的有效配置中假定完成合約服務。輸出 `summary.csv` 為隊內配對後的隊平均，`owner_ranges.csv` 保存個人跨 15 配置的金額範圍，`audit.json` 記錄來源與驗收。輸出目錄須為新路徑。

[結果與驗收](results/audit.json)對應[八個配對條件](results/summary.csv)。分析使用既有原始紀錄，新增模型查詢及求解皆為零。數值為會計規則重算，服務履行為指定情境，實際合約及獎金發放不屬於此入口。
