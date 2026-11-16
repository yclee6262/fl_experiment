# 下一輪 α 初始化比較：執行與驗收

本輪比較 A（Uniform）、B（Stage0 inverse-MSE）、既有 OOB 品質收縮，以及兩項新增方法：最佳組合 OOB 收縮、保守品質。凍結 pooled evaluator 只在交付 X 後評分，truth 僅診斷；不更新 α、不 pruning、不付款。

## 固定設定

- 情境：reference_clean、low_quality_25pct、sample_imbalance_4x。
- target 0、二维、固定三人；相同 seed 共用資料、模型、proposal、20 筆資格資料、evaluator。
- Bootstrap 1000 次，共同抽樣列；OOB 少於兩筆跳過。保守品質固定 MSE 的第90百分位。
- Pilot：歷史模型 seeds4400–4404，75條件，只作實作驗收、不調參、不併入正式。
- 正式：新 seeds4600–4629，450條件；313個既有 manifests 查核無重疊。
- 樣本數固定30，沒有保證統計檢定力。18項候選對A/B配對比較；10,000次 seed bootstrap、RNG20260924、Bonferroni家族0.05。
- 成功優勢：完整有效配對、校正區間完全低於0、失敗不增加且保存驗收通過。
- Feasibility tolerance 1e-8；reload tolerance 1e-7。所有失敗保留。

## 目前證據

Pilot75/75有效且可行，權重、模型預測及pooled重載最大差異0。44項回歸測試通過。

正式來源準備第一次在開始前遭遇 OpenMP SHM 啟動錯誤，未建立來源目錄。保留原錯誤記錄，採沙箱外同一Python、同一參數繼續；研究設定未更改。活躍狀態以 pipeline resume_status.json 與實際process為準，原 status.json 保留歷史中斷。

## 結果交付門檻

1. 來源準備與模型保存驗收通過。
2. 450完整唯一識別條件；全部狀態及成本保存，不刪失敗。
3. 從交付X重算box；來源與artifact雜湊、模型預測、α、pooled/truth重載通過。
4. 新方法fresh API／cached qualification產生相同α、X與Stage3 API數；新增計算成本獨立計入。
5. 報告均值、中位、Q1/Q3、失敗、配對有效n、18項區間及pooled/truth衝突。
6. 新Notion文件清楚列出目的、方法步驟／例子、各欄位定義、結果與適用限制；寫入後回讀確認。
7. 研究結果為負面仍完成交付；不得將未顯著解讀成等效或保證无效。

## 保存位置

- 固定登錄：docs/next_alpha_preregistration_20261010.json。
- Pilot：alpha_weight_outputs/next_alpha_pilot_v1_20261010。
- 正式来源：alpha_weight_outputs/next_alpha_source_v1_20261010。
- 正式比較：alpha_weight_outputs/next_alpha_formal_v1_20261010。
- 執行日誌：alpha_weight_outputs/next_alpha_pipeline_20261010。
- 補充驗收：docs/next_alpha_extra_acceptance_20261010.json（執行後產生）。

## 重現命令

使用專案相同Python與套件，OMP_NUM_THREADS、MKL_NUM_THREADS、OPENBLAS_NUM_THREADS均為1。下列命令的輸出目錄必須為新的空目錄。

```text
python run_alpha_initialization_study.py --output-dir NEW_SOURCE --seeds 4600,4601,...,4629 --scenarios reference_clean,low_quality_25pct,sample_imbalance_4x --methods A_uniform --primary-baseline A_uniform
python verify_alpha_initialization_study.py NEW_SOURCE
python run_next_alpha_comparison.py --source NEW_SOURCE --output NEW_STUDY --seeds 4600,4601,...,4629
python analyze_next_alpha_comparison.py NEW_STUDY
```

省略號在此表示seed清單，實際執行須展開；完整精確參數保存於pipeline命令與登錄JSON。正式驗收另加 --registration docs/next_alpha_preregistration_20261010.json。

## 最終交付（已完成）

450/450條件有效且可行；18項配對比較完成。兩個新增方法未建立校正後穩定優勢；既有OOB品質收縮僅在乾淨情境對A符合本輪判準，對B沒有。重新載入模型／α／pooled差異0，完整重播seed4600三情境五方法共15條件差異0。44項測試与補充fresh/cache、OOB分離、fallback驗收通過。

新版Notion：https://app.notion.com/p/3f5bf435d47481dd907bd5ce6449b6cb。原始CSV與配對下載檔均已置入；remote fetch確認11張原生表格、2個檔案、欄位定義與完整內容。逐項證據：docs/next_alpha_requirement_audit_20261010.json。
