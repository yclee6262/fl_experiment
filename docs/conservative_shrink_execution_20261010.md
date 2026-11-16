# 保守品質＋留出收縮：固定執行與驗收

本轮只研究單次倒推的初始α，不更新α、不pruning、不結算。凍結pooled evaluator只評判交付X，不能選收縮、權重、參與者或交付輪次；truth僅診斷。

## 方法與配對設計

A、B、原保守品質（全20筆bootstrap1000次、第90百分位反比）、重複五折保守品質不收縮（診斷）、重複五折保守品質＋留出收縮（主方法）。同seed五方法共用模型、三人IDs1/2/20、proposal與原20筆資格資料。

20次隨機排列，每次五折，每折16筆估權重、4筆選λ。16筆內bootstrap1000次，取MSE第90百分位，加1e-5取倒數正規化q。λ在heldout4筆上用解析公式選混合預測MSE最小值，權重(1−λ)q＋λUniform，平坦損失取λ1。平均100折，單次Stage3 best_of（受限SLSQP β路徑及Custom30）。

診斷方法同一切分與風險bootstrap，固定λ0。留出列在該折完全排除於風險估計；不是先用全部20筆估q再留出。100折並非獨立樣本，統計單位仍是seed。

## 固定版本與樣本

Pilot4600–4604沿用前輪模型，三情境×五方法×五seed=75條件，不調參、不併入正式。正式4800–4829，450條件，三情境乾淨／低品質／四倍樣本不平衡，target0、二維。324份既有manifests查核無新seed重疊。固定30seed，不宣稱保證檢定力。

12項新方法對四基準的配對比較，seed-bootstrap10000次、RNG20260924、Bonferroni家族0.05。穩定優勢要求同時區間完全低於0、完整有效配對、失敗不增加且驗收通過。正式資料不得選參數；新版本須另登錄。

## 驗收與交付門檻

- Pilot75條件、正式450條件完整唯一，保留所有研究失敗。
- 成功交付finite且box最大違規≤1e-8；從X重算，不能只信旗標。
- α、資格預測、pooled/truth重载容差1e-7；來源與model/data SHA核對。
- 所有新方法fold的train16/heldout4不重疊，每列恰驗證20次；同fold診斷q完全一致。
- 新方法fresh/cache實際α、X及Stage3呼叫數一致。
- 一seed完整重播：資料切分、local/pooled模型、資格與proposal、離散決策、α、X與評分一致；時間不要求一致。
- 分開平均／中位／Q1／Q3、失敗、成本、有效配對n、一般與校正CI、pooled/truth衝突。
- Notion明確列方法、例子、欄位定義、全部數據下載與適用界線；寫入後回讀驗證。

## 保存位置

登錄docs/conservative_shrink_preregistration_20261010.json；環境docs/conservative_shrink_environment_20261010.json；執行alpha_weight_outputs/conservative_shrink_pipeline_20261010；正式source/study目錄名稱分別為conservative_shrink_source_v1_20261010及conservative_shrink_formal_v1_20261010。所有舊輪次保留，不合併樣本、不覆蓋。

## Pilot驗收紀錄

49項測試、Pilot75/75有效可行，α、預測與pooled重載差異0。正式版本在pilot之後維持固定。

## 最終交付：已完成

正式450/450條件有效可行，重評差異0；12項新方法比較的校正区間全部跨零，未建立穩定優勢。新方法初始化＋Stage3約原保守品質5倍。49項回歸測試、273項來源/切分/配對/fresh-cache檢查通過；重新生成seed4800所有資料、local/pooled模型、資格、proposal與15個交付結果，最大差異0。

Notion已寫入並回讀核對11張原生表格、2份CSV檔案與完整方法、例子、欄位定義：https://app.notion.com/p/3f5bf435d47481a89cbedc0cd4c10669。逐項證據：docs/conservative_shrink_requirement_audit_20261010.json。
