"""Evidence-bounded report generated after all fixed diagnostics and audit."""
import argparse
from pathlib import Path
from contribution_analysis.custom_realized_gain_v1.schema import read,save,sha
WORK=Path(__file__).resolve().parents[2]


def build(root):
    s=read(root/'summary.json');audit=read(root/'independent_audit.json');teams=s['teams'];table=[];numbers=[];paths=[];cost=[]
    for t in teams:
        a=t['base_allocation'];label='乾淨'if'poison'not in t['case']else'受控污染'
        table.append(f"| {label} | {t['base_rounds']} | {t['base_owner_rounds']} | "+'；'.join(f"{o}: {100*float(a['shares'][o]):.4f}%"for o in a['owners'])+' | '+str(a['payments'])+' |')
        for v in ('repeat','split_adjacent','split_separated','budget60'):
            r=t['comparisons'][v];numbers.append(f"| {label} | {v} | {r['passed']} | {r['max_solution_difference']:.3g} | {r['max_net_difference']:.3g} | {r['max_share_difference']:.3g} | {r['payments_equal']} | {r['other_rounds']} |")
        n=t['base_numerical'];paths.append(f"- {label}：原30輪上限實際{t['base_rounds']}輪，60上限實際{t['extended_rounds']}輪。最後{t['prefix_tail']['last_quarter_rounds']}輪的|C|總和{t['prefix_tail']['last_quarter_abs_C']:.6g}，連續份額最大改變{t['prefix_tail']['last_quarter_delta_share']}; 主數值擾動maxΔq={n['arithmetic']['max_delta_share']:.3g}，±1e−8壓力maxΔq={n['stress_1e-8']['max_delta_share']:.3g}，±1e−6壓力maxΔq={n['stress_1e-6']['max_delta_share']:.3g}。")
        c=t['costs'];b=c['base'];control=(c['control0']['search_seconds']+c['control1']['search_seconds'])/2
        cost.append(f"| {label} | {c['control0']['main_api_calls']} | {b['branch_api_calls']} | {t['base_calls_vs_control_ratio']:.3f}× | {control:.4f} | {b['algorithm_processing_seconds']+b['save_seconds']:.4f} | {t['base_processing_vs_control_ratio']:.3f}× |")
    splitpass=all(t['comparisons'][v]['passed']for t in teams for v in('split_adjacent','split_separated'))
    numericpass=all(t['all_primary_numerical_passed']for t in teams)
    return f'''# 錨點池：多輪流程、數值穩定與輸入複製驗證結果

日期：2026-10-28。執行前規約：`docs/anchor_pool_end_to_end_design_v1_20261028.md`。輸出：`{root.relative_to(WORK)}`。

**已完成：兩個既有模型團隊的端到端重算、重現、輸入端同owner錨點複製／權重守恆拆分、數值壓力、prefix與完整成本紀錄。輸入等價gate：{splitpass}；主數值gate：{numericpass}。未建立：新應用泛化、公平、非等價操弄及低成本。只產生影子分配。**

## 1. 本輪規則與資料邊界

使用者目前只要求固定預算的相對比例。因此本輪不再沿用上一輪測試Σw≥0.001的發放gate；固定500整數測試單位，先累積有號S，再以S>τ的正淨值正規化。τ=max(1e−8,Σρ)，ρ=max(1e−12,32eps64(1+|L_full|+|L_minus|))。無可靠正值保留預算、q=null；缺任何應計槽位整池held，未知不填零。T1基本保障與模型池不變，500不是定價。

資料只用10/27兩個已讀CPU模型團隊，3／5個owner，模型和原起點／權重／目標／版本固定，不訓練、不開pooled或truth。它們是歷史研究情境，不能稱新真實應用或獨立品質確認。本輪技術測試先通過才載入模型。沒有以LOO相關性作驗收。

## 2. 原路徑累積與分配

| 團隊 | 實際輪 | 完整owner×輪 | 連續比例 | 500單位取整 |
|---|---:|---:|---|---|
{chr(10).join(table)}

每一實際輪包含拒絕／mode切換與最後停止輪；完整結果使用接受端點，排除分支也完成原最多10次縮步。保留所有模型，僅扣同owner已形成H，其他a不重分配；分支不回流主路徑。原兩個counter/digest控制與觀察base結果、查詢digest一致；base／repeat的事件及比例精確相同。原17輪69個保存C亦全部回歸核對，不回填正式結果。

## 3. 真正在輸入端複製錨點

每個原錨點在搜尋輸入端複製兩份、每份一半權重、owner不變；原模型清單及v不複製。adjacent相鄰排列，separated先排所有第一份再排第二份，以測不同加總順序。各變體都重新執行完整模型搜尋及所有owner消融，不是只把H除二。模型raw回覆、初始化錨點查詢、dynamic probe與trial均保存。

| 團隊 | 變體 | gate | maxΔx | maxΔS | maxΔq | 整數金額相同 | 實際輪 |
|---|---|---|---:|---:|---:|---|---:|
{chr(10).join(numbers)}

q差為0到1的比例差；1e−6等於0.0001百分點。輸入複製允許查詢數增加，因重複錨點實際觸發初始化／探測。這不是新owner、任意增加權重、不同位置錨點、Sybil或串通測試；只在這兩個團隊與本輪固定變換範圍內判讀。

## 4. 數值與輪長診斷

獨立Fraction依保存raw回覆重算loss，最大float64聚合差{audit['max_saved_reply_aggregation_error']:.3g}；主搜尋與消融的嚴格接受比較共{audit['exact_reply_acceptance_flips']}次與精確保存回覆比較異號，{audit['near_acceptance_margins']}個消融trial的接受差落入事前算術容差。這是同一份回覆後的算術診斷，不驗證float32模型準確率、硬體漂移或查詢座標擾動。

每個owner的所有C同向加／減ρ，窮舉2^owner最有利／最不利組合，完整保存重算結果；另按每C±1e−8／±1e−6作壓力。主gate要求資格不翻、maxΔq≤1e−6、整數金額相同；較大壓力只診斷，沒有依結果放寬門檻。

{chr(10).join(paths)}

若30與60上限都在30前自然停止，則只證明延長預算沒有新增有效輪；不能宣稱已測持續更長路徑。prefix份額隨新輪改變本身不算失敗，須看新增C；本輪沒有設計可自利拖長路徑的攻擊。

## 5. 獨立稽核與帳務邊界

原12項新規則測試通過（含小正值、累積後套容差、全零／負、null／缺失、重播／衝突、取整與零輪）。獨立reference不使用生產net/q作預期，從owner×輪C槽位以Fraction重建。{audit['observed_runs']}個完整觀察run、{audit['main_iterations']}輪及{audit['owner_rounds']}個消融全部原始回覆／幾何／接受核對；原未插樁solver以保存回覆重播，沒有新模型推論。另核對{audit['prefix_settlements']}個prefix、{audit['stress_settlements']}個壓力分配及{audit['ledger_variant_checks']}個缺失／重播／逆序／null變體；{audit['mutation_detections']}個故意C／比例／金額篡改全部被拒絕。非法owner及衝突槽位另在主流程拒絕，無實際轉帳。

## 6. 成本

| 團隊 | 原控制API | 新增消融API | 完整API/原控制 | 原控制秒（2次平均） | 觀察＋消融＋分析＋寫檔秒 | 時間比 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(cost)}

完整API含base主搜尋與消融，未新增共同端點補評，完整分支重用原run。模型載入及原／分支獨立audit時間另列原始cost，不混入算法處理比；完整處理秒包含本輪prefix與壓力分析，屬本實作研究流程，不是最小結算延遲。對照帶最小counter/digest，也不是裸搜尋。兩次控制不足以作性能顯著性推論。unique input只是統計，沒有快取省呼叫。

本輪共{s['new_main_searches']}次主搜尋（含4個最小控制、10個完整觀察）、{s['total_main_rounds']}個實際主輪、新模型API{s['total_new_model_api_calls']}次；主流程{s['seconds']:.3f}秒（含測試、hash、載入、搜尋、分析、內嵌核對與寫檔，不含最後audit／報告）。新訓練0、pooled0、truth未載入、付款0。主流程末輸出{s['bytes_before_audit']}bytes。程序生命期峰值RSS原值{s['process_lifetime_peak_rss_raw']} {s['peak_rss_unit']}，含Python／PyTorch載入，不是單算法增量。成本只描述這兩個CPU舊模型團隊，不宣稱低成本方法通過，也沒有拿離線消融要求15%門檻。

## 7. 結論與下一步

**已完成：**固定預算相對比例的研究規約更新；原資料端到端重算與精確重現；真實輸入端同owner權重守恆複製、數值算術與壓力、prefix、基本缺失處理及成本紀錄。技術與等價gate依上述原始結果判定，不把記帳成功稱公平。

**未建立有效性：**兩團隊以外泛化、新任務、隨機／不同硬體推論、非等價錨點與任意權重、新身份／串通、真實多輪拖長攻擊、獨立品質、公平或低成本。相對分配只對固定內部目標成立，各人C可有互補重疊，不等於總成果拆帳。

**下一步提案：**若等價／主數值gate通過，可保留C^A有號累積與固定預算比例，將本輪列為限定資料及變換範圍內的穩定可重現影子方法；若未通過，先依保存差異找原因，不改原solver或事後重選門檻。後續需要更多已凍結團隊／維度／權重形狀的工程確認，或另立錨點操弄威脅模型；不自動擴大模型池、調alpha、pruning或付款。

來源、模型hash、兩個舊團隊清單、本規約於推論前封存；T0–T3、上一輪影子結果與Notion未修改。原始來源與失敗／近零事件保留，新結果不回填歷史正式確認。

## 8. 重現

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/yclee/anaconda3/bin/python -m contribution_analysis.anchor_pool_end_to_end_v1.run --output contribution_analysis/anchor_pool_end_to_end_replay_v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/yclee/anaconda3/bin/python -m contribution_analysis.anchor_pool_end_to_end_v1.audit --root contribution_analysis/anchor_pool_end_to_end_replay_v1
/Users/yclee/anaconda3/bin/python -m contribution_analysis.anchor_pool_end_to_end_v1.report --root contribution_analysis/anchor_pool_end_to_end_replay_v1
```

輸出資料夾必須不存在；重現不覆寫專案docs。
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve();name='anchor_pool_end_to_end_results_20261028.md'
    with(root/name).open('x')as f:f.write(build(root))
    save(root/'completion_receipt.json',dict(complete=True,file_sha256={str(p.relative_to(root)):sha(p)for p in root.rglob('*')if p.is_file()and p.name!='completion_receipt.json'},actual_payments=0,new_formal_confirmation=False))
    print(root/name)


if __name__=='__main__':main()
