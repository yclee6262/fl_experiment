"""Generate the T0/T1 research report without executing settlements."""
import argparse
import datetime
from pathlib import Path
import shutil
import statistics
from contribution_analysis.custom_realized_gain_v1.schema import read,save,sha

WORK=Path(__file__).resolve().parents[2]


def render(root):
    s=read(root/'summary.json');audit=read(root/'independent_audit.json');rows={r['case']:r['result'] for r in read(root/'results.json')}
    if not audit['accepted']:raise ValueError('audit_not_passed')
    t=['# 三池 T0／T1 實作與驗證結果', '',
       '日期：2026-10-28。設計：`docs/three_pool_t0_t1_contract_v1_20261028.md`。輸出：`contribution_analysis/three_pool_t0_t1_experiment_v1_20261028/`。', '',
       '**已完成：研究用三池服務／預算規約及基本保障影子結算引擎；97案符合執行前golden、410項事件不變性檢查通過，獨立核對通過。未建立有效性：成果分潤、公平性、真實身份與事件認證、模型／錨點品質及部署付款。**', '',
       '## 1. 本輪結論', '',
       '在固定服務承諾及可信收件事件假設下，基本保障可以按合約可靠地判定：合格履約依事前固定費支付影子金額；負貢獻、沒有改善或反向作用不取消已完成的服務費；額外呼叫或重複身份別名不增加承諾金額。這是履約規則的計算正確性，不是完整經濟分潤機制已驗證。', '',
       'T0已將研究用基本費、預算、交付、時間、退出、重試、版本、確定性與缺資料規則具體化。成果池的個人價值算法及部署品質裁判仍未建立。T1只生成基本池的影子金額，兩個成果池維持reserved且金額null，不用0冒稱已算出無貢獻。', '',
       '## 2. T0固定的研究規則', '',
       '示例為無貨幣意義的整數測試單位：總預算2000，基本池1000，錨點池500，模型池500。這不是建議價格、最適比例或實際付款契約。', '',
       '| owner | 事前服務承諾 | 費用規則 | 最大基本費 |', '|---|---|---|---:|',
       '| A | 錨點＋模型 | 兩者全完成才成立；不事後拆分 | 300 |',
       '| B | 錨點＋模型 | 事前約定分段：錨點100、完整評分200 | 300 |',
       '| C | 只有錨點 | 錨點合格交付 | 100 |',
       '| D | 只有模型 | 兩個授權請求皆完成且無版本／確定性違反 | 200 |', '',
       '最大基本費承諾900≤1000，預留100。固定承諾不依新身份、參與人數或分數重新除分母；未准入者不增加資格。若新合約最大承諾超出基本池，直接拒絕，不按比例削減已承諾費用。', '',
       '- 錨點：指定slot、二維有限值、[-1,1]邊界、按時交付，且已有可信的服務准入品質資格。准入資格由外部系統提供，本輪不驗證實際品質。',
       '- 評分：指定請求、精確輸入、凍結版本、有限單值，兩個授權請求都完成。未授權呼叫、重試、重複回覆不加費。',
       '- 允許無效回覆在期限內補正；同一授權請求有不同有效值，或授權期間版本改變，整項模型服務不合格。這是服務承諾檢查，沒有用E或其他貢獻分數扣費。',
       '- 截止時間包含等號；退出同時交付不補計。退出前已完成的資格保留；事前分段者可保留已完成階段。',
       '- 沒有可行錨點或缺資料時保留原因，不偽造交付。沒有另訂無解搜尋努力費，不能自動領取錨點基本費。',
       '- 同event_id完全相同內容視為重播；同ID不同內容整批拒絕影子結算。不同ID的重複服務不加費。', '',
       '預算性質由規則直接限制：每位基本金額≤該owner固定承諾，故Σ基本影子金額≤900≤1000；兩成果池保留，不跨池挪補，不強迫將剩餘基本池分完。', '',
       '## 3. T1實際驗證範圍', '',
       f'- 共{s["case_count"]}個執行前固定案例：65個手工標記情境及32個原子／分段履約布林窮舉。golden輸入在執行engine前封存及hash鎖定。',
       f'- {s["successful_settlements"]}個可結算案例：每位金額與角色完成狀態符合golden，獨立reference另從原始事件核對完整角色細節、退出與預算。',
       f'- {s["expected_rejections"]}個預期拒絕：無效合約／事件輸入正確拒絕，沒有產出金額。這些不是未解決的測試失敗。',
       f'- {s["invariance_checks"]}項事件不變性：82案各5變體（反轉順序、完全重播、未授權噪音、同owner別名、診斷分數改變）。金額、履約狀態、預算與成果池狀態不變；事件原因及重播數可以不同。這些由執行期斷言核對，保存每項結果。',
       f'- {s["audit_mutation_checks"]}項篡改檢查：82案各故意增加一個單位金額／翻轉一項角色完成狀態，全部被reference拒絕；獨立audit重做這164項檢查。',
       '- 獨立audit不匯入engine，核對封存來源與golden hash、全部82項原始事件重建、15項拒絕結果及完整案例覆蓋。拒絕案例按事前golden原因核對，不把紀錄中的拒絕字串稱為另一次engine重跑。', '',
       '## 4. 代表結果', '',
       '下表金額均為測試單位。每一列是各自獨立的模擬合約情境，不能跨列相加當成同一次付款。', '',
       '| 案例 | A | B | C | D | 基本池影子支出 | 基本池剩餘 |', '|---|---:|---:|---:|---:|---:|---:|']
    named=[('complete_all','全部履約'),('negative_contribution','貢獻診斷為負但服務合格'),('atomic_anchor_only','A只完成錨點，未約定分段'),
           ('staged_anchor_only','B只完成錨點，已約定分段'),('staged_model_only','B只完成模型，已約定分段'),
           ('same_id_exact_replay','完全相同事件重播'),('version_changed_after_completion','D收帳前發現授權期版本變更'),
           ('staged_withdraw_after_anchor','B完成錨點後退出'),('withdraw_after_completion','A全部完成後退出'),
           ('no_feasible_anchor_notice','B無可行錨點，但完成模型服務')]
    for key,label in named:
        r=rows[key];a={v['owner']:v['basic_shadow_amount'] for v in r['owners']}
        t.append(f'| {label} | {a["A"]} | {a["B"]} | {a["C"]} | {a["D"]} | {r["basic"]["spent_shadow"]} | {r["basic"]["remaining"]} |')
    t += ['', '這些例子反映事前契約差異：A不具部分費用權利；B事前已有分段承諾。不能看到交付結果後才替A補設分段規則，也不能把B沒完成的部分分給其他人。', '',
          '沒有改善、反向作用、低成果品質但具服務准入資格的案例，基本費均與完整履約案例相同。這只是把服務履約與成果價值分開，沒有證明這些參與者應取得多少貢獻獎金。', '',
          '## 5. 拒絕與缺失結果', '',
          '無效合約包括：基本池不足、三池總和錯誤、負費／小數／bool費用、owner或別名重複、缺少模型版本或版本欄位、重複授權請求、未知費用政策、請求時間逆序、啟用真實付款。另有同事件ID內容衝突、未知事件type，整批拒絕。共15項均符合預期。', '',
          '合約有效但交付無效者，例如NaN／Infinity標記、bool值、錯維度、超界、缺准入品質、錯slot、錯輸入、未授權請求、提前／晚交、退出後交付，保留各自原因及未完成狀態。影子可付0不代表品質差、負貢獻或個人沒有價值。', '',
          '## 6. 成本與修改範圍', '',
          f'主流程共{s["total_engine_invocations"]}次engine呼叫（97個原案＋410個變體＋1次記憶體探針），未執行搜尋、推論、訓練或實際轉帳。主流程約{s["costs"]["seconds"]:.3f}秒，含golden檢查、不變性、篡改核對與部分輸出，不含後續獨立audit及報告。', '',
          f'原案單次時間中位數約{statistics.median(s["costs"]["per_case_seconds"])*1000:.3f}ms，含預期拒絕，僅是本地小事件帳的描述，不是部署延遲保證。完整示例單次Python配置峰值{s["costs"]["python_peak_bytes_single_demo"]}bytes；不是RSS/native memory。', '',
          '新增獨立模組 `contribution_analysis/three_pool_t0_t1_v1/`；原host_server、切線法／割線法、A/E/B算法、alpha、歷史實驗輸出均未修改。本輪未啟動T2–T7、LOO、pooled、truth或Notion更新。研究進度文件新增T0/T1結果；原T0–T7規劃文件維持執行前版本。', '',
          '## 7. 已完成、未建立有效性與下一步', '',
          '**已完成：**研究用T0服務契約及可機器核對的示例合約；T1固定任務費影子引擎、原子／分段／單角色履約、身份別名歸戶、冪等、窗口與退出、版本／確定性、預算守恆與獨立檢查。97案均符合預期，沒有未解決的技術失敗。', '',
          '**未建立有效性：**真實報價與三池比例、真實收件日誌接入、身份認證／抗Sybil、版本證明、准入品質如何可信取得、錨點／模型成果獎金、激勵相容、部署成果驗收及實際付款。研究reference_measure只鎖定共同尺類型，T2/T3仍須凍結具體任務模型與權重；沒有事先可部署成果尺，不能稱三池正式機制可用。', '',
          'registry與received_tick由可信系統提供是明確假設。本模組不驗證簽章、不查真人身份；同owner別名不重領不等於不同假身份無法重領。回覆值有限／版本欄位相符也不證明模型預測準確或版本聲明真實。', '',
          '**下一步提案：**依原計畫進T2「錨點有益制衡」，先核對已保存分支，再建立解析案例與角色對齊基準；不重跑舊資料當新確認。基本保障可作後續影子機制的固定底座，但錨點／模型成果池維持未結算，不能拿A/E/B補缺。本次沒有自動啟動T2。', '',
          '## 8. 重現', '',
          '```sh',
          '/Users/yclee/anaconda3/bin/python -m contribution_analysis.three_pool_t0_t1_v1.run --output contribution_analysis/three_pool_t0_t1_experiment_v1_replay',
          '/Users/yclee/anaconda3/bin/python -m contribution_analysis.three_pool_t0_t1_v1.audit --root contribution_analysis/three_pool_t0_t1_experiment_v1_replay',
          '```', '',
          '輸出資料夾須不存在。來源與golden在執行前封存；執行後報告器另放delivery快照。完整事件、每事件處置、每owner狀態、金額、拒絕原因、所有檢查清單及完成receipt均另存，不覆寫舊結果。']
    return '\n'.join(t)+'\n'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root.resolve()
    output=render(root);name='three_pool_t0_t1_results_20261028.md'
    for path in [root/name,WORK/'docs'/name]:
        with path.open('x') as f:f.write(output)
    delivery=root/'delivery_source_snapshot';delivery.mkdir()
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ['.py','.md']:shutil.copy2(p,delivery/p.name)
    save(root/'completion_receipt.json',dict(complete=True,t0_research_contract_complete=True,t1_technical_validation_passed=True,
        contribution_payment_validity_established=False,deployment_contract_complete=False,actual_transfers=0,notion_modified=False,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),report='docs/'+name,report_sha256=sha(WORK/'docs'/name),
        file_sha256={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    print('REPORT',WORK/'docs'/name)


if __name__=='__main__':main()
