"""Generate evidence-bounded Traditional Chinese report and sealed receipt."""
import argparse
from pathlib import Path
from contribution_analysis.custom_realized_gain_v1.schema import read,save,sha

WORK=Path(__file__).resolve().parents[2]


def build(root):
    s=read(root/'summary.json');a=read(root/'independent_audit.json');ledger=read(root/'ledger_results.json');history=read(root/'historical_settlement.json')
    named={r['id']:r for r in ledger};rows=[]
    for name in ('signed_example','cycle_positive_part','gate_exact','below_gate','undefined'):
        r=named[name]['results'];rows.append(f"| {name} | {r['net']['status']} | {r['net']['payments']} | {r['positive_rounds']['payments']} | {r['equal']['payments']} |")
    hist=[]
    for h in history:
        r=h['results'];nets={o:v['net'] for o,v in r['net']['rows'].items()}
        hist.append(f"- `{h['case']}`：淨值 `{nets}`；淨值候選 `{r['net']['payments']}`，逐輪正部 `{r['positive_rounds']['payments']}`，等額 `{r['equal']['payments']}`；狀態 {r['net']['status']}。")
    interactions=[r for d in s['analytic'] for r in d['cases'] if r['factor']==1.]
    overlap=sum(abs(r['interaction'])>1e-8 for r in interactions)
    sensitivity=sum(any(v['payments']!=d['cases'][1]['payments'] or v['status']!=d['cases'][1]['status'] for v in d['cases']) for d in s['analytic'])
    helpful=[];harmful=[]
    for d in s['analytic']:
        if d['id'].startswith('opposing_helpful_') or d['id'].startswith('opposing_harmful_'):
            target='owner_1';r=d['cases'][1];(helpful if d['id'].startswith('opposing_helpful_') else harmful).append((r['C'][target],r['payments'][target]))
    text=f'''# 錨點池：有號累積與影子獎金驗證結果

日期：2026-10-28。執行前規約：`docs/anchor_pool_shadow_design_v1_20261028.md`。新輸出：`{root.relative_to(WORK)}`。

**已完成：內部當輪C^A的有號累積、完整性、釋出條件、整數影子分配與獨立核對。未建立有效性：獨立成果品質、公平、部署門檻、抗操弄及低成本。這不是實際付款。**

## 1. 凍結候選及判讀

主候選先對每個owner累積S=ΣC，再以w=max(S,0)分配；累積後|S|≤1e−8視為零。保留正向／負向／淨值與覆蓋率。固定名冊所有應計槽位完整且Σw≥0.001才釋出500整數測試單位，按最大餘數法取整，餘數同分按事前owner順序。500與0.001沒有貨幣、最適價格或品質門檻意義。等額與逐輪正部對照使用相同釋出gate。

缺任何owner/輪次或有null：整池held，個人金額null。完整但全零、全負或未達gate：reserved，金額0。重複同槽相同值不加分，衝突值拒絕。T1基本費不受負C取消。A/B沒有作付款輸入。

## 2. 帳務規則結果

共{s['ledger_cases']}個執行前固定案例，{s['rejected']}個預期拒絕，其餘{s['ledger_cases']-s['rejected']}可接受輸入（含held），先開發後保留均符合手算／獨立有理數reference。{s['invariance_checks']}項重排、重播、換event_id、加零輪、授權同槽二等分重組、合法共同單位換算不變性通過。二等分由固定前處理收齊兩半後重組，尚未建立任意分片輸入／缺片認證協定。

| 案例 | 主候選狀態 | 淨值後取正部 | 逐輪取正部 | 等額 |
|---|---|---|---|---|
{chr(10).join(rows)}

signed_example是A:+3/−2/+1，B:+1/+1/+1：淨分2/3，影子金額200/300；逐輪正部變4/3，金額286/214。cycle_positive_part的A:+2/−2、B:+1，淨值法A=0；逐輪正部卻讓A領333。這是手算循環紀錄反例，不宣稱原嚴格下降solver真的可製造該循環。純事件複製不能增加任何一種候選金額。

門檻反例：同一owner由0.000999變0.001，影子釋出由0跳500。這是事前硬gate的結果，證明規則可執行，**沒有證明這個跳躍合理**。極小作用如何對應可釋出預算仍待需求方訂契約與驗證，不能因技術通過就採用此門檻。

## 3. T2既有48解析狀態的新分支診斷

48設定全部是已讀開發資料，沒有重新稱為獨立保留或正式確認。本輪新增所有owner消融、同owner方向等權拆分、共同移除全部錨點，以及固定輪前狀態eta的0.5/1/2倍，共144個局部情境。每個分支保留所有模型、固定其他權重，完成原最多10次縮步，原eta時T2目標C及完整端點吻合。

獨立60位Decimal重算H、方向、邊界、所有trial、接受、C及互動差，144情境通過，最大誤差{a['max_analytic_error']:.3g}。同owner等權方向拆分先聚合後消融，C和影子金額不變；這是已形成方向分量的等價拆分，不是新增錨點資源或換owner的抗操弄證明。

原eta的反向有益6例C均正，其中{sum(p>0 for c,p in helpful)}/6取得正影子金額；反向有害6例C均負，{sum(p==0 for c,p in harmful)}/6成果金額0。它們的基本保障仍由履約決定。有益作用可能因其他owner分數、取整、釋出gate而沒有正整數金額，不能只看「是否發到錢」判斷C有無辨識作用。

原eta共有{overlap}/48情境的ΣC與共同移除差J不相等。各人的邊際作用存在互補／重疊，ΣC不能當成新創造的總改善。分成500上限內的比例不會自動消除公平或重複計價疑慮。

步長診斷：{sensitivity}/48個原狀態在0.5/1/2倍eta間的影子金額或狀態不同。這是固定狀態的局部敏感度，並非重新跑多輪路徑，也不能當成原設計技術失敗；若獎勵定義為實際當輪作用，步長改變本來可能改變作用。尚未驗證參與者能否藉拖長真實路徑提高獎金。

## 4. 兩個歷史團隊影子重算：僅探索

獨立重核原兩團隊17輪69個owner-round、599個保存objective事件及原solver回覆重播；新訓練模型查詢0。每團隊完整名冊×輪次單獨結算，不將不同任務混成一池：

{chr(10).join(hist)}

這些是原路徑上的累積局部作用與規約性分配，不是全程LOO、實際支付、獨立品質或公平性的確認。69個事件巢狀於兩團隊，不當69個獨立任務。

## 5. 稽核與成本

獨立Fraction reference不匯入影子engine，從原始槽位核對{a['ledger_reference_checks']}項原案分配；{a['mutation_detections']}項故意改金額／淨值／狀態全部被發現。不變性結果亦逐項從保存輸入重建。拒絕案例對照執行前golden原因，不冒稱另一次獨立parser實作。解析原始回覆核對{a['analytic_reply_checks']}筆、trial核對{a['analytic_trial_checks']}筆。

新增合成API {s['new_synthetic_api_calls']}次；新主搜尋0、已訓練模型呼叫0、訓練0、pooled0、實際付款0。主流程約{s['seconds']:.3f}秒，含來源／舊receipt核對、規則變體、分支及內嵌獨立檢查，不含後續完整audit和報告。這不是部署成本或低成本估計成功。沒有測RSS/native memory或在真實模型上比較完整LOO。

## 6. Gate及下一步

**已完成：**有號累積主候選及兩個對照、全池缺失暫緩、零／負預算保留、gate、取整、重播／分拆紀錄不變性、等價owner方向拆分與獨立稽核。固定內部尺下可產生可重現的影子比例。

**未建立有效性：**0.001的合理性、500釋出規則的價格依據、實際成果品質、局部作用累積作獎勵是否符合需求者偏好、多輪路徑操弄、不同owner／新身份／串通、非等價錨點選擇、真實應用泛化和低成本。品質門檻與經濟效度尚未通過，因此結論是「技術影子分配通過，可研究候選」，不是可部署分潤。模型池保持未結算，T1規則不變。

**下一步提案：**先由需求方界定錨點成果獎金是固定事前服務競賽額度，還是隨可驗收成果釋出；事前凍結適當釋出規約，再做錨點限定的T5壓力測試（等價複製、權重守恆拆分與非等價極端錨點分開）及真實多輪路徑診斷。若要求獨立成果品質，補離線品質證據且不讓pooled進搜尋／分數／付款。不得用調小池比例或門檻掩蓋分數效度問題。

原算法、T0–T3封存結果與Notion未改；新研究不能回填為原正式確認。

## 7. 重現

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/yclee/anaconda3/bin/python -m contribution_analysis.anchor_pool_shadow_v1.run --output contribution_analysis/anchor_pool_shadow_replay_v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/yclee/anaconda3/bin/python -m contribution_analysis.anchor_pool_shadow_v1.audit --root contribution_analysis/anchor_pool_shadow_replay_v1
/Users/yclee/anaconda3/bin/python -m contribution_analysis.anchor_pool_shadow_v1.report --root contribution_analysis/anchor_pool_shadow_replay_v1
```

輸出目錄須不存在，來源和fixtures於執行前封存。報告器只在新輸出中寫報告；專案docs的報告由交付步驟另外保存，避免重現覆寫歷史。
'''
    return text


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root.resolve()
    name='anchor_pool_shadow_results_20261028.md'
    with (root/name).open('x') as f:f.write(build(root))
    files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='completion_receipt.json'}
    save(root/'completion_receipt.json',dict(complete=True,file_sha256=files,new_formal_confirmation=False,actual_payments=0))
    print(root/name)


if __name__=='__main__':main()
