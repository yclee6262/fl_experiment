"""Render T2 evidence without executing searches or payment logic."""
import argparse
import datetime
from pathlib import Path
import shutil
from contribution_analysis.custom_realized_gain_v1.schema import read,save,sha

WORK=Path(__file__).resolve().parents[2]


def count(rows,eps):return sum(r['C']>eps for r in rows),sum(r['C']< -eps for r in rows),sum(abs(r['C'])<=eps for r in rows)


def render(root):
    s=read(root/'summary.json');audit=read(root/'independent_audit.json');old=read(root/'historical_reanalysis.json');cs=s['analytic_cases']
    if not audit['accepted'] or not read(root/'additional_account_link_checks.json')['passed']:raise ValueError('audit_required')
    t=['# T2：錨點有益制衡驗證結果', '',
       '日期：2026-10-28。規約：`docs/three_pool_t2_anchor_design_v1_20261028.md`；輸出：`contribution_analysis/three_pool_t2_anchor_experiment_v1_20261028/`。', '',
       '**已完成：48個解析案例、69組已保存歷史分支重讀、獨立Decimal／原始回覆核對。結論：完整當輪分支差可以記錄反向有益作用；A及B不能直接代替此結果價值。未建立：低成本估計器、外部品質、公平分潤或實際付款。**', '',
       '## 1. 主要結論', '',
       '本輪建立並驗證的是研究用局部結果參考：在同一輪狀態，保留或拿掉某owner的錨點方向分量，各自按原縮步與接受規則完成一輪，再用固定相同內部loss比較接受端點。', '',
       'C^A = loss(沒有該方向分量的結果) − loss(完整方向的結果)。正值表示保留該分量讓這一輪的結果較好；負值表示保留後較差。分支不進下一輪，因此不是整段搜尋或完整參與者LOO價值。', '',
       '6個反向有益解析例全部C>0，而原A均為0；6個反向有害例全部C<0。兩類使用相同方向來源幾何，B都為20%，因此B的相同強度不能區分有益與有害。6個共線反向例B>0，但實際方向與結果相同，C=0。', '',
       '這證明基準能正確測出事先定義的局部決策差，不是低成本分數成功預測獨立真值的實驗。所有案例的好壞仍依固定內部目標，不宣稱外部品質或經濟報酬已成立。', '',
       '## 2. 方法與實際操作', '',
       '錨點與模型、目標、起點、原方向權重、模式、eta、delta和box固定。從完整g中扣掉指定owner已形成的H；其他方向權重不再正規化，模型不移除。完整及排除分支各最多10次縮步，只有嚴格loss下降才接受；沒有接受就保持原點。', '',
       '完整控制使用未插樁原solver與既有Observer各跑一次，完整結果、呼叫數與query/reply digest精確相同。分支引擎形成的完整端點、mode與eta轉換亦對齊原一輪。A/B維持原計算，沒有把兩者加總或轉成獎金。', '',
       '零方向處理：norm≤1e−8的分支不試走、保持原點。q/p/B可能null，但若端點已知，C仍可計算；不把方向未定義誤當結果差未定義。本輪補齊這個覆蓋範圍，沒有回填或覆寫舊檔對零替代方向的狀態。', '',
       '## 3. 解析驗證：24開發＋24保留', '',
       '8類各6組，前3開發、後3保留。全部參數與預期符號在執行前封存；開發24全過才執行保留24，沒有修正fixture、補種子或挑結果。模型為固定已知二次多項式，方向由原solver實際查詢產生，沒有任意指定與目標無關的H。', '',
       '獨立reference使用60位Decimal及有理數參數，重算模型、錨點斜率、方向、每個trial、接受點、mode/eta與C。方向／座標誤差門檻1e−10，其他數值1e−9。所有48案通過，最大絕對誤差約1.54×10⁻¹³。', '',
       '| 類別 | 開發／保留通過 | 正C | 負C | 近零C | 發現 |', '|---|---|---:|---:|---:|---|']
    labels={'positive_helpful':('正向有益','正向訊號確有局部改善'), 'positive_harmful':('正向有害','原A>0，但保留後結果較差'),
            'opposing_helpful':('反向有益','A=0仍可記錄正C'), 'opposing_harmful':('反向有害','不能因抵銷就發保護獎勵'),
            'collinear_no_effect':('共線無作用','B>0但方向及結果相同'), 'rejected_equal':('拒絕等值','兩邊都拒絕，C=0'),
            'boundary':('邊界','排除後零方向，仍有可定義的端點差'), 'cancellation':('精確／近抵銷','原零方向也可能有非零C')}
    for cat,(label,note) in labels.items():
        rows=[r for r in cs if r['category']==cat];p,n,z=count(rows,1e-8)
        t.append(f'| {label} | 3/3＋3/3 | {p} | {n} | {z} | {note} |')
    t += ['', '保留集是固定解析機制案例，不是新真實應用資料；分類成功不是八類情況在實務中的出現率。', '',
          '## 4. 可直接理解的代表案例', '',
          '### 4.1 反向但有益', '',
          '在 `opposing_helpful_1`，指定錨點的A=0、B=20%。兩邊都首試接受：', '',
          '- 保留：loss約999.825559。',
          '- 拿掉方向分量：loss約999.930000。',
          '- C約+0.104441，代表這個分量讓當輪接受結果更好。', '',
          '因此，原A的正向支持分帳沒有涵蓋這項作用；局部結果比較則能記下。此處說「有益」依固定內部尺，不是已建立可付0.104441比例或金額。', '',
          '### 4.2 相同B卻有害', '',
          '`opposing_harmful_0`的B同樣20%、A同樣0；保留loss約999.949723，拿掉約999.930000，C約−0.019723。同樣是抵銷訊號，這次保留反而較差，不能單用B決定獎勵。', '',
          '### 4.3 有抵銷但沒有結果價值', '',
          '`collinear_no_effect_1`兩邊的正規化方向與端點相同，loss都999.970000，C=0。B大於0只表示方向訊號有抵銷。', '',
          '### 4.4 原本零方向也可能不是沒有作用', '',
          '`cancellation_0`完整方向精確抵銷，保持原點loss=1000，p/B未定義。拿掉指定分量後接受loss約999.873333，C約−0.126667。這個分量在此局部比較中妨礙了改善；不能因q未定義而填C=0。', '',
          '另6個邊界例，拿掉唯一有效分量後零方向，完整分支則走到邊界並改善，C為正。原方向為零與替代方向為零都已涵蓋。', '',
          '## 5. 重讀舊69組分支：不是新效果確認', '',
          '核對10/27實驗與原pilot receipt全部列檔hash，重讀兩案17輪、69個owner×輪分支；從原始回覆獨立重建H、p、B、A、trial幾何、共同loss與接受端點。599個保存objective事件與兩次原solver保存回覆重播通過，未載入訓練模型或新增其推論。', '',
          '下面按原投影分層，只比較各自完成縮步後的接受結果。正C＝保留較好；負C＝保留較差；主要死區±1e−8：', '',
          '| 舊案例 | 原投影 | 正C | 負C | 近零C | 合計 |', '|---|---|---:|---:|---:|---:|']
    for case in sorted(set(r['case'] for r in old)):
        for sign,label in [(1,'正向'),(-1,'反向')]:
            rows=[r for r in old if r['case']==case and r['p']*sign>0];p,n,z=count(rows,1e-8)
            t.append(f'| {case} | {label} | {p} | {n} | {z} | {len(rows)} |')
    t += ['', '原A=0但C>1e−8：clean 4組、controlled_poison 10組；原A>0但C<−1e−8：clean 6組、controlled_poison 2組。這些差異是兩種分數回答不同問題的證據，不能把A直接改稱完整局部結果貢獻。', '',
          '死區改成±1e−6的敏感度：', '', '| 舊案例 | 原投影 | 正C | 負C | 近零C |', '|---|---|---:|---:|---:|']
    for case in sorted(set(r['case'] for r in old)):
        for sign,label in [(1,'正向'),(-1,'反向')]:
            p,n,z=count([r for r in old if r['case']==case and r['p']*sign>0],1e-6);t.append(f'| {case} | {label} | {p} | {n} | {z} |')
    t += ['', '全部保留原資料及近零事件。69組巢狀於兩團隊，不當69個獨立任務；C不跨角色或跨輪加總成全程收益。舊結果與本次重讀一致，沒有回填成正式確認。', '',
          '## 6. 技術與成本帳', '',
          '- 新增8項測試通過：零方向不查詢、非法／錯形狀梯度拒絕、dynamic停止、secant切模式、縮步接受、邊界與同owner先聚合／缺方向拒絕。未修改原模組，因此沒有冒稱重跑原49項測試。',
          '- 48個原始紀錄搜尋完成原scalar A/E重建、原solver重播及B獨立核對；額外核對各case的target A與原帳本連結、Decimal方向分帳一致。',
          '- 獨立audit不匯入分支引擎，重算48個解析分支；故意把C加1的48項篡改全部被拒絕。',
          f'- 主實驗96次原單輪搜尋（48控制＋48觀察）；96次分支決策（48完整＋48消融）。共{s["new_synthetic_api_calls"]}次新合成API，其中原搜尋808次、分支完整驗證118次、消融96次。完整驗證與新增消融成本分開，不能用平均數保證部署成本。',
          f'- 主流程約{s["seconds"]:.3f}秒，包含來源／舊receipt核對、單元測試、解析檢查與落盤，不含後續獨立audit及報告。未量RSS/native memory，未進行部署延遲測量。',
          '- 已訓練模型新呼叫0、新訓練0、pooled0、實際付款0。沒有載入真實任務truth；已知解析多項式用於機制診斷。', '',
          '未來若完整搜尋已有原始來源回覆，可不重做完整控制；但每個owner消融仍可能產生新試走點並需模型評分，最壞每分支10個trial×模型數。這個基準不因本輪合成查詢少就變成低成本估計器。', '',
          '## 7. T2驗收與後續建議', '',
          '| 項目 | 結果 |', '|---|---|',
          '| 開發24案、保留24案解析分支與事前符號 | 全通過 |',
          '| 反向有益6案能記正C，A=0不漏記 | 通過 |',
          '| 共線／雙拒絕不虛構正結果價值 | 通過 |',
          '| 原／替代零方向與null分開處理 | 通過 |',
          '| 舊69組方向、原回覆及結果重讀 | 通過，舊資料探索 |',
          '| 低成本估計器／公平獎金／外部品質 | 未建立 |', '',
          '**已完成：**可核對、能涵蓋有益制衡的「固定內部尺、原路徑、完整當輪決策差」參考基準。這是技術與解析機制驗證，不是付款效度通過。', '',
          '**未建立有效性：**是否符合獨立外部品質、如何低成本近似、每輪如何累積轉分配、公平、抗操弄、新真實任務泛化及三池分潤。A維持方向支持分帳，B維持抵銷紀錄，C另列結果差，沒有改成新付款分數。', '',
          '**下一步提案：**可依三池計畫進T3模型決策價值，沿用同樣固定共同尺、完整當輪與自身／他人評分診斷。T2參考基準可先固定，低成本錨點估計與T4分配須另設計；不直接把B變成獎勵，不自行啟動T3或成果付款。', '',
          '## 8. 重現與保存', '',
          '```sh',
          'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/yclee/anaconda3/bin/python -m contribution_analysis.three_pool_t2_anchor_v1.run --output contribution_analysis/three_pool_t2_anchor_experiment_v1_replay',
          'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/yclee/anaconda3/bin/python -m contribution_analysis.three_pool_t2_anchor_v1.audit --root contribution_analysis/three_pool_t2_anchor_experiment_v1_replay',
          '```', '',
          '輸出須是新資料夾。執行前來源、48個有理數參數及標籤、控制／觀察回覆、分支trial、成本與獨立核對均保存；舊檔與原演算法未修改。執行後報告器及補充A連結檢查另標delivery階段。研究索引更新，但執行前規約不覆寫。']
    return '\n'.join(t)+'\n'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root.resolve()
    output=render(root);name='three_pool_t2_anchor_results_20261028.md'
    for path in [root/name,WORK/'docs'/name]:
        with path.open('x') as f:f.write(output)
    delivery=root/'delivery_source_snapshot';delivery.mkdir()
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ['.py','.md']:shutil.copy2(p,delivery/p.name)
    save(root/'completion_receipt.json',dict(complete=True,technical_and_analytic_reference_passed=True,low_cost_estimator_established=False,
         economic_payment_validity_established=False,new_formal_confirmation=False,actual_payments=0,notion_modified=False,
         created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),report='docs/'+name,report_sha256=sha(WORK/'docs'/name),
         file_sha256={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    print('REPORT',WORK/'docs'/name)


if __name__=='__main__':main()
