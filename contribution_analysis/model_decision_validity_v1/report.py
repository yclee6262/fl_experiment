import argparse
from pathlib import Path
from contribution_analysis.custom_realized_gain_v1.schema import read,save,sha

WORK=Path(__file__).resolve().parents[2]
NAME='model_decision_validity_results_20261028.md'


def generate(root):
    s=read(root/'summary.json');a=read(root/'independent_audit.json');groups=s['analysis']['groups']
    if not a['passed']:raise ValueError('audit_failed')
    labels=dict(effective_information='有效資訊',erroneous_information='錯誤資訊',effective_restraint='有效制衡',
                erroneous_block='錯誤阻擋',redundant_information='資訊重複')
    rows=[]
    for cat,label in labels.items():
        g=groups['all'][cat];rows.append(f"| {label} | {g['raw_C']['matches']}/6 | {g['C']['matches']}/6 | {g['own']['matches']}/6 | {g['peer']['matches']}/6 | {groups['heldout_parameter'][cat]['C']['matches']}/3 |")
    examples=[]
    for cat,label in labels.items():
        r=next(r for r in s['rows']if r['id']==cat+'_0')
        examples.append(f"| {label} | {r['full'][0]:.8g} | {r['minus'][0]:.8g} | {r['raw_C']:.8g} | {r['C']:.8g} | {r['own']:.8g} | {r['others']:.8g} | {r['D']:.8g} |")
    match=groups['all']['all'];dev=groups['development']['all']['C']['matches'];held=groups['heldout_parameter']['all']['C']['matches']
    status='通過'if s['analysis']['quality_gate_passed']else'未通過'
    # Report states evidence and its limits, even when the preregistered validity gate fails.
    return f'''# 模型池換算後貢獻：單輪決策作用辨識驗證結果

日期：2026-10-28。執行前設計：[model_decision_validity_design_v1_20261028.md](model_decision_validity_design_v1_20261028.md)。輸出：`{root.relative_to(WORK) if root.is_relative_to(WORK) else root}`。

**已完成：30個新解析設定、90次單輪求解及180個固定端點尺度變體；技術稽核通過。決策改善辨識gate{status}：換算C與已知答案D同號／同為零共{match['C']['matches']}/30，開發{dev}/15、參數保留{held}/15。尺度一致不能建立品質有效性。**

## 1. 實際驗證內容

每類6案，15開發先完成並封存checkpoint，再執行15個未參與調整的參數保留。全部是新構造的小型解析機制案例，刻意包含反例，非實務隨機樣本；比例不是實際部署準確率，也不是新的正式外部確認。舊T3反例早已讀過，本輪不把相同機制換參數包裝成未知發現。

使用原切線法／割線法的一輪求解：錨點、a、起點、eta=.1及box固定；full保留兩模型，minus只移除模型0評分、保留其錨點並正規化剩餘v。每案另執行一次未插樁full控制，確認查詢序列與結果不變。兩分支各自用原內部objective接受；共同評分不重新決定接受，也不是完整參與者LOO。

所有設定先固定25個共同probe及s=Q90−Q10，封存後才啟動第一個新搜尋。原搜尋未使用此尺度。端點鎖定後以完整團隊原v計算C_norm=Σv_m(loss_minus−loss_full)/s_m；own、others分開保存，peer=others/剩餘原v總和。對照raw C不換尺度。D=reference(minus)−reference(full)只作事後診斷，不進方向、接受、C或分配。

## 2. 五類作用的辨識

表內為分數正負／近零與診斷D符合的案數。正負死區±1e−10；本輪沒有未定義案例，未定義處理另由尺度技術測試覆蓋。own與peer都是診斷，並非已驗證替代方案。

| 情境 | 原raw C | 換算C | own部分 | peer其他模型 | 換算C參數保留 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

總計raw C為{match['raw_C']['matches']}/30；換算C為{match['C']['matches']}/30；peer為{match['peer']['matches']}/30。不按較高總符合率事後挑方案；每類都通過才達本輪必要gate。判錯ID與完整三分類混淆矩陣保存在summary.json。

換算C的有害作用仍正分共{len(s['analysis']['harmful_positive_ids'])}案、有益作用負分共{len(s['analysis']['helpful_negative_ids'])}案、有益／有害被歸近零共{len(s['analysis']['nonzero_missed_ids'])}案。這些錯誤保留，不調門檻或換probe補救。

## 3. 可逐項核對的例子

下表取每類第0案，列x0；另一座標均為0。較低診斷loss較好，故D>0表示模型0使決策改善。

| 情境 | full位置 | minus位置 | raw C | 換算C | own | others | 診斷D |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(examples)}

有效資訊與錯誤資訊使用完全相同模型、權重、probe、搜尋查詢、端點與內部評分，只讓事後已知任務答案不同。換算C仍相同，D一正一負。有效制衡／錯誤阻擋也是相同可觀測資料、相反D的一對。

有效制衡時，full拒絕有害移動，minus卻接受移動。尺度換算會改變兩個評分者在共同尺中的相對影響，因此可能讓raw正分變成換算負分；不能把它解讀成這個模型實際造成傷害。應同時讀D、own與others，區分內部判斷與獨立任務品質。

12組matched pair逐項驗證可觀測資料完全一致。只依這些相同資料的確定性分數，無法在兩個不同正確答案的世界給出不同判斷。對本輪每組相反D，任何這種判斷至少錯一案；加上6個重複資訊零作用案，30案最多能符合18案。這是此刻意配對集合的資訊上限，不是所有實務資料的上限，也不證明所有外部品質訊號都無效。

peer同樣受限：只讓其他模型評價，可能錯罰提供正確少數資訊者，不能以刪去自評宣稱解決。own/others分解能說明誰認為變好，不能單憑分解得知誰正確。

## 4. 尺度、技術與獨立稽核

每案每模型將probe、full與minus loss一起乘.1/10/100，180個固定端點變體全部通過，最大C／own／others變化{s['max_invariance_error']:.3g}。未重跑縮放後搜尋，因此仍不主張完整路徑尺度不變。

16項單元測試（既有尺度12項＋新統計4項）通過。獨立60位Decimal interpreter檢查{a['branch_decisions']}個分支決策的梯度、端點、trial接受、eta與mode；30組控制查詢重播一致。獨立Fraction分位數及分數重算{a['score_panels']}個panel，並核對{a['raw_reply_checks']}個raw回覆、{a['probe_reply_checks']}個probe與有理數多項式。最大獨立數值差{a['max_reference_error']:.3g}。每案故意C+1、scale+1、端點+.01、D翻號／零改1，{a['mutations_detected']}項篡改全被偵測。

來源／design／fixtures執行前封存；calibration在搜尋前鎖定，開發checkpoint早於保留搜尋；原T3、尺度實驗、錨點池receipt逐檔未變。原演算法、alpha、錨點池及Notion未改，新分析不回填舊正式確認。

## 5. 成本與適用範圍

新解析搜尋及共同評分API {s['synthetic_search_and_common_api']}次：control {s['stage_api']['control']}、full {s['stage_api']['full']}、minus {s['stage_api']['minus']}、common {s['stage_api']['common']}。另有1500次解析probe；舊adapter的RMSE診斷含1260次模型多項式及1320次reference多項式評估（含每案D的兩端點）。獨立Decimal／Fraction核對為另行算術，不冒稱模型API。180倍增變體只有數字換算，沒有新增模型查詢。

90次單輪求解包含30控制、30觀察full與30排除；不是90次完整多輪搜尋。主流程{s['seconds']:.3f}秒，含測試、載入、校準、求解、內嵌獨立核對及寫檔，未含最後audit／報告。稽核前產物{s['bytes_before_audit']} bytes，未測RSS或實務延遲；此小解析成本不能推論部署低成本。新已訓練模型呼叫、訓練、pooled、真實truth資料、實際付款皆0。

## 6. 判讀與下一步

**已完成：**尺度一致的工程性質延伸至新固定端點；模型-only單輪作用、五類機制診斷、own／others分解及反例資訊限制已核對。技術正確與品質不通過可以同時成立。

**未建立有效性：**換算C代表獨立決策品質、可信性／公平、原搜尋尺度不變、真實任務泛化、模型池有號累積／固定預算比例與低成本。不能稱本輪完成了有效模型報酬公式。

**下一步提案：**依設計停止把目前換算C升級模型池比例，也不直接改peer。需先決定獎勵目標：若獎勵固定內部共同偏好，可保留C作「共同尺下的局部決策作用」，接受它與任務品質可能相反；若堅持獎勵實際品質改善，需先定義在實際情境可取得、可信且可驗收的品質證據，再另行規劃。pooled及truth維持離線診斷角色，不因本輪失敗自動加入付款。這個選擇確定前，暫停擴大真實模型／分配實驗。

## 7. 重現

參見程式README。使用全新output依序執行run、audit、report；全部輸出exclusive-create，保留品質失敗與原反例，不覆寫此次封存。
'''


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--publish-docs',action='store_true');args=ap.parse_args()
    root=args.root.resolve();content=generate(root)
    with (root/NAME).open('x') as f:f.write(content)
    if args.publish_docs:
        with (WORK/'docs'/NAME).open('x')as f:f.write(content)
    files={str(p.relative_to(root)):sha(p)for p in root.rglob('*')if p.is_file()and p.name!='completion_receipt.json'}
    save(root/'completion_receipt.json',dict(complete=True,technical_audit_passed=True,
        quality_gate_passed=read(root/'summary.json')['analysis']['quality_gate_passed'],file_sha256=files,new_formal_confirmation=False))
    print('SEALED',len(files),'files; report',root/NAME)


if __name__=='__main__':main()
