"""Evidence-only T3 report; no settlement or new model calls."""
import argparse
import datetime
from pathlib import Path
import shutil
from contribution_analysis.custom_realized_gain_v1.schema import read,save,sha

WORK=Path(__file__).resolve().parents[2]


def counts(rows,key,eps):
    values=[r['score'][key] for r in rows if r['score'][key] is not None]
    return sum(v>eps for v in values),sum(v< -eps for v in values),sum(abs(v)<=eps for v in values)


def render(root):
    s=read(root/'summary.json');audit=read(root/'independent_audit.json');old=read(root/'historical_reanalysis.json');cs=s['analytic_cases']
    if not audit['accepted']:raise ValueError('audit_required')
    labels={'agreement':'模型一致','useful_information':'有用不同資訊','accurate_redundant':'準確但冗餘','local_error':'局部錯誤',
            'fixed_bias':'固定偏差','individual_scale':'個別尺度放大','optimistic_acceptance':'過度樂觀接受',
            'prevent_bad_step':'有效阻止壞步','both_no_update':'雙方不動','concentrated_weights':'權重集中／退化'}
    t=['# T3：模型評分的當輪決策價值驗證結果', '',
       '日期：2026-10-28。規約：`docs/three_pool_t3_model_design_v1_20261028.md`。輸出：`contribution_analysis/three_pool_t3_model_experiment_v1_20261028/`。', '',
       '**已完成：60個解析設定、既有模型單輪基準重用、19項測試、獨立Decimal及舊interpreter核對。技術基準通過；直接將內部C當獨立品質成果獎金依據的效度未通過。**', '',
       '## 1. 主要結論', '',
       'C^M可以正確衡量「該模型評分參與後，當輪結果是否更符合原完整團隊內部目標」。它能區分冗餘、局部有害決策與阻止壞步，並比E或呼叫次數更直接對齊決策作用。', '',
       '但共同評分尺也包含該模型本身。已知解析案例顯示，模型可取得正C，同時讓獨立設定的解析參考結果更差。個別回覆尺度變大時，兩個決策端點完全不變，C卻從0.1增到0.7甚至0.9。這是自評／尺度敏感性反例，不能以技術重建通過稱為成果分潤有效。', '',
       'peer去自身評分也不是自動解法：真正提供正確不同資訊的案例，其他模型恰好偏錯，peer會給負值。故本輪維持原共同尺為主定義，自己／他人及解析品質分開診斷，沒有事後換裁判。', '',
       '## 2. 本輪做法與比較尺', '',
       '同一輪前位置、模式、eta、delta、box、錨點與原錨點方向權重，完整與排除模型分支各自重算決策loss、形成方向、縮步及接受。被排除模型不參與該分支的評分，但其錨點保留。兩個端點鎖定後才補齊共同評分，排除分支不回流原主路徑。', '',
       '主分數：C^M=L_ref(x_minus)−L_ref(x_full)。L_ref固定原完整模型與原模型權重。', '',
       '分解：C=own+others；own為被排除owner模型的加權評價差，others為其餘模型的加權差。peer=others／剩餘權重，只作敏感度。模型服務次數另帳，E仍是完整分支端點改善評價。', '',
       '使用者所要的「評分讓決策改善多少」需要同時說清楚誰判定改善。這個C只回答固定內部偏好；解析診斷D=r(x_minus)−r(x_full)使用已知函數、鎖定後才算，不進搜尋、C或付款。真實任務沒有因此取得已知真值或部署裁判。', '',
       '重用10/21的AST engine與branch_state，以及其獨立one_round interpreter；不修改舊程式。舊engine的s.alphas介面只作錨點方向權重傳入，本輪模型v另外保存在objective，兩種角色不混用。K=2、M=3及多模型owner群組移除另有技術測試。', '',
       '## 3. 60個固定解析設定', '',
       '10類各6組：先30個開發全通過，再30個保留。兩維幾何的主要變動在第一維，兩個模型；不能以這種簡單案例代表高維真實應用。參數、預期C／D符號在執行前封存，沒有依結果換案例。', '',
       '60設定中59個可計算；1個移除後剩餘有效模型權重為零，按規約保留未定義（C=null，不是0）。獨立60位Decimal從原模型函數重算方向、trial、接受、C與分解／E／RMSE，全部符合預期；最大絕對誤差約9.10×10⁻¹³。', '',
       '| 類別 | 開發／保留通過 | C正／負／近零 | 未定義 | 解析診斷D正／負／近零 |', '|---|---|---|---:|---|']
    for key,label in labels.items():
        rows=[r for r in cs if r['category']==key];cc=counts(rows,'C',1e-8);dd=counts(rows,'D_analytic',1e-8);u=sum(r['score']['C'] is None for r in rows)
        t.append(f'| {label} | 3/3＋3/3 | {cc[0]}/{cc[1]}/{cc[2]} | {u} | {dd[0]}/{dd[1]}/{dd[2]} |')
    t += ['', '解析技術通過包含「如預期出現C與D異號」，不是60個案例全部證明分數品質有效；未定義案例也不是漏失或技術失敗。解析保留案例不是新真實任務確認，不能以反例數除59宣稱實務誤判率。', '',
          '## 4. 四個關鍵發現', '',
          '### 4.1 準確、改善評價高，都可能沒有當輪額外價值', '',
          '準確但冗餘的6案，模型0對解析參考RMSE=0、E=+0.05，但保留與移除得到同一端點，C=0。模型一致類亦有正E而C=0。表示評分品質、服務履約與這一輪額外決策價值是不同問題；C=0不能取消T1已成立的基本服務費。', '',
          '局部錯誤6案的E為正（0.15–0.20），但C均−0.2，解析D也−0.2：原完整搜尋仍能改善內部loss，不表示該模型參與比排除更好。', '',
          '### 4.2 阻止壞步可以有正價值，即使E=0', '',
          '`prevent_bad_step_0`完整分支10次都不接受，保持原點；排除模型後其他評分接受x=+0.1。共同內部及解析參考在排除點較差0.01，因此C=D=+0.01，E=0。其餘5案同樣記到正值。', '',
          '此處排除分支依不同目標接受，因此能出現「不動勝過排除後更新」。沒有用共同尺把排除點重新拒絕，也沒有把被拒絕trial重複分帳。', '',
          '### 4.3 相同的內部分數分解，可對應相反的解析品質', '',
          '| 案例 | C | own | others | peer | 解析D |', '|---|---:|---:|---:|---:|---:|',
          '| 有用不同資訊_0 | +0.10 | +0.15 | −0.05 | −0.20 | +0.20 |',
          '| 過度樂觀接受_0 | +0.10 | +0.15 | −0.05 | −0.20 | −0.20 |', '',
          '前者模型0較準，完整走−0.1、排除走+0.1，完整較好；後者完整走+0.1、排除走−0.1，完整反而較差。C／own／others／peer與E可相同，但依同一解析參考r_plus，結果好壞相反。單靠這些內部數值，沒有自動辨認誰的評分可信的依據。', '',
          '因此不能因自身項主導就一律剔除，也不能認定其他模型的共識就是獨立真值。', '',
          '### 4.4 尺度放大可提高C，而決策結果沒有變好', '',
          '尺度類模型0=s(1000+x)，模型1=1000−x，原模型權重各1/2。所有6案的完整端點都−0.1、排除端點都+0.1，解析參考r_minus的D皆−0.2。', '',
          '| s | C | 解析D |', '|---:|---:|---:|',
          '| 2 | +0.10 | −0.20 |', '| 4 | +0.30 | −0.20 |', '| 8 | +0.70 | −0.20 |', '',
          '完整共同尺包含被放大的模型，所以跨尺度案例的L_ref也改變；每個案例內兩分支仍用同一尺。這個差異揭露分數對自評尺度的依賴，不能解讀成同一獨立品質尺下貢獻真的上升。', '',
          '若付款規則單調獎勵高C，這是必須處理的風險，但本輪沒有套用任何付款公式，也未證明參與者能在T0版本與准入限制下實際進行此操作。它是可稽核的效度／威脅模型反例，抗操弄仍屬後續T5。', '',
          '## 5. 既有資料重讀與新診斷', '',
          '核對10/21完成receipt、來源與鎖定端點hash；不載入訓練模型或pooled。舊獨立interpreter重新核對24團隊的286個完整輪及1174個模型分支，包含4pilot的141筆與20開發團隊的1033筆。從保存原始回覆重建C、own、others，新增E／peer診斷。', '',
          '以下僅列原20開發團隊，死區±1e−4；不同欄可重疊，不是獨立事件樣本：', '',
          '| 情境 | 比較數 | own/others異號（兩者皆明確） | C正但peer負 | E正但C負 | 完整未接受但C正 |',
          '|---|---:|---:|---:|---:|---:|']
    for scenario,label in [('clean','乾淨'),('controlled_poison','受控污染')]:
        rows=[r for r in old if r['phase']=='development' and r['scenario']==scenario];e=1e-4
        fields=[len(rows),sum(r['own']*r['others']<0 and abs(r['own'])>e and abs(r['others'])>e for r in rows),
                sum(r['C']>e and r['peer']< -e for r in rows),sum(r['E']>e and r['C']< -e for r in rows),sum(not r['full_accepted'] and r['C']>e for r in rows)]
        t.append('| '+label+' | '+' | '.join(str(v) for v in fields)+' |')
    t += ['', '這些只說明內部評價衝突與E／C語義不同，不證明C或peer誰比較符合外部品質。近零、未接受輪均保留，不刪除支持較弱的事件。團隊與輪次巢狀，不能用1033當獨立任務樣本。', '',
          '舊外部結果保留：乾淨／污染平均排名相關約0.3255／0.3930，正負一致率67.50%／72.00%；兩情境均未通過原整體門檻。這些數字引用已核對hash的舊結果，本輪沒有新pooled推論、重新選門檻或宣稱外部效度改善。', '',
          '### 舊敘述計數差異', '',
          '舊文字報告把pilot比較數寫成201、全輪比較寫成1234；原始四pilot分別24、27、45、45，合計141；舊verification.json也列同樣數量，24團隊合計1174。原20開發1033筆與其效度結論不變。本輪以原始檔及舊獨立核對檔為準，另存historical_count_note.json，不改舊報告或封存結果。執行前規約引用的201亦保留原貌，這是本輪核對後的明確更正說明。', '',
          '## 6. 技術驗收與成本', '',
          '- 原12項模型基準測試以unittest discovery重跑通過，未執行會寫舊technical_results的script入口。新增7項群組owner、K≠M、未定義、零權重、phase鎖定、排除模型禁查、共同尺不二次接受及權重分離測試通過。',
          '- 60案的原未插樁控制與AST engine完整結果／原始查詢序列精確相同；可定義分支均通過舊獨立interpreter和新Decimal雙重核對。',
          '- 稽核原始API函數回覆、鎖定前／後stage，排除模型在minus決策中無呼叫；common補評均在端點鎖定後。',
          '- 對60案各篡改C、own、peer（含把未定義改0），180項全部被稽核發現。', '',
          f'主實驗120次完整一輪搜尋（60控制＋60engine）、59次可定義模型排除決策。合成API共{s["new_synthetic_api_calls"]}次，分項：', '',
          '| 項目 | 合成API次數 |', '|---|---:|',
          '| 原控制 | 1056 |', '| engine完整分支 | 1056 |', '| 排除分支（含目標重設） | 408 |', '| 鎖定後共同評分／E端點補評 | 358 |', '',
          f'21點品質診斷另有2520次解析模型函數計算，使用1260個「案例×參考網格點」，不作模型服務API請領。主流程約{s["seconds"]:.3f}秒，含舊資料核對、測試、解析驗證與保存，不含後續獨立audit／報告，不是部署延遲保證。', '',
          '新已訓練模型呼叫0、新訓練0、pooled0、實際付款0。真實任務truth未載入，已知解析參考只作診斷。沒有建立低成本近似，也沒有與完整LOO效果作替代性比較。', '',
          '## 7. Gate與下一步', '',
          '| 驗收項目 | 結果 |', '|---|---|',
          '| 原控制／engine／Decimal／獨立interpreter | 通過 |',
          '| 開發30＋保留30，包含1個預期未定義 | 通過 |',
          '| own+others=C、peer/E/原回覆重建 | 通過 |',
          '| 模型不參與決策但可在鎖定後補共同評分 | 通過 |',
          '| C直接代表獨立品質的模型成果價值 | 未通過：尺度與樂觀評分反例 |',
          '| 去自己改peer即可修復 | 未建立；有用不同資訊例亦會被peer否定 |',
          '| 低成本估計、公平、激勵與三池付款 | 未建立 |', '',
          '**已完成：**模型局部內部決策差的可核對基準、分解與反例。它能記錄有用與阻止壞步，也揭露內部偏好不等於獨立品質；技術成功沒有消除舊外部效度失敗。', '',
          '**未建立有效性：**可信的部署品質裁判、評分準入／校準、低成本分數、成果獎金分配、抗操弄、新真實任務泛化。T1基本保障按履約不受本輪C正負取消；模型成果池仍未結算。', '',
          '**下一步提案：**T3基準先固定，不宜直接把C或peer帶入T4模型成果比例。先設計可取得且可稽核的任務品質／模型準入與尺度一致性驗證，再決定主張只限內部偏好還是獨立品質；如研究低成本近似，仍須保留這個效度限制。T2錨點同樣依內部尺，不能因本輪模型問題改稱已具外部效度。', '',
          '本輪停止模型付款候選升級，不自動訓練、新增pooled、調alpha、改錨點、pruning、啟動T4–T7或更新Notion。', '',
          '## 8. 重現與保存', '',
          '```sh',
          'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/yclee/anaconda3/bin/python -m contribution_analysis.three_pool_t3_model_v1.run --output contribution_analysis/three_pool_t3_model_experiment_v1_replay',
          'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/yclee/anaconda3/bin/python -m contribution_analysis.three_pool_t3_model_v1.audit --root contribution_analysis/three_pool_t3_model_experiment_v1_replay',
          '```', '',
          'run資料夾必須不存在。原模型基準及T0/T1/T2未修改；執行前來源、60參數與標籤、逐API stage、決策、鎖定端點、分解、未定義及成本另存；報告器與舊計數更正說明屬delivery階段。所有失敗與未定義保留，不回填正式確認。']
    return '\n'.join(t)+'\n'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root.resolve()
    output=render(root);name='three_pool_t3_model_results_20261028.md'
    for path in [root/name,WORK/'docs'/name]:
        with path.open('x') as f:f.write(output)
    delivery=root/'delivery_source_snapshot';delivery.mkdir()
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ['.py','.md']:shutil.copy2(p,delivery/p.name)
    save(root/'completion_receipt.json',dict(complete=True,technical_reference_passed=True,payment_candidate_validity_passed=False,
        low_cost_estimator_established=False,new_formal_confirmation=False,actual_payments=0,notion_modified=False,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),report='docs/'+name,report_sha256=sha(WORK/'docs'/name),
        file_sha256={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    print('REPORT',WORK/'docs'/name)


if __name__=='__main__':main()
