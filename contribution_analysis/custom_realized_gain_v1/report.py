"""Render the saved pilot evidence. Does not run a model or modify scores."""
import argparse
from pathlib import Path
from .schema import read, lines, sha


def render(root, output):
    summary=read(root/'summary.json');audit=read(root/'independent_audit.json');text=[]
    def add(s=''):text.append(s)
    add('# Custom 實現改善歸屬 v1：實作與兩案 pilot 結果')
    add('\n日期：2026-10-27。狀態：**實作與有限技術驗證完成；經濟有效性尚未建立。**')
    add('\n## 1. 本輪結論')
    add('\n依原設計完成唯讀觀察器、逐次回覆證據、離線 A/E/P 計分、獨立 scalar 重建與原求解器重播。32 項測試通過，兩個固定舊 pilot 的查詢序列、回覆與求解結果和 control 完全一致；分帳及端點改善恆等式通過。')
    add('\n新增參與者模型呼叫為0，並非零運算成本。兩案搜尋時間相對有最小計數／摘要紀錄的 control 約增加29.3%及26.9%，高於設計的15%工程目標；每次搜尋本身僅約7–13毫秒，且只有3組量測，這些比例是本機描述，尚未建立穩定延遲結論。離線計分與落盤成本另計。')
    add('\n本輪只驗證本次程序的成果歸屬可計算且可重現；沒有以 LOO、pooled、truth、模型品質或付款結果替代驗收。J保留原定義。本版不宣稱公平、激勵相容或完整參與者價值。')
    add('\n## 2. 固定方法與實際資料')
    add('\n- 錨點 A：依合成方向的投影，先在固定 owner 內有號相加，再取正部，按份額分配各接受步的 G。')
    add('- 模型 E：既有權重乘以接受前後的個別 loss 下降，保留負值。它是改善評價，不是品質或決策因果貢獻。')
    add('- P：有號「錨點 × 模型」方向投影表；服務成本另列。A/E不可相加，也不輸出支付比例。')
    add('- 固定沿用 layer2 pilot 每情境首案，seed均為2026101600；clean三人、controlled_poison五人，2D、custom30、原固定起點、原權重、CPU單執行緒。新確認=false。')
    add('- 每案先1組control/record技術核對，再3組交替順序計時，共16次原搜尋。沒有額外刪人分支或重訓。')
    add('\n## 3. 已完成：計算與搜尋忠實性')
    add('\n| 案例 | 外層輪／接受步 | 保存objective／模型呼叫 | 獨立重建最大絕對差 | 原solver重播 |')
    add('|---|---:|---:|---:|---|')
    for c in summary['cases']:
        ver=next(z['verification'] for z in audit['cases'] if z['case']==c['case'])
        add(f"| {c['scenario']} | {c['iterations']}／{c['accepted']} | {ver['evaluations']}／{ver['queries']} | {ver['max_absolute_error']:.3g} | 通過 |")
    add('\n32項測試涵蓋：手算矩陣、owner先聚合、K≠M、非均勻／零權重、負E、小改善不刪除、強抵銷、正交訊號、同owner等分複製、固定trace尺度、重複／缺失事件拒絕、首試／縮步成功、secant切dynamic、dynamic失敗、零方向、相同錨點、box探測截短及跳過、無退火、不同history metric、零輪／正常輪次預算、隨機回覆拒絕。')
    add('\n獨立驗證器不匯入accounting.py；以保存的逐模型回覆重建loss、方向、P、A/E，再將完整objective序列餵給未插樁原solver，逐點核對精確輸入及完整回傳。它不新增模型推論，不是獨立資料泛化。')
    add('\n開發階段曾有一個強抵銷合成fixture因生成負absolute loss被正確拒絕；已提高fixture共同基準而保持差商／分配公式不變，之後32項測試通過，才凍結來源並執行pilot。沒有失敗pilot、沒有以修改公式救結果。')
    add('\n## 4. 實現改善與個人帳')
    for c in summary['cases']:
        add(f"\n### {c['scenario']}：{c['case']}")
        add(f"\n內部loss：{c['initial_loss']:.12f} → {c['final_loss']:.12f}，G={c['G']:.12f}。")
        add('\n| 參與者ID | 錨點 A | A占本次總改善 | 模型改善評價 E |')
        add('|---|---:|---:|---:|')
        for i,A,s,E in zip(c['owners'],c['A'],c['share'],c['E']):add(f'| {i} | {A:.12f} | {100*s:.4f}% | {E:+.12f} |')
        add(f"\nΣA與ΣE分別等於G（容差內）；兩者各是一份完整的改善視角，不能合計成2G。敏感標記輪數={c['sensitive']}，不代表已證明所有擾動下穩健。")
    add('\n## 5. 代表案例的正確解讀')
    add('\nclean第一步G=0.007379269848。錨點9/10/5的投影約為+0.014654/+0.017338/−0.009425，因此當步份額約45.805%/54.195%/0%；agent5的抵銷仍保存在P及方向帳。零份額只表示這套正向支持分配規則沒有給它本步成果，不證明它沒有制衡價值。')
    add('\nclean的agent5最終E約−0.004389，意思是它認為最終解較起點更差；團隊加權目標仍改善。不能推論它是壞模型，也不能直接扣款。')
    add('\ncontrolled_poison的agent15錨點A=0，但模型E約+0.098566，是本案最大的模型改善評價。這說明角色帳有必要分開；不能以A=0宣稱該參與者無價值，也不能因E最大宣稱其模型最準確。這些只是事前固定兩案的描述，不是情境間因果比較。')
    add('\n## 6. 抵銷與微擾診斷')
    add('\n固定登錄診斷：對每個已接受步、每個未跳過錨點與模型，將單一來源model loss分別加減 `1e-6*max(1,|來源loss|,|目前loss|)`，重建合成方向及份額，G、原幾何、h及權重固定。這是分配映射的局部敏感性，不是合法重跑搜尋或品質反事實；不改正式分數、不新增查詢。')
    add('\n| 案例 | 擾動測試／未定義 | 最大單輪份額變動 | 超過1個百分點次數 |')
    add('|---|---:|---:|---:|')
    for row in audit['cases']:
        p=row['sensitivity'];label=next(c['scenario'] for c in summary['cases'] if c['case']==row['case'])
        add(f"| {label} | {p['attempted']}／{p['undefined']} | {100*p['max_owner_share_change']:.6f}個百分點 | {p['changes_above_one_percentage_point']} |")
    clean=summary['cases'][0];p=audit['cases'][0]['sensitivity'];ds=lines(root/clean['case']/'diagnostics.jsonl');worst=ds[p['worst']['t']]
    add(f"\nclean最大局部變動出現在第{p['worst']['t']+1}輪，該步G={worst['G']:.12f}，只占全程改善{100*worst['G']/clean['G']:.4f}%；這次擾動對該步A的最大變化約{worst['G']*p['max_owner_share_change']:.3g}，折合全程份額約{100*worst['G']/clean['G']*p['max_owner_share_change']:.6f}個百分點。這只是局部最大案例的全程換算，不聲稱是所有擾動下全程最大值。")
    add('\n兩案沒有rho<1e−6的輪；但clean仍有單輪份額敏感，表示極端抵銷標記不是完整穩定性保證。合成強抵銷測試亦證明：份額有界，不代表方向判斷抗擾動。')
    add('\n同owner、固定總權重的重複錨點測試通過；只代表此受控分拆不增加其份額。若換成不同owner或增加權重，不保證不變。例：投影[3,1,−1]若正負項同owner，其份額2/3；拆成不同owner後原正向項份額可到3/4，不能宣稱抗虛假身分。')
    add('\n## 7. 成本與工程目標')
    add('\n| 項目 | clean | controlled_poison |')
    add('|---|---:|---:|')
    c,p=summary['cases']
    rows=[('原搜尋模型呼叫',c['costs']['single_search']['api_calls'],p['costs']['single_search']['api_calls']),
          ('唯一模型／輸入鍵',c['costs']['single_search']['unique_inputs'],p['costs']['single_search']['unique_inputs']),
          ('新增模型呼叫',0,0),
          ('control搜尋中位數 ms',1000*c['costs']['control_median_seconds'],1000*p['costs']['control_median_seconds']),
          ('record搜尋中位數 ms',1000*c['costs']['record_median_seconds'],1000*p['costs']['record_median_seconds']),
          ('record/control',c['costs']['record_control_ratio'],p['costs']['record_control_ratio']),
          ('離線計分 ms（單次）',1000*c['costs']['offline_accounting_seconds'],1000*p['costs']['offline_accounting_seconds']),
          ('主要事件／分數落盤 ms',1000*c['costs']['events_serialization_seconds'],1000*p['costs']['events_serialization_seconds']),
          ('原始事件加分數輸出 KB（補充前）',c['bytes_before_supplement']/1000,p['bytes_before_supplement']/1000),
          ('觀察版Python峰值 bytes',c['costs']['observer_search_python_peak_bytes'],p['costs']['observer_search_python_peak_bytes']),
          ('離線計分Python峰值 bytes',c['costs']['offline_python_peak_bytes'],p['costs']['offline_python_peak_bytes'])]
    for name,x,y in rows:
        fmt=lambda z:f'{z:.6f}' if isinstance(z,float) else str(z)
        add(f'| {name} | {fmt(x)} | {fmt(y)} |')
    add('\n搜尋時間的15%目標在此次中位數均未達成；三組結果保存在costs.json。由於量測短、重複少，沒有時間分布或跨硬體效率保證。計分與落盤為分開單次量測，不能冒充三次端到端中位數。control也包含最小API計數／hash，並非完全沒有診斷開銷的產品基準。')
    add('\nPython峰值來自技術配對的tracemalloc；第一案觀察器首次AST建置／編譯亦在外層記憶體量測範圍內。它不是總記憶體，也不應解讀成兩情境固有記憶體差。未量PyTorch原生配置或RSS。')
    add(f"\n本輪pilot參與者模型實際推論共{summary['research_input_samples']:,}次，來自16次完整搜尋；這些驗證重跑有研究成本。合成函數測試不計成已訓練參與者模型推論。pooled呼叫0、truth未載入、新訓練0。每案原生模型呼叫無省略，重複輸入仍照原程序呼叫。")
    add('\n## 8. 未建立有效性')
    add('\n- 守恆是代數與程式驗收，不證明公平或外部品質。')
    add('- A取正部是一項規約；抵銷與拒絕資訊的保護價值仍未定價。')
    add('- E具有端點恆等式，不衡量模型評分對中間決策的全部作用，也不是可靠度。')
    add('- 只有兩個舊pilot，沒有獨立資料或泛化驗證。沒有比較LOO；也不對J作淘汰結論。')
    add('- 不同任務／人數／權重下的原始A/E不能直接當同一經濟單位比較。')
    add('\n## 9. 下一步提案')
    add('\n1. 先檢視這種正向支持分帳是否符合研究目的，尤其agent15的A=0及抵銷角色未獲分的含義。這是規則選擇，不能用計算通過取代。')
    add('2. 若保留此規則，下一個有限工作先改善觀察器成本與記錄體積，維持相同公式；用同一控制驗收證明優化沒有改變搜尋與分數。新增推論0與低延遲分開驗收。')
    add('3. 模型品質與拒絕／制衡價值另訂估計對象和驗證資料；本輪不自行加入品質乘數、付款、LOO或大型實驗。')
    add('\n## 10. 證據位置與重現')
    add(f"\n- 設計：`docs/{'custom_realized_gain_attribution_design_v1_20261027.md'}`。")
    add('- 程式與命令：`contribution_analysis/custom_realized_gain_v1/README.md`。')
    add(f'- 原始結果：`{root.relative_to(output.parents[1])}`。')
    add('- 來源凍結：source_snapshot/、implementation/、source_lock.json；設計與模型來源hash：inputs.json。')
    add('- 每案：manifest、逐次queries/evaluations/iterations、accounts、diagnostics、costs、verification、sensitivity。')
    add('- 全體：technical_tests.json、summary.json、independent_audit.json；交付receipt另行封存，包含本報告及交付版本來源。')
    add('\n設計文件保留執行前狀態；本報告是本輪新結果，不回填歷史確認、不修改Notion。')
    with output.open('x') as f:f.write('\n'.join(text)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    render(args.root.resolve(),args.output.resolve())
