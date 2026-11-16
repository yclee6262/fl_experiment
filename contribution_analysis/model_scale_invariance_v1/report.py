"""Generate report after independent audit; seal all new artifacts."""
import argparse
from pathlib import Path
from contribution_analysis.custom_realized_gain_v1.schema import read,save,sha
WORK=Path(__file__).resolve().parents[2]


def build(root):
    s=read(root/'summary.json');a=read(root/'independent_audit.json');by={c['id']:c for c in s['cases']};example=read(root/'cases/useful_information_0.json')
    scale=[c for c in s['cases']if c['category']=='individual_scale'];table=[]
    for c in scale:
        d=read(root/'cases'/(c['id']+'.json'));factor=d['fixture']['models'][0]['linear'][0];table.append(f"| {factor} | {c['raw_C']:.6g} | {c['normalized_C']:.6g} | {c['D_analytic']:.6g} |")
    def sign(v):return None if v is None else 1 if v>1e-10 else -1 if v< -1e-10 else 0
    signchanges=sum(sign(c['raw_C'])!=sign(c['normalized_C'])for c in s['cases']if c['raw_C']is not None)
    b=example['baseline'];ctrl=example['inconsistent_control']['score'];useful=by['useful_information_0'];optimistic=by['optimistic_acceptance_0']
    return f'''# 模型池第一階段：固定端點尺度換算驗證結果

日期：2026-10-28。執行前規約：`docs/model_scale_invariance_design_v1_20261028.md`。新輸出：`{root.relative_to(WORK)}`。

**已完成：固定端點下，一致縮放模型全部loss並同步換算探測尺度時，標準化貢獻維持一致。354個可比較變體通過，6個原未定義變體仍為null。這是尺度換算的代數／數值驗證，尚未證明模型決策品質或搜尋尺度不變。**

## 1. 實際做法

重用T3的60個已讀解析設定、原接受full/minus端點、原common階段loss。59設定有minus端點，1設定無剩餘正權重，保留原未定義。這些資料不是新的獨立保留或實務確認。

共同探測固定25點[-1,−0.5,0,0.5,1]²，非依C選點。各模型s=Q90(loss)−Q10(loss)，線性插值。固定完整團隊v與尺度，用C_norm=Σv_m(loss_minus−loss_full)/s_m，另存own和others。正權重模型s≤1e−12則未定義，不用epsilon硬除。零權重模型不參與評分。

每次只換一個模型，將其探測及兩端點loss一起乘0.1、10、100，其他模型、原端點與v不變。所有360變體中354可比較、6原無端點未定義。量的是同一對結果，不重跑方向／縮步／接受，也不重新產生排除端點。

## 2. 一致性結果與反例控制

60個baseline與360個一致換算變體完成；C/own/others最大baseline差{s['max_invariance_error']:.3g}，符合事前|Δ|≤1e−10+1e−9|baseline|；可辨識非零符號沒有翻轉。標準化尺度在放大時一起改變，正比例因子可相消，這是方法構造的預期性質，不是新品質能力。

例useful_information_0：原C約{example['raw_score']['C']:.6g}；換算後C={b['C']:.6g}，own={b['own']:.6g}，others={b['others']:.6g}。把任一模型全部loss乘10且重估對應尺度，這些換算後數字維持相同（容差內）。若故意只把target端點loss乘10、不換尺度，C變成{ctrl['C']:.6g}。這個控制確實能揭露不一致換算，但不代表正式鎖版本後允許此操作。

## 3. 原尺度反例及品質限制

T3個別尺度類六個設定的full端點均−0.1、minus均+0.1。以固定25點校準後：

| 原模型0尺度倍數 | 原C | 標準化C | 原解析D |
|---:|---:|---:|---:|
{chr(10).join(table)}

標準化不只是把原C除上一個共同常數；它調整每個模型在共同尺中的相對影響。59個可計算設定有{signchanges}個原C與標準化C的正／負／近零分類不同（±1e−10），不能要求新分數保留原排名。這不是權重alpha更新，而是新的離線評分尺，仍須另驗證是否符合付款理由。

重要反例仍存在：有用不同資訊_0的標準化C={useful['normalized_C']:.6g}、原解析D={useful['D_analytic']:.6g}；過度樂觀接受_0的標準化C={optimistic['normalized_C']:.6g}、原解析D={optimistic['D_analytic']:.6g}。分數仍可相同而解析品質好壞相反。尺度消除不等於準確度、可信性或公平性成立；D只引用舊鎖定診斷，不進新公式。

## 4. 邊界與獨立核對

12項技術測試通過：有號／分解、一致倍增、零／近零尺度、無minus、零權重、非有限／負loss、probe缺失、端點長度、非零target下預測倍增≠loss倍增，以及絕對尺度可靠性門檻的邊界。近零s在更換單位後可能跨絕對1e−12門檻，因此主不變性只限所有panel都具可靠尺度；不宣稱任意精度、退化尺度的資格永遠不變。

獨立Fraction reference不匯入生產score，從raw probe／endpoint loss重算分位數、尺度與C分解，共核對{a['reference_panels']}個panel，最大絕對差{a['max_reference_absolute_error']:.3g}。{a['analytic_probe_replies']}個解析探測回覆另用有理數多項式核對；common端點與原API回覆、座標、鎖定順序連結通過。120個故意C+1／尺度+1篡改（含null改0）全被發現。原未定義不是被漏掉或填0。

## 5. 成本與資料角色

新解析探測函數評分{s['analytic_probe_calls']}次（60×2×25），變體只是保存數字乘法與重算分位數，不重查模型；不是已測神經模型校準延遲。新訓練模型呼叫0、搜尋0、訓練0、pooled0、實際付款0；主流程{s['seconds']:.3f}秒，含測試／hash／探測／數字變換／內嵌reference與保存，不含最終audit／報告。未測RSS/native memory或部署成本。

探測點規則於本輪執行前固定，但舊搜尋端點早已存在；本輪是現在離線校準，不能說原搜尋前已校準。若未來要用於新搜尋，需要先鎖模型版本及尺度。只有原target=0的一組模型消融端點，沒有同一團隊所有owner的排除端點，本輪沒有生成或聲稱驗證模型池付款比例。

## 6. 狀態與下一步

**已完成：**固定端點／固定probe規則下的正比例尺度換算一致性、獨立回覆連結與算術核對、故意不一致控制、零尺度及原未定義處理。可以保留此標準化共同尺作下一輪候選。

**未建立有效性：**原搜尋決策不受尺度影響、分數代表獨立品質、探測點對實際搜尋區域的代表性、選擇性回覆操弄、模型版本／准入、模型池累積與比例、低成本及新任務泛化。不能從這輪通過推論「自評問題已解決」。

**下一步提案：**若先保留原搜尋，需另用所有owner固定端點驗證新尺的局部作用與相對比例；若要主張整條搜尋也不受純尺度支配，則需另設計獨立研究版，讓完整／排除分支都使用同一標準化loss，再測重算後端點、C與分配。兩者都保留原有用少數／過度樂觀／阻止壞步反例，不先修改歷史solver、不自動啟動新實驗。

原T3、錨點池、模型、alpha、原演算法與Notion未改。新分析不回填原正式確認。

## 7. 重現

```sh
/Users/yclee/anaconda3/bin/python -m contribution_analysis.model_scale_invariance_v1.run --output contribution_analysis/model_scale_invariance_replay_v1
/Users/yclee/anaconda3/bin/python -m contribution_analysis.model_scale_invariance_v1.audit --root contribution_analysis/model_scale_invariance_replay_v1
/Users/yclee/anaconda3/bin/python -m contribution_analysis.model_scale_invariance_v1.report --root contribution_analysis/model_scale_invariance_replay_v1
```

新目錄必須不存在；不覆寫專案docs或舊來源。
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();name='model_scale_invariance_results_20261028.md'
    with(root/name).open('x')as f:f.write(build(root))
    save(root/'completion_receipt.json',dict(complete=True,file_sha256={str(p.relative_to(root)):sha(p)for p in root.rglob('*')if p.is_file()and p.name!='completion_receipt.json'},new_formal_confirmation=False,actual_payments=0))
    print(root/name)


if __name__=='__main__':main()
