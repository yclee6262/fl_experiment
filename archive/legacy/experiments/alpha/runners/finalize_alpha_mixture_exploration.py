import json,itertools,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from alpha_mixture_exploration import ROOT,read,dump,SCENARIOS
import oob_shrinkage_research as shared

a=json.loads((ROOT/'analysis.json').read_text());st=a['methods'];rows=read(ROOT/'results.jsonl');prep=read(ROOT/'shared.jsonl');replay=read(ROOT/'replay.jsonl')
# Complete costs: measured legacy redundancies are distinct from cached deployment work.
common={k:float(np.mean([r['preparation']['preparation'][k] for r in prep])) for k in ['training_seconds','qualification_seconds','proposal_seconds','qualification_requests','qualification_input_samples']}
common.update({k:float(np.mean([r['preparation'][k] for r in prep])) for k in ['fixed_proposal_seconds','fixed_proposal_requests','fixed_proposal_input_samples','evaluator_seconds']})
common_seconds=sum(common[k] for k in ['training_seconds','qualification_seconds','proposal_seconds','fixed_proposal_seconds'])
cost={}
for m,s in st.items():
    rr=[r for r in rows if r['method']==m];support=6 if rr[0]['w'][3]>0 else 0
    cost[m]={'actual_recorded_init_requests':float(np.mean([r['initialization_requests'] for r in rr])),
       'actual_recorded_init_samples':float(np.mean([r['initialization_input_samples'] for r in rr])),
       'support_requests':support,'support_samples':support,
       'cached_deployment_method_requests':s['mean_method_requests'],
       'cached_deployment_method_samples':s['mean_stage3_input_samples']+support,
       'historical_common_plus_method_seconds_descriptive':common_seconds+s['mean_method_seconds'],
       'full_workflow_requests_cached':common['qualification_requests']+common['fixed_proposal_requests']+s['mean_method_requests'],
       'full_workflow_samples_cached':common['qualification_input_samples']+common['fixed_proposal_input_samples']+s['mean_stage3_input_samples']+support}
# Approximate power planning only, without peeking at any new confirmation outcomes.
d=np.array(a['seed_differences']['UQES_equal']);sd=float(d.std(ddof=1));n=math.ceil(((1.95996398454+.84162123357)*sd/.001)**2)
dump('cost_and_power.json',{'shared_mean':common,'shared_seconds':common_seconds,'methods':cost,'power_sd':sd,'normal_approx_n_80pct_delta_0001':n,'planning_only':True})
fig,axes=plt.subplots(2,3,figsize=(12,7),sharey=True)
for ax,(i,j) in zip(axes.flat,itertools.combinations('UQES',2)):
    names=[j,i+j+'_0.250',i+j+'_0.500',i+j+'_0.750',i];x=[0,.25,.5,.75,1]
    for k,label in enumerate(['Clean','Low quality','Imbalance']):ax.plot(x,[st[m]['scenario_delta'][k] for m in names],marker='.',alpha=.6,label=label)
    ax.plot(x,[st[m]['delta'] for m in names],color='black',marker='o',lw=2,label='Seed mean')
    ax.axhline(0,color='gray',ls='--');ax.set_title(i+' + '+j);ax.set_xlabel('Weight on '+i);ax.set_ylabel('Pooled error minus EU');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8);fig.suptitle('Development only: 30 seeds, paired with fixed EU');fig.tight_layout();fig.savefig(ROOT/'pair_surfaces.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5));names=['UQ_0.500','ES_0.500','UE_0.250','UE_0.750','QS_0.750','QES_equal','UQES_equal']
for i,m in enumerate(names):
    s=st[m];lo,hi=s['exploratory_ci95'];ax.errorbar(s['delta'],i,xerr=[[s['delta']-lo],[hi-s['delta']]],fmt='o',color='navy',capsize=3)
ax.set_yticks(range(len(names)),names);ax.axvline(0,color='gray',ls='--');ax.set_xlabel('Paired seed-mean error difference versus EU');ax.set_title('Exploratory 95% bootstrap intervals; not selection-adjusted');fig.tight_layout();fig.savefig(ROOT/'paired_differences.png',dpi=160);plt.close(fig)
report=ROOT/'report.md';body=report.read_text();head='''
**結論：維持固定 EU，不啟動局部擴張。沒有候選通過事前推薦門檻。**

- U+Q、E+S各半已於11/7補齊，本輪重用並逐案核對。這30案例對EU差分別為 +0.000540、+0.000187，尚未見品質收益。
- EU附近：0.25U+0.75E與0.75U+0.25E對EU差分別 +0.000384、+0.001222；批次／情境方向交錯，不能稱0.5已最優，也未找到一致較佳的寬廣區域。
- 六條邊均未達預設局部細查門檻。沒有測0.375/0.625等細點，不把未測區域說成無效。
- 四法等權 w=(0.25,0.25,0.25,0.25) 是事後平均最佳，改善0.001617（本子集EU均值的8.74%），三情境平均均改善，但第二批差+0.000977、勝16/30低於55%門檻，95%探索區間[−0.004070,+0.000613]跨零。尚未測其鄰域，不能稱穩定平台。只列觀察線索，不升級為通過篩選候選。
- 三法等權最好的(Q+E+S)/3平均改善0.000773，乾淨情境變差且第二批變差；未超過四法等權，但也不能據此認定方法越多越好。
- 最好的兩法點0.75Q+0.25S改善0.000896，未達0.001門檻，僅贏13/30；它的0.5Q+0.5S鄰點平均反而較差，不構成已證實穩定區域。

本輪使用開發pooled結果比較與篩選，並非完全不依賴pooled資訊的方法發現。部署時固定w後仍只使用local與盲測資訊計算α，不逐任務用pooled選比例。
'''
body=body.replace('\n\n30個歷史', '\n'+head+'\n30個歷史',1)
body+='''
## 品質尾端與集中度

- E+S各半在seed22008低品質情境，pooled誤差0.118887，EU為0.011719，惡化0.107168；其α約(0.084,0.404,0.512)，並非極端集中在單一模型。
- 四法等權在seed21003乾淨情境，pooled=0.030117、EU=0.001951，惡化0.028166；其α約(0.495,0.307,0.198)。在seed22001的三情境平均也惡化0.010797。
- 這些是品質尾端案例，不是solver失敗：全部2430筆有效／box可行。集中度不能單獨解釋品質。完整最差seed與任務、交付X、α、truth均已保存，未依失敗尾端建立新切換規則。

## 完整成本口徑

'''
body+=f"歷史共享準備平均：訓練{common['training_seconds']:.3f}s、資格{common['qualification_seconds']:.3f}s、原提案{common['proposal_seconds']:.3f}s、固定三人提案{common['fixed_proposal_seconds']:.3f}s，合計{common_seconds:.3f}s；離線evaluator訓練{common['evaluator_seconds']:.3f}s另列研究成本。共享準備沿用歷史計時，本輪沒有重訓。資格請求平均{common['qualification_requests']:.1f}、樣本{common['qualification_input_samples']:.1f}；固定提案的API計數為{common['fixed_proposal_requests']:.1f}，不能解讀為提案訓練／梯度運算免費。\n\n"
body+='歷史EU初始化原始紀錄有3 requests/60 samples重查盲測；新runner重用完全相同形狀的資格預測，沒有重查。上表「方法請求」採可部署快取口徑＝Stage3＋必要支持查詢，避免把歷史重查浪費當成新法改善；實際原始查詢與完整流程總量另見cost_and_power.json。U/Q/E成分不需新標籤，S需6次非自身scalar預測；四法等權需要一次E擬合＋Q/U/S計算，而非四次Stage3。\n\n'
body+=f"四法等權方法請求為EU的{st['UQES_equal']['requests_ratio_to_EU']:.3f}倍（含6次支持），未超過1.10工作量門檻；品質門檻仍未過。它與EU在不同時段的wall time不能用來宣稱更快，整機其他工作無法由受限程序清單完全核實。後續確認須交錯同機計時。共享成本巨大時總倍率接近1，也不能掩蓋初始化增量。\n\n"
body+='## 圖與驗收\n\n![六條邊的開發品質曲線](pair_surfaces.png)\n\n![部分候選配對差](paired_differences.png)\n\n全部27候選×30 seeds×3情境=2430筆，缺漏0、重複0、失敗0、fallback0。全量pooled重評差0、α重建最大差2.22e−16。90筆基準重播最大差4.45e−10，低於1e−7。舊結果雜湊不變。4項初始化／快取／回退測試通過。\n\n只新增獨立runner與分析，沒有更改既有Stage3、α／β更新或歷史輸出。完整設計、註冊、來源快照、結果與驗收同目錄。\n'
report.write_text(body)
plan=f'''# 鎖定方法後的獨立確認方案（提案，未執行）

2026-11-16。本輪推薦是維持EU；無方法通過預設篩選，不因平均排名再擴大探索。

若研究仍希望檢驗唯一最有訊號的四法等權，可把它當成「未通過開發穩定門檻的單一研究假說」另行鎖定；不是本輪成功選出的替代法。不再微調比例或以新確認資料測鄰域。若先研究鄰域，必須另標開發且其後再換全新確認資料。

## 精確鎖定

對照：原固定EU=0.5U+0.5E，保存原legacy擬合失敗回退U。
唯一可能實驗臂：α=(U+Q+E+S)/4，w順序U/Q/E/S；E失敗回退Q、Q無效MSE回退U；S非自身預測無效則交付準備失敗，不補抽、不改方法。保留requested/effective方法與fallback。若要統一EU回退公式，那是另一個版本，必須在註冊說明，不能偷偷替換。
兩臂相同IDs1/2/20、2D、target0、20盲測、三情境、best_of/custom30、box/tolerance。新資料訓練local與凍結pooled evaluator，不使用舊artifact；先封存兩臂X再pooled評分。新w不能依任務pooled變動。

## 資料隔離與固定預算

舊90案例（含本次30）、C/P90案例及其他已做alpha研究資料全部視為已用。確認前跑既有used_seeds掃描，核對所有manifest/registration/結果資料seed與artifact雜湊；整數不重複僅必要非充分條件，同時核對生成來源。尚未分配新seed，所以不宣稱已有未見確認集。

先做3個全新完整案例的流程pilot，只看完整性、失敗與成本，不用pooled品質改比例；pilot不併入確認。再封存新seed清單與完整分析，三批全部完成才一次分析；不逐批看效果，不補抽，不加樣到顯著。若更改規則，現有確認即轉開發，另建新的確認批。

## 樣本量與品質門檻

本開發樣本四法等權−EU的seed差標準差={sd:.6f}。若預期真實改善0.001、雙側α=.05、80% power，常態近似所需n={n}個完整seed。這是30個開發案例變異的粗估，不是保證；不採用事後最好點0.001617當預期效果，且未把三情境當獨立樣本。

可用固定三批、每批{math.ceil(n/3)}案，共{math.ceil(n/3)*3}案作資源估算（{math.ceil(n/3)*3*3*2}方法任務；另計pilot），正式開跑前用開發seed bootstrap模擬整體門檻與跨批符號一致率，寫下功效的不確定性與接受的資源上限，然後鎖定。不應在結果不顯著後擴樣。若預算只允許90案，應明列低功效的估計性確認，不能當等效試驗。

主比較只有四法等權−EU；完整seed先等權平均三情境，200,000次paired bootstrap，固定分析RNG20261116。通過需95%區間上界<0、點估計≤−0.001、三批平均皆負、完整配對且失敗率不增加。這支持平均改善與實質點估計，不等於已證明改善至少0.001；若要證明後者，需區間上界<−0.001並另作樣本量規劃。情境、勝負、尾端、集中與truth只作預定描述，不作新selector。若增加第二候選，先改比較家族與樣本量，不能事後加測。

## 成本確認與決策

兩臂逐任務用預定亂數交錯先後順序，單執行緒、同機、不平行重訓；記初始化／Stage3／完整方法及共同準備的wall time、requests與samples。對必要支持查詢與資格快取明確記帳，研究驗收重播另外計。建議事前成本界線：方法requests及samples各≤EU的1.10倍、同機方法時間≤1.25倍；時間用完整seed配對bootstrap並報區間。這些是研究資源取捨，不是真實服務延遲或金錢成本證據。

品質及成本都通過才考慮採用；任何一項不通過維持EU。沒有顯著差不代表等效。不主張泛化到更多Agent、不同維度、目標或真實資料；外部有效性是後續單獨研究。
'''
Path('docs/alpha_mixture_independent_confirmation_plan_20261116.md').write_text(plan)
# Freeze analysis code and test evidence separately from the pre-experiment runner snapshot.
snap=ROOT/'analysis_source_snapshot';snap.mkdir(exist_ok=True)
for name in ['analyze_alpha_mixture_exploration.py','finalize_alpha_mixture_exploration.py','test_alpha_mixture_exploration.py']:
    (snap/name).write_bytes(Path(name).read_bytes())
dump('analysis_source_hashes.json',{p.name:shared.digest(p) for p in snap.iterdir()})
print(json.dumps({'planning_n':n,'sd':sd,'shared_seconds':common_seconds}))
