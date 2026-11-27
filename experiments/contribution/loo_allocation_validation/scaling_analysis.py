"""Paired descriptive scaling summaries, retaining reserve states and tails."""
import numpy as np
from .allocation import compare,allocate
from .analyze import stats
from .run import read,save


def analyze(out):
    data=read(out/'scores.json');timing=read(out/'timing.json');pools=read(out/'qualification.json')
    ref={r['case']['key']:r for r in data if r['result']['method']=='FULL_LOO'}
    teams=[];cells=[];stable=[]
    for r in data:
        v=r['result'];c=r['case'];reference=ref[c['key']]['result']
        if v['method']=='FULL_LOO':continue
        for scorer in ['peer','full']:
            cmp=compare(v[scorer],reference[scorer]);pe=None
            if v['x'] is not None:pe=float(np.linalg.norm(np.asarray(v['x'])-reference['x'],axis=1).mean())
            teams.append(dict(key=c['key'],condition=c['scenario'],n=len(c['ids']),seed=c['seed'],method=v['method'],scorer=scorer,
              **cmp,position_mean=pe,repairs=None if v['repairs'] is None else sum(x>0 for x in v['repairs']),
              primary_samples=r['cost']['primary'],reference_samples=ref[c['key']]['cost']['primary'],
              extra_samples=r['cost']['primary']-r['cost']['baseline']))
    conditions=list(dict.fromkeys(r['case']['scenario'] for r in data))
    for condition in conditions:
        for n in [3,5,10,20]:
            for scorer in ['peer','full']:
                for method in ['LAST_J','CUM_A']:
                    rows=[r for r in teams if r['condition']==condition and r['n']==n and r['scorer']==scorer and r['method']==method]
                    if not rows:continue
                    t=[r for r in timing if r['condition']==condition and r['n']==n and r['method']==method]
                    rt=[r for r in timing if r['condition']==condition and r['n']==n and r['method']=='FULL_LOO']
                    cells.append(dict(condition=condition,n=n,scorer=scorer,method=method,completed=len(rows),planned=5,
                       tv=stats([r['tv'] for r in rows]),augmented_tv=stats([r['augmented_tv'] for r in rows]),max_share=stats([r['max_share_error'] for r in rows]),
                       distributed=sum(r['predicted']['status']=='distributed' for r in rows),reference_distributed=sum(r['reference']['status']=='distributed' for r in rows),
                       reserved=sum(r['predicted']['status']=='reserved' for r in rows),invalid=sum(r['predicted']['status']=='invalid' for r in rows),
                       fragile=sum(r['predicted']['fragile'] is True for r in rows),reference_fragile=sum(r['reference']['fragile'] is True for r in rows),
                       both_reserved=sum(r['both_reserved'] for r in rows),disagreement=sum(r['status_disagreement'] for r in rows),
                       tv_pass={str(th):sum(r['tv'] is not None and r['tv']<=th for r in rows) for th in [.05,.1,.2]},
                       robust_pass={str(th):sum(r['tv'] is not None and r['tv']<=th and not r['predicted']['fragile'] and not r['reference']['fragile'] for r in rows) for th in [.05,.1,.2]},
                       combined_pass={str(th):sum(r['tv'] is not None and r['tv']<=th and r['max_share_error']<=.1 for r in rows) for th in [.05,.1,.2]},
                       primary_mean=float(np.mean([r['primary_samples'] for r in rows])),extra_mean=float(np.mean([r['extra_samples'] for r in rows])),
                       saving=1-sum(r['primary_samples'] for r in rows)/sum(r['reference_samples'] for r in rows),
                       representative_median_seconds=float(np.median([r['seconds'] for r in t])) if t else None,
                       representative_time_ratio=float(np.median([r['seconds'] for r in t])/np.median([r['seconds'] for r in rt])) if t and rt else None,
                       repairs=sum(r['repairs'] or 0 for r in rows)))
    for r in read(out/'stability.json'):
        c=r['case'];v=r['result'];thirty=ref[c['key']]['result']
        for scorer in ['peer','full']:
            stable.append(dict(key=c['key'],condition=c['scenario'],n=len(c['ids']),scorer=scorer,
              **compare(v[scorer],thirty[scorer]),base_motion=float(np.linalg.norm(np.array(v['base']['solution'])-thirty['base']['solution'])),
              removed_motion_mean=float(np.linalg.norm(np.array(v['x'])-thirty['x'],axis=1).mean()),
              base30_history=len(thirty['base']['history']),base60_history=len(v['base']['history'])))
    differences=[]
    def contrast(rows_a,rows_b,label,method):
        aa={r['seed']:r for r in rows_a};bb={r['seed']:r for r in rows_b};ds=[]
        for seed in sorted(aa.keys()&bb.keys()):
            if aa[seed]['augmented_tv'] is not None and bb[seed]['augmented_tv'] is not None:
                ds.append(dict(seed=seed,difference=aa[seed]['augmented_tv']-bb[seed]['augmented_tv']))
        vals=[r['difference'] for r in ds]
        differences.append(dict(contrast=label,method=method,pairs=ds,mean=float(np.mean(vals)) if vals else None,min=min(vals) if vals else None,max=max(vals) if vals else None))
    for method in ['LAST_J','CUM_A']:
        rs=[r for r in teams if r['scorer']=='peer' and r['method']==method]
        for co in conditions:contrast([r for r in rs if r['condition']==co and r['n']==20],[r for r in rs if r['condition']==co and r['n']==3],co+':n20-n3',method)
        for n in [3,5,10,20]:
            for a,b in [('d10_low','d2_low'),('d5_high','d5_low')]:
                contrast([r for r in rs if r['condition']==a and r['n']==n],[r for r in rs if r['condition']==b and r['n']==n],f'{a}-{b}:n{n}',method)
    difficulty=[]
    for condition in conditions:
        for n in [3,5,10,20]:
            rs=[r for r in data if r['case']['scenario']==condition and len(r['case']['ids'])==n and r['result']['method']=='FULL_LOO']
            difficulty.append(dict(condition=condition,n=n,teams=len(rs),base_initial_F=stats([r['result']['base']['history'][0] for r in rs]),
              base_final_F=stats([r['result']['base']['loss'] for r in rs]),
              history30_limit=sum(len(r['result']['base']['history'])>=31 for r in rs),
              accepted_steps=stats([sum(b<a for a,b in zip(r['result']['base']['history'],r['result']['base']['history'][1:])) for r in rs]),
              loo_history_changed=sum(sum(x!=len(r['result']['base']['history']) for x in r['result']['removal_history_lengths']) for r in rs)))
    for name,obj in [('team_metrics.json',teams),('summary.json',cells),('stability_metrics.json',stable),('paired_differences.json',differences),('difficulty.json',difficulty)]:save(out/name,obj)
    # Independent overlap identity for every defined allocation.
    for row in teams+stable:
        if row['augmented_tv'] is not None:
            overlap=sum(min(a,b) for a,b in zip(row['predicted']['augmented'],row['reference']['augmented']))
            assert abs(1-overlap-row['augmented_tv'])<1e-12
    render_report(out,cells,stable,differences,difficulty,pools,teams)


def render_report(out,cells,stable,differences,difficulty,pools,teams):
    pct=lambda x:'NA' if x is None else f'{100*x:.1f}%'
    lines=['# 增加合格agent與問題複雜度的分配近似探索','',
      '主分析為F₋ᵢ，完整F為固定對照；LAST_J=最後接受輪同時移除錨點與模型，CUM_A=原路徑累積錨點位移。無pooled評分。全員與完整LOO模型均固定，不重訓移除後模型。新分布探索，不宣稱獨立確認或公平。',
      '', '3/5/10/20人與2/5/10維、5維高交互四個條件，各5seed，預排80隊。相同seed/條件共用合格模型池前綴與固定起點，alpha uniform。每人512訓練與256資格驗證，架構與訓練預算固定。資料函數已按維度正規化，但維度/交互改變仍可能影響模型擬合、資格篩選與搜尋；不是純粹solver難度的因果隔離。','',
      '資格條件為validation RMSE<=.20且anchor loss<=.02、有限且box可行；沒有毒化資料。這是本輪合成入場篩選，不是沿用生產Stage1或已驗證防毒。全部候選的通過／拒絕原因均保留。',
      '', '主要TV只在雙方可分配時定義；擴展TV把留池列一個份額，所有有效團隊可比較。雙方均留池另列，不能當成個人比例成功。5/10/20%只是探索門檻，沒有業務接受宣稱。','',
      '| 條件 | 人數 | 方法 | 平均TV／可算n | 全體擴展TV均值／P90 | 狀態不同 | 近似／LOO可分配 | TV<=10%（非脆弱） | 模型評分節省 |',
      '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for c in cells:
        if c['scorer']!='peer':continue
        lines.append(f"| {c['condition']} | {c['n']} | {c['method']} | {pct(c['tv']['mean'])}／{c['tv']['n']} | {pct(c['augmented_tv']['mean'])}／{pct(c['augmented_tv']['p90'])} | {c['disagreement']}/{c['completed']} | {c['distributed']}／{c['reference_distributed']} | {c['tv_pass']['0.1']}（{c['robust_pass']['0.1']}）/5 | {pct(c['saving'])} |")
    lines+=['','## 配對因素變化','','正值表示較多人／較高維度／較強交互條件的擴展TV更大；負值表示更小。列5seed配對均值與範圍，非信賴區間或顯著性宣稱。','','| 對比 | 方法 | seed數 | 平均差（百分點） | seed最小／最大差 |','|---|---|---:|---:|---:|']
    for r in differences:
        lines.append(f"| {r['contrast']} | {r['method']} | {len(r['pairs'])} | {None if r['mean'] is None else round(100*r['mean'],2)} | {None if r['min'] is None else round(100*r['min'],2)}／{None if r['max'] is None else round(100*r['max'],2)} |")
    lines+=['','## 搜尋困難與參考穩定性','','較高維度不自動代表本solver實際更難。下表列全員起終點內部F、接受步數及到30輪上限；更少改善或更多上限可能揭露搜尋困難，但不等同真正全域解誤差。不同模型/維度下的F只作同設定程序診斷。','','| 條件 | 人數 | 初始F均值 | 最終F均值 | 接受步均值 | 到30輪上限隊數 |','|---|---:|---:|---:|---:|---:|']
    for r in difficulty:lines.append(f"| {r['condition']} | {r['n']} | {r['base_initial_F']['mean']:.5f} | {r['base_final_F']['mean']:.5f} | {r['accepted_steps']['mean']:.1f} | {r['history30_limit']}/{r['teams']} |")
    lines+=['','60輪檢查固定每cell首seed，共16隊。將全員及所有LOO搜尋一同改成60輪，監測整個參考分配程序的預算敏感性，不將60輪當全域真值、不事後換參考。','','| 條件 | 人數 | LOO30→60擴展TV | 狀態不同 | 全員解移動 |','|---|---:|---:|---:|---:|']
    for r in stable:
        if r['scorer']=='peer':lines.append(f"| {r['condition']} | {r['n']} | {pct(r['augmented_tv'])} | {r['status_disagreement']} | {r['base_motion']:.5f} |")
    lines+=['','## 資格與時間','','| 條件 | 訓練候選 | 合格總數 | 合格驗證RMSE均值 |','|---|---:|---:|---:|']
    for co in dict.fromkeys(p['condition'] for p in pools):
        pp=[p for p in pools if p['condition']==co];rr=[a['rmse'] for p in pp for a in p['agents'] if a['accepted']]
        lines.append(f"| {co} | {sum(p['attempted'] for p in pp)} | {sum(p['accepted'] for p in pp)} | {np.mean(rr):.5f} |")
    lines+=['','重複計時只覆蓋首seed的16隊：第一遍暖機，後3次中位數。其餘隊為單次成本資料。時間包含線上觀測、生成、peer評分，不含模型訓練、載入、檔案I/O與完整F補評；微型CPU模型不能外推服務API。完整F端點補查成本獨立保留在records成本帳。','','| 條件 | 人數 | 方法 | 主peer平均樣本／新增樣本 | 代表隊中位秒數 | 代表隊對LOO時間比 |','|---|---:|---|---:|---:|---:|']
    for c in cells:
        if c['scorer']=='peer':lines.append(f"| {c['condition']} | {c['n']} | {c['method']} | {c['primary_mean']:.1f}／{c['extra_mean']:.1f} | {c['representative_median_seconds']:.4f} | {c['representative_time_ratio']:.3f} |")
    lines+=['','## 固定完整F對照','','| 條件 | 人數 | 方法 | 平均TV／有效隊 | 全體擴展TV |','|---|---:|---|---:|---:|']
    for c in cells:
        if c['scorer']=='full':lines.append(f"| {c['condition']} | {c['n']} | {c['method']} | {pct(c['tv']['mean'])}／{c['tv']['n']} | {pct(c['augmented_tv']['mean'])} |")
    lines+=['','## 嚴重失敗與成功個案','','個案依擴展TV排序，不能作事先適用條件。完整raw signed分數、q與保留池見team_metrics和records。']
    for method in ['LAST_J','CUM_A']:
        rr=sorted([t for t in teams if t['method']==method and t['scorer']=='peer' and t['augmented_tv'] is not None],key=lambda t:t['augmented_tv'])
        for tag,chosen in [('小誤差',rr[:2]),('大誤差',rr[-2:])]:
            for t in chosen:lines.append(f"- {tag} {method} {t['key']}：擴展TV {pct(t['augmented_tv'])}，近似={t['predicted']['status']}，LOO={t['reference']['status']}；近似q={t['predicted']['q']}，LOOq={t['reference']['q']}。")
    lines+=['','判讀須以同cell五個團隊與配對差為準；不能把人數、難度增加後平均降低就說全部個案變準，也不能把雙方留池當成個人分配驗證成功。只有5seed，沒有穩定的尾端估計或統計效力保證。若30→60參考本身變動大，只能說近似指定30輪程序的分配，不能擴張成精確的經濟價值。','', '完整來源、模型、訓練資料、資格紀錄、原始query events、端點、分配及成本均獨立保存；舊研究檔案未覆寫。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    plot(out,cells)


def plot(out,cells):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for method,ax in zip(['LAST_J','CUM_A'],axs):
        for co in dict.fromkeys(c['condition'] for c in cells):
            rs=[c for c in cells if c['condition']==co and c['method']==method and c['scorer']=='peer']
            ax.plot([c['n'] for c in rs],[100*c['augmented_tv']['mean'] for c in rs],'-o',label=co)
        ax.set(title=method,xlabel='Qualified agents',ylabel='Mean allocation TV incl. reserve (%)',xticks=[3,5,10,20],ylim=(0,105));ax.legend(fontsize=8)
    fig.suptitle('Clean scaling exploration: five paired seeds per condition')
    fig.savefig(out/'allocation_scaling.png',dpi=180);plt.close(fig)
