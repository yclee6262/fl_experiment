"""Allocation-first summaries; raw signed scores remain available."""
import json
import csv
from pathlib import Path
import numpy as np
from .allocation import compare


def stats(values):
    x=np.array([v for v in values if v is not None],float)
    return dict(n=len(x),mean=float(x.mean()),p50=float(np.median(x)),p90=float(np.quantile(x,.9)),max=float(x.max())) if len(x) else dict(n=0,mean=None,p50=None,p90=None,max=None)


def analyze(out):
    from .run import read,save
    data=read(out/'scores_and_costs.json');timing=read(out/'timing.json')
    refs={r['case']['key']:r for r in data if r['method']=='FULL_LOO'};teams=[];summaries=[];people=[]
    for r in data:
        if r['method']=='FULL_LOO':continue
        c=r['case'];ref=refs[c['key']]
        for scorer in ['peer','full']:
            cmp=compare(r[scorer],ref[scorer]);row=dict(key=c['key'],scenario=c['scenario'],method=r['method'],scorer=scorer,**cmp)
            if r[scorer] is not None:
                a=np.array(r[scorer]);b=np.array(ref[scorer]);sig=lambda x:np.where(x>1e-4,1,np.where(x< -1e-4,-1,0))
                nonzero=abs(b)>1e-4
                row['signed_mae']=float(abs(a-b).mean());row['sign_match']=float((sig(a)[nonzero]==sig(b)[nonzero]).mean()) if nonzero.any() else None
                row['position_mean']=float(np.linalg.norm(np.array(r['x'])-ref['x'],axis=1).mean())
            else:row.update(signed_mae=None,sign_match=None,position_mean=None)
            teams.append(row)
        for i,aid in enumerate(c['ids']):
            if r['x'] is None:continue
            diag=r['checks']['diagnostics'][i]
            people.append(dict(key=c['key'],method=r['method'],agent_id=aid,
                peer=r['peer'][i],reference_peer=ref['peer'][i],full=r['full'][i],reference_full=ref['full'][i],
                position_error=float(np.linalg.norm(np.array(r['x'][i])-ref['x'][i])),
                early_ratio=diag['early_ratio'],model_dominates=diag['model_dominates'],
                path_length_changed=diag['loo_history_length']!=diag['base_history_length'],
                strong_cancellation=any(x is not None and x<.1 for x in r['checks']['cancellation']),
                repair_distance=float(np.linalg.norm(np.array(r['raw'][i])-r['x'][i]))))
    for scenario in ['all','clean','controlled_poison']:
        for scorer in ['peer','full']:
            for method in ['LAST_J','CUM_A']:
                rows=[r for r in teams if r['method']==method and r['scorer']==scorer and (scenario=='all' or r['scenario']==scenario)]
                costs=[r for r in data if r['method']==method and (scenario=='all' or r['case']['scenario']==scenario)]
                samples=sum(r['cost']['primary_samples'] for r in costs);ref_samples=sum(refs[r['case']['key']]['cost']['primary_samples'] for r in costs)
                total_time=sum(float(np.median([t['seconds'] for t in timing if t['key']==r['case']['key'] and t['method']==method])) for r in costs)
                ref_time=sum(float(np.median([t['seconds'] for t in timing if t['key']==r['case']['key'] and t['method']=='FULL_LOO'])) for r in costs)
                coverage=[]
                for th in [.05,.1,.2]:
                    for share in [None,.05,.1]:
                        passing=[r for r in rows if r['tv'] is not None and r['tv']<=th and (share is None or r['max_share_error']<=share)]
                        robust=[r for r in passing if not r['predicted']['fragile'] and not r['reference']['fragile']]
                        coverage.append(dict(tv=th,max_share=share,passed=len(passing),nonfragile_passed=len(robust),all_denominator=len(rows),
                          reference_distributed_denominator=sum(r['reference']['q'] is not None for r in rows),both_distributed_denominator=sum(r['tv'] is not None for r in rows)))
                summaries.append(dict(scenario=scenario,scorer=scorer,method=method,teams=len(rows),
                  **{k:stats([r[k] for r in rows]) for k in ['tv','max_share_error','augmented_tv','signed_mae','position_mean','sign_match']},
                  statuses={s:sum(r['predicted']['status']==s for r in rows) for s in ['distributed','reserved','invalid']},
                  reference_statuses={s:sum(r['reference']['status']==s for r in rows) for s in ['distributed','reserved','invalid']},
                  disagreements=sum(r['status_disagreement'] for r in rows),both_reserved=sum(r['both_reserved'] for r in rows),
                  fragile=sum(r['predicted']['fragile'] is True for r in rows),reference_fragile=sum(r['reference']['fragile'] is True for r in rows),
                  coverage=coverage,primary_samples=samples,reference_samples=ref_samples,saving=1-samples/ref_samples,
                  extra_samples=sum(r['cost']['extra_samples'] for r in costs),full_supplement_samples=sum(r['cost']['supplement_samples'] for r in costs),
                  median_time_sum=total_time,reference_median_time_sum=ref_time,time_ratio=total_time/ref_time,
                  per_team_time_ratio=stats([float(np.median([t['seconds'] for t in timing if t['key']==r['case']['key'] and t['method']==method]))/float(np.median([t['seconds'] for t in timing if t['key']==r['case']['key'] and t['method']=='FULL_LOO'])) for r in costs])))
    diagnostics=[]
    for method in ['LAST_J','CUM_A']:
        for condition in ['early','strong_cancellation','model_dominates','path_length_changed']:
            for value in [False,True]:
                selected=[]
                for r in data:
                    if r['method']!=method:continue
                    c=r['checks'];ds=c['diagnostics']
                    flag=(any(d['early_ratio'] is not None and d['early_ratio']<.1 for d in ds) if condition=='early' else
                          any(x is not None and x<.1 for x in c['cancellation']) if condition=='strong_cancellation' else
                          any(d['model_dominates'] for d in ds) if condition=='model_dominates' else
                          any(d['loo_history_length']!=d['base_history_length'] for d in ds))
                    if flag==value:selected.append(next(t for t in teams if t['key']==r['case']['key'] and t['method']==method and t['scorer']=='peer'))
                diagnostics.append(dict(method=method,condition=condition,value=value,teams=len(selected),tv=stats([r['tv'] for r in selected])))
    save(out/'team_metrics.json',teams);save(out/'participant_metrics.json',people);save(out/'summary.json',summaries);save(out/'diagnostics.json',diagnostics)
    with (out/'participant_metrics.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(people[0]));w.writeheader();w.writerows(people)
    lines=['# 近似完整 LOO 分配的舊資料探索結果','',
      '主評分F₋ᵢ，完整F作對照。20個既有團隊、80人；相同端點與分配規則配對比較。沒有pooled載入、訓練或推論。這是舊資料探索與技術核對，新案例獨立確認未啟動。',
      '', '主要TV只在兩方可分配時定義；擴展TV把未分配獎池當獨立份額，避免單方留池被刪除。雙方均留池另列，不當作個人份額成功。分配規則：正分截取，總正分>1e-8才正規化，<=1e-4標脆弱。',
      '', '| 尺／情境 | 方法 | TV均值／有效隊數 | 擴展TV均值 | 中位／P90／最大TV | 分配／留池／無效 | 狀態不一致 | 脆弱隊 |', '|---|---|---:|---:|---:|---|---:|---:|']
    pct=lambda x:'未定義' if x is None else f'{100*x:.2f}%'
    for s in summaries:
        lines.append(f"| {s['scorer']}／{s['scenario']} | {s['method']} | {pct(s['tv']['mean'])}／{s['tv']['n']} | {pct(s['augmented_tv']['mean'])} | {'／'.join(pct(s['tv'][k]) for k in ['p50','p90','max'])} | {s['statuses']} | {s['disagreements']} | {s['fragile']} |")
    lines+=['','## 探索誤差門檻','', '以下是比例門檻，沒有加入符號一致率要求，亦不是使用者已接受的合格門檻。每列同時公開全部／參考可分配／雙方可分配分母。','', '| 尺 | 方法 | TV上限 | 單人上限 | 通過／非脆弱通過 | 全部／參考可分配／雙方可分配 |','|---|---|---:|---:|---:|---|']
    for s in summaries:
        if s['scenario']!='all':continue
        for cov in s['coverage']:
            lines.append(f"| {s['scorer']} | {s['method']} | {pct(cov['tv'])} | {pct(cov['max_share'])} | {cov['passed']}／{cov['nonfragile_passed']} | {cov['all_denominator']}／{cov['reference_distributed_denominator']}／{cov['both_distributed_denominator']} |")
    lines+=['','## 主peer成本與時間','','每方法獨立baseline及cache，完整F補查另帳；一次部署模型評分與研究暖機／重複分開。時間包含實際線上向量累積或最後接受輪狀態擷取，排除模型載入及檔案I/O；只描述本機微型神經網路。','','| 方法 | 主評分總樣本 | 新增樣本 | 對LOO節省 | F對照補查 | 20隊中位秒數之和 | 對LOO時間比 |','|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        if s['scenario']=='all' and s['scorer']=='peer':lines.append(f"| {s['method']} | {s['primary_samples']} | {s['extra_samples']} | {pct(s['saving'])} | {s['full_supplement_samples']} | {s['median_time_sum']:.4f} | {s['time_ratio']:.3f} |")
    loo=[r for r in data if r['method']=='FULL_LOO'];lines.append(f"| FULL_LOO | {sum(r['cost']['primary_samples'] for r in loo)} | {sum(r['cost']['extra_samples'] for r in loo)} | 參考 | {sum(r['cost']['supplement_samples'] for r in loo)} | {next(s['reference_median_time_sum'] for s in summaries):.4f} | 1 |")
    lines+=['','## 診斷與適用範圍','','LAST_J從最後接受輪前狀態排除錨點與模型，保留前期路徑；CUM_A精確重建全員原路徑錨點位移，但不模擬移除後的其他人反應。兩者與完整LOO的差異是操作定義限制。','', '低成本是否值得取決於所需TV、最大單人差及覆蓋率。若參考本身沒有正分，正確輸出可能是留池，不能要求近似法一定創造分配。原始signed分數、份額、保留池、缺失、模型計費及方法適用性完整保留，不從結果挑選較好的評分器。','']
    for method in ['LAST_J','CUM_A']:
        worst=sorted([r for r in teams if r['scorer']=='peer' and r['method']==method and r['augmented_tv'] is not None],key=lambda r:r['augmented_tv'],reverse=True)[:3]
        for r in worst:lines.append(f"- {method} {r['key']}：擴展TV={pct(r['augmented_tv'])}，預測={r['predicted']['status']}，LOO={r['reference']['status']}，signed預測及參考見scores_and_costs.json，分配={r['predicted']['q']}／{r['reference']['q']}。")
    lines+=['','記憶體：CUM方法新增h及D陣列為O(nd)，LAST_J保存一份輪前數值狀態，不存全程trace；observer_array_bytes提供核心陣列實際bytes。查詢稽核事件與cache為研究／推論管理成本，並非O(nd)的一部分。diagnostics.json按團隊報預定失敗條件；樣本不足或全部屬同組時，不能推論因果或可靠適用族群。','', '未以pooled效果或signed符號作比例合格條件。平均和尾端分開報告；本批結果不能作新案例獨立確認或公平性證明。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
