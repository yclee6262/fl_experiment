"""Paired comparison, integrity, feasibility, replay and cost audit."""
import torch
import json,itertools
from pathlib import Path
import numpy as np
from oob_shrinkage_research import dump,digest
from analyze_four_alpha import valid
from research_analysis import audit_coverage,paired_bootstrap_mean_ci
ROOT=Path('alpha_weight_outputs/se_qu_extension_v1_20261107')
LABELS=dict(A_uniform='A',B_stage0='Q',stacked_full='E',ensemble_shrink='EU',B_stacked_half='EQ',pure_support='S',peer_support_shrink='SU',support_quality_mix50='SQ',support_ensemble_mix50='SE',quality_shrink='QU')
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def main():
    reg=json.loads((ROOT/'registration.json').read_text());batches=reg['batches'][:3]
    methods=reg['original_methods']+reg['methods'];idx={};errors=[];replay_max=0.;raw_max=0.;seal_max=0.
    for path,h in reg['original_results_sha256'].items():
        if digest(path)!=h:errors.append('old result changed '+path)
    for path,h in reg['source_sha256'].items():
        if digest(path)!=h:errors.append('source changed '+path)
    for b in batches:
        for key in ['pure_results','six_method_results','results']:
            root=Path(b[key]);v=json.loads((root/'verification.json').read_text())
            if not v['passed']:errors.append('reload failed '+str(root))
            for row in lines(root/'results.jsonl'):
                k=(row['scenario'],row['seed'],row['method'])
                if k in idx:errors.append('duplicate '+str(k))
                idx[k]=row
        for d in lines(Path(b['results'])/'decisions.jsonl'):
            r=idx[d['scenario'],d['seed'],d['method']]
            if d.get('pooled_error') is not None:errors.append('decision scored before seal')
            if d['status']=='valid':
                seal_max=max(seal_max,float(np.max(abs(np.array(d['X'])-r['X']))))
    seeds=[s for b in batches for s in b['seeds']];scenarios=reg['scenarios']
    coverage=audit_coverage(list(idx.values()),list(itertools.product(scenarios,seeds,methods)),['scenario','seed','method'])
    if not coverage['complete']:errors.append('coverage incomplete')
    for b in reg['batches'][3:]:
        if not json.loads((Path(b['results'])/'verification.json').read_text())['passed']:errors.append('replay verification failed')
        for r in lines(Path(b['results'])/'results.jsonl'):
            original=idx[r['scenario'],r['seed'],r['method']]
            if r['status']!=original['status']:errors.append('replay status differs')
            if valid(r):
                for k in ['alpha','X','pooled_error','truth_error']:
                    replay_max=max(replay_max,float(np.max(abs(np.array(r[k])-original[k]))))
    for sc,s in itertools.product(scenarios,seeds):
        get=lambda m:idx[sc,s,m]
        if all(valid(get(m)) for m in methods):
            for m,a,b in [('support_ensemble_mix50','pure_support','stacked_full'),('quality_shrink','B_stage0','A_uniform')]:
                raw_max=max(raw_max,float(np.max(abs(np.array(get(m)['alpha'])-(np.array(get(a)['alpha'])+get(b)['alpha'])/2))))
    if max(raw_max,replay_max,seal_max)>1e-7:errors.append('component/replay/seal mismatch')
    common=[s for s in seeds if all(valid(idx[sc,s,m]) for sc in scenarios for m in methods)]
    groups=[]
    for endpoint in ['equal_scenario_mean']+scenarios:
        ss=scenarios if endpoint=='equal_scenario_mean' else [endpoint]
        for m in methods:
            values=[float(np.mean([idx[sc,s,m]['pooled_error'] for sc in ss])) for s in common]
            groups.append(dict(endpoint=endpoint,method=m,label=LABELS[m],n=len(values),mean=float(np.mean(values)),median=float(np.median(values))))
    cs=[];mat=[]
    for m,base in reg['contrasts']:
        delta={s:float(np.mean([idx[sc,s,m]['pooled_error']-idx[sc,s,base]['pooled_error'] for sc in scenarios])) for s in common}
        ds=list(delta.values());mat.append(ds);bs=[]
        for b in batches:
            bd=[delta[s] for s in b['seeds'] if s in delta]
            bs.append(dict(n=len(bd),mean_delta=float(np.mean(bd)),CI95=paired_bootstrap_mean_ci(bd)))
        conflicts=sum((idx[sc,s,m]['pooled_error']-idx[sc,s,base]['pooled_error'])*(idx[sc,s,m]['truth_error']-idx[sc,s,base]['truth_error'])<0 for sc in scenarios for s in common)
        cs.append(dict(method=m,baseline=base,label=LABELS[m]+'−'+LABELS[base],n=len(ds),mean_delta=float(np.mean(ds)),CI95=paired_bootstrap_mean_ci(ds),batches=bs,pooled_truth_direction_conflicts=int(conflicts),win_fraction=float(np.mean(np.array(ds)<0))))
    matrix=np.array(mat).T;n=len(common);rng=np.random.default_rng(reg['analysis_rng']);rep=reg['bootstrap_replicates'];boot=np.empty((rep,len(cs)))
    for start in range(0,rep,1000):
        w=rng.multinomial(n,np.full(n,1/n),size=min(1000,rep-start));boot[start:start+len(w)]=w@matrix/n
    ci=np.quantile(boot,[.05/(2*len(cs)),1-.05/(2*len(cs))],axis=0).T
    for c,q in zip(cs,ci):
        c['CI17']=q.tolist();c['stable_in_this_cohort']=bool(q[1]<0 and len(common)==len(seeds) and all(b['mean_delta']<0 for b in c['batches']) and sum(b['CI95'][1]<0 for b in c['batches'])>=2)
    costs={}
    for m in methods:
        rs=[r for r in idx.values() if r['method']==m];vs=[r for r in rs if valid(r)]
        costs[m]=dict(valid=len(vs),failed=len(rs)-len(vs),mean_stage3_seconds=float(np.mean([r['stage3_seconds'] for r in vs])),mean_stage3_requests=float(np.mean([r['stage3_requests'] for r in vs])),mean_initialization_seconds=float(np.mean([r['initialization_seconds'] for r in vs])),fallbacks=sum(bool(r.get('alpha_audit',{}).get('fallback_reason') or r.get('alpha_audit',{}).get('ensemble',{}).get('fallback_reason')) for r in rs))
    accept=dict(passed=not errors,errors=errors,coverage=coverage,rows=len(idx),new_rows=540,common_seeds=len(common),excluded_seeds=[s for s in seeds if s not in common],original_results_unchanged=True,max_component_difference=raw_max,max_replay_difference=replay_max,max_decision_difference=seal_max,replay_rows=18)
    dump(ROOT/'acceptance.json',accept);dump(ROOT/'analysis.json',dict(groups=groups,comparisons=cs,costs=costs,inference=reg['kind']))
    if errors:raise RuntimeError(errors)
    text=['# SE／QU 與原八種初始化方法：補充比較（2026-11-07）','','## 已完成','','SE＝0.5S＋0.5E；QU＝0.5Q＋0.5U。皆先固定 α，再呼叫一次原 Stage 3 best_of（內含既有兩個求解引擎），不把兩個交付 X 混合。QU 早期已有 quality_shrink 實驗；本次補上報告八方法同批比較。','','沿用報告的 90 seeds × 三情境、相同 local models／20 筆盲測／proposal／凍結 pooled evaluator，新增 540 條件；連同既有 2160 條件合計 2700。無新標籤、不調比例、不重訓、不更新 α、不 pruning／付款。pooled 僅交付後評分，truth 僅診斷。','','主要指標：每 seed 先平均三情境 pooled 絕對目標誤差，再對 seeds 平均。SE／QU 各對原八方法，加上 SE 對 QU，共 17 比較；200,000 次 seed 配對 bootstrap，Bonferroni 家族錯誤率 0.05。情境表僅描述，不另作顯著性主張。','','| 方法 | 整體平均 | 乾淨 | 低品質 25% | 樣本不平衡 4× |','|---|---:|---:|---:|---:|']
    lookup={(g['method'],g['endpoint']):g['mean'] for g in groups}
    for m in sorted(methods,key=lambda m:lookup[m,'equal_scenario_mean']):
        text.append('| '+LABELS[m]+' | '+' | '.join(f'{lookup[m,e]:.8f}' for e in ['equal_scenario_mean']+scenarios)+' |')
    text+=['','平均排名不等於穩定優於。以下差值為前者減後者，負值有利前者。','','| 比較 | 平均差 | 17 比較校正區間 | 通過批次一致性條件 |','|---|---:|---|---|']
    for c in cs:text.append(f"| {c['label']} | {c['mean_delta']:.8f} | [{c['CI17'][0]:.8f}, {c['CI17'][1]:.8f}] | {'是' if c['stable_in_this_cohort'] else '否'} |")
    text+=['','批次一致性條件：校正上界 < 0、三批平均差均 < 0、至少兩批各自普通 95% 區間上界 < 0、完整共同有效配對。即使通過，仍只是既有資料的補充證據。','','### 驗收與成本','',f'驗收：{len(idx)} 筆完整；共同有效 seeds {len(common)}/90；新增有效 {sum(costs[m]["valid"] for m in reg["methods"])}/540。原結果雜湊未變；三批各首 seed 共 18 筆重新求解，α／X／pooled／truth 最大差 {replay_max:g}；組成權重對原 E/S/Q/A 最大差 {raw_max:g}；決策先保存再評分差 {seal_max:g}。有效只表示有限且 box 可行，不證明全域最優。','','SE 需要六筆非自身 proposal 預測；QU 不需要支持訊號。共同盲測與訓練成本不重複收費。runner 為同批 SE 蒐集支持矩陣，不代表 QU 部署時也需這六次。診斷用對角線三次另存，不計部署必需成本。','','| 方法 | 有效／失敗 | Stage 3 平均查詢數 | Stage 3 平均秒數 | 初始化平均秒數 |','|---|---:|---:|---:|---:|']
    for m in methods:
        c=costs[m];text.append(f"| {LABELS[m]} | {c['valid']}／{c['failed']} | {c['mean_stage3_requests']:.2f} | {c['mean_stage3_seconds']:.4f} | {c['mean_initialization_seconds']:.6f} |")
    text+=['','原八法耗時取自歷史執行，SE／QU 取自本輪；只作描述，不能視為同時同負載的速度比較或嚴格成本門檻驗收。模型查詢數、單次倒推結構及追加標籤數可直接核對。各比較 pooled／truth 排序衝突數、fallback、批次區間與失敗資訊見 analysis.json。','','## 待驗證','','這是已檢視歷史 cohort 上的固定 50/50 補充比較，不是新獨立確認；多重比較校正不消除歷史資料反覆使用的限制。本輪不證明共同尺度最優、不驗證單筆盲測、惡意模型穩健性或真實世界效果。','','## 下一步提案','','若值得延伸，另行固定主要比較、效果門檻、獨立 seeds 與同機交錯成本測試；此次結果不得冒充獨立驗證。保留全部十法，不依平均排名淘汰。','','## 可重現產物','',f'- 註冊、來源快照、逐筆輸出、重評、重播、17 比較統計與驗收：`{ROOT}`','- 程式：`se_qu_initialization.py`、`run_se_qu_batch.py`、`run_se_qu_extension.py`、`analyze_se_qu_extension.py`','- 執行：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run_se_qu_extension.py`（輸出目錄必須不存在，禁止覆蓋）。']
    Path('docs/se_qu_extension_results_20261107.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(acceptance=accept,means={LABELS[m]:lookup[m,'equal_scenario_mean'] for m in methods}),ensure_ascii=False))
if __name__=='__main__':main()
