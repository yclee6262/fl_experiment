"""Post-hoc paired extension: alpha = .5 support + .5 inverse-quality."""
import contextlib, json, time
from pathlib import Path
import numpy as np
import oob_shrinkage_research as shared
from peer_support_alpha import support_signal, METHODS, SCENARIOS
from research_evaluation import load_evaluator_artifact, score_solution, synthetic_truth
from research_analysis import audit_coverage, paired_bootstrap_mean_ci

METHOD = 'support_quality_mix50'
ROOT = Path('alpha_weight_outputs/support_quality_mix50_v1_20261013')
REG = Path('docs/support_quality_mix50_registration_20261013.json')

def mixed_weights(mse, prediction_matrix, target):
    mse = np.asarray(mse, float)
    if mse.ndim != 1 or not np.isfinite(mse).all() or np.any(mse < 0):
        raise ValueError('Invalid quality evidence')
    s, audit = support_signal(prediction_matrix, target)
    if len(s) != len(mse):
        raise ValueError('Agent order mismatch')
    b = 1 / (mse + 1e-5)
    b /= b.sum()
    alpha = .5 * s + .5 * b
    audit.update(quality=b.tolist(), support_fraction=.5, quality_fraction=.5,
                 uniform_fraction=0., formula='0.5*support + 0.5*B',
                 signal_semantics='candidate support, not participant contribution')
    return alpha, audit

def initializer(matrix):
    def initial(method, agents, mse, x, y, proposals, target, **kwargs):
        if method != METHOD:
            raise ValueError(method)
        w, a = mixed_weights(mse, matrix, target)
        a['agent_ids'] = [g.agent_id for g in agents]
        return w, a
    return initial

def readlines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]

def register():
    old = json.loads(Path('docs/peer_support_formal_registration_v2_20261013.json').read_text())
    if ROOT.exists() or REG.exists():
        raise FileExistsError('Preserve existing extension output')
    reg = dict(output=str(ROOT), method=METHOD, baselines=METHODS, scenarios=SCENARIOS,
        batches=old['batches'], formula='alpha=0.5*support+0.5*B', uniform_fraction=0,
        inference='post-hoc extension of previously inspected data; not independent confirmation',
        pooled_role='only after delivery', truth_role='diagnostic only',
        alpha_update=False, pruning=False, settlement='not_run',
        box_tolerance=1e-8, reload_tolerance=1e-7, bootstrap_replicates=10000,
        analysis_rng=20261013, comparison_family=6,
        primary_endpoint='equal-weight three-scenario mean within each seed',
        old_registration_sha256=shared.digest('docs/peer_support_formal_registration_v2_20261013.json'),
        source_sha256={p.name:shared.digest(p) for p in Path('.').glob('*.py')},
        baseline_sha256={b['results']:shared.digest(Path(b['results'])/'results.jsonl') for b in old['batches']})
    # Original frozen sources must still match before any new execution.
    for name, digest in old['source_sha256'].items():
        if shared.digest(name) != digest:
            raise RuntimeError('Original frozen source changed: '+name)
    shared.dump(REG, reg)
    ROOT.mkdir()
    snapshot=ROOT/'source_snapshot';snapshot.mkdir()
    for name in reg['source_sha256']:
        (snapshot/name).write_bytes(Path(name).read_bytes())
    shared.dump(ROOT/'manifest.json',dict(reg, status='running', expected=1089))
    return reg

def run():
    reg=register()
    expected=[]
    manifest=json.loads((ROOT/'manifest.json').read_text())
    try:
        with (ROOT/'solver.log').open('w') as log:
            for bi,batch in enumerate(reg['batches']):
                evidence={(r['scenario'],r['seed']):r for r in readlines(Path(batch['results'])/'peer_evidence.jsonl')}
                for scenario in SCENARIOS:
                    for seed in batch['seeds']:
                        identity=dict(scenario=scenario,seed=seed,batch=bi+1,method=METHOD)
                        expected.append((scenario,seed,METHOD))
                        peer=evidence.get((scenario,seed))
                        if peer is None:
                            shared.append(ROOT/'results.jsonl',dict(identity,status='preparation_failure',reason='missing common peer evidence',X=None,alpha=None,pooled_error=None,truth_error=None))
                            continue
                        server,_,artifact=shared.load_base(batch['source'],scenario,seed)
                        p=np.array([[float(np.asarray(a.api_predict(x)).reshape(-1)[0]) for x in server.I_list] for a in server.trusted_agents])
                        if not np.allclose(p,peer['prediction_matrix'],atol=1e-7,rtol=0):
                            raise RuntimeError('Peer matrix reload mismatch')
                        if [a.agent_id for a in server.trusted_agents]!=peer['agent_ids']:
                            raise RuntimeError('Agent order mismatch')
                        mse=np.asarray(peer['stage0_mse'])
                        shared.initial_alpha=initializer(p)
                        try:
                            with contextlib.redirect_stdout(log):
                                row=shared.solve(server,METHOD,mse,server.test_X,server.test_y,np.asarray(peer['blind_prediction']),seed,1000)
                        except Exception as exc:
                            shared.append(ROOT/'results.jsonl',dict(identity,**getattr(exc,'partial_row',{}),status='interrupted',reason=repr(exc)))
                            raise
                        row.update(identity,additional_labels=0,agent_ids=peer['agent_ids'],artifact=str(artifact.resolve()),
                            artifact_sha256={q.name:shared.digest(q) for q in artifact.iterdir() if q.is_file()},
                            support_independent_cost=peer['support_cost'],common_preparation='reused previous paired batch',
                            research_reload_queries=9,research_reload_input_samples=9)
                        if row['status']=='valid':
                            evaluator,_=load_evaluator_artifact(artifact)
                            start=time.perf_counter()
                            _,score=score_solution(evaluator,np.array(row['X']),0)
                            row.update(pooled_error=score,truth_error=abs(float(synthetic_truth(np.array(row['X'])[None,:])[0])),external_scoring_seconds=time.perf_counter()-start)
                        shared.append(ROOT/'results.jsonl',row)
                    print(f'batch {bi+1}: {scenario} complete',flush=True)
        manifest['status']='complete'
    except Exception as exc:
        manifest.update(status='interrupted',reason=repr(exc));raise
    finally:
        shared.dump(ROOT/'manifest.json',manifest)
    verify(reg)
    analyze(reg)

def valid(row):
    x=row.get('X')
    return row['status']=='valid' and x is not None and np.asarray(x).shape==(2,) and np.isfinite(x).all() and np.max(np.abs(x))<=1+1e-8 and row.get('pooled_error') is not None and np.isfinite(row['pooled_error'])

def verify(reg):
    rows=readlines(ROOT/'results.jsonl');errors=[];maxima=dict(alpha=0.,pooled=0.,matrix=0.);valid_n=0
    expected=[(s,k,METHOD) for b in reg['batches'] for s in SCENARIOS for k in b['seeds']]
    cover=audit_coverage(rows,expected,['scenario','seed','method'])
    if not cover['complete']:errors.append('incomplete coverage')
    for name,h in reg['source_sha256'].items():
        if shared.digest(name)!=h or shared.digest(ROOT/'source_snapshot'/name)!=h:errors.append('source hash mismatch '+name)
    for b in reg['batches']:
        if shared.digest(Path(b['results'])/'results.jsonl')!=reg['baseline_sha256'][b['results']]:errors.append('baseline changed')
    for row in rows:
        if row['status']!='valid':continue
        b=reg['batches'][row['batch']-1]
        server,_,artifact=shared.load_base(b['source'],row['scenario'],row['seed'])
        for name,h in row['artifact_sha256'].items():
            if shared.digest(artifact/name)!=h:errors.append('artifact changed')
        blind=np.column_stack([a.api_predict(server.test_X) for a in server.trusted_agents]);mse=((blind-server.test_y[:,None])**2).mean(axis=0)
        p=np.array([[float(np.asarray(a.api_predict(x)).reshape(-1)[0]) for x in server.I_list] for a in server.trusted_agents])
        alpha,_=mixed_weights(mse,p,0)
        maxima['alpha']=max(maxima['alpha'],float(np.max(abs(alpha-row['alpha']))))
        if not valid(row) or server.box_violation(np.asarray(row['X']))>1e-8:errors.append('invalid delivery')
        evaluator,_=load_evaluator_artifact(artifact);_,score=score_solution(evaluator,np.asarray(row['X']),0)
        maxima['pooled']=max(maxima['pooled'],abs(score-row['pooled_error']));valid_n+=1
    if maxima['alpha']>1e-7 or maxima['pooled']>1e-7:errors.append('reload exceeds tolerance')
    result=dict(passed=not errors,errors=errors,coverage=cover,saved=len(rows),valid=valid_n,max_reload_difference=maxima)
    shared.dump(ROOT/'verification.json',result)
    if errors:raise RuntimeError(str(errors))

def analyze(reg):
    rows=readlines(ROOT/'results.jsonl');index={(r['scenario'],r['seed'],r['method']):r for r in rows}
    for b in reg['batches']:
        for r in readlines(Path(b['results'])/'results.jsonl'):
            key=(r['scenario'],r['seed'],r['method'])
            if key in index:raise RuntimeError('Duplicate baseline key')
            index[key]=r
    methods=METHODS+[METHOD];groups=[];comparisons=[]
    allseeds=[k for b in reg['batches'] for k in b['seeds']]
    for endpoint in ['equal_scenario_mean']+SCENARIOS:
        scenarios=SCENARIOS if endpoint=='equal_scenario_mean' else [endpoint]
        for method in methods:
            values=[np.mean([index[s,k,method]['pooled_error'] for s in scenarios]) for k in allseeds if all(valid(index[s,k,method]) for s in scenarios)]
            groups.append(dict(endpoint=endpoint,method=method,n=len(values),excluded=len(allseeds)-len(values),mean=float(np.mean(values)) if values else None,median=float(np.median(values)) if values else None,Q1=float(np.quantile(values,.25)) if values else None,Q3=float(np.quantile(values,.75)) if values else None))
        for base in METHODS:
            ds=[];batches=[];excluded=[];conflicts=0
            for bi,b in enumerate(reg['batches']):
                bd=[]
                for k in b['seeds']:
                    if not all(valid(index[s,k,METHOD]) and valid(index[s,k,base]) for s in scenarios):excluded.append(k);continue
                    d=np.mean([index[s,k,METHOD]['pooled_error']-index[s,k,base]['pooled_error'] for s in scenarios]);bd.append(d)
                    for s in scenarios:
                        conflicts+=int((index[s,k,METHOD]['pooled_error']-index[s,k,base]['pooled_error'])*(index[s,k,METHOD]['truth_error']-index[s,k,base]['truth_error'])<0)
                ds.extend(bd);batches.append(dict(batch=bi+1,n=len(bd),mean_delta=float(np.mean(bd)) if bd else None,CI95=paired_bootstrap_mean_ci(bd,seed=reg['analysis_rng'])))
            comparisons.append(dict(endpoint=endpoint,baseline=base,n=len(ds),excluded=excluded,mean_delta=float(np.mean(ds)) if ds else None,CI95=paired_bootstrap_mean_ci(ds,seed=reg['analysis_rng']),CI_family=paired_bootstrap_mean_ci(ds,confidence=1-.05/6,seed=reg['analysis_rng']) if endpoint=='equal_scenario_mean' else None,batches=batches,pooled_truth_conflicts=conflicts,inference='exploratory reused-data comparison, not independent confirmation'))
    results=dict(groups=groups,comparisons=comparisons,new_conditions=len(rows),new_valid=sum(valid(r) for r in rows),inference=reg['inference'])
    shared.dump(ROOT/'analysis.json',results)
    report=['# 支持度與 B 品質各 50%：追加配對比較','', '**本輪完成；使用前輪資料追加新方法，屬探索性比較，不能宣稱已獨立建立穩定優勢。**','',
        '新方法 M：αᵢ = 0.5 × sᵢ + 0.5 × Bᵢ。支持度 s 與 B 先各自正規化，再相加；沒有額外向均勻收縮。',
        'Bᵢ 是原 20 筆盲測 MSE 加 0.00001 後取倒數並正規化。sᵢ 是其他模型對第 i 位提出者候選點的預測距離目標之中位數，依前輪固定尺度轉成指數支持度並正規化；排除自己的評分。',
        '例如 B=[0.6,0.3,0.1]、s=[0.2,0.3,0.5]，M=[0.4,0.3,0.3]。這是兩種信號的折衷，不是貢獻或付款比例。','',
        '三種情境為乾淨、低品質（local labels 噪聲）與樣本量 4 倍不平衡。沿用同一模型、候選點、固定三人與凍結 evaluator，新增 1,089 次單次倒推；既有六方法的 6,534 列保留不改。',
        '各情境 363 個配對案例，來自前輪三批。2D、target=0、box=[−1,1]、best_of/custom30；不更新 α、不剪枝、不结算。pooled evaluator 僅在交付後評分；真實函數只作診斷。'.replace('结算','結算'),'',
        f'新增結果 {len(rows)} 列，其中有效且可行 {results["new_valid"]} 列。驗收與重載差異見 verification.json。','',
        '## pooled 品質總比較','', '主要表格先在每個案例平均三種情境，再跨案例彙總。pooled error = |f_pooled(X)−target|，越低越好；中位數為第 50 百分位，Q1/Q3 為第 25/75 百分位。','',
        '|方法|平均誤差|中位數|Q1|Q3|有效案例|','|---|---:|---:|---:|---:|---:|']
    names=dict(zip(METHODS,['A 均勻','B 品質反比','Q 品質向均勻收縮','S 支持度向均勻收縮','QS 品質乘支持後向均勻收縮','E 最佳組合向均勻收縮']));names[METHOD]='M 支持度與 B 各 50%'
    for g in sorted([g for g in groups if g['endpoint']=='equal_scenario_mean'],key=lambda g:g['mean'] if g['mean'] is not None else float('inf')):
        report.append(f'|{names[g["method"]]}|{g["mean"]:.6f}|{g["median"]:.6f}|{g["Q1"]:.6f}|{g["Q3"]:.6f}|{g["n"]}|')
    report+=['','## M 相對現有方法','', '差值 = M 誤差 − 對照誤差；負值代表 M 較好。95% CI 以案例為配對抽樣單位，三情境保留同一組；六比較的校正區間採 Bonferroni。由於重用已看過的資料，即使區間未跨零仍需新資料確認。','', '|對照|平均差值|95% CI|六比較校正 CI|pooled／truth 方向衝突數|','|---|---:|---|---|---:|']
    for c in comparisons:
        if c['endpoint']=='equal_scenario_mean':
            report.append(f'|{names[c["baseline"]]}|{c["mean_delta"]:+.6f}|{c["CI95"]}|{c["CI_family"]}|{c["pooled_truth_conflicts"]}|')
    report+=['','方向衝突指同一方法配對在 pooled 與真實函數上得到相反的改善／惡化方向，不是求解失敗。','',
        '## 成本與保存','', '每列保存 α、X、求解狀態、可行性、pooled/truth、Stage 3 呼叫數／輸入樣本数／時間、支持度獨立準備成本，以及模型檔案雜湊。共同模型訓練沿用舊資料，不冒充零成本；本輪 9 次候選矩陣重載查詢另記為研究驗證成本。'.replace('样','樣').replace('数','數'),
        '舊六方法資料不覆蓋。來源快照、registration、manifest、逐條結果、重評與完整分析存於本輪獨立目錄。','',
        '## 下一步','', '若 M 有值得確認的訊號，固定本輪 50/50，不根據這批結果再調比例，改用未使用過的資料與 evaluator 做獨立配對確認。M 與 S/Q 的差別同時包含訊號與均勻收縮方式，不能單憑這兩組差異分別歸因。']
    (ROOT/'report.md').write_text('\n'.join(report)+'\n')
    Path('docs/support_quality_mix50_results_20261013.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(new_conditions=len(rows),valid=int(results['new_valid']),means={g['method']:g['mean'] for g in groups if g['endpoint']=='equal_scenario_mean'})))

if __name__=='__main__':run()
