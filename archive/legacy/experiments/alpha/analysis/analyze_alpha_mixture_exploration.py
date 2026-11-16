"""Verification and descriptive analysis; never called by a deployable initializer."""
import argparse,contextlib,itertools,json,time
import numpy as np
from pathlib import Path
from alpha_mixture_exploration import ROOT,SCENARIOS,BASE,read,dump,candidates,initializer,matrix
import oob_shrinkage_research as shared
from research_evaluation import load_evaluator_artifact,score_solution

def verify(label):
    reg=json.loads((ROOT/'registration.json').read_text());rows=read(ROOT/'results.jsonl');methods=sorted(set(r['method'] for r in rows));seeds=[s for b in reg['batches'] for s in b['seeds']]
    keys=[(r['seed'],r['scenario'],r['method']) for r in rows];expected=set(itertools.product(seeds,SCENARIOS,methods));errors=[];maxima={'alpha':0.,'pooled':0.};fallback=[]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:errors.append('coverage')
    for p,h in reg['old_result_hashes'].items():
        if shared.digest(p)!=h:errors.append('historical file changed '+p)
    for p,h in reg['source_sha256'].items():
        if shared.digest(ROOT/'source_snapshot'/p)!=h:errors.append('snapshot changed '+p)
        if shared.digest(p)!=h:errors.append('live source changed '+p)
    for b in reg['batches']:
        for seed in b['seeds']:
            for scenario in SCENARIOS:
                with contextlib.redirect_stdout(None):server,_,artifact=shared.load_base(b['source'],scenario,seed)
                pred=np.column_stack([a.api_predict(server.test_X) for a in server.trusted_agents]);mse=((pred-server.test_y[:,None])**2).mean(0)
                peer=np.full((3,3),np.nan)
                for j,a in enumerate(server.trusted_agents):
                    for i,x in enumerate(server.I_list):
                        if j!=i:peer[j,i]=float(np.asarray(a.api_predict(x)).reshape(-1)[0])
                evaluator,_=load_evaluator_artifact(artifact)
                for r in [r for r in rows if r['seed']==seed and r['scenario']==scenario]:
                    if r['status']!='valid':errors.append('failure '+str((seed,scenario,r['method'])));continue
                    alpha,a=initializer(r['w'],peer)('verify',server.trusted_agents,mse,server.test_X,server.test_y,server.I_list,0,cached_blind_predictions=pred)
                    da=float(np.max(abs(alpha-np.asarray(r['alpha']))));maxima['alpha']=max(maxima['alpha'],da)
                    if da>1e-7 or min(r['alpha'])<0 or abs(sum(r['alpha'])-1)>1e-8:errors.append('alpha')
                    for k,v in a['component_audits'].items():
                        if v.get('fallback_reason'):fallback.append([seed,scenario,r['method'],k,v['fallback_reason']])
                    x=np.asarray(r['X']);_,q=score_solution(evaluator,x,0);dq=abs(q-r['pooled_error']);maxima['pooled']=max(maxima['pooled'],dq)
                    if dq>1e-7 or not np.isfinite(x).all() or server.box_violation(x)>1e-8:errors.append('delivery')
    replay=read(ROOT/'replay.jsonl');replay_max=max(r['max_replay_difference'] for r in replay)
    if len(replay)!=90 or replay_max>1e-7:errors.append('replay')
    if fallback:errors.append('fallback needs baseline review')
    if (ROOT/'shared.jsonl').exists():
        for r in read(ROOT/'shared.jsonl'):
            for p,h in r['sha256'].items():
                if shared.digest(Path(r['artifact'])/p)!=h:errors.append('artifact changed')
    for stage in ['coarse','expansion']:
        p=ROOT/f'{stage}_decisions.jsonl'
        if p.exists():
            ds={(r['seed'],r['scenario'],r['method']):r for r in read(p)}
            for r in rows:
                if r['origin']==stage:
                    old=ds[r['seed'],r['scenario'],r['method']]
                    if old['X']!=r['X'] or old['alpha']!=r['alpha'] or old['pooled_error'] is not None:errors.append('decision integrity')
    result={'passed':not errors,'errors':errors,'rows':len(rows),'methods':len(methods),'seeds':len(seeds),'max_difference':maxima,'replay_rows':len(replay),'replay_max':replay_max,'fallbacks':fallback,'old_results_unchanged':all(shared.digest(p)==h for p,h in reg['old_result_hashes'].items())}
    dump(label+'_verification.json',result);print(json.dumps(result));return result

def analyze():
    rows=read(ROOT/'results.jsonl');names=sorted(set(r['method'] for r in rows));seeds,a=matrix(rows,names);base=a[names.index('UE_0.500')];rng=np.random.default_rng(20261116);ix=rng.integers(0,len(seeds),(20000,len(seeds)))
    stats={};all_diffs={};recommended=[]
    for m,z in zip(names,a):
        d=z-base;sd=d.mean(1);boot=sd[ix].mean(1);rr=[r for r in rows if r['method']==m]
        alpha=np.array([r['alpha'] for r in rr]);quality=np.array([r['pooled_error'] for r in rr]);concentration=alpha.max(1)
        cost=np.array([r['stage3_requests']+(6 if r['w'][3]>0 else 0) for r in rr],float)
        basecost=np.array([r['stage3_requests'] for r in rows if r['method']=='UE_0.500'])
        neighbor=[]
        if '_' in m and not m.endswith('equal'):
            edge,t=m.split('_');t=float(t)
            # Immediate observed neighbors on this edge, including endpoints.
            points={0.:edge[1],1.:edge[0]}
            points.update({float(n.split('_')[1]):n for n in names if n.startswith(edge+'_') and not n.endswith('equal')})
            tt=sorted(points);i=tt.index(t)
            neighbor=[points[tt[j]] for j in [i-1,i+1] if 0<=j<len(tt)]
        effect=float(sd.mean());batch=[float(sd[i:i+10].mean()) for i in [0,10,20]];scenario=d.mean(0).tolist();win=float(np.mean(sd< -1e-12))
        gate=effect<=-.001 and max(batch)<0 and max(scenario)<=0 and win>=.55 and cost.mean()/basecost.mean()<=1.10 and bool(neighbor) and any((a[names.index(n)]-base).mean()<0 for n in neighbor)
        if gate:recommended.append(m)
        corr=lambda x,y:float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-12 and np.std(y)>1e-12 else None
        paired_flat=np.array([r['pooled_error']-next(b['pooled_error'] for b in rows if b['method']=='UE_0.500' and b['seed']==r['seed'] and b['scenario']==r['scenario']) for r in rr])
        stats[m]={'w':rr[0]['w'],'mean':float(z.mean()),'seed_median':float(np.median(z.mean(1))),'delta':effect,'delta_median':float(np.median(sd)),
          'delta_p10_p90':np.quantile(sd,[.1,.9]).tolist(),'exploratory_ci95':np.quantile(boot,[.025,.975]).tolist(),
          'win_tie_loss':[int(sum(sd< -1e-12)),int(sum(abs(sd)<=1e-12)),int(sum(sd>1e-12))],
          'batch_delta':batch,'scenario_mean':z.mean(0).tolist(),'scenario_delta':scenario,'neighbors':neighbor,'passes_screen':bool(gate),
          'mean_stage3_requests':float(np.mean([r['stage3_requests'] for r in rr])),
          'mean_method_requests':float(cost.mean()),'requests_ratio_to_EU':float(cost.mean()/basecost.mean()),
          'mean_stage3_input_samples':float(np.mean([r['stage3_input_samples'] for r in rr])),
          'mean_init_seconds':float(np.mean([r['initialization_seconds']+r.get('support_seconds',r.get('support_independent_cost',{}).get('seconds',0)) for r in rr])),
          'mean_search_seconds':float(np.mean([r['stage3_seconds'] for r in rr])),
          'mean_method_seconds':float(np.mean([r['initialization_seconds']+r['stage3_seconds']+r.get('support_seconds',r.get('support_independent_cost',{}).get('seconds',0)) for r in rr])),
          'mean_max_alpha':float(concentration.mean()),'max_alpha_gt_08_fraction':float(np.mean(concentration>.8)),
          'mean_effective_models':float(np.mean(1/(alpha**2).sum(1))),
          'max_alpha_quality_correlation':corr(concentration,quality),'max_alpha_delta_correlation':corr(concentration,paired_flat),
          'truth_mean':float(np.mean([r['truth_error'] for r in rr])),
          'worst_seed_deltas':[{'seed':seeds[i],'delta':float(sd[i])} for i in np.argsort(sd)[-3:][::-1]],
          'worst_tasks':[{'seed':r['seed'],'scenario':r['scenario'],'pooled':r['pooled_error'],'delta':float(paired_flat[i]),'alpha':r['alpha'],'X':r['X']} for i,r in sorted(enumerate(rr),key=lambda ir:paired_flat[ir[0]],reverse=True)[:3]]}
        all_diffs[m]=sd.tolist()
    recommended=sorted(recommended,key=lambda m:(stats[m]['delta'],m))[:2]
    replay=read(ROOT/'replay.jsonl');timing={}
    for m in BASE:
        r=[x for x in replay if x['method']==m];timing[m]={'n':len(r),'mean_init_seconds':float(np.mean([x['initialization_seconds']+x['support_seconds'] for x in r])),'mean_search_seconds':float(np.mean([x['stage3_seconds'] for x in r]))}
    shared_rows=read(ROOT/'shared.jsonl');new=[r for r in rows if r['origin']!='historical']
    audit={'methods':stats,'recommended':recommended,'seed_differences':all_diffs,'oracle_task_mean':float(a.min(0).mean()),'EU_mean':float(base.mean()),
       'same_session_replay_timing':timing,'new_solves':len(new),'research_stage3_requests':sum(r['stage3_requests'] for r in new),'research_search_seconds':sum(r['stage3_seconds'] for r in new),
       'shared_preparation_examples':shared_rows[0]['preparation'],'shared_blind_requests':270,'shared_blind_samples':5400,
       'notes':['CI is descriptive, unadjusted, not selection-valid.','Historical and new wall times not a strict speed comparison.','Truth diagnostic separate from pooled.']}
    dump('analysis.json',audit)
    lines=['# α 初始化混合比例：小規模探索結果（2026-11-16）','','30個歷史開發案例，每案三情境；沒有新獨立確認。數值不可與原90案例均值或C/P cohort混用。',
      f"共{len(names)}候選、{len(rows)}筆比較，其中新增{len(new)}次求解，另90次基準核驗。EU在本子集平均={base.mean():.8f}。",'',
      '負配對差有利候選。95%區間為20,000次完整seed bootstrap，未校正且不處理事後選擇偏差，不能當正式顯著性。',
      '', '|方法（前字母比例）|pooled均值|對EU差|探索95%區間|勝/平/負（30 seeds）|方法請求/EU|', '|---|---:|---:|---|---|---:|']
    for m in sorted(names,key=lambda m:stats[m]['mean']):
        s=stats[m];lo,hi=s['exploratory_ci95'];lines.append(f"|{m}|{s['mean']:.8f}|{s['delta']:+.8f}|[{lo:+.6f},{hi:+.6f}]|{'/'.join(map(str,s['win_tie_loss']))}|{s['requests_ratio_to_EU']:.3f}|")
    lines+=['','## 分布與情境','','|方法|配對差中位|配對差p10/p90|乾淨差|低品質差|不平衡差|三批子集差|','|---|---:|---|---:|---:|---:|---|']
    for m in names:
        s=stats[m];lines.append(f"|{m}|{s['delta_median']:+.6f}|{s['delta_p10_p90'][0]:+.6f}/{s['delta_p10_p90'][1]:+.6f}|"+'|'.join(f'{v:+.6f}' for v in s['scenario_delta'])+'|'+ '/'.join(f'{v:+.6f}' for v in s['batch_delta'])+'|')
    lines+=['','## 成本與集中診斷','','初始化包含活躍成分計算及S所需六次非自身查詢。共同盲測3 requests/60 samples屬資格流程；訓練、資格、anchor成本另保存在shared.jsonl，不重複計入每方法。下列歷史／新計時不同時段，僅描述；9任務同機基準重播另見analysis.json。',
      '', '|方法|初始化ms|搜尋ms|方法總ms|搜尋requests|方法requests|搜尋samples|max α均值|有效模型數|max α>0.8|集中/pooled相關|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in names:
        s=stats[m];c=s['max_alpha_quality_correlation'];lines.append(f"|{m}|{s['mean_init_seconds']*1000:.3f}|{s['mean_search_seconds']*1000:.3f}|{s['mean_method_seconds']*1000:.3f}|{s['mean_stage3_requests']:.2f}|{s['mean_method_requests']:.2f}|{s['mean_stage3_input_samples']:.2f}|{s['mean_max_alpha']:.3f}|{s['mean_effective_models']:.3f}|{s['max_alpha_gt_08_fraction']:.1%}|{c if c is not None else 'NA'}|")
    lines+=['','## 篩選與限制','',f"通過預先篩選規則：{recommended or '無；維持固定EU'}。門檻與局部擴張規則見registered_design.md；完整逐案分布、失敗尾端、truth獨立診斷見analysis.json。",f"逐任務pooled oracle平均={audit['oracle_task_mean']:.8f}，這是不可部署的事後診斷，不能據此宣稱任何固定w可達此值。",'',
      f"新增搜尋實際總模型請求={audit['research_stage3_requests']:,}，搜尋總秒數={audit['research_search_seconds']:.2f}；這是整批探索成本，不是固定w單次部署成本。",
      '', 'alpha 是三位Agent權重；w是U/Q/E/S方法比例。集中/品質相關不代表因果，truth與pooled可能不同，不用truth選候選。尚未測高密度格點、其他三/四法非等權比例、其他維度/人數/目標或新資料確認。']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'recommended':recommended,'top':sorted([(m,s['mean'],s['delta']) for m,s in stats.items()],key=lambda x:x[1])[:8],'new_solves':len(new)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action');a=p.parse_args()
    if a.action=='analyze':analyze()
    elif not verify(a.action)['passed']:raise SystemExit(1)
