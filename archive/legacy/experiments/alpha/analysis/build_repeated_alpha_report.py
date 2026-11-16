"""Readable pooled-only main report; seeds and inference details stay folded."""
import argparse,json,hashlib
from pathlib import Path
from audit_repeated_alpha_evidence import audit
from repeated_alpha_candidates import CANDIDATES,DEFINITIONS

def table(headers,rows):
    def row(xs):return '<tr>'+''.join('<td>'+str(x)+'</td>' for x in xs)+'</tr>'
    return '<table header-row="true" fit-page-width="true">\n'+row(headers)+'\n'+'\n'.join(row(r) for r in rows)+'\n</table>'

def number(x):return '無有效資料' if x is None else f'{x:.6f}'

def interval(x):
    return '未計算（有效配對不完整）' if x is None else '['+', '.join(number(v) for v in x)+']'

def report(registration):
    evidence=audit(registration,final=True)
    if not evidence['completion_proven']:raise RuntimeError('Full independent evidence audit must pass before publication')
    reg=json.loads(Path(registration).read_text());root=Path(reg['output']);result=json.loads((root/'final_analysis.json').read_text())
    group={(g['endpoint'],g['method']):g for g in result['groups']}
    claim={c['method']:c for c in evidence['candidate_claims']}
    paired={(c['endpoint'],c['method'],c['baseline']):c for c in result['comparisons']}
    diagnostic_path=root/'qualified_case_diagnostic/analysis.json'
    diagnostic=json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else None
    if diagnostic and diagnostic['original_analysis_sha256']!=hashlib.sha256((root/'final_analysis.json').read_bytes()).hexdigest():
        raise RuntimeError('Diagnostic no longer corresponds to unchanged registered analysis')
    diag={} if diagnostic is None else {(c['endpoint'],c['method'],c['baseline']):c for c in diagnostic['comparisons']}
    signals=[m for m in CANDIDATES if all(diag.get(('equal_scenario_mean',m,b),{}).get('qualified_case_replication_signal') for b in ['A_uniform','B_stage0'])]
    names=dict(zip(reg['scenarios'],['乾淨','低品質','樣本量不平衡']))
    ordered=sorted(CANDIDATES,key=lambda m:group['equal_scenario_mean',m]['mean'] if group['equal_scenario_mean',m]['mean'] is not None else float('inf'))
    winners=[m for m in CANDIDATES if claim[m]['stable_primary_vs_A_and_B']]
    text='<mention-date start="2026-10-12"/>\n# 全部 α 初始化方法：三批獨立確認結果\n'
    text+='本次固定全部 27 種候選，另外加入兩個資訊量相同的對照，完成三批全新資料比較。原探索結果沒有併入本次確認，也沒有因中途名次而刪除方法。\n'
    text+='**主要品質為交付 X 的 pooled 絕對目標誤差，越低越好。**整體終點事前固定為三情境等權平均；各情境分別呈現。成本不影響品質排序，真實函數評分只作診斷。\n'
    valid_total=sum(b['statuses'].get('valid',0) for b in evidence['batches'])
    text+='執行完整：'+str(reg['expected_conditions'])+' 個條件全部保存，'+str(valid_total)+' 個解有效可行。第三批低品質情境有一次共同資格失敗，所有 29 個方案都在倒推前失敗；這不是某一種 α 的求解失敗。權重與 pooled 重載差均為 0。\n'
    if winners:
        text+='\n## 本次建立的穩定優勢\n'
        for m in winners:
            text+='- **'+DEFINITIONS[m][0]+'**：整體 pooled 品質同時相對 A／B 通過最終校正區間與逐批重現門檻。\n'
            if m in ['proposal_local_9','local_bias_risk_100']:
                text+='  - '+('也通過資訊量相同的對照，支持超出新增資訊本身的初始化收益。' if claim[m]['stable_matched_information_increment'] else '尚未通過資訊量相同的對照；可支持整套方案的收益，不能單獨歸因於初始化公式。')+'\n'
    else:
        text+='\n## 目前結論：正式門檻與方法潛力分開看\n原先登錄要求「所有計畫案例都有完整有效配對」。一次共同資格失敗使整體終點與低品質情境各少一個配對，原正式分析因此沒有計算這些比較的整批同時區間，也沒有給出整體優勢宣稱。**不能把這個門檻未通過寫成所有方法的區間都跨零，或所有方法都沒效果。**\n'
        text+='乾淨與樣本量不平衡情境仍有完整配對；新增 9 筆方案在這兩個情境對 A 通過原登錄的穩定改善條件，但對 B 未通過，不能擴大成普遍優勢。\n'
    if diagnostic:
        text+='\n## 合格案例診斷：哪些方法仍值得保留？\n為避免誤殺，另以所有方法共用的 563 組完整三情境合格案例計算相同的配對與跨批指標。所有原始失敗列都保留，原正式判讀也未改寫。**這是事後、以共同資格通過為條件的診斷，不能當作原先正式確認已通過。**\n'
        for m in signals:text+='- **'+DEFINITIONS[m][0]+'**：相對 A／B 的平均差在三批都為負，兩組診斷同時區間全負，具跨批改善線索。\n'
        text+='- **最佳組合向均勻收縮**：使用原 20 筆資訊，三批對 B 的平均差均較好；整批診斷區間仍跨零，保留為有潛力候選。\n'
        text+='- **自適應相關權重**：這次整體平均只略低於 B，診斷區間跨零；上一輪平均第一沒有轉成較強的跨批優勢證據。仍保留資料，沒有直接判為無效。\n'
    text+='\n## 所有候選的 pooled 表現\n依整體平均排序；平均名次與穩定優勢分開判讀。整體平均使用相同 563 組可評分案例，乾淨與不平衡各 564 組，低品質 563 組。資格失敗與有效解品質分開報告。\n'
    rows=[]
    for m in ordered:
        c=claim[m]
        state='既有比較基準' if c['role']=='baseline' else '整體穩定優勢' if c['stable_primary_vs_A_and_B'] else '僅特定情境穩定優勢' if c['stable_scenarios_vs_A_and_B'] else '尚未建立穩定優勢，保留候選'
        if not winners and c['role']!='baseline':
            if m in signals:state='合格案例有跨批改善線索；不是原正式通過'
            elif m=='ensemble_shrink':state='有潛力；對 B 的診斷區間仍跨零'
            elif m=='proposal_local_9':state='部分情境對 A 改善；未建立對 B 的正式優勢'
            else:state='保留候選；整體原正式配對門檻未通過'
        if c['stable_scenarios_vs_A_and_B']:state+='；'+ '／'.join(names[s] for s in c['stable_scenarios_vs_A_and_B'])
        rows.append([DEFINITIONS[m][0]]+[number(group[s,m]['mean']) for s in reg['scenarios']]+[number(group['equal_scenario_mean',m]['mean']),state])
    text+=table(['方法','乾淨','低品質','樣本量不平衡','整體平均','目前證據'],rows)+'\n'
    text+='\n## 三批是否都有相同方向？\n下表為相對 B 的平均 pooled 差值，負值較好；A／B 屬既有基準，不需要勝過自己。\n'
    rows=[]
    for m in ordered:
        c=paired.get(('equal_scenario_mean',m,'B_stage0'))
        if c:rows.append([DEFINITIONS[m][0]]+[number(b['mean_delta']) for b in c['batches']]+[number(c['mean_delta'])])
    text+=table(['方法','第一批差值','第二批差值','第三批差值','合併差值'],rows)+'\n'
    text+='\n## 新增資訊的對照\n新增 9 筆方案與 29 筆一般品質反比使用完全相同的新標記；100 筆局部風險與 100 筆一般品質反比也使用完全相同的資料。兩者都只在交付後接受 pooled 評分。\n'
    text+=table(['方案','新增標記','整體 pooled 平均'],[[DEFINITIONS[m][0],9 if m in ['proposal_local_9','B_augmented29'] else 80,number(group['equal_scenario_mean',m]['mean'])] for m in ['proposal_local_9','B_augmented29','local_bias_risk_100','B_augmented100']])+'\n'
    text+='两個新增標記方案雖然平均較好，相對資訊量相同對照的診斷區間都跨零。因此目前不能把收益歸因為局部風險／初始化公式較好，也不能宣稱公式有經濟最適或付款公平保證。\n'.replace('两','兩')
    if diagnostic:
        text+=table(['合格案例診斷比較','平均差','整批診斷區間','判讀'],[[DEFINITIONS[m][0]+' − '+DEFINITIONS[b][0],number(diag['equal_scenario_mean',m,b]['mean_delta']),interval(diag['equal_scenario_mean',m,b]['CI_simultaneous']),'跨零，未建立公式的額外收益'] for m,b in [('proposal_local_9','B_augmented29'),('local_bias_risk_100','B_augmented100')]])+'\n'
    text+='\n## 保留有潛力的方法\n沒有通過的候選仍保存完整結果。平均改善但不確定性較大的方法、只在部分情境改善的方法、合併有改善但跨批未重現的方法，分開列出，不直接標為「無效」。任何後續改公式、參數或情境的實驗都須另立版本，不能回頭改本次成功標準。\n'
    text+='下一輪優先方向：原 20 筆資訊先保留「最佳組合向均勻收縮」；可取得額外標記時，保留新增 9 筆與 100 筆局部風險方案及各自的相同資訊對照。其他候選不刪除。若要正式確認合格參與者中的改善，必須事前定義資格失敗的分析方式、固定獨立資料與有效改善門檻，不能沿用這次事後診斷直接宣稱勝出。此段是後續建議，尚未啟動新實驗。\n'
    text+='\n## 方法定義\n'+table(['方法','具體做法'],[DEFINITIONS[m] for m in CANDIDATES])+'\n'
    appendix=['## 固定設計','三批各 '+str(len(reg['batches'][0]['seeds']))+' 次，全新資料重複總數 '+str(len(reg['seeds']))+'；27 候選與兩個對照共 '+str(reg['expected_conditions'])+' 條件。',
        '三批 seeds：'+str([b['seeds'] for b in reg['batches']]),
        '主要終點為每個 seed 的三情境等權平均；不同情境不是獨立樣本。53 組對照 × 四個終點，212 組 Bonferroni 同時區間。最終使用 500,000 次配對 bootstrap、RNG 20261012；各批一般 95% 區間沿用 10,000 次、RNG 20260924。',
        '最終區間完全低於零、三批平均差都負、至少兩批一般 95% 區間全負、完整有效配對、未增加失敗，才標為該比較的穩定優勢。新候選需同時對 A 與 B 通過；單一情境的主張限定於該情境。',
        '研究解析度 0.003 只用於樣本數規劃；歷史 SD 與常態近似不保證較小效果或各情境的檢定力。跨零不是等效或沒有作用的證明。',
        '## 完整驗收','獨立稽核完成：'+str(evidence['completion_proven'])+'；所有條件完整保存，無重複或缺漏，來源快照與批次驗收一致。',table(['批次','預期／保存','有效交付','資格失敗','完成'],[[b['batch'],str(b['expected'])+'／'+str(b['saved']),b['statuses'].get('valid',0),b['statuses'].get('qualification_failure',0),b['complete']] for b in evidence['batches']]),
        '## 比較差與信賴區間',table(['終點','方法','基準','平均差','95% 區間','整批同時區間','穩定'],[[c['endpoint'],DEFINITIONS[c['method']][0],DEFINITIONS[c['baseline']][0],number(c['mean_delta']),interval(c['CI95']),interval(c['CI_simultaneous']),'通過' if c['stable_advantage'] else '未通過'] for c in result['comparisons']]),
        '## 失敗與有效配對',table(['方法','終點','有效數','失敗數'],[[DEFINITIONS[g['method']][0],g['endpoint'],g['n'],g['failures']] for g in result['groups']]),
        '## 重現與保存','登錄：'+registration+'；完整結果：'+str(root)+'；每批保存 source、results、calibration、模型與來源快照、verification。completion_evidence_audit.json 為獨立完成稽核，不僅依賴流程狀態。']
    if diagnostic:
        appendix.extend(['## 原正式判讀受阻的原因','第三批 low_quality_25pct、seed 10115 的 agent 20 未通過資格；同一條件的 29 法全部失敗。原正式比較的整體終點與低品質各有 53 組、共 106 組未計算同時區間；保留原分析及所有失敗，沒有以成功案例回填成完整有效配對。',
            '## 合格案例診斷：非原正式通過','只排除事前資格階段、所有方案共同失敗的案例；不能排除任何差異性的 solver 失敗。原 formal analysis SHA256：'+diagnostic['original_analysis_sha256']+'。使用相同 212 組區間校正與 500,000 次配對重抽樣，並維持原方法與參數；事後補充不升格為原確認結果。',
            table(['終點','方法','基準','有效配對','平均差','整批診斷區間','跨批線索'],[[c['endpoint'],DEFINITIONS[c['method']][0],DEFINITIONS[c['baseline']][0],c['n'],number(c['mean_delta']),interval(c['CI_simultaneous']),'有' if c['qualified_case_replication_signal'] else '未建立'] for c in diagnostic['comparisons']]),
            '診斷檔：'+str(diagnostic_path)+'；診斷程式：diagnose_repeated_alpha_qualified_cases.py。'])
    text+='\n<details>\n<summary>三批確認附錄｜統計、失敗與完整驗收</summary>\n'+'\n'.join('\t'+line for entry in appendix for line in entry.splitlines())+'\n</details>\n'
    path=Path('docs/repeated_alpha_confirmation_results_20261012_notion.md');path.write_text(text)
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('registration');a=p.parse_args();print(report(a.registration))
