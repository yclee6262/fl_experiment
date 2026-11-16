"""Post-lock audit and reporting. No feedback to learner."""
import contextlib,itertools,json
from pathlib import Path
import numpy as np
from four_way_bayes import ROOT,read,metrics,propose
import four_way_bayes as run
import alpha_mixture_exploration as prior
import oob_shrinkage_research as shared
from research_evaluation import load_evaluator_artifact,score_solution

def analyze():
    reg=json.loads((ROOT/'registration.json').read_text());lock=json.loads((ROOT/'locked_candidate.json').read_text());rows=read(ROOT/'results.jsonl');old=read(prior.ROOT/'results.jsonl');errors=[]
    train=[s for b in reg['training_batches'] for s in b['seeds']];audit=[s for b in reg['audit_batches'] for s in b['seeds']]
    keys=[(r['method'],r['seed'],r['scenario']) for r in rows];methods=sorted({r['method'] for r in rows});expected=set()
    for m in methods:
        expected.update(itertools.product([m],audit if m.startswith('audit_') else train,prior.SCENARIOS))
    if len(keys)!=len(set(keys)) or set(keys)!=expected or len(rows)!=2970 or len(methods)!=29:errors.append('coverage')
    if set(train)&set(audit):errors.append('split overlap')
    for p,h in reg['old_hashes'].items():
        if shared.digest(p)!=h:errors.append('old result changed')
    for p,h in reg['source_hashes'].items():
        if shared.digest(ROOT/'source_snapshot'/p)!=h or shared.digest(p)!=h:errors.append('source changed '+p)
    decisions={(r['method'],r['seed'],r['scenario']):r for r in read(ROOT/'decisions.jsonl')};max_alpha=0.;max_pooled=0.;fallback=[]
    all_batches=reg['training_batches']+reg['audit_batches']
    for b in all_batches:
        for seed in b['seeds']:
            for sc in prior.SCENARIOS:
                with contextlib.redirect_stdout(None):server,item,artifact=shared.load_base(b['source'],sc,seed)
                p=np.column_stack([a.api_predict(server.test_X) for a in server.trusted_agents]);mse=((p-server.test_y[:,None])**2).mean(0);peer=np.full((3,3),np.nan)
                for j,a in enumerate(server.trusted_agents):
                    for i,x in enumerate(server.I_list):
                        if i!=j:peer[j,i]=float(np.asarray(a.api_predict(x)).reshape(-1)[0])
                evaluator,_=load_evaluator_artifact(artifact)
                for r in [r for r in rows if r['seed']==seed and r['scenario']==sc]:
                    if r['status']!='valid':errors.append('invalid output');continue
                    alpha,info=prior.initializer(r['w'],peer)('verify',server.trusted_agents,mse,server.test_X,server.test_y,server.I_list,0,cached_blind_predictions=p)
                    da=float(np.max(abs(alpha-r['alpha'])));max_alpha=max(max_alpha,da)
                    _,q=score_solution(evaluator,np.array(r['X']),0);dq=abs(q-r['pooled_error']);max_pooled=max(max_pooled,dq)
                    if da>1e-7 or dq>1e-7 or min(alpha)<0 or abs(sum(alpha)-1)>1e-8 or server.box_violation(r['X'])>1e-8:errors.append('reload')
                    d=decisions[r['method'],seed,sc]
                    if d['X']!=r['X'] or d['alpha']!=r['alpha'] or d['pooled_error'] is not None:errors.append('decision integrity')
                    for k,v in info['component_audits'].items():
                        if v.get('fallback_reason'):fallback.append([r['method'],seed,sc,k,v['fallback_reason']])
    # Rebuild learning objectives and acquisition decisions using only observations then available.
    baseline={(r['seed'],r['scenario']):r for r in old if r['method']=='UE_0.500'};ws=[];ys=[];obs=[]
    for m in sorted({r['method'] for r in old}):
        rr=[r for r in old if r['method']==m];d,j=metrics(rr,baseline);ws.append(rr[0]['w']);ys.append(j);obs.append({'method':m,'w':rr[0]['w'],'J':j})
    proposals=read(ROOT/'proposals.jsonl');max_proposal=0
    for k in range(16):
        if k<4:w=np.full(4,.2);w[k]=.4
        else:w,_=propose(np.asarray(ws),np.array(ys),k-4)
        max_proposal=max(max_proposal,float(np.max(abs(w-np.array(proposals[k]['w'])))))
        rr=[r for r in rows if r['method']==f'learn_{k:02d}'];d,j=metrics(rr,baseline);ws.append(rr[0]['w']);ys.append(j);obs.append({'method':f'learn_{k:02d}','w':rr[0]['w'],'J':j})
    chosen=min([o for o in obs if min(o['w'])>=.05-1e-12],key=lambda o:(o['J'],o['method']))
    if chosen['method']!=lock['method'] or chosen['w']!=lock['w'] or max_proposal>1e-6:errors.append('adaptive decision rebuild')
    # Audit EU versus saved original EU, not used for learning.
    full=json.loads(Path('alpha_weight_outputs/se_qu_extension_v1_20261107/registration.json').read_text())['batches'][:3]
    original={(r['seed'],r['scenario']):r for b in full for r in read(Path(b['six_method_results'])/'results.jsonl') if r['method']=='ensemble_shrink'}
    eu_replay=max(abs(r['pooled_error']-original[r['seed'],r['scenario']]['pooled_error']) for r in rows if r['method']=='audit_EU_canonical')
    if eu_replay>1e-7:errors.append('EU audit replay')
    # New locked-candidate full solver replay for one seed and three scenarios.
    if not (ROOT/'replay.jsonl').exists():
        first=reg['audit_batches'][0];run.ROOT=ROOT/'replay';run.ROOT.mkdir(exist_ok=False)
        replayed=run.evaluate('replay_candidate',np.array(lock['w']),[{**first,'seeds':first['seeds'][:1]}]);run.ROOT=ROOT
        shared.dump(ROOT/'replay.jsonl',replayed) # JSON array, separate from primary rows
    else:replayed=json.loads((ROOT/'replay.jsonl').read_text())
    idx={(r['method'],r['seed'],r['scenario']):r for r in rows}
    replay_max=max(float(np.max(abs(np.asarray(r[k])-np.asarray(idx['audit_candidate',r['seed'],r['scenario']][k])))) for r in replayed for k in ['X','alpha','pooled_error'])
    if replay_max>1e-7:errors.append('candidate replay')
    verification={'passed':not errors,'errors':errors,'expected_rows':2970,'rows':len(rows),'fallbacks':fallback,'alpha_max':max_alpha,'pooled_max':max_pooled,'proposal_max':max_proposal,'EU_saved_replay_max':eu_replay,'candidate_replay_max':replay_max,'replay_rows':3}
    shared.dump(ROOT/'verification.json',verification)
    if errors:raise RuntimeError(errors)
    rng=np.random.default_rng(20261116);stats={}
    auditbase={(r['seed'],r['scenario']):r for r in rows if r['method']=='audit_EU_canonical'}
    for m in methods+['train_equal','train_selected']:
        rr=([r for r in old if r['method']=='UQES_equal'] if m=='train_equal' else [r for r in old+rows if r['method']==lock['method']] if m=='train_selected' else [r for r in rows if r['method']==m])
        base=auditbase if m.startswith('audit_') else baseline;d,j=metrics(rr,base);ss=sorted({r['seed'] for r in rr});boot=d[rng.integers(0,len(d),(20000,len(d)))].mean(1)
        req=np.mean([r['stage3_requests']+r.get('support_requests',6 if r['w'][3]>0 else 0) for r in rr]);sam=np.mean([r['stage3_input_samples']+(6 if r['w'][3]>0 else 0) for r in rr]);br=list(base.values());breq=np.mean([r['stage3_requests'] for r in br]);bsam=np.mean([r['stage3_input_samples'] for r in br])
        delta=lambda filt:float(np.mean([r['pooled_error']-base[r['seed'],r['scenario']]['pooled_error'] for r in rr if filt(r)]))
        taskd=[r['pooled_error']-base[r['seed'],r['scenario']]['pooled_error'] for r in rr];alphas=np.array([r['alpha'] for r in rr])
        stats[m]={'w':rr[0]['w'],'n':len(d),'mean':float(np.mean([r['pooled_error'] for r in rr])),'delta':float(d.mean()),'J':j,'ci95':np.quantile(boot,[.025,.975]).tolist(),'median':float(np.median(d)),'p10_p90':np.quantile(d,[.1,.9]).tolist(),'win_tie_loss':[int(sum(d< -1e-12)),int(sum(abs(d)<=1e-12)),int(sum(d>1e-12))],'batch_delta':[delta(lambda r:r['batch']==b) for b in [1,2,3]],'scenario_delta':[delta(lambda r:r['scenario']==s) for s in prior.SCENARIOS],'method_requests':float(req),'requests_ratio':float(req/breq),'samples_ratio':float(sam/bsam),'init_ms':float(np.mean([r['initialization_seconds']+r.get('support_seconds',0) for r in rr])*1000),'search_ms':float(np.mean([r['stage3_seconds'] for r in rr])*1000),'max_alpha_mean':float(alphas.max(1).mean()),'effective_models_mean':float(np.mean(1/(alphas**2).sum(1))),'truth_mean':float(np.mean([r['truth_error'] for r in rr])),'worst_tasks':[{'seed':r['seed'],'scenario':r['scenario'],'pooled':r['pooled_error'],'delta':float(taskd[i]),'alpha':r['alpha']} for i,r in sorted(enumerate(rr),key=lambda ir:taskd[ir[0]],reverse=True)[:3]]}
    neighbors=[s for m,s in stats.items() if (m.startswith('neighbor') or m.startswith('tangent')) and np.linalg.norm(np.array(s['w'])-np.array(lock['w']))>1e-6];ng=sum(s['delta']<0 for s in neighbors)>=4 and np.median([s['delta'] for s in neighbors])<0
    s=stats['audit_candidate'];gate=s['delta']<=-.001 and max(s['batch_delta'])<0 and max(s['scenario_delta'])<0 and s['win_tie_loss'][0]/60>=.55 and s['requests_ratio']<=1.1 and s['samples_ratio']<=1.1 and ng
    equal={(r['seed'],r['scenario']):r for r in rows if r['method']=='audit_equal'};deq,_=metrics([r for r in rows if r['method']=='audit_candidate'],equal)
    result={'lock':lock,'methods':stats,'neighbor_gate':bool(ng),'neighbors_improving':sum(s['delta']<0 for s in neighbors),'recommend_new_confirmation':bool(gate),'audit_candidate_minus_equal':float(deq.mean()),'research_rows':2970,'research_search_seconds':sum(r['stage3_seconds'] for r in rows),'research_requests':sum(r['stage3_requests']+r['initialization_requests']+r['support_requests'] for r in rows),'verification':verification}
    shared.dump(ROOT/'analysis.json',result);print(json.dumps({'lock':lock,'audit':s,'neighbor_gate':bool(ng),'recommend':bool(gate)},ensure_ascii=False))
if __name__=='__main__':analyze()
