from pathlib import Path
import csv,json,hashlib,itertools
import numpy as np
R=Path('/Users/yclee/Documents/GitHub/fl_experiment');A=R/'contribution_analysis';OUT=A/'contribution_search_summary_20261020';CONF=A/'adaptive_response_confirmation_experiment_v2_20261020'
FAMILIES=[('共享位置庫','shared_bank_experiment_v2_20261020'),('固定回應代理','shared_response_experiment_v2_20261020'),('固定模型面板','distributed_panel_experiment_v1_20261020'),('既有驗證可靠度','validation_judge_experiment_v1_20261020'),('可靠度誤差聚合','reliability_risk_experiment_v1_20261020'),('路徑幾何','geometric_replay_experiment_v1_20261020'),('穩健端點評分','robust_endpoint_experiment_v2_20261020'),('排名共識','rank_consensus_experiment_v1_20261020'),('當前殘差回饋','current_task_residual_experiment_v1_20261020'),('當前局部回應','local_response_panel_experiment_v1_20261020'),('自適應回應代理','adaptive_shared_response_experiment_v2_20261020')]
def read(p):return json.loads(p.read_text())
def rows(p):
 with p.open() as f:return list(csv.DictReader(f))
def save(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
allrows=[];best=[]
for label,folder in FAMILIES:
 ss=read(A/folder/'development_summary.json');keys={(s.get('config',''),s.get('panel',''),s['method']) for s in ss};candidates=[]
 for config,panel,method in sorted(keys):
  pair=[s for s in ss if (s.get('config',''),s.get('panel',''),s['method'])==(config,panel,method)];cl=next(s for s in pair if s['scenario']=='clean');po=next(s for s in pair if s['scenario']=='controlled_poison');rec=dict(family=label,folder=folder,config=config,panel=panel,method=method,clean=cl['mean'],poison=po['mean'],clean_coverage=cl['coverage'],poison_coverage=po['coverage'],offline_only=method=='offline_E');allrows.append(rec)
  if not rec['offline_only'] and cl['mean'] is not None and po['mean'] is not None:candidates.append(rec)
 picked=max(candidates,key=lambda t:(min(t['clean'],t['poison']),t['clean']+t['poison'],t['method']));best.append(dict(**picked,n_configurations=len(candidates)))
with (OUT/'development_all_configurations.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
save(OUT/'development_family_summaries.json',best)
team=rows(CONF/'team_metrics.csv');values=rows(CONF/'participant_comparison.csv');examples=[]
for scenario in ['clean','controlled_poison']:
 ts=[t for t in team if t['scenario']==scenario]
 for label,selected in [('best_rank',max(ts,key=lambda t:(float(t['spearman']),-int(t['seed'])))),('worst_rank',min(ts,key=lambda t:(float(t['spearman']),int(t['seed']))))]:examples.append(dict(category=label,team=selected,participants=[r for r in values if r['key']==selected['key']]))
save(OUT/'representative_cases.json',examples)
ratios=[];scanned=0
for c in read(CONF/'cases.json'):
 sc=read(CONF/'scores'/(c['key']+'.json'));base=np.array(read(CONF/'normal'/(c['key']+'.json'))['result']['solution'])
 for s in sc['provider_summaries']:
  for p in sc['participants']:ratios.append(float(np.linalg.norm(np.array(p['solution'])-base)/s['h']))
 # Full own-data scan to select nearest neighbors; only ownership and X are needed here.
 with np.load(Path(c['models']).parent/'pooled_data.npz') as data:owner=data['agent_ids']
 scanned+=sum(int(np.sum(owner==i)) for i in sc['provider_ids'])
save(OUT/'client_data_scope.json',dict(client_rows_scanned=scanned,client_rows_selected_for_fitting=15360,endpoint_provider_pairs=len(ratios),endpoint_distance_over_h_median=float(np.median(ratios)),endpoint_distance_over_h_max=float(max(ratios)),endpoint_provider_pairs_outside_radius=sum(x>1 for x in ratios),new_observations_collected=0,notes='Descriptive after locked analysis; no exclusions or score changes.'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font=Path('/System/Library/Fonts/PingFang.ttc')
zh=font.exists()
if zh:plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus']=False
S=read(CONF/'summary.json');cost=read(CONF/'cost.json');bench=read(CONF/'benchmark_results_v2.json');fig,ax=plt.subplots(1,3,figsize=(16,5.1),gridspec_kw={'width_ratios':[1.1,1,1.2]})
means=[s['mean'] for s in S];ci=np.array([s['ci'] for s in S]);ys=np.array([1,0]);ax[0].errorbar(means,ys,xerr=np.array([np.array(means)-ci[:,0],ci[:,1]-means]),fmt='o',capsize=6,color='#2266aa',lw=2);ax[0].axvline(.5,color='#b54d42',ls='--',label='0.5 threshold');ax[0].set_yticks(ys,['Clean','Controlled poison']);ax[0].set_xlim(-.1,1.05);ax[0].set_ylim(-.45,1.45);ax[0].set_xlabel('Mean within-team Spearman');ax[0].set_title('Independent data: 20 seeds per scenario')
for x,y,s in zip(means,ys,S):ax[0].text(x,y+.18,f"{x:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]",ha='center',fontsize=9)
ax[0].grid(axis='x',alpha=.2);ax[0].legend(loc='lower right',fontsize=8)
xs=np.arange(2);baseline=[cost['normal_samples']]*2;extra=[cost['additional_loo_samples'],cost['additional_probe_samples']];ax[1].bar(xs,baseline,label='Normal solve',color='#aeb7c1');ax[1].bar(xs,extra,bottom=baseline,label='LOO / shared probes',color=['#8b6a94','#3c8d8c']);ax[1].set_xticks(xs,['Full LOO','New method']);ax[1].set_ylabel('Model input samples, all 40 teams');ax[1].set_title(f"Model queries: {100*cost['saving']:.1f}% fewer")
for i,total in enumerate([cost['normal_plus_loo_samples'],cost['normal_plus_proxy_samples']]):ax[1].text(i,total+1600,f'{total:,}',ha='center')
ax[1].set_ylim(0,80000);ax[1].legend(fontsize=8);ax[1].grid(axis='y',alpha=.2)
t=bench['median_seconds_summed_across_four_cases'];labels=['Full LOO\n(reused models)','Original\nproxy','Cached\nproxy'];times=[t['LOO_cached_models']*1000,t['original']*1000,t['cached']*1000];bars=ax[2].bar(np.arange(3),times,color=['#8b6a94','#b8c4c7','#3c8d8c']);ax[2].set_xticks(np.arange(3),labels);ax[2].set_ylabel('Sum of four case medians (ms)');ax[2].set_title('Paired end-to-end timing: 10 repeats')
for bar,value in zip(bars,times):ax[2].text(bar.get_x()+bar.get_width()/2,value+25,f'{value:.1f}',ha='center')
ax[2].set_ylim(0,max(times)*1.17);ax[2].grid(axis='y',alpha=.2)
fig.suptitle('Ranking threshold met by point estimates; local runtime remains higher',fontsize=14);fig.text(.5,.015,'Score uses current client-data summaries; pooled only evaluates locked reference solutions. CI does not establish population correlation >0.5.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.06,1,.92]);assert not (OUT/'rank_and_cost.png').exists();fig.savefig(OUT/'rank_and_cost.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,6))
for j,(label,folder) in enumerate(FAMILIES):
 rs=[r for r in allrows if r['folder']==folder and not r['offline_only'] and r['clean'] is not None and r['poison'] is not None];ax.scatter([r['clean'] for r in rs],[r['poison'] for r in rs],s=24,alpha=.6,label=label if zh else folder.split('_experiment')[0].replace('_',' '))
ax.axvline(.5,color='#b54d42',ls='--');ax.axhline(.5,color='#b54d42',ls='--');ax.scatter([.6],[.56],marker='*',s=180,facecolor='gold',edgecolor='black',zorder=9,label='Selected for independent test');ax.set_xlim(-.65,1.02);ax.set_ylim(-.65,1.02);ax.set_xlabel('Clean: development mean Spearman');ax.set_ylabel('Poison: development mean Spearman');ax.set_title('260 configurations, including duplicates: development only');ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=6.4);fig.tight_layout();fig.savefig(OUT/'development_search.png',dpi=160);plt.close(fig)
print(json.dumps(dict(configurations=sum(not r['offline_only'] for r in allrows),offline_diagnostics=sum(r['offline_only'] for r in allrows),client_rows_scanned=scanned),indent=2))
