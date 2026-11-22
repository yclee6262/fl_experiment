from pathlib import Path
import json,hashlib,subprocess,sys,os
import argparse
parser=argparse.ArgumentParser(description='Run three historical batches with corrected solver policy.')
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[3]
out=args.output_dir.resolve()
out.mkdir(parents=True,exist_ok=False)
reg=json.loads((root/'alpha_weight_outputs/se_qu_extension_v1_20261107/registration.json').read_text())
m={'status':'running','kind':'historical_reanalysis_after_numeric_correction','policy':'float32_slsqp_v1','batches':[],'input_sha256':{},'code_sha256':{}}
for f in ['core/host_server.py','core/agent_client.py','experiments/alpha/compare_initializations.py','experiments/alpha/replay.py','experiments/shared/solver_policy_v1.json']:
 m['code_sha256'][f]=hashlib.sha256((root/f).read_bytes()).hexdigest()
for i,b in enumerate(reg['batches'][:3],1):
 source=root/f'alpha_weight_outputs/four_alpha_v1_20261014/formal_batch_{i}/source'
 for p in source.rglob('*'):
  if p.is_file() and (p.name in ['local_models.pt','global_evaluator.pt','shared_costs.jsonl']):m['input_sha256'][str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
 m['batches'].append({'batch':i,'seeds':b['seeds'],'source':str(source),'output':str(out/f'batch_{i}')})
(out/'registration.json').write_text(json.dumps(m,indent=2)+'\n')
for b in m['batches']:
 cmd=[sys.executable,'run.py','initialization','--source',b['source'],'--seeds',','.join(map(str,b['seeds'])),'--solver-policy','float32_slsqp_v1','--output-dir',b['output']]
 with (out/f"batch_{b['batch']}.log").open('w') as f:
  subprocess.run(cmd,cwd=root,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=f,stderr=subprocess.STDOUT,check=True)
 print('batch',b['batch'],'complete',flush=True)
m['status']='complete'
(out/'execution.json').write_text(json.dumps(m,indent=2)+'\n')
