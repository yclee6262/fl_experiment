"""Correct an observation-only nested-break labeling bug, without editing raw data."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from worker import classify_custom_stop

root=Path(sys.argv[1]).resolve()
source=(root/'source_snapshot/host_server.py').read_text().splitlines()
outer=[i+1 for i,line in enumerate(source) if 420<i<540 and line.strip()=='break' and len(line)-len(line.lstrip())<=20]
assert len(outer)==2,outer
mapping=dict(zip(outer,['direction_gradient_below_threshold','no_improving_step_after_search']))
rows=[]
for phase in ['pilot','confirmation']:
    for p in sorted((root/phase).glob('*/*/solves/*.json')):
        r=json.loads(p.read_text())
        if r.get('status')!='valid_delivery':continue
        metadata=r['engines']['custom'];trace=metadata['observed_trace']
        corrected=classify_custom_stop(trace,mapping)
        assert corrected!='unclassified_stop',p
        rows.append(dict(phase=phase,scenario=r['scenario'],seed=r['seed'],file=p.name,
          original_label=metadata['stop_reason'],audited_label=corrected,
          iterations=metadata['iterations_attempted'],limit=r['iterations'],last_break_line=trace.get('break_line'),
          changed=corrected!=metadata['stop_reason']))
out=root/'stop_label_audit';out.mkdir(exist_ok=False)
with (out/'records.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=dict(kind='posthoc_metadata_correction_only',n_rows=len(rows),changed=sum(r['changed'] for r in rows),
    confirmation_changed=sum(r['changed'] for r in rows if r['phase']=='confirmation'),
    numerical_results_edited=False,raw_labels_preserved=True,
    reason='Frozen observer mistook last inner line-search break for outer termination; raw frame line/iteration permits exact reconstruction.',
    corrected_confirmation_counts={s+'_'+label:dict(Counter(r['audited_label'] for r in rows
        if r['phase']=='confirmation' and r['scenario']==s and r['file'].startswith(label+'_')))
        for s in ['clean','controlled_poison'] for label in ['A','B']},
    audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    corrected_worker_sha256=hashlib.sha256((Path(__file__).parent/'worker.py').read_bytes()).hexdigest())
with (out/'summary.json').open('x') as f:json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
