"""End-to-end failure accounting and resume corruption checks."""
import json
import sys
import pytest
import torch
from experiments.baselines.paired_benchmark.run import execute_case,PROTOCOL,sha,main


def test_unqualified_case_keeps_rows_and_resume_rejects_corruption(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    protocol=json.loads(PROTOCOL.read_text())
    directory=tmp_path/'seed39999_d2_rho0'
    rr=execute_case(39999,2,0.,directory,protocol,smoke=True)
    assert len(rr)==20
    assert sum(r['status']=='no_coalition' for r in rr)==19
    selected=next(r for r in rr if r['method']=='fedavg_selected')
    assert selected['training_communication_rounds']==0 and selected['training_communication_bytes']==0
    assert selected['qualification_host_requests']==20
    assert selected['qualification_host_samples']==400
    assert selected['host_prediction_requests']==20
    assert selected['host_prediction_input_samples']==400
    assert (directory/'qualification.log').exists()
    manifest=dict(protocol=protocol,protocol_sha256=sha(PROTOCOL),smoke=True,validation_case=False,code_sha256={},status='complete')
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(sys,'argv',['run','--output-dir',str(tmp_path),'--resume','--smoke'])
    (directory/'results.jsonl').write_text('corrupted\n')
    with pytest.raises(AssertionError): main()
