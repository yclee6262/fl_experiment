import hashlib,json
from pathlib import Path
import numpy as np
WORK=Path(__file__).resolve().parents[2];CA=WORK/'contribution_analysis'
SRC=CA/'j_loo_start_stability_experiment_v1_20261027';DIAG=CA/'j_self_other_diagnosis_v2_20261027'
ROOT=CA/'j_peer_experiment_v1_20261027';PLAN=WORK/'docs/j_peer_experiment_plan_20261027.md';EPS=1e-4
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,allow_nan=False,indent=2)
def key(i,x):return str(i)+':'+np.asarray(x,np.float64).tobytes().hex()
def near(a,b):
    np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10);return float(np.max(abs(np.asarray(a)-b)))
