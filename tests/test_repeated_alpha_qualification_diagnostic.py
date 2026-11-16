from diagnose_repeated_alpha_qualified_cases import shared_qualification_failure
from repeated_alpha_candidates import METHODS

def test_common_failure_requires_every_method_and_same_upstream_reason():
    rows=[{'method':m,'status':'qualification_failure','reason':[20]} for m in METHODS]
    assert shared_qualification_failure(rows)
    assert not shared_qualification_failure(rows[:-1])
    rows[-1]['reason']=[2]
    assert not shared_qualification_failure(rows)

def test_differential_solver_failure_must_not_be_excluded_as_common():
    rows=[{'method':m,'status':'qualification_failure','reason':[20]} for m in METHODS]
    rows[-1]['status']='solver_failure'
    assert not shared_qualification_failure(rows)
