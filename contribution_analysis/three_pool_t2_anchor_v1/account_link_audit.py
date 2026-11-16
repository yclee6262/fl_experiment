"""Delivery-stage A/account linkage check, independently rerunnable after audit."""
import argparse
from pathlib import Path
from contribution_analysis.custom_realized_gain_v1.schema import read,save
from .reference import floats,oracle


def run(root):
    checks=[]
    for c in read(root/'fixtures.json'):
        doc=read(root/'cases'/(c['id']+'.json'));ref=floats(oracle(c));r=doc['result'];i=c['target_owner']
        if r['A']!=doc['accounts']['steps'][0]['A_increment_owner'][i]:raise ValueError('account_owner_link')
        expected=0.
        if ref['full']['q'] is not None:
            ps=[sum(a*b for a,b in zip(h,ref['full']['q'])) for h in ref['H']]
            before=doc['run']['evaluations'][doc['run']['initial_eval_id']]['objective_value'];G=before-ref['full']['loss']
            expected=G*max(ps[i],0)/sum(max(p,0) for p in ps)
        error=abs(r['A']-expected)
        if error>1e-9:raise ValueError('independent_A_allocation')
        checks.append(dict(id=c['id'],account_link_passed=True,independent_A_error=error))
    result=dict(passed=True,cases=checks,note='post-run linkage of target A to independently verified baseline and Decimal projections; no new model calls')
    path=root/'additional_account_link_checks.json'
    if path.exists():
        if read(path)!=result:raise ValueError('existing_account_link_check_mismatch')
    else:save(path,result)
    print('ACCOUNT LINKS PASSED',len(checks))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);run(ap.parse_args().root.resolve())
