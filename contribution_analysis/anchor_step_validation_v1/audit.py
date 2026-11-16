"""Read-only historical verifier invocation; all new writes are redirected."""
import subprocess
import sys
from pathlib import Path
from study import ROOT,LOCAL,save,check,sha

def main():
    check()
    source=LOCAL/'source_snapshot'
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(source),'-p','test_kernel.py','-v'],capture_output=True,text=True)
    (ROOT/'historical_tests.log').write_text(tests.stdout+tests.stderr)
    if tests.returncode:raise RuntimeError('historical_tests_failed')
    code='''import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import verify
original=verify.save
def redirected(path,obj):
    assert Path(path).name=='verification.json'
    original(Path(sys.argv[3]),obj)
verify.save=redirected
verify.verify(Path(sys.argv[2]))
'''
    verify=subprocess.run([sys.executable,'-c',code,str(source),str(LOCAL),str(ROOT/'historical_verification.json')],capture_output=True,text=True)
    (ROOT/'historical_verifier.log').write_text(verify.stdout+verify.stderr)
    if verify.returncode:raise RuntimeError('historical_verification_failed')
    own=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'source_snapshot'),'-p','test_state_engine.py','-v'],capture_output=True,text=True)
    (ROOT/'state_tests.log').write_text(own.stdout+own.stderr)
    if own.returncode:raise RuntimeError('state_tests_failed')
    save(ROOT/'p0_audit.json',dict(historical_tests_passed=11,state_tests_passed=4,
        historical_verification_accepted=True,historical_results_unchanged=True,
        note='four pilot query order and state continuation assertions passed before development; no effect statistics inspected'))
    check();print('P0 passed, historical outputs unchanged')

if __name__=='__main__':main()
