"""Check the current layout without running research or changing evidence."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from project_support.legacy import prepare
flat = prepare()
layout = json.loads((root / 'docs/reorganization_20261116/layout.json').read_text())
errors = []
for row in layout['files']:
    link, source = flat / row['original'], root / row['canonical']
    if not link.is_symlink() or not source.is_file() or link.resolve() != source.resolve():
        errors.append('source link: ' + row['original'])
        continue
    if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256_current']:
        errors.append('source hash: ' + row['canonical'])
    ast.parse(source.read_text())
for name in layout['active_contribution_packages']:
    link = root / 'experiments/contribution/packages' / name
    if not link.is_symlink() or not link.is_dir():
        errors.append('package link: ' + name)
for path, digest in layout['new_source_hashes'].items():
    source = root / path
    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        errors.append('new source hash: ' + path)
    else:
        ast.parse(source.read_text())
for doc in [root / 'README.md', root / 'experiments/README.md',
            *list((root / 'docs/reorganization_20261116').glob('*.md'))]:
    for target in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
        if not target.startswith(('http', '/', '#')) and not (doc.parent / target.split('#')[0]).exists():
            errors.append('doc link: ' + str(doc) + ': ' + target)
report = {'passed': not errors, 'errors': errors,
          'entries': len(layout['files']),
          'archived_scripts': sum(r['status'] == 'archive' for r in layout['files']),
          'active_contribution_packages': len(layout['active_contribution_packages'])}
print(json.dumps(report, indent=2))
raise SystemExit(bool(errors))
