"""Check the current layout without running research or changing evidence."""
import ast
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parents[2]
layout = json.loads((root / 'docs/reorganization_20261116/layout.json').read_text())
errors = []
for row in layout['files']:
    link, source = root / row['original'], root / row['canonical']
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
# The proposal inventory was captured before the first move; verify unchanged content.
original = json.loads((root / 'docs/handoff_20261116/migration_manifest.json').read_text())
proposal_present = (root / 'paper/proposal').is_dir()
for name, digest in (original['proposal_files'].items() if proposal_present else []):
    source = root / 'paper/proposal' / name
    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        errors.append('proposal hash: ' + name)
for doc in [root / 'README.md', root / 'experiments/README.md',
            *list((root / 'docs/reorganization_20261116').glob('*.md'))]:
    for target in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
        if not target.startswith(('http', '/', '#')) and not (doc.parent / target.split('#')[0]).exists():
            errors.append('doc link: ' + str(doc) + ': ' + target)
report = {'passed': not errors, 'errors': errors,
          'entries': len(layout['files']),
          'archived_scripts': sum(r['status'] == 'archive' for r in layout['files']),
          'active_contribution_packages': len(layout['active_contribution_packages']),
          'proposal_files_unchanged': len(original['proposal_files']) if proposal_present else None,
          'proposal_bundle_present': proposal_present}
print(json.dumps(report, indent=2))
raise SystemExit(bool(errors))
