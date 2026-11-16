"""Generate the ignored flat script view required by historical research code."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    directory = ROOT / '.local_scripts'
    directory.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / 'docs/reorganization_20261116/layout.json').read_text())
    for row in manifest['files']:
        link = directory / row['original']
        target = Path(os.path.relpath(ROOT / row['canonical'], directory))
        if link.is_symlink():
            if os.readlink(link) == str(target):
                continue
            link.unlink()
        elif link.exists():
            raise FileExistsError(f'Refusing to replace local file: {link}')
        link.symlink_to(target)
    # Historical scripts refer to data/docs relative to their flat working directory.
    for target in ROOT.iterdir():
        if target.is_dir() and not target.name.startswith('.'):
            link = directory / target.name
            if not link.exists() and not link.is_symlink():
                link.symlink_to(Path('..') / target.name, target_is_directory=True)
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return directory
