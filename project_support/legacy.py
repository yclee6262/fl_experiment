"""Provide the small flat import view needed by retained historical modules."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    """Build only versioned aliases; archived experiments are never registered."""
    directory = ROOT / '.local_scripts'
    directory.mkdir(exist_ok=True)
    modules = json.loads((Path(__file__).with_name('legacy_modules.json')).read_text())
    for stale in directory.glob('*.py'):
        if stale.is_symlink() and stale.name not in modules:
            stale.unlink()
    for name, canonical in modules.items():
        source = ROOT / canonical
        if not source.is_file():
            raise FileNotFoundError(f'Missing retained source: {canonical}')
        link = directory / name
        target = Path(os.path.relpath(source, directory))
        if link.is_symlink():
            if os.readlink(link) == str(target):
                continue
            link.unlink()
        elif link.exists():
            raise FileExistsError(f'Refusing to replace local file: {link}')
        link.symlink_to(target)
    for target in ROOT.iterdir():
        if target.is_dir() and not target.name.startswith('.'):
            link = directory / target.name
            if not link.exists() and not link.is_symlink():
                link.symlink_to(Path('..') / target.name, target_is_directory=True)
    for path in (ROOT, directory):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return directory
