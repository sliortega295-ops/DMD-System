"""Explicitly fetch pinned external CC-BY-NC-SA source, NEVER model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def fetch_source(root, *, oracle=False):
    from distillschedule.training.sd import SOURCE_FILES, REPOSITORY, REVISION
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    files = dict(SOURCE_FILES)
    if oracle:
        files['oracle/train_sd.py'] = '55a8fca2e4fbf2f36c0c3a205f6fa18e92bfa7be06c8249561ce7780a4c68f7f'
    for name, expected in files.items():
        remote = 'main/train_sd.py' if name == 'oracle/train_sd.py' else name
        path = root / name
        payload = path.read_bytes() if path.exists() else urllib.request.urlopen(
            f'https://raw.githubusercontent.com/tianweiy/DMD2/{REVISION}/{remote}', timeout=60).read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('source SHA256 mismatch; existing files are never replaced: ' + name)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(payload)
    manifest = dict(repository=REPOSITORY, revision=REVISION, license='CC-BY-NC-SA-4.0', files=files)
    (root / 'source-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    # Validate bytes without importing optional model dependencies.
    from distillschedule.training.upstream import PinnedSource
    PinnedSource(root, files).validate()
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_root'); parser.add_argument('--with-test-oracle', action='store_true')
    args = parser.parse_args()
    print(json.dumps(fetch_source(args.source_root, oracle=args.with_test_oracle), indent=2))
