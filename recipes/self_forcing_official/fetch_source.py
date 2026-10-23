"""Fetch only the licensed small pinned official source closure; never weights."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from distillschedule.training.self_forcing_source import REPOSITORY, REVISION, SOURCE_FILES


def fetch_source(output):
    root = Path(output).expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    for name, expected in SOURCE_FILES.items():
        target = root / name
        if not target.resolve().is_relative_to(root):
            raise ValueError('source target escapes output root: ' + name)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError('refusing to replace a different source file: ' + name)
            continue
        url = 'https://raw.githubusercontent.com/guandeh17/Self-Forcing/' + REVISION + '/' + name
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read(2_000_001)
        if len(payload) > 2_000_000 or hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('upstream source size/SHA256 mismatch: ' + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(payload)
    result = dict(repository=REPOSITORY, revision=REVISION, files=SOURCE_FILES, license='Apache-2.0', weights='NOT_FETCHED')
    (root / 'source-manifest.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('output')
    args = parser.parse_args(); print(json.dumps(fetch_source(args.output), indent=2))
