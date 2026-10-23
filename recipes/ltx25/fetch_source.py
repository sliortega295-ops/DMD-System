"""Fetch only the hash-locked LTX-2.5 licensed source closure, NEVER weights."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from distillschedule.training.ltx_source import REPOSITORY, REVISION, SOURCE_FILES, SOURCE_MANIFEST_SHA256


def fetch_source(output):
    root = Path(output).expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    for name, expected in SOURCE_FILES.items():
        target = root / name
        if not target.resolve().is_relative_to(root):
            raise ValueError('source target escapes output root: ' + name)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError('refusing to replace changed source: ' + name)
            continue
        url = REPOSITORY.replace('github.com', 'raw.githubusercontent.com') + '/' + REVISION + '/' + name
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read(2_000_001)
        if len(payload) > 2_000_000 or hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('upstream source size/SHA256 mismatch: ' + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(payload)
    result = dict(repository=REPOSITORY, revision=REVISION, files=SOURCE_FILES,
                  purpose='external upstream CPU source qualification; no weights')
    content = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if hashlib.sha256(content.encode()).hexdigest() != SOURCE_MANIFEST_SHA256:
        raise ValueError('source manifest serialization differs from the reviewed lock')
    manifest = root / 'source-manifest.json'
    if manifest.exists():
        if manifest.read_text() != content:
            raise ValueError('refusing to replace a different source manifest')
    else:
        with manifest.open('x') as stream:
            stream.write(content)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('output')
    print(json.dumps(fetch_source(parser.parse_args().output), indent=2))
