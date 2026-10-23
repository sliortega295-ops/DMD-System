"""Optional two-file Apache source fetch for the native Wan2.1 decoder, no weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def fetch_source(output):
    from distillschedule.training.wan21_codec import REPOSITORY, REVISION, SOURCE_FILES
    root = Path(output).expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    for name, expected in SOURCE_FILES.items():
        target = root/name
        if not target.resolve().is_relative_to(root):
            raise ValueError('source target escapes output root')
        if target.exists():
            payload = target.read_bytes()
        else:
            with urllib.request.urlopen(f'https://raw.githubusercontent.com/Wan-Video/Wan2.1/{REVISION}/{name}', timeout=30) as response:
                payload = response.read(2_000_001)
        if len(payload) > 2_000_000 or hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('source SHA256 mismatch; existing files are never replaced: ' + name)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(payload)
    result = dict(repository=REPOSITORY, revision=REVISION, files=SOURCE_FILES,
                  license='Apache-2.0', model_weights='NOT_FETCHED')
    (root/'source-manifest.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('output')
    args=parser.parse_args(); print(json.dumps(fetch_source(args.output), indent=2))
