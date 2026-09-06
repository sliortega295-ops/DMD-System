from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib
import subprocess
import tarfile

ROOT = pathlib.Path.cwd()
EXPECTED_PAYLOAD_SHA = '529bed8d9a716e28a3ef9403c64f383276093b2c2170b6bc2f031f07929aedd8'
MAP = ROOT / 'docs/workloads/source_import_map.json'
BLOCKED_TARGETS = {
    'src/distillschedule/runtime/helios_frozen_cross_norm_kernel.py',
    'src/distillschedule/runtime/helios_paired_rope_kernel.py',
}
SKIP_SOURCE_OVERWRITE = {
    'scripts/distillschedule/launch_wan13b_fair_world16_node.sh',
    'scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh',
    'scripts/distillschedule/launch_livewan_crossscale_world16_node.sh',
}
GITLINKS = {
    'third_party/lightx2v': 'd034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be',
    'third_party/livewan': '1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5',
    'third_party/wan21': '9737cba9c1c3c4d04b33fcad41c111989865d315',
}


def run(*args: str, capture: bool = False):
    return subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# Chunks were individually Git-object verified when uploaded. Reconstruct and
# verify the complete payload again before extracting anything.
parts = []
for chunk in sorted((ROOT / '.integration/chunks').glob('chunk*.b64')):
    parts.append(base64.b64decode(chunk.read_text().strip()))
payload = b''.join(parts)
actual_payload = sha256(payload)
if actual_payload != EXPECTED_PAYLOAD_SHA:
    raise SystemExit(f'payload hash mismatch: {actual_payload} != {EXPECTED_PAYLOAD_SHA}')
payload_gz = ROOT / '.integration/direct_payload.tar.gz'
payload_gz.write_bytes(payload)
with tarfile.open(payload_gz, 'r:gz') as tf:
    tf.extractall(ROOT, filter='data')

# The historical payload predates two later GAN canaries. They are explicitly
# excluded rather than reassigned to a newer source because canonical GAN is C25.
manifest = json.loads(MAP.read_text())
manifest['files'] = [
    entry for entry in manifest['files'] if entry.get('target') not in BLOCKED_TARGETS
]
MAP.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
for target_name in BLOCKED_TARGETS:
    target = ROOT / target_name
    if target.exists():
        target.unlink()

entries = manifest['files']
source_entries = [
    entry
    for entry in entries
    if entry['source_repository'] in {
        'sliortega295-ops/DMD-System',
        'sliortega295-ops/H3-DMD-for-A100',
    }
    and entry['target'] not in SKIP_SOURCE_OVERWRITE
]
for repo in sorted({entry['source_repository'] for entry in source_entries}):
    commits = sorted(
        {entry['source_commit'] for entry in source_entries if entry['source_repository'] == repo}
    )
    for commit in commits:
        if repo == 'sliortega295-ops/DMD-System':
            run('git', 'fetch', '--no-tags', 'origin', commit)
        else:
            run(
                'git',
                'fetch',
                '--no-tags',
                'https://github.com/sliortega295-ops/H3-DMD-for-A100.git',
                commit,
            )

for entry in source_entries:
    data = run(
        'git', 'show', f"{entry['source_commit']}:{entry['source_path']}", capture=True
    ).stdout
    actual = sha256(data)
    expected = entry['integrated_sha256']
    if actual != expected:
        raise SystemExit(f"source hash mismatch {entry['target']}: {actual} != {expected}")
    target = ROOT / entry['target']
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)

for entry in entries:
    target = ROOT / entry['target']
    if not target.is_file():
        raise SystemExit(f"missing mapped target: {entry['target']}")
    actual = sha256(target.read_bytes())
    expected = entry['integrated_sha256']
    if actual != expected:
        raise SystemExit(f"integrated hash mismatch {entry['target']}: {actual} != {expected}")

# Remove all bootstrap-only material before creating the candidate commit.
for chunk in (ROOT / '.integration/chunks').glob('chunk*.b64'):
    chunk.unlink()
for rel in [
    '.integration/direct_payload.b64',
    '.integration/direct_payload.tar.gz',
    '.integration/bootstrap_final.py',
    '.github/workflows/bootstrap-final-integration.yml',
]:
    path = ROOT / rel
    if path.exists():
        path.unlink()
for directory in [ROOT / '.integration/chunks', ROOT / '.integration']:
    try:
        directory.rmdir()
    except OSError:
        pass

run('git', 'add', '-A')
for path, commit in GITLINKS.items():
    run('git', 'update-index', '--add', '--cacheinfo', f'160000,{commit},{path}')
trigger = ROOT / '.integration/finalize-trigger.txt'
trigger.parent.mkdir(parents=True, exist_ok=True)
trigger.write_text(
    'Delete this file after bootstrap to trigger final PR CI on the completed integration tree.\n'
)
run('git', 'add', str(trigger.relative_to(ROOT)))

# source_layout validates gitlinks from HEAD, not the staging index. Create the
# candidate commit locally first; if any validation below fails, it is never pushed.
run('git', 'config', 'user.name', 'github-actions[bot]')
run(
    'git', 'config', 'user.email',
    '41898282+github-actions[bot]@users.noreply.github.com',
)
run('git', 'commit', '-m', 'Integrate five canonical best-known DMD workloads')

env = dict(os.environ)
env['PYTHONPATH'] = str(ROOT / 'src')
subprocess.run(
    ['python', 'scripts/distillschedule/check_best_known_integration.py'],
    cwd=ROOT, env=env, check=True,
)
subprocess.run(
    ['python', '-m', 'unittest', 'discover', '-s', 'tests',
     '-p', 'test_refactor_best_known.py', '-v'],
    cwd=ROOT, env=env, check=True,
)
subprocess.run(
    ['python', '-m', 'compileall', '-q', 'src', 'scripts/distillschedule'],
    cwd=ROOT, env=env, check=True,
)

run('git', 'push', 'origin', f"HEAD:{os.environ['GITHUB_REF_NAME']}")
print('BOOTSTRAP_FINAL_INTEGRATION_PASS')
