from __future__ import annotations
import base64, hashlib, json, os, pathlib, shutil, subprocess, tarfile

ROOT = pathlib.Path.cwd()
PAYLOAD_B64 = ROOT / '.integration/direct_payload.b64'
PAYLOAD_GZ = ROOT / '.integration/direct_payload.tar.gz'
EXPECTED_PAYLOAD_SHA = '529bed8d9a716e28a3ef9403c64f383276093b2c2170b6bc2f031f07929aedd8'
MAP = ROOT / 'docs/workloads/source_import_map.json'
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
TEMP_PATHS = [
    ROOT / '.integration/direct_payload.b64',
    ROOT / '.integration/direct_payload.tar.gz',
    ROOT / '.integration/bootstrap_final.py',
    ROOT / '.github/workflows/bootstrap-final-integration.yml',
]

def run(*args: str, check: bool = True, capture: bool = False):
    return subprocess.run(args, cwd=ROOT, check=check,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None)

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

raw = base64.b64decode(PAYLOAD_B64.read_text().strip())
assert sha256(raw) == EXPECTED_PAYLOAD_SHA, (sha256(raw), EXPECTED_PAYLOAD_SHA)
PAYLOAD_GZ.write_bytes(raw)
with tarfile.open(PAYLOAD_GZ, 'r:gz') as tf:
    tf.extractall(ROOT, filter='data')

manifest = json.loads(MAP.read_text())
entries = manifest['files']
source_entries = [e for e in entries if e['source_repository'] in {
    'sliortega295-ops/DMD-System', 'sliortega295-ops/H3-DMD-for-A100'
} and e['target'] not in SKIP_SOURCE_OVERWRITE]
for repo in sorted({e['source_repository'] for e in source_entries}):
    commits = sorted({e['source_commit'] for e in source_entries if e['source_repository'] == repo})
    for commit in commits:
        if repo == 'sliortega295-ops/DMD-System':
            run('git', 'fetch', '--no-tags', 'origin', commit)
        else:
            run('git', 'fetch', '--no-tags', 'https://github.com/sliortega295-ops/H3-DMD-for-A100.git', commit)
for e in source_entries:
    spec = f"{e['source_commit']}:{e['source_path']}"
    data = run('git', 'show', spec, capture=True).stdout
    actual = sha256(data)
    if actual != e['integrated_sha256']:
        raise SystemExit(f"source hash mismatch {e['target']}: {actual} != {e['integrated_sha256']}")
    target = ROOT / e['target']
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)

for e in entries:
    target = ROOT / e['target']
    if not target.is_file():
        raise SystemExit(f"missing mapped target: {e['target']}")
    actual = sha256(target.read_bytes())
    if actual != e['integrated_sha256']:
        raise SystemExit(f"integrated hash mismatch {e['target']}: {actual} != {e['integrated_sha256']}")

for p in TEMP_PATHS:
    if p.exists():
        p.unlink()
try:
    (ROOT / '.integration').rmdir()
except OSError:
    pass

run('git', 'add', '-A')
for path, commit in GITLINKS.items():
    run('git', 'update-index', '--add', '--cacheinfo', f'160000,{commit},{path}')

trigger = ROOT / '.integration/finalize-trigger.txt'
trigger.parent.mkdir(parents=True, exist_ok=True)
trigger.write_text('Delete this file after bootstrap to trigger final PR CI on the completed integration tree.\n')
run('git', 'add', str(trigger.relative_to(ROOT)))

env = dict(os.environ)
env['PYTHONPATH'] = str(ROOT / 'src')
subprocess.run(['python', 'scripts/distillschedule/check_best_known_integration.py'], cwd=ROOT, env=env, check=True)
subprocess.run(['python', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_refactor_best_known.py', '-v'], cwd=ROOT, env=env, check=True)
subprocess.run(['python', '-m', 'compileall', '-q', 'src', 'scripts/distillschedule'], cwd=ROOT, env=env, check=True)

run('git', 'config', 'user.name', 'github-actions[bot]')
run('git', 'config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com')
run('git', 'commit', '-m', 'Integrate five canonical best-known DMD workloads')
branch = os.environ['GITHUB_REF_NAME']
run('git', 'push', 'origin', f'HEAD:{branch}')
print('BOOTSTRAP_FINAL_INTEGRATION_PASS')
