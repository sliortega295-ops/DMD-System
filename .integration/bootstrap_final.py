from __future__ import annotations
import base64, hashlib, json, os, pathlib, subprocess, tarfile

ROOT = pathlib.Path.cwd()
CHUNKS = [('.integration/chunks/chunk00.b64', '686c930090a7c6e105e30bafef4a6b70563a7b4590632972f55cd7f23ad3d067'), ('.integration/chunks/chunk01.b64', '91791e60aa36f94aef4c5415a426c4edd55ea63da470d00e588f9fb7d5080e25'), ('.integration/chunks/chunk02.b64', '7dde35ed164e5cd531879df4cb841a818ac40029d25bdb53f3b6533d49b1d161'), ('.integration/chunks/chunk03.b64', 'ec7d862d78aa7eb6afa458cba828ce37f32955cd475ecb7d41bd0557302a0bb0'), ('.integration/chunks/chunk04.b64', 'cace323cfc47552c8ad0a842afac277b53e1db190085047dfa7876ad47ad0f87'), ('.integration/chunks/chunk05.b64', '0ff69005541b21975362fb917f9315bbac3e3bc3648b997edbc714d5fbb021f5'), ('.integration/chunks/chunk06.b64', '72e9dc812b00a264ce1389f1d78080563f1cffff5f796900afeb3f0b7a7a248e'), ('.integration/chunks/chunk07.b64', '53e35a48ab4f771060eb32f2d220435619f4b1c727ad0db4ebe6b4c0f3af594b'), ('.integration/chunks/chunk08.b64', 'dd61e176120470a1b451a8984762e2f59d0709ad9747cd3f9d8ea1a7bea908e4'), ('.integration/chunks/chunk09.b64', '07ac65190ff6638dd7bb946ebc16b925418ea4368548054ca8288d02c6906910'), ('.integration/chunks/chunk10.b64', 'a7032e4072f4010b52a14504b985aced2cb26c790e9adb516a6c100b7b807315'), ('.integration/chunks/chunk11.b64', 'b974ecf2e3059c35f21b63c462f890eb8893c424f18f7d930732ca9a497bed83'), ('.integration/chunks/chunk12.b64', 'b7f7b081806cff98e66cc0daeda6218fc847232d249ee2228e94e4930269814c'), ('.integration/chunks/chunk13.b64', 'cb34e167be957ebade2e915e1edafc6e7ece3604be97b7ae36cd0bfab7f7c1b0')]
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

def run(*args: str, check: bool = True, capture: bool = False):
    return subprocess.run(args, cwd=ROOT, check=check,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None)

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

parts=[]
for rel, expected in CHUNKS:
    raw=base64.b64decode((ROOT/rel).read_text().strip())
    actual=sha256(raw)
    if actual != expected:
        raise SystemExit(f'chunk hash mismatch {rel}: {actual} != {expected}')
    parts.append(raw)
payload=b''.join(parts)
actual_payload=sha256(payload)
if actual_payload != EXPECTED_PAYLOAD_SHA:
    raise SystemExit(f'payload hash mismatch: {actual_payload} != {EXPECTED_PAYLOAD_SHA}')
PAYLOAD_GZ.write_bytes(payload)
with tarfile.open(PAYLOAD_GZ, 'r:gz') as tf:
    tf.extractall(ROOT, filter='data')

manifest=json.loads(MAP.read_text())
# Exclude the unvalidated C27-only canary from the canonical C25 integration.
blocked_target='src/distillschedule/runtime/helios_frozen_cross_norm_kernel.py'
manifest['files']=[e for e in manifest['files'] if e.get('target') != blocked_target]
MAP.write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
blocked_path=ROOT/blocked_target
if blocked_path.exists(): blocked_path.unlink()
entries=manifest['files']
source_entries=[e for e in entries if e['source_repository'] in {
    'sliortega295-ops/DMD-System','sliortega295-ops/H3-DMD-for-A100'
} and e['target'] not in SKIP_SOURCE_OVERWRITE]
for repo in sorted({e['source_repository'] for e in source_entries}):
    commits=sorted({e['source_commit'] for e in source_entries if e['source_repository']==repo})
    for commit in commits:
        if repo=='sliortega295-ops/DMD-System':
            run('git','fetch','--no-tags','origin',commit)
        else:
            run('git','fetch','--no-tags','https://github.com/sliortega295-ops/H3-DMD-for-A100.git',commit)
for e in source_entries:
    data=run('git','show',f"{e['source_commit']}:{e['source_path']}",capture=True).stdout
    actual=sha256(data)
    if actual != e['integrated_sha256']:
        raise SystemExit(f"source hash mismatch {e['target']}: {actual} != {e['integrated_sha256']}")
    target=ROOT/e['target']; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
for e in entries:
    target=ROOT/e['target']
    if not target.is_file(): raise SystemExit(f"missing mapped target: {e['target']}")
    actual=sha256(target.read_bytes())
    if actual != e['integrated_sha256']:
        raise SystemExit(f"integrated hash mismatch {e['target']}: {actual} != {e['integrated_sha256']}")

# Remove all bootstrap-only files before committing the integration.
for rel,_ in CHUNKS:
    p=ROOT/rel
    if p.exists(): p.unlink()
for rel in ['.integration/direct_payload.b64','.integration/direct_payload.tar.gz','.integration/bootstrap_final.py','.github/workflows/bootstrap-final-integration.yml']:
    p=ROOT/rel
    if p.exists(): p.unlink()
for d in [ROOT/'.integration/chunks', ROOT/'.integration']:
    try: d.rmdir()
    except OSError: pass

run('git','add','-A')
for path,commit in GITLINKS.items():
    run('git','update-index','--add','--cacheinfo',f'160000,{commit},{path}')
trigger=ROOT/'.integration/finalize-trigger.txt'
trigger.parent.mkdir(parents=True,exist_ok=True)
trigger.write_text('Delete this file after bootstrap to trigger final PR CI on the completed integration tree.\n')
run('git','add',str(trigger.relative_to(ROOT)))

env=dict(os.environ); env['PYTHONPATH']=str(ROOT/'src')
subprocess.run(['python','scripts/distillschedule/check_best_known_integration.py'],cwd=ROOT,env=env,check=True)
subprocess.run(['python','-m','unittest','discover','-s','tests','-p','test_refactor_best_known.py','-v'],cwd=ROOT,env=env,check=True)
subprocess.run(['python','-m','compileall','-q','src','scripts/distillschedule'],cwd=ROOT,env=env,check=True)

run('git','config','user.name','github-actions[bot]')
run('git','config','user.email','41898282+github-actions[bot]@users.noreply.github.com')
run('git','commit','-m','Integrate five canonical best-known DMD workloads')
branch=os.environ['GITHUB_REF_NAME']
run('git','push','origin',f'HEAD:{branch}')
print('BOOTSTRAP_FINAL_INTEGRATION_PASS')
