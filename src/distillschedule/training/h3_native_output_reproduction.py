"""Explicit R375 C50 reproduction through its ORIGINAL two-node owner.

Metadata only until exec; never mix the historical runtime into this package.
Not a generic sampler, launcher, owner, or ownership authorization.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

SCHEMA = 'distillschedule.h3_native_output_reproduction.v1'
SCOPE = 'R375_C50_AP40_NEW232_REPRODUCTION_NOT_GENERAL_INPUT'
HEAD = '228cfdc093dd5e6c5fa4d130c7ef88f4f8a5656b'
KIND = 'INFERENCE_FROM_RESTORED_NATIVE_CHECKPOINT_WITH_OUTPUT_SCOPE_V1'
PACKET_SCHEMA = 'r375.w6.native_generation_original_owner.v1'
C50_SHA = '184326ab71dd82ee3f888b79820d666bea82f033d211bd2647d053ee87734112'
INPUT_SHA = 'e2d51e99b752cb03fcf7a5c1842a555b3306a25ac99c1cda19d9fc438e6304ce'
ORIGINAL_LAUNCH_SHA = 'c68fd1802851fc22ae13469ddbf33172e51ca87a9218a1678c1055ff9ca3ca66'
COMPOSITION = 'r375.w6.reused_AP_history_new_text_composition_receipt.v2'
SAMPLE = 'CURRENT_C50_NEW232_TEXT_REUSED_AP40_HISTORY_FUNCTIONAL_ONLY'
HISTORY_SHA = '15888a75d0c8cf68a1e6d132fa7e91cdd1a70f5eb37205390d65db52b112a7d2'
TEXT_SHA = '17128562fdc53abaa7dd57dd6cb390152056c7875bc277532602c1d08e8737fa'
EXTENSIONS = dict(scope='4107606aacdb6a26ec2fac02609644286ef7bbc084e60ad11d404335ca476c38',
                  generation='a41a8081441fa2aef5881c0b1bcbf0dd2db7eb29b4b5202959e5e3f5812c095d')
PACKET_FILES = {
    'native_output_node_owner.py': 'e851ab3643ce5d74e72c14c59b937467bb35d6d22b878a728ff60e26cf64e8ed',
    'native_output_packet.py': 'a103dc22505b58c0197120aae790e1c2ec03c1c2010742dca0bcc5dd5f5b6773',
    'native_fd_binding.py': '85e406bf145031d2990b6818e9c9bafb1023390e4e7b0202d79d57bcfa26d4a3',
    'rank_native_output.py': '106cae4a84f1afe7b15cb0a8e1cd49649d6a1592f0c6d83a2dc53e9164e35c08',
}


def require(ok, message):
    if not ok:
        raise ValueError('native-output reproduction: ' + message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def checked(ref, *, executable=False):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}
            and isinstance(ref['path'], str) and isinstance(ref['sha256'], str)
            and re.fullmatch('[0-9a-f]{64}', ref['sha256']), 'exact absolute file pin required')
    path = Path(ref['path'])
    require(path.is_absolute() and path.is_file() and (executable or path.resolve() == path)
            and 0 < path.stat().st_size <= (64 << 20 if executable else 2 << 20),
            'missing/noncanonical/oversized metadata or executable')
    require(sha(path) == ref['sha256'], 'file SHA256 changed: ' + str(path))
    if executable:
        require(os.access(path, os.X_OK), 'bound Python is not executable')
    return path


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON field: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)


def owner_environment(cfg, node, caller):
    """Bound original owner environment, never the public package's imports."""
    bound, source = cfg['nodes'][node]['environment'], cfg['source_root']
    require(isinstance(bound, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in bound.items())
            and bound.get('OMP_NUM_THREADS') == '4' and bound.get('PYTHONPATH', '').split(':')[:2] == [source+'/src', source]
            and bound.get('CUDA_VISIBLE_DEVICES') == ','.join(cfg['nodes'][node]['gpu_uuids'])
            and not any(k in bound for k in ('PYTHONHOME', 'LD_PRELOAD', 'WORLD_SIZE', 'RANK', 'LOCAL_RANK')),
            'exact original owner Python/threads/device environment required')
    # Same safe inheritance as the pinned original h3_common.environment.
    env = {key: caller[key] for key in ('HOME', 'USER', 'LOGNAME', 'LANG') if key in caller}
    env.update(bound)
    return env


def _bindings(path):
    path = Path(path).expanduser().resolve()
    require(path.is_file() and path.stat().st_size <= 2 << 20, 'bounded public request required')
    request = read(path)
    require(isinstance(request, dict) and set(request) == {'schema_version', 'scope', 'launch', 'runtime',
            'owner', 'checkpoint_manifest', 'extensions', 'conditions', 'input_witness'}
            and request['schema_version'] == SCHEMA and request['scope'] == SCOPE,
            'explicit versioned R375 reproduction request only')
    cfg = read(checked(request['launch']))
    p, runtime, owner = cfg['w6_native_output'], request['runtime'], request['owner']
    require(set(runtime) == {'python', 'source_root', 'source_head'}
            and runtime == {k: cfg[k] for k in runtime} and runtime['source_head'] == HEAD,
            'real historical native runtime/source/head binding required')
    source = Path(runtime['source_root'])
    require(source.is_absolute() and source.resolve() == source and source.is_dir(), 'canonical native source root required')
    checked(runtime['python'], executable=True)
    require(p['schema_version'] == PACKET_SCHEMA and p['kind'] == KIND
            and p['status'] == 'ROOT_ADMITTED_NATIVE_OUTPUT_ONLY'
            and p['capture_c0'] is False and p['reexport'] is False
            and p['checkpoint_manifest_sha256'] == C50_SHA, 'newkind admitted output only; no C0/reexport/update')
    require(set(owner) == {'entry', 'original_launch', 'admission', 'nodes', 'run_root'}
            and owner['original_launch'] == p['original_launch'] and owner['admission'] == p['scope_v2_gate']
            and owner['run_root'] == cfg['run_root'], 'actual original owner admission/run binding required')
    require(owner['original_launch']['sha256'] == ORIGINAL_LAUNCH_SHA, 'frozen original owner launch changed')
    original = read(checked(owner['original_launch']))
    require(all(cfg[key] == original[key] for key in ('source_root', 'source_head', 'python',
            'owner_helpers', 'target_cpu_gates')), 'original runtime/owner/CPU-admission pins changed')
    entry = checked(owner['entry'])
    require(entry.name == 'native_output_node_owner.py' and owner['entry']['sha256'] == PACKET_FILES[entry.name],
            'only the original node owner may execute; direct rank entry refused')
    require(sha(entry.parent.parent/'configs/original-launch.json') == ORIGINAL_LAUNCH_SHA,
            'owner import-time original launch closure changed')
    for name, digest in PACKET_FILES.items():
        ref = dict(path=str(entry.with_name(name)), sha256=digest)
        require(ref in p['bound_packet_files'], 'missing reviewed owner/rank/FD packet source pin: ' + name)
        checked(ref)
    require(p['rank_entry'] == dict(path=str(entry.with_name('rank_native_output.py')),
            sha256=PACKET_FILES['rank_native_output.py']), 'foreign rank entry refused')
    require(isinstance(owner['nodes'], list) and len(owner['nodes']) == len(cfg['nodes']) == 2,
            'both original node roles/admission records required')
    for node, binding in enumerate(owner['nodes']):
        actual = cfg['nodes'][node]
        prior = original['nodes'][node]
        cache_keys = {'TMPDIR', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR'}
        require({k: v for k, v in actual.items() if k != 'environment'}
                == {k: v for k, v in prior.items() if k != 'environment'}
                and {k: v for k, v in actual['environment'].items() if k not in cache_keys}
                == {k: v for k, v in prior['environment'].items() if k not in cache_keys}
                and set(actual['environment']) == set(prior['environment'])
                and all(isinstance(actual['environment'][k], str)
                        and actual['environment'][k].startswith('/tmp/lyy-experiments/DMD-System/')
                        for k in cache_keys & set(actual['environment'])),
                'original node/environment changed beyond its existing disposable cache paths')
        expected = {k: actual[k] for k in ('node_rank', 'hostname', 'gpu_uuids')}
        expected['target_cpu_gate'] = cfg['target_cpu_gates'][str(node)]
        require(binding == expected and type(actual['node_rank']) is int and actual['node_rank'] == node
                and isinstance(actual['hostname'], str)
                and len(set(actual['gpu_uuids'])) == 8, 'two-node hostname/UUID/node-role binding changed')
        gate = read(checked(binding['target_cpu_gate']))
        require(gate['source_head'] == HEAD and gate['returncode'] == 0 and gate['CUDA_initialized'] is False
                and gate['hostname'] == actual['hostname'] and gate['native_continuation_tests_executed'] is True,
                'actual original owner target CPU admission missing')
        owner_environment(cfg, node, {})
    require(cfg['nodes'][0]['hostname'] != cfg['nodes'][1]['hostname'], 'distinct owning hosts required')
    require(request['extensions'] == p['scope_v2_pins'] and set(request['extensions']) == set(EXTENSIONS),
            'separate reviewed extension source pair required')
    for role, ref in request['extensions'].items():
        require(ref['sha256'] == EXTENSIONS[role], 'frozen v2 extension source changed')
        checked(ref)
    gate = read(checked(owner['admission']))
    require(gate['status'] == 'PASS_SCOPE_V2_REVIEW_AND_ACTUAL_INPUT_CPU'
            and gate['source_pins'] == request['extensions'] and gate['prepared_sha256'] == INPUT_SHA
            and gate['CUDA_initialized'] is False and gate['model_constructed'] is False and gate['GPU_started'] == 0,
            'actual reviewed scope/input CPU admission required')
    require(request['checkpoint_manifest'] == dict(path=p['checkpoint']+'/manifest.json', sha256=C50_SHA)
            and request['conditions'] == p['conditions'], 'old C50 and actual input binding required')
    conditions = read(checked(request['conditions']))
    require(conditions['schema_version'] == 'distillschedule.h3_native_parsed_conditions.v1'
            and conditions['prepared']['sha256'] == INPUT_SHA and conditions['conversion_witness'] == request['input_witness'],
            'actual v2 input witness missing')
    witness_path = checked(request['input_witness'])
    witness = read(witness_path)
    require((witness_path.parent/'_SUCCESS').read_text().strip() == request['input_witness']['sha256']
            and witness['schema_version'] == COMPOSITION and witness['sample_id'] == SAMPLE
            and witness['status'] == 'COMPLETE_FUNCTIONAL_INPUT_NOT_GENERATED_MEDIA'
            and witness['scope'] == 'FUNCTIONAL_CURRENT_C50_OUTPUT_NOT_HELDOUT_QUALITY'
            and witness['prepared'] == conditions['prepared'] and witness['retained_full_raw'] == conditions['full_raw_sidecar']
            and witness['original_history']['sha256'] == HISTORY_SHA and witness['new_text_features']['sha256'] == TEXT_SHA
            and witness['transport']['parsed_BF16_content_sha256'] == conditions['native_parsed_sha256'],
            'frozen AP40/new232 composition/witness changed; not generic assets')
    native = read(checked(p['native_request']))
    require(native['schema_version'] == 'distillschedule.h3_native_generation_request.v1' and native['kind'] == KIND
            and native['cycle'] == 50 and native['historical_training_source_head'] == HEAD
            and native['checkpoint_manifest'] == request['checkpoint_manifest'] and native['extension_source'] == request['extensions']
            and native['output'] == p['output'], 'actual native request differs from owner launch')
    return path, request, cfg


def _owner_argv(request, node):
    return [request['runtime']['python']['path'], '-B', request['owner']['entry']['path'],
            '--launch', request['launch']['path'], '--launch-sha256', request['launch']['sha256'],
            '--node', str(node), '--segment', 'native_output50', '--execute']


def inspect_request(path):
    path, request, _ = _bindings(path)
    return dict(schema_version=SCHEMA, status='INSPECTED_REPRODUCTION_NOT_OWNER_ADMISSION', scope=SCOPE,
                public_request=dict(path=str(path), sha256=sha(path)),
                public_front_door=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
                launch=request['launch'], runtime=request['runtime'],
                owner=request['owner'], owner_commands=[dict(node=n, argv=_owner_argv(request, n)) for n in (0, 1)],
                native_input_byte_shape_RNG_admission='DEFERRED_UNCHANGED_ORIGINAL_RUNNER_AND_V2_API',
                resource_FD_lock_peer_admission='DEFERRED_UNCHANGED_ORIGINAL_RUN_OWNED',
                GPU_started=0, model_constructed=False, actual_generation='NOT_RUN')


def _source_check(root):
    try:
        head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError('historical runtime must be a real clean source checkout') from exc
    require(head == HEAD and not dirty, 'wrong actual historical source/head or dirty runtime; never retag')


def dispatch(path, *, execute=False, node=None):
    plan = inspect_request(path)
    if not execute:
        return plan
    require(type(node) is int and node in (0, 1), 'execute requires an explicit node 0 or 1')
    require(all(os.environ.get(k, v) == v for k, v in (('WORLD_SIZE', '1'), ('LOCAL_WORLD_SIZE', '1'),
            ('RANK', '0'), ('LOCAL_RANK', '0'))) and 'TORCHELASTIC_RUN_ID' not in os.environ,
            'node owner entry only; arbitrary torchrun rank cannot bypass original ownership')
    _, request, cfg = _bindings(path)
    require(socket.gethostname() == cfg['nodes'][node]['hostname'], 'wrong actual owning hostname/node role')
    _source_check(request['runtime']['source_root'])
    require(inspect_request(path) == plan, 'public request/bindings changed before owner exec')
    argv = _owner_argv(request, node)
    print('NATIVE_OUTPUT_OWNER_HANDOFF=' + json.dumps(dict(status='HANDOFF_NOT_OWNER_SUCCESS',
          public_request=plan['public_request'], public_front_door=plan['public_front_door'],
          launch=request['launch'], node=node), sort_keys=True), file=sys.stderr, flush=True)
    previous = os.getcwd()
    try:
        os.chdir(request['runtime']['source_root'])
        os.execve(argv[0], argv, owner_environment(cfg, node, os.environ))
        raise RuntimeError('original owner exec unexpectedly returned')
    finally:
        os.chdir(previous)
