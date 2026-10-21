"""Bounded W6 recovery. Must inherit the real node lease; never preempts."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time
import traceback

import run_w6_native_three_arm_r3 as base

base.REPO = base.ROOT / 'repos/DMD-System-h100-w6-lora-unit-inplace-20261015-r1'
base.HEAD = 'f5d66e96665bb92e3f7f112e24b064b2710903a2'
base.WORKER = base.REPO / 'scripts/distillschedule/probe_h3_diffsynth_h100_world8.py'
base.ACTIVATE = base.OWNER / 'scripts/h3_w6_lora_unit_environment_r1.sh'


def own_command(command, output, *, gpu):
    output.mkdir(exist_ok=False)
    before = base.host()
    owned = {}
    process = None
    result = {'command': command, 'GPU': 'PRIMITIVE_ONLY' if gpu else 'NOT_RUN_CPU_ONLY'}
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0' if gpu else '',
                   PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
        with (output / 'command.log').open('x') as log:
            process = subprocess.Popen(command, env=env, cwd=base.REPO, stdout=log,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       start_new_session=True)
            owned[process.pid] = base.proc_record(process.pid)
            assert owned[process.pid] is not None
            base.write(output / 'owned_handle.json', owned[process.pid])
            deadline = time.monotonic() + 300
            while process.poll() is None:
                base.discover_owned_children(owned)
                if time.monotonic() > deadline:
                    raise TimeoutError('bounded primitive/CPU comparison deadline')
                time.sleep(1)
            if process.returncode:
                raise RuntimeError(f'gate command failed ({process.returncode}); preserve log')
        result['status'] = 'PASS_COMMAND_REQUIRES_RECEIPT'
    except BaseException as exc:
        result.update(status='FAIL_PRESERVED', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        alive = base.cleanup_owned(owned) if owned else []
        after = base.host()
        result.update(owned_alive_after=alive,
                      cgroup_oom_delta={k: after['events'][k] - before['events'][k] for k in ('oom', 'oom_kill')})
        if alive or any(result['cgroup_oom_delta'].values()):
            result['status'] = 'FAIL_TEARDOWN_OR_HOST_OOM'
        base.write(output / 'health.json', result)
    assert result['status'].startswith('PASS_')


def validate_native_boundary(out, *, tiny, lora):
    for rank in range(8):
        row = json.loads((out / f'rank_receipts/rank_{rank:02d}.json').read_text())
        for cycle in row['cycles']:
            boundary = cycle['checkpoint_boundary']
            if lora:
                assert boundary['enabled'] and boundary['counters']['checkpoint_calls'] == (8 if tiny else 200)
                assert boundary['counters']['D2H_count'] == boundary['counters']['H2D_count']
                assert boundary['counters']['D2H_bytes'] == boundary['counters']['H2D_bytes'] > 0
            else:
                assert not boundary['enabled']
            op = cycle['lora_sum']
            assert op['enabled'] is lora
            if lora:
                assert op['counters']['physical_dual_lora_forward'] > 0
                assert op['counters']['elided_unit_multiply'] == op['counters']['inplace_base_add']
            assert not op['parameter_dtype_changed'] and not op['optimizer_or_loss_changed']


def compare_tiny(parent, tiny_out):
    reference = base.CAMPAIGN / 'h100_w6_native_B16_GPU_tiny_gate_20261014_r2/tiny_resident_correctness/rank_receipts'
    code = f'''
import sys,json,hashlib
from pathlib import Path
import torch
sys.path.insert(0,{str(base.REPO / 'src')!r})
from distillschedule.runtime.p0_parity import tensor_parity_metrics
torch.set_num_threads(1)
reference=Path({str(reference)!r});current=Path({str(tiny_out / 'rank_receipts')!r});checks=[]
def read(root,rank):
 receipt=json.loads((root/f'rank_{{rank:02d}}.json').read_text())
 path=root/f'rank_{{rank:02d}}_tiny_trainable_state.pt'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['tiny_state_sha256']
 assert receipt['status']=='PASS_TINY_NATIVE_INTEGRATION_ONLY'
 return torch.load(path,map_location='cpu',weights_only=True),receipt
for rank in range(8):
 a,ar=read(reference,rank);b,br=read(current,rank)
 assert ar['initial_LoRA_sha256']==br['initial_LoRA_sha256'] and ar['logical_slots']==br['logical_slots']
 assert ar['final_versions']==br['final_versions']
 assert a['parameters'].keys()==b['parameters'].keys()
 for name,value in a['parameters'].items():
  checks.append({{'rank':rank,'kind':'parameter','name':name,**tensor_parity_metrics(value,b['parameters'][name])}})
 for role,states in a['Adam'].items():
  assert len(states)==len(b['Adam'][role])
  for index,state in enumerate(states):
   other=b['Adam'][role][index];assert state.keys()==other.keys()
   for key,value in state.items():checks.append({{'rank':rank,'kind':'Adam','role':role,'state_index':index,'field':key,**tensor_parity_metrics(value,other[key])}})
assert not torch.cuda.is_initialized()
result={{'status':'PASS_TINY_NATIVE_PARAMETERS_ADAM' if all(c['status']=='PASS' for c in checks) else 'FAIL_TINY_PARITY','rows':checks,'thresholds':'unchanged DEFAULT_P0_PARITY','full_model_equivalence':'NOT_RUN','GPU':'NOT_RUN_CPU_POSTPROCESS'}}
Path({str(parent / 'tiny_state_parity.json')!r}).open('x').write(json.dumps(result,indent=2))
print(json.dumps({{'status':result['status'],'comparisons':len(checks),'failures':sum(c['status']!='PASS' for c in checks)}}))
raise SystemExit(0 if result['status'].startswith('PASS_') else 1)
'''
    own_command(['bash', str(base.ACTIVATE), '-c', code], parent / 'tiny_state_comparison', gpu=False)


def main():
    def interrupted(number, frame):
        raise InterruptedError(f'owned W6 runner interrupted by signal {number}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parent = base.CAMPAIGN / 'h100_w6_lora_unit_inplace_20261015_r1'
    parent.mkdir(exist_ok=False)
    report = {'run_id': parent.name, 'head': base.HEAD, 'status': 'FAIL_PRESERVED',
              'new_mechanism': 'exact unit-scale LoRA sum; no dtype, loss, RNG or count change',
              'Resident_capacity_fallback': 'same native checkpoint-boundary CPU + unit-scale in-place LoRA',
              'sigma': 'native continuous shifted logit-normal', 'Grid1000': 'NOT_APPLICABLE',
              'replications': 1, 'strict_real_state_equivalence': 'NOT_RUN', 'arms': []}
    try:
        base.source_gate()
        assert os.environ.get('DISTILLSCHEDULE_INHERITED_GPU_LOCKS') == '1'
        subprocess.run([str(base.PYTHON), str(base.REPO / 'scripts/distillschedule/verify_inherited_gpu_locks.py'),
                        '--project-root', str(base.ROOT), '--asset-fd', '7'], check=True, pass_fds=(7, 8, 9))
        assert not base.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])
        probe = parent / 'GPU_primitive.json'
        own_command(['bash', str(base.ACTIVATE), str(base.OWNER / 'scripts/w6_lora_unit_gpu_canary_r1.py'),
                     '--repo', str(base.REPO), '--output', str(probe)], parent / 'GPU_primitive', gpu=True)
        assert json.loads(probe.read_text())['status'] == 'PASS_GPU_NATIVE_LORA_CHECKPOINT_PARITY'
        inputs = json.loads(base.INPUT.read_text())
        assert base.sha(inputs['cache_manifest']) == inputs['cache_manifest_sha256']
        assert base.sha(inputs['config_path']) == inputs['config_sha256']
        base.WORKER_EXTRA_ARGS = ('--checkpoint-boundary-cpu', '--lora-unit-inplace')
        tiny, tiny_out = base.arm(parent, 'resident', 'correctness', inputs, True)
        assert tiny['status'] == 'PASS_TINY_8_RANK_INTEGRATION_ONLY'
        validate_native_boundary(tiny_out, tiny=True, lora=True)
        compare_tiny(parent, tiny_out)
        # Only after the novel Resident arm fits do we spend on fresh same-head
        # unchanged FSDP/cache controls. Never rerun an unchanged failed arm.
        for placement in ('resident', 'fsdp', 'candidate'):
            lora = placement == 'resident'
            base.WORKER_EXTRA_ARGS = ('--checkpoint-boundary-cpu', '--lora-unit-inplace') if lora else ()
            for mode in ('correctness', 'timing'):
                summary, out = base.arm(parent, placement, mode, inputs, False)
                report['arms'].append({'placement': placement, 'mode': mode, 'path': str(out), **summary})
                assert summary['status'] == 'PASS_REAL_8_RANK_16_LOGICAL_SLOT_NATIVE_GATE'
                validate_native_boundary(out, tiny=False, lora=lora)
        report['status'] = 'PASS_W6_THREE_ARM_SAME_HEAD_SINGLE_DIAGNOSTIC'
        report['timings'] = {x['placement']: x['timed_C2_seconds'] for x in report['arms'] if x['mode'] == 'timing'}
        report['claim_boundary'] = 'Resident uses a named CPU-boundary/operator capacity fallback, not a pure placement ablation. Full-model numerical comparison remains NOT_RUN.'
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        base.write(parent / 'result.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'arms'}), flush=True)
    return 0 if report['status'].startswith('PASS_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
