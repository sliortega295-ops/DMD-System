"""Explicit torchrun worker. Never SSHs, starts a launcher, or changes a lease."""
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time
from datetime import timedelta

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import file_sha256, write_new_json
from distillschedule.search.resources import _capacity_observation
from .h3_ar_assets import checked
from .h3_ar_task import H3ARImplementation
from .h3_ar_planning import source_identity, validate_plan, validate_policy, policy_for_node


def topology(environment):
    names = ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE')
    require(all(str(environment.get(k, '')).isdigit() for k in names),
            'explicit torchrun topology required', path='h3_ar.worker')
    rank, local, world, local_world = (int(environment[k]) for k in names)
    require(world == 16 and local_world == 8 and 0 <= rank < 16 and local == rank % 8,
            'W6 is one native world16, two nodes of eight', path='h3_ar.worker')
    return rank, local


def runtime_identity(torch):
    require(torch.__version__ == '2.10.0+cu128' and torch.version.cuda == '12.8',
            'W6 first bridge requires the validated real Torch 2.10.0+cu128 runtime', path='h3_ar.worker')
    return dict(python=platform.python_version(), executable_sha256=file_sha256(Path(sys.executable).resolve()),
        torch=torch.__version__, cuda=torch.version.cuda,
        packages={name: importlib.metadata.version(name) for name in
                  ('accelerate', 'deepspeed', 'triton', 'transformers', 'modelscope', 'peft', 'einops', 'safetensors')},
        attention=os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION'),
        environment={key: os.environ.get(key) for key in ('NCCL_ALGO', 'NCCL_PROTO', 'NCCL_IB_DISABLE',
            'NCCL_SOCKET_IFNAME', 'NCCL_P2P_DISABLE', 'OMP_NUM_THREADS', 'USE_TF', 'USE_FLAX', 'PYTHONNOUSERSITE')},
        deterministic=torch.are_deterministic_algorithms_enabled(), matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32, matmul_precision=torch.get_float32_matmul_precision())


def capacity_ok(memory, policy, *, node_rank=None):
    validate_policy(policy)
    deltas = memory.get('oom_event_delta', {})
    return (memory['peak_allocated_bytes'] <= policy['gpu_limit_bytes']
            and memory['driver_free_bytes'] >= policy['gpu_reserve_bytes']
            and host_capacity_ok(memory, policy, node_rank=node_rank)
            and set(deltas) == {'oom', 'oom_kill'} and all(type(v) is int and v == 0 for v in deltas.values()))


def host_capacity_ok(memory, policy, *, node_rank=None):
    """Same evidence policy as the common ResourceObserver; never invent a peak."""
    resolved = policy_for_node(policy, node_rank)
    if ('schema_version' in policy or 'policy_node_rank' in memory):
        if (type(memory.get('policy_node_rank')) is not int or memory['policy_node_rank'] != node_rank
                or type(memory.get('policy_host_limit_bytes')) is not int
                or memory['policy_host_limit_bytes'] != resolved['host_limit_bytes']):
            return False
    policy = resolved
    cap = memory.get('capacity_observation', {})
    current = memory.get('host_current_bytes')
    effective = cap.get('effective_host_limit_bytes')
    if (type(current) is not int or not 0 <= current <= policy['host_limit_bytes']
            or type(effective) is not int or effective < policy['host_limit_bytes']
            or cap.get('scope') != 'OBSERVED_PROCESS_CGROUP_HIERARCHY_AND_NODE_MEMTOTAL'):
        return False
    events = memory.get('cgroup_events', {})
    if not all(type(events.get(k)) is int and events[k] >= 0 for k in ('oom', 'oom_kill')):
        return False
    peak = memory.get('host_peak_bytes')
    if memory.get('host_peak_status') == 'UNAVAILABLE':
        return (peak is None and policy['host_reserve_bytes'] == 0
            and type(cap.get('cgroup_hard_limit_bytes')) is int
            and cap['cgroup_hard_limit_bytes'] == policy['host_limit_bytes']
            and memory.get('host_peak_scope') == 'UNAVAILABLE_CGROUP_V2_MEMORY_PEAK'
            and memory.get('host_capacity_evidence') == 'FINITE_CGROUP_HARD_LIMIT_AND_UNCHANGED_OOM_NOT_OBSERVED_PEAK')
    return (memory.get('host_peak_status') == 'OBSERVED' and type(peak) is int and peak >= current
        and peak + policy['host_reserve_bytes'] <= policy['host_limit_bytes']
        and memory.get('host_peak_scope') == 'CGROUP_LIFETIME_NOT_RESET_BY_OBSERVER'
        and memory.get('host_capacity_evidence') == 'OBSERVED_CGROUP_LIFETIME_PEAK')


def host_capacity_signature(memory):
    return {key: memory[key] for key in ('capacity_observation', 'host_peak_status',
        'host_peak_scope', 'host_capacity_evidence', 'policy_node_rank', 'policy_host_limit_bytes') if key in memory}


def observe_host_memory(policy, *, node_rank=None, root=None, capacity=None):
    resolved = policy_for_node(policy, node_rank)
    if root is None:
        root, capacity = _capacity_observation()
    require(isinstance(capacity, dict) and capacity.get('cgroup_path') == str(root),
            'actual cgroup capacity observation required', path='h3_ar.worker')
    current = int((root/'memory.current').read_text())
    events = {k: int(v) for k, v in (line.split() for line in (root/'memory.events').read_text().splitlines())}
    try:
        peak = int((root/'memory.peak').read_text())
    except FileNotFoundError:
        peak = None
    result = dict(host_current_bytes=current, host_peak_bytes=peak, cgroup_events=events,
        capacity_observation=capacity,
        host_peak_status='UNAVAILABLE' if peak is None else 'OBSERVED',
        host_peak_scope='UNAVAILABLE_CGROUP_V2_MEMORY_PEAK' if peak is None else 'CGROUP_LIFETIME_NOT_RESET_BY_OBSERVER',
        host_capacity_evidence='FINITE_CGROUP_HARD_LIMIT_AND_UNCHANGED_OOM_NOT_OBSERVED_PEAK' if peak is None else 'OBSERVED_CGROUP_LIFETIME_PEAK')
    if node_rank is not None:
        result.update(policy_node_rank=node_rank, policy_host_limit_bytes=resolved['host_limit_bytes'])
    require(host_capacity_ok(result, policy, node_rank=node_rank),
            'host capacity unproved: unavailable peak requires zero reserve and exact finite cgroup hard limit', path='h3_ar.worker')
    return result


def host_admission(policy, root=None, *, node_rank=None, capacity=None):
    """Original owner entry gate, not a sustained 64 GiB headroom claim."""
    observation = observe_host_memory(policy, node_rank=node_rank, root=root, capacity=capacity)
    policy = policy_for_node(policy, node_rank)
    current = observation['host_current_bytes']
    maximum = observation['capacity_observation']['cgroup_hard_limit_bytes']
    require(type(maximum) is int, 'original startup requires observed finite cgroup cap', path='h3_ar.worker')
    require(maximum-current >= 64 << 30, 'original startup cgroup reserve64GiB not met', path='h3_ar.worker')
    require(current + policy['host_reserve_bytes'] <= policy['host_limit_bytes'] <= maximum,
            'configured host capacity/reserve not admitted by actual cgroup', path='h3_ar.worker')
    return dict(current_bytes=current, maximum_bytes=maximum, startup_reserve_bytes=64 << 30,
                scope='ENTRY_ONLY_ORIGINAL_OWNER_GATE_NOT_SUSTAINED_RESERVE',
                initial_events={key: observation['cgroup_events'][key] for key in ('oom', 'oom_kill')},
                cgroup_path=observation['capacity_observation']['cgroup_path'],
                host_capacity=host_capacity_signature(observation))


def execute_cycles(session, *, mode, torch, distributed, capacity_policy, memory, node_rank=None, order=('native_update',), clock=time.perf_counter_ns):
    """Tested common dispatcher. Setup and state hashing are outside both timers."""
    require(mode in ('calibrate', 'run', 'manual'), 'invalid native mode', path='h3_ar.worker')
    initial = session.versions(); records = []
    require(initial == dict(student=0, fake=0, ema=0), 'native session must start fresh', path='h3_ar.worker')
    for cycle in (1, 2):
        prepared = session.prepare(cycle, tuple(order))
        distributed.barrier(); torch.cuda.synchronize(session.device)
        start = clock()
        prepared.run()
        torch.cuda.synchronize(session.device); distributed.barrier()
        duration = clock() - start
        checks = session.finish(cycle)
        require(checks['census'] == session.task.census
                and checks['updates'] == checks['adam_updates'] == dict(student=1, critic=1)
                and checks['final_versions'] == dict(student=cycle, fake=cycle, ema=0),
                'native runtime census/versions failed', path='h3_ar.worker')
        finite = all(type(v) not in (float, int) or math.isfinite(v) for v in checks['metrics'].values())
        require(finite, 'native nonfinite metric', path='h3_ar.worker')
        mem = memory(); passed = capacity_ok(mem, capacity_policy, node_rank=node_rank)
        require(passed, 'common GPU/host/reserve/OOM capacity policy failed', path='h3_ar.worker')
        records.append(dict(cycle=cycle, duration_ns=duration, **checks, memory=mem, capacity_pass=passed,
            timing_scope='warmup_first_update' if cycle == 1 else 'steady_second_native_update',
            prepared=prepared.receipt(), state=session.state_metadata(cycle), RNG_reseeded_between_cycles=False))
    return dict(initial_versions=initial, final_versions=session.versions(), cycles=records, finite=True)


def run_worker(task, assets, output, *, mode, capacity_policy, implementation=None, plan_path=None):
    validate_policy(capacity_policy)
    rank, local = topology(os.environ)
    saved = collection = None
    order = ('native_update',)
    if mode == 'run':
        require(plan_path is not None and implementation is None, 'run must consume one saved plan', path='h3_ar.worker')
        saved, collection = validate_plan(task, plan_path)
        implementation = H3ARImplementation.from_dict(saved['selected']['implementation'])
        order = tuple(saved['selected']['order'])
        require(saved['capacity_policy'] == capacity_policy, 'runtime capacity differs from saved common policy', path='h3_ar.worker')
    else:
        require(mode in ('calibrate', 'manual') and isinstance(implementation, H3ARImplementation)
                and plan_path is None, 'explicit measured implementation required', path='h3_ar.worker')
    host_entry = host_admission(capacity_policy, node_rank=rank//8)
    if saved is not None:
        require(all(host_entry['host_capacity'] == capability for capability in collection['host_capacity_bindings'][rank]),
                'actual host capacity/capability differs from measured plan', path='h3_ar.worker')
    asset_validation = assets.validate()
    identity = dict(source=source_identity(), assets_sha256=assets.digest, task_sha256=task.digest)
    if saved is not None:
        require(identity == saved['identity'], 'saved source/assets/task differs before model load', path='h3_ar.worker')
    allocator = 'expandable_segments:True,pinned_use_cuda_host_register:True,pinned_num_register_threads:4' if implementation.optimized else 'expandable_segments:True'
    require(os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION') == 'torch', 'native torch attention environment required', path='h3_ar.worker')
    require(os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == allocator
            and os.environ.get('PYTORCH_ALLOC_CONF', allocator) == allocator,
            'allocator environment must match selected actual native strategy before CUDA initialization', path='h3_ar.worker')
    import torch
    import torch.distributed as dist
    from accelerate import Accelerator, DeepSpeedPlugin
    from accelerate.utils import DistributedType
    import accelerate.utils.environment as accelerate_environment
    assets.install_import_paths()
    import json
    startup_path = checked(assets.startup_gpu_info[str(rank//8)])
    startup = json.loads(startup_path.read_text())
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == ','.join(startup['ordered_gpu_uuids']),
            'rank visibility differs from fresh owner-qualified GPU metadata', path='h3_ar.worker')
    assets.helper('native_accelerate_gpu_info_cache_r1').install(accelerate_environment, startup_path)
    runtime = runtime_identity(torch)
    torch.cuda.set_device(local)
    prop = torch.cuda.get_device_properties(local)
    hardware = dict(name=prop.name, total_memory=prop.total_memory, capability=[prop.major, prop.minor],
                    uuid=str(getattr(prop, 'uuid', 'UNAVAILABLE')), hostname=socket.gethostname())
    require(capacity_policy['gpu_limit_bytes'] + capacity_policy['gpu_reserve_bytes'] <= prop.total_memory,
            'GPU capacity/reserve policy exceeds device', path='h3_ar.worker')
    if saved is not None:
        require(runtime == collection['runtimes'][rank] and hardware == collection['hardware'][rank],
                'runtime/software/device differs from measured plan', path='h3_ar.worker')
        free, _ = torch.cuda.mem_get_info(local)
        required = next(r['peak_bytes'] for r in saved['selected']['memory'] if r['rank'] == rank)
        require(required + capacity_policy['gpu_reserve_bytes'] <= free,
                'current GPU headroom does not admit saved plan', path='h3_ar.worker')
    # This worker owns only its process group, never cluster process cleanup.
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=True)
    target = root/f'rank_{rank:03d}.json'
    require(not target.exists(), 'refuse overwriting another native receipt', path='h3_ar.worker')
    initialized = False
    before = host_entry['initial_events']
    try:
        dist.init_process_group('nccl', timeout=timedelta(seconds=300)); initialized = True
        plugin = DeepSpeedPlugin(gradient_accumulation_steps=1, zero_stage=3,
            offload_optimizer_device='none', offload_param_device='none', zero3_init_flag=True,
            zero3_save_16bit_model=False)
        accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1, deepspeed_plugin=plugin)
        require(accelerator.distributed_type == DistributedType.DEEPSPEED, 'original native DeepSpeed plugin not active', path='h3_ar.worker')
        assets.helper('native_ptxas_fd_isolation_r1').install(enabled=True)
        from .h3_ar_session import H3ARSession
        session = H3ARSession(task, assets, implementation, rank=rank, local_rank=local, torch=torch, accelerator=accelerator)
        def memory():
            free, _ = torch.cuda.mem_get_info(local)
            host = observe_host_memory(capacity_policy, node_rank=rank//8)
            require(host_capacity_signature(host) == host_entry['host_capacity'],
                    'host capacity/capability changed during native execution', path='h3_ar.worker')
            now = host['cgroup_events']
            peak = torch.cuda.max_memory_allocated(local)
            if saved is not None:
                require(peak <= next(r['limit_bytes'] for r in saved['selected']['memory'] if r['rank'] == rank),
                        'observed native allocation exceeds selected tighter capacity', path='h3_ar.worker')
            return dict(peak_allocated_bytes=peak,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(local), driver_free_bytes=free,
                **host,
                oom_event_delta={k: now[k]-before[k] for k in before})
        executed = execute_cycles(session, mode=mode, torch=torch, distributed=dist,
            capacity_policy=capacity_policy, memory=memory, node_rank=rank//8, order=order)
        assets.audit_imports()
        require(runtime_identity(torch) == runtime and source_identity() == identity['source'],
                'source/runtime changed during native work', path='h3_ar.worker')
        result = dict(schema_version='distillschedule.original_h3_ar_execution.v1',
            status='COMPLETED_ORIGINAL_H3_AR_WORK', mode=mode, rank=rank, task=task.to_dict(), identity=identity,
            implementation=implementation.to_dict(), runtime=runtime, hardware=hardware,
            capacity_policy=capacity_policy, static_bytes=session.static_bytes, build=session.build,
            input_identity=session.input_identity, asset_validation=asset_validation, host_admission=host_entry, **executed,
            plan_sha256=saved['plan_sha256'] if saved else None, selected_order=list(order),
            gpu_execution='NATIVE_CUDA_RUN', profiled=False, numerical_status='REVIEW_REQUIRES_MATCHED_REFERENCE',
            claimable_speedup=False, quality_validation='NOT_RUN')
        write_new_json(target, result)
        return result
    except BaseException as exc:
        write_new_json(root/f'failure_{rank:03d}.json', dict(status='FAIL', mode=mode, error=str(exc), rank=rank, identity=identity))
        raise
    finally:
        if initialized:
            dist.destroy_process_group()
