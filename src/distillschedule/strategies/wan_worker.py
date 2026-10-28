"""Execute a task-bound plan on the ORIGINAL Wan world16 helpers.

This module never starts remote machines. It is entered by the existing CLI
under torchrun. NCCL is created only after the selected strategy environment
is installed. Calibration is a separate diagnostic mode, not a speedup claim.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import timedelta
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import time

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint, repository_root
from .wan_task import OriginalWanTask, NATIVE_HISTORY_POLICY
from .wan_strategy import WanStrategy, strategy_environment
from .wan_bridge import load_original_backend, OriginalWanSession
from .wan_calibration import OriginalWanCalibrator, trace_work
from .wan_planning import source_fingerprint, validate_plan
from .wan_stream import MAX_COHORTS


@dataclass(frozen=True)
class OriginalWanAssets:
    model_root: str
    bootstrap: str
    expected_bootstrap_hash: str
    trace_bundle: str
    expected_trace_hash: str
    negative_prompt: str
    schema_version: str = 'distillschedule.original_wan_assets.v1'

    @classmethod
    def from_dict(cls, value):
        require(isinstance(value, dict) and set(value) == set(cls.__dataclass_fields__),
                'exact original Wan asset fields required', path='wan.assets')
        result = cls(**value)
        require(result.schema_version == cls.__dataclass_fields__['schema_version'].default,
                'unknown assets schema', path='wan.assets')
        for name, text in asdict(result).items():
            require(isinstance(text, str) and bool(text), f'invalid {name}', path='wan.assets')
        for value in (result.expected_bootstrap_hash, result.expected_trace_hash):
            require(len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                    'asset hash must be SHA256', path='wan.assets')
        return result


def topology(environment):
    """No inferred world size, rank, or localhost defaults."""
    names = ('WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE')
    require(all(str(environment.get(k, '')).isdigit() for k in names),
            'explicit torchrun world16 rank environment required', path='wan.worker')
    world, rank, local, local_world = (int(environment[k]) for k in names)
    require(world == 16 and local_world == 8 and 0 <= rank < 16 and local == rank % 8,
            'original Wan backend supports two nodes of eight ranks', path='wan.worker')
    return rank, local


def gather_metadata(distributed, value, metadata_group):
    """Control-plane objects must not use the Tree-only gradient NCCL group."""
    require(metadata_group is not None and distributed.get_backend(metadata_group) == 'gloo',
            'metadata requires an explicit Gloo process group', path='wan.worker')
    rows = [None] * 16
    distributed.all_gather_object(rows, value, group=metadata_group)
    return rows


def admit_run_controls(distributed, metadata_group, *, mode, cycles, checkpoint_every, resume_from):
    """Agree on optional collective branches before any rank enters them.

    Checkpoint paths may be mount aliases. Compare the completed manifest's
    content, not the local path spelling. Read failures are exchanged too, so a
    missing file on one node cannot strand peers in another process group.
    """
    try:
        manifest_sha = None
        if resume_from:
            root = Path(resume_from)
            manifest_sha = file_sha256(root / 'manifest.json')
            require((root / '_SUCCESS').read_text().strip() == manifest_sha,
                    'checkpoint is not complete', path='wan.worker')
        local = {'controls': {'mode': mode, 'cycles': cycles,
                             'checkpoint_every': checkpoint_every,
                             'resume_manifest_sha256': manifest_sha}, 'error': None}
    except (OSError, ValueError) as exc:
        local = {'controls': None, 'error': f'{type(exc).__name__}: {exc}'}
    rows = gather_metadata(distributed, local, metadata_group)
    require(not any(row['error'] for row in rows),
            'rank checkpoint is not complete: ' + '; '.join(str(row['error']) for row in rows if row['error']),
            path='wan.worker')
    require(all(row['controls'] == rows[0]['controls'] for row in rows),
            'rank run/restart control mismatch', path='wan.worker')


def destroy_process_groups(distributed, metadata_group):
    """Attempt tensor-group cleanup even if control-plane teardown fails."""
    try:
        if metadata_group is not None:
            distributed.destroy_process_group(metadata_group)
    finally:
        distributed.destroy_process_group()


def load_assets(task, spec, *, rank):
    """Validate source assets before constructing the large model."""
    from distillschedule.runtime.wan13b_upper_bound import Wan13BBootstrap
    from runtime.gst_pool_contract import NegativePromptAsset
    from .wan_trace import load_original_trace, validate_loaded_history
    bootstrap = Wan13BBootstrap.load(spec.bootstrap, expected_hash=spec.expected_bootstrap_hash,
                                    expected_lora_rank=task.lora_rank, verify_payloads=True)
    require(Path(bootstrap.manifest['model_root']).resolve() == Path(spec.model_root).resolve(),
            'bootstrap/model root mismatch', path='wan.assets')
    model_dir = Path(spec.model_root).resolve() / 'transformer'
    require((model_dir / 'config.json').is_file(), 'original transformer config missing', path='wan.assets')
    weights = sorted(model_dir.glob('*.safetensors'))
    require(weights, 'original local safetensors weights required', path='wan.assets')
    model_files = [model_dir / 'config.json', *weights, *sorted(model_dir.glob('*.index.json'))]
    model_identity = {p.name: file_sha256(p) for p in model_files}
    trace = load_original_trace(task, spec.trace_bundle, expected_hash=spec.expected_trace_hash)
    if task.stage_schedule_policy == NATIVE_HISTORY_POLICY:
        validate_loaded_history(task, trace._load(rank))
    negative = NegativePromptAsset.load(spec.negative_prompt)
    identity = {'model_files': model_identity, 'bootstrap_hash': bootstrap.manifest_hash,
                'trace_bundle_hash': trace.bundle_hash, 'trace_binding_hash': trace.binding_hash,
                'negative_manifest_hash': negative.manifest['manifest_hash']}
    return bootstrap, trace, negative, identity


def _json(path, payload):
    """Atomic per-rank progress receipt; never overwrite another job's directory."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def execute_cycles(session, *, cycles, mode, torch, distributed, observer=None,
                   operation_order=None, progress=None, validate_result=None, before_cycle=None, clock=time.perf_counter_ns,
                   start_cycle=0, after_cycle=None):
    """Shared tested orchestration; original handlers perform all model math."""
    limit = MAX_COHORTS if before_cycle is not None and mode == 'run' else 64
    require(type(cycles) is int and 2 <= cycles <= limit, 'bounded two-or-more cycle run required', path='wan.worker')
    require(mode in {'calibrate', 'run'} and (mode == 'calibrate') == (observer is not None),
            'calibration observer/mode mismatch', path='wan.worker')
    require(type(start_cycle) is int and 0 <= start_cycle < cycles
            and (not start_cycle or mode == 'run' and before_cycle is not None),
            'resume requires an admitted advancing native input stream', path='wan.worker')
    initial = session.versions(); warm = None; records = []
    for cycle in range(start_cycle + 1, cycles + 1):
        before = session.versions()
        cohort = None; current_order = operation_order
        wall_start = clock() if before_cycle is not None else None
        if before_cycle is not None:
            require(mode == 'run', 'fresh cohorts cannot masquerade as fixed-trace calibration', path='wan.worker')
            engines = (session.g, session.c, session.ema)
            cohort = before_cycle(cycle, session)
            require(session.versions() == before and all(a is b for a, b in
                    zip(engines, (session.g, session.c, session.ema))),
                    'cohort transition reset model/optimizer/EMA', path='wan.worker')
            current_order = tuple(cohort['operation_order'])
        if cycle == 2:
            warm = before
        scope = None if observer is None else lambda op, c=cycle: observer.stage(c, op.operation_id)
        prepared = session.prepare(operation_scope=scope, operation_order=current_order)
        distributed.barrier(); torch.cuda.synchronize(session.device)
        start = clock()
        result = prepared.run()
        torch.cuda.synchronize(session.device)
        duration = clock() - start
        checks = session.finite
        finite = bool(torch.stack(checks).all().detach().cpu()) if checks else True
        require(finite, 'native finite check failed', path='wan.worker')
        checks.clear()
        expected = {'generator': before['generator'] + 1,
                    'critic': before['critic'] + session.task.fake_updates,
                    'ema': before['ema'] + len(session.task.ema_after)}
        require(session.versions() == expected and len(result['rollouts']) == session.task.fake_updates
                and len(result['critic']) == session.task.fake_updates,
                'native updates differ from requested protocol', path='wan.worker')
        require(all(int(v.get('batch_size', 1)) == 1 for v in result['rollouts']),
                'native rollout batch changed', path='wan.worker')
        record = {'cycle': cycle, 'duration_ns': duration, 'versions': expected,
                  'prepared': prepared.receipt(), 'rollouts': result['rollouts'],
                  'generator': result['generator'], 'critic': result['critic'],
                  'calibration_perturbed': observer is not None, 'finite': finite}
        if validate_result is not None:
            record['call_census'] = validate_result(result)
        if cohort is not None:
            record['cohort'] = dict(cohort['receipt'])
            record['input_and_execution_wall_ns'] = clock() - wall_start
        del prepared, result
        lifecycle = getattr(session, 'bundle_lifecycle', None)
        if lifecycle is not None:
            lifecycle.finish(record)
            record['input_and_execution_wall_ns'] = clock() - wall_start
        records.append(record)
        if progress is not None:
            progress(records, expected)
        if after_cycle is not None:
            after_cycle(cycle, session, record)
    return {'cycles': records, 'initial_versions': initial, 'warm_initial_versions': warm,
            'final_versions': session.versions(), 'finite': True}


def runtime_info(torch):
    versions = {}
    for name in ('diffusers', 'peft', 'transformers', 'triton', 'safetensors'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'packages': versions, 'nccl_algo': os.environ.get('NCCL_ALGO', 'default'),
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'matmul_precision': torch.get_float32_matmul_precision(),
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'tf32_cublas_override': os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'NOT_SET'),
            'nvidia_tf32_override': os.environ.get('NVIDIA_TF32_OVERRIDE', 'NOT_SET')}


def save_final_state(session, output):
    """Reuse original full-state evidence writers, outside all cycle timers."""
    from safetensors.torch import save_file
    from torch.distributed.tensor import DTensor
    base = session.fair.base; rank = session.rank
    output = Path(output); (output / 'correctness').mkdir(parents=True, exist_ok=True)
    evidence = {}
    for name, engine in (('generator', session.g), ('critic', session.c)):
        evidence[name] = base._save_trainable_evidence(output_dir=output, rank=rank,
                                                     component=name, module=engine.module)
        evidence[name + '_optimizer'] = base._save_optimizer_evidence(output_dir=output, rank=rank,
                                                                       component=name, engine=engine)
    ema_values = {}
    for i, (shadow, parameter) in enumerate(zip(session.ema._shadows, session.ema._parameters, strict=True)):
        if isinstance(parameter, DTensor):
            full = DTensor.from_local(shadow, device_mesh=parameter.device_mesh,
                                     placements=parameter.placements, run_check=False,
                                     shape=parameter.shape, stride=parameter.stride()).full_tensor()
        else:
            full = shadow
        if rank == 0:
            ema_values[f'ema/{i:06d}'] = full.detach().cpu().float().contiguous()
        del full
    if rank == 0:
        path = output / 'correctness' / 'final_generator_ema.safetensors'
        save_file(ema_values, str(path))
        evidence['ema'] = {'path': str(path), 'sha256': file_sha256(path),
                           'optimization_step': session.ema.optimization_step}
    else:
        evidence['ema'] = {'writer_rank': 0, 'status': 'PARTICIPATED_IN_GATHER'}
    return evidence


def run_worker(task, assets, *, output, mode, strategy=None, plan_path=None, cycles=2, stream_path=None,
               record_memory_highwater=False, checkpoint_every=0, resume_from=None, native_output=None):
    """Called under torchrun; no implicit launch, external commands, or downloads."""
    require(isinstance(task, OriginalWanTask) and isinstance(assets, OriginalWanAssets),
            'validated task/assets required', path='wan.worker')
    require(mode in {'calibrate', 'run', 'manual', 'train', 'export'}, 'invalid worker mode', path='wan.worker')
    require((mode == 'export') == (native_output is not None),
            'output-only mode requires its explicit native output callback', path='wan.worker')
    require(mode != 'export' or resume_from is not None and not checkpoint_every and not record_memory_highwater,
            'output-only mode requires a complete checkpoint and no training/capture/memory work', path='wan.worker')
    require(type(cycles) is int and 2 <= cycles <= (MAX_COHORTS if stream_path is not None else 64),
            'only fresh streams support 2..100 cycles; fixed-trace diagnostics remain bounded to 64', path='wan.worker')
    rank, local_rank = topology(os.environ)
    document = None
    if mode == 'calibrate':
        require(plan_path is None and isinstance(strategy, WanStrategy) and strategy.rollout_window == 1
                and cycles == 2, 'calibration is one primitive arm, two cycles, window1', path='wan.worker')
    elif mode == 'manual':
        require(plan_path is None and isinstance(strategy, WanStrategy) and cycles == 2
                and stream_path is None,
                'manual reference requires explicit strategy, exactly two fixed-trace cycles and no plan/stream',
                path='wan.worker')
    elif mode == 'export' and plan_path is not None:
        require(strategy is None and stream_path is not None,
                'planned output requires the same selected plan and stream', path='wan.worker')
        document = validate_plan(task, plan_path)
        strategy = WanStrategy.from_dict(document['selected']['strategy'])
    elif mode in {'train', 'export'}:
        require(plan_path is None and isinstance(strategy, WanStrategy) and stream_path is not None,
                'explicit native training requires a strategy and fresh input stream', path='wan.worker')
    else:
        require(plan_path is not None and strategy is None, 'run must use a validated selected plan', path='wan.worker')
        document = validate_plan(task, plan_path)
        strategy = WanStrategy.from_dict(document['selected']['strategy'])
    strategy.validate_for(task)
    require(not strategy.retained_critic_blocks or mode == 'manual'
            or strategy.retention_execution == 'planner-retain4.v1',
            'experimental block retention requires an explicit manual candidate; no uncalibrated saved-plan fallback',
            path='wan.worker')
    require(not strategy.candidate_bundle.continuous or (mode in {'train', 'export'} and stream_path is not None),
            'stream bundle requires explicit fresh native training; no fixed replay/calibration/old plan', path='wan.worker')
    require(not strategy.candidate_bundle.enabled or strategy.candidate_bundle.continuous or (cycles == 2 and stream_path is None),
            'incumbent literal graph bundle supports only two fixed-trace cycles, not streams', path='wan.worker')
    stream = None
    if stream_path is not None:
        require(mode in {'run', 'train', 'export'}, 'cohort training requires a selected or explicit strategy', path='wan.worker')
        from .wan_stream import load_stream
        stream = load_stream(task, stream_path)
        require(cycles == len(stream['cohorts']), 'training cycles must match the complete fresh input sequence', path='wan.worker')
    require(type(checkpoint_every) is int and checkpoint_every >= 0,
            'checkpoint interval must be nonnegative', path='wan.worker')
    require(not (checkpoint_every or resume_from) or stream is not None,
            'restart checkpoints require a fresh native cohort stream, never fixed-trace replay', path='wan.worker')
    if checkpoint_every or resume_from:
        require(str(os.environ.get('GROUP_RANK', '')).isdigit()
                and int(os.environ['GROUP_RANK']) == rank//8,
                'native restart requires explicit matching torchrun node ranks', path='wan.worker')
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=True)
    claim = root / f'rank_{rank:02d}.claim'
    with claim.open('x') as f:
        f.write(f'{socket.gethostname()} {os.getpid()}\n')
    path = root / f'rank_{rank:02d}.json'
    receipt = {'status': 'STARTED_ORIGINAL_WAN_WORK', 'mode': mode, 'rank': rank,
               'task': task.to_dict(), 'strategy': strategy.to_dict(), 'gpu_execution': 'NOT_RUN',
               'claimable_speedup': False, 'host': socket.gethostname()}
    if mode == 'manual':
        receipt.update(execution_kind='MANUAL_REFERENCE_DIRECT_NATIVE_ORDER',
                       manual_strategy_sha256=fingerprint(strategy.to_dict()),
                       search_performed=False)
    _json(path, receipt)
    require(type(record_memory_highwater) is bool, 'explicit memory recorder boolean required', path='wan.worker')
    observer = None; initialized = False; metadata_group = None; highwater = None
    try:
        with strategy_environment(strategy):
            import torch
            import torch.distributed as dist
            require(torch.cuda.is_available() and torch.cuda.device_count() >= 8,
                    'eight visible CUDA devices per node required', path='wan.worker')
            require(not dist.is_initialized(), 'strategy must be selected before NCCL initialization', path='wan.worker')
            torch.cuda.set_device(local_rank); device = torch.device('cuda', local_rank)
            if record_memory_highwater:
                from .wan_memory import CudaHighwater
                highwater = CudaHighwater(torch.cuda, device)
                highwater.start()
            bootstrap, trace, negative, asset_identity = load_assets(task, assets, rank=rank)
            if strategy.candidate_bundle.enabled:
                strategy.candidate_bundle.validate_model_files(bootstrap.manifest, asset_identity["model_files"])
                strategy.candidate_bundle.validate_assets({**asset_identity,
                    'model_identity_sha256': bootstrap.manifest['model_identity_sha256']})
                if strategy.candidate_bundle.continuous:
                    strategy.candidate_bundle.validate_bootstrap(bootstrap, task)
                    require(trace.bundle_hash == stream['cohorts'][0]['trace_hash'],
                            'initial asset trace differs from fresh stream cohort zero', path='wan.worker')
            fair = load_original_backend(repository_root())
            from distillschedule.search.wan_implementation import verify_execution_precision
            verify_execution_precision(torch)
            runtime_seed = int(bootstrap.component('generator')['seed']) + 1000003 + rank
            import random
            random.seed(runtime_seed); torch.manual_seed(runtime_seed); torch.cuda.manual_seed_all(runtime_seed)
            import subprocess
            source_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository_root(), text=True).strip()
            identity = {'source_head': source_head, 'source_fingerprint': source_fingerprint(),
                        'task_sha256': task.digest, 'assets': asset_identity}
            runtime = runtime_info(torch)
            runtime['rank_seed'] = runtime_seed
            prop = torch.cuda.get_device_properties(device)
            hardware = {'gpu_name': prop.name, 'uuid': str(getattr(prop, 'uuid', 'UNAVAILABLE')),
                        'total_memory': prop.total_memory, 'capability': [prop.major, prop.minor],
                        'host': socket.gethostname(), 'local_rank': local_rank}
            if document is not None:
                require(document['identity'] == identity, 'source/assets differ from planned calibration', path='wan.worker')
                from .wan_planning import load_collections, calibrated_variant, _read
                _, variants = load_collections(task, [c['path'] for c in document['collections']])
                variant, _ = calibrated_variant(strategy, variants)
                observed = [_read(s['path']) for s in variant['rank_sources']]
                old = next(v for v in observed if v['rank'] == rank)
                require(old['runtime'] == runtime and old['hardware'] == hardware,
                        'runtime/hardware changed: recalibration required', path='wan.worker')
            dist.init_process_group('nccl', timeout=timedelta(minutes=15)); initialized = True
            # CPU identity/trace bookkeeping is outside cycle timers. NCCL
            # remains unchanged for real tensor/FSDP/reducer communication.
            metadata_group = dist.new_group(backend='gloo', timeout=timedelta(minutes=15))
            seen = gather_metadata(dist, {'identity': identity, 'strategy': strategy.to_dict(),
                'stream_sha256': None if stream is None else stream['stream_sha256']}, metadata_group)
            require(all(row == seen[0] for row in seen), 'rank task/strategy/asset mismatch', path='wan.worker')
            admit_run_controls(dist, metadata_group, mode=mode, cycles=cycles,
                               checkpoint_every=checkpoint_every, resume_from=resume_from)
            checkpoint_identity = checkpoint_policy = None
            checkpoint_state = None
            start_cycle = 0
            resume_header = None
            if checkpoint_every or resume_from:
                from distillschedule.open_workload.replicated import control_plane, checked
                from distillschedule.training.checkpoint import read_manifest
                environments = gather_metadata(dist, {k: v for k, v in runtime.items() if k != 'rank_seed'}, metadata_group)
                checkpoint_identity = {'native': identity, 'strategy': strategy.to_dict(),
                    'stream_sha256': stream['stream_sha256'], 'rank_runtimes': environments}
                checkpoint_policy = {'schema_version': 'distillschedule.wan_stream_checkpoint_policy.v1',
                    'strategy_sha256': fingerprint(strategy.to_dict()),
                    'stream_sha256': stream['stream_sha256'],
                    'selected_plan_sha256': None if document is None else document['plan_sha256']}
                if resume_from:
                    with control_plane(metadata_group):
                        resume_header = checked(lambda: read_manifest(resume_from, identity=checkpoint_identity),
                                                'native restart header')
                    require(resume_header['policy'] == checkpoint_policy
                            and type(resume_header['cycle']) is int
                            and 0 < resume_header['cycle'] == resume_header['cursor']
                            and (resume_header['cycle'] <= cycles if mode == 'export' else resume_header['cycle'] < cycles),
                            'native restart policy/progress differs from the supplied stream', path='wan.worker')
            if native_output is not None:
                with control_plane(metadata_group):
                    native_output.admit()
            negative_tensor = negative.embedding.to(device=device, dtype=torch.bfloat16)
            from distillschedule.profiling.static_profile_diagnostics import StaticProfileDiagnostics
            session = OriginalWanSession(task=task, strategy=strategy, fair=fair, bootstrap=bootstrap,
                model_root=assets.model_root, device=device, rank=rank, trace=trace, negative=negative_tensor,
                diagnostics=StaticProfileDiagnostics(enabled=False, rank=rank),
                asset_identity={**asset_identity, 'model_identity_sha256': bootstrap.manifest['model_identity_sha256']})
            from .wan_evidence import validate_build, validate_cycle
            strategy.candidate_bundle.validate_versions(session.versions())
            effects = validate_build(strategy, session.build)
            if checkpoint_every or resume_from:
                from distillschedule.training.native_state import engine_state
                checkpoint_state = engine_state({'generator': session.g, 'critic': session.c}, ema=session.ema)
                if resume_from:
                    from distillschedule.training.checkpoint import restore_checkpoint
                    from .wan_stream import restore_stream_cursor
                    with control_plane(metadata_group):
                        restored = restore_checkpoint(resume_from, checkpoint_state, identity=checkpoint_identity)
                    require(session.versions() == restored['versions'],
                            'native restored engine/EMA clocks differ from manifest', path='wan.worker')
                    start_cycle = restored['cycle']
                    if native_output is None:
                        restore_stream_cursor(session, task, stream, completed_cycles=start_cycle)
                    del restored
                receipt['training_restart'] = {'schema_version': 'distillschedule.wan_native_training_restart.v1',
                    'resume_from': str(Path(resume_from).resolve()) if resume_from else None,
                    'resume_cycle': start_cycle, 'checkpoint_every': checkpoint_every,
                    'checkpoint_identity': checkpoint_identity, 'checkpoint_policy': checkpoint_policy,
                    'latest_checkpoint': str(Path(resume_from).resolve()) if resume_from else None}
            receipt.update(identity=identity, runtime=runtime, hardware=hardware, build=session.build, backend_effects=effects)
            _json(path, receipt)
            if native_output is not None:
                # A terminal complete checkpoint is valid for export, not for
                # another training cycle. Never bind/consume the next cohort,
                # prepare a graph, run loss handlers or advance native clocks.
                with control_plane(metadata_group):
                    exported = native_output.publish(session,manifest=resume_header,identity=checkpoint_identity)
                receipt.update(status='COMPLETED_ORIGINAL_WAN_EXPORT',inference_export=exported,
                    final_versions=session.versions(),checkpoint_cycle=start_cycle,
                    generator_selection='CURRENT_FULL_UNMERGED_NOT_EMA',training_cycles_executed=0,
                    gpu_execution='NATIVE_CUDA_CHECKPOINT_RESTORE_AND_EXPORT_NO_TRAINING',quality_validation='NOT_RUN')
                _json(path,receipt)
                return receipt
            if mode == 'calibrate':
                observer = OriginalWanCalibrator(task=task, initial_versions=session.versions(),
                    work=trace_work(task, trace), torch=torch, distributed=dist, device=device,
                    rank=rank, implementation=strategy, source_head=source_head)
            elif document is not None and stream is None:
                free, _ = torch.cuda.mem_get_info(device)
                predicted = next(r for r in document['selected']['memory'] if r['rank'] == rank)
                require(predicted['peak_bytes'] <= free + torch.cuda.memory_allocated(device),
                        'current free capacity does not admit plan', path='wan.worker')
            def progress(records, versions):
                receipt.update(status='PARTIAL_ORIGINAL_WAN_WORK', cycles=records,
                               final_versions=versions, gpu_execution='PARTIAL_NATIVE_CUDA_RUN')
                if observer is not None:
                    receipt['calibration'] = observer.snapshot()
                _json(path, receipt)
            before_cycle = None
            if stream is not None:
                from .wan_stream import bind_cohort
                receipt.update(stream_sha256=stream['stream_sha256'],
                    policy_transfer='SAME_GEOMETRY_STRATEGY_ONLY_NO_TIME_OR_CAPACITY_PREDICTION',
                    stream_source=str(Path(stream_path).resolve()))
                def before_cycle(cycle, current):
                    if document is not None:
                        selected_order = tuple(document['selected']['order'])
                    else:
                        from distillschedule.runtime.parameterized_cycle import (
                            build_parameterized_cycle_graph, parameterized_window_order)
                        selected_order = parameterized_window_order(build_parameterized_cycle_graph(
                            source_steps=task.source_steps, initial_versions=current.versions(), ema_after=task.ema_after),
                            max_live_rollouts=strategy.rollout_window)
                    event = bind_cohort(current, task, stream, cycle-1, selected_order)
                    peers = gather_metadata(dist, {'cohort': event['receipt'], 'versions': current.versions(),
                        'order': event['operation_order']}, metadata_group)
                    require(all(p == peers[0] for p in peers), 'rank cohort/order/version mismatch', path='wan.worker')
                    return event
            after_cycle = None
            if checkpoint_every:
                def after_cycle(cycle, current, record):
                    if cycle % checkpoint_every and cycle != cycles:
                        return
                    from distillschedule.training.checkpoint import save_checkpoint
                    started = time.perf_counter_ns()
                    destination = root/'checkpoints'/f'cycle-{cycle:06d}'
                    with control_plane(metadata_group):
                        if strategy.candidate_bundle.continuous:
                            checked(current.bundle_lifecycle.validate_checkpoint_boundary,
                                    'incumbent stream checkpoint quiescence')
                        save_checkpoint(destination, checkpoint_state, identity=checkpoint_identity,
                            cycle=cycle, cursor=cycle, versions=current.versions(), policy=checkpoint_policy)
                    record['checkpoint_outside_compute_ns'] = time.perf_counter_ns() - started
                    receipt['training_restart']['latest_checkpoint'] = str(destination)
                    _json(path, receipt)
            if mode == 'train':
                mode = 'run'  # The same original handlers/timer; no calibration claim.
            result = execute_cycles(session, cycles=cycles, mode='run' if mode == 'manual' else mode, torch=torch, distributed=dist,
                observer=observer, operation_order=None if document is None else tuple(document['selected']['order']),
                before_cycle=before_cycle, progress=progress,
                validate_result=lambda raw: validate_cycle(session.task, session.trace, raw, rank=rank),
                start_cycle=start_cycle, after_cycle=after_cycle)
            receipt.update(result)
            # Host metadata after both native timers, before CPU state export.
            # Native helpers may reset counters; only the opt-in recorder spans them.
            free, total = torch.cuda.mem_get_info(device)
            receipt['gpu_memory'] = dict(allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                driver_free_bytes=free, total_bytes=total,
                scope='SINCE_LAST_NATIVE_RESET_NOT_PROCESS_CUMULATIVE')
            if highwater is not None:
                receipt['gpu_memory'].update(highwater.close())
            if start_cycle:
                receipt['backend_effects'] = validate_build(strategy, session.build, executed=True, cycles=cycles-start_cycle)
            else:
                receipt['backend_effects'] = validate_build(strategy, session.build, executed=True, cycles=cycles)
            if observer is not None:
                receipt['calibration'] = observer.finish()
            receipt['state_evidence'] = save_final_state(session, root)
            dist.barrier()
            receipt.update(status='COMPLETED_ORIGINAL_WAN_WORK', gpu_execution='NATIVE_CUDA_RUN',
                plan_sha256=None if document is None else document['plan_sha256'],
                numerical_comparison='REQUIRES_MATCHED_REFERENCE', quality_validation='NOT_RUN',
                measurement_scope=('FRESH_COHORT_SEQUENCE_PERSISTENT_OPTIMIZERS' if stream is not None else
                                   ('TWO_CYCLE_REPLAY_OF_FIXED_TRACE' if cycles == 2 else 'REPEATED_FIXED_TRACE')),
                formal_speedup_eligible=False)
            _json(path, receipt)
            return receipt
    except BaseException as exc:
        receipt.update(status='FAILED_ORIGINAL_WAN_WORK', error_type=type(exc).__name__, error=str(exc))
        if observer is not None:
            receipt['calibration'] = observer.snapshot()
        _json(path, receipt)
        raise
    finally:
        if highwater is not None:
            highwater.close()
        if initialized:
            import torch.distributed as dist
            destroy_process_groups(dist, metadata_group)
