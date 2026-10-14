"""Full-parameter GPU-resident Helios reference with GPU ZeRO-1 state.

This is not FSDP or a Route-A fallback. It reuses the validated ZeRO-1 logical
restore and the existing official Helios loss/rollout helpers. No parameter,
activation, optimizer, residual or EMA CPU offload is enabled. The two logical
B1 rows/rank are accumulated by MicrobatchCommitEngine, with one loss /2 and
the native DP average. DeepSpeed's B2 is the outer logical batch, not a B2 DiT.
"""
from __future__ import annotations

import json
import weakref
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def resident_zero1_config() -> dict[str, Any]:
    return {
        'train_batch_size': 16,
        'train_micro_batch_size_per_gpu': 2,
        'gradient_accumulation_steps': 1,
        'bf16': {'enabled': True},
        'communication_data_type': 'fp32',
        'data_types': {'grad_accum_dtype': 'fp32'},
        'gradient_clipping': 1.0,
        'zero_allow_untested_optimizer': True,
        'zero_optimization': {
            'stage': 1,
            'overlap_comm': True,
            'contiguous_gradients': True,
            'reduce_bucket_size': 1000000000,
            'allgather_bucket_size': 536870912,
        },
        'steps_per_print': 1000000,
        'wall_clock_breakdown': False,
    }


def zero1_master_partitions(engine: Any) -> list[Any]:
    zero = engine.optimizer
    if hasattr(zero, 'single_partition_of_fp32_groups'):
        return zero.single_partition_of_fp32_groups
    # DeepSpeed 0.18.4 selects native BF16_Optimizer for GPU-only stage1 +
    # FP32 accumulation. It owns the same flat FP32 master/Adam partitions
    # under a different API name; never silently switch accumulation to BF16.
    if type(zero).__name__ == 'BF16_Optimizer' and hasattr(zero, 'fp32_groups_flat_partition'):
        return zero.fp32_groups_flat_partition
    raise TypeError('unrecognized GPU ZeRO-1 master-partition API')


def initialize_native_bf16_scaling_api(optimizer: Any) -> list[str]:
    """Supply the unscaled BF16 defaults missing from 0.18.4's legacy class.

    The new ZeROOptimizer.scale_if_loss/needs_scaler API expects these fields;
    BF16_Optimizer does not initialize them. Use stock unity LossScaler, not a
    replacement backward or an FP16/dynamic scaling policy.
    """
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    defaults = {'custom_loss_scaler': False, 'torch_autocast_gradscaler': None,
                'loss_scaler': LossScaler(1.0)}
    added = []
    for name, value in defaults.items():
        if not hasattr(optimizer, name):
            setattr(optimizer, name, value)
            added.append(name)
    if optimizer.custom_loss_scaler or optimizer.torch_autocast_gradscaler is not None:
        raise AssertionError('native BF16 reference cannot enable custom/FP16 loss scaling')
    if optimizer.loss_scaler.loss_scale != 1.0:
        raise AssertionError('native BF16 reference requires exact unity loss scale')
    return added


@contextmanager
def gpu_zero1_bf16_route_compat(*, engine_module=None, torch_module=None):
    """Fix the installed 0.18.4 BFLOAT16/DDP_BFLOAT16 selector collision.

    Both constants are 'bf16', so the earlier DDP branch incorrectly calls
    FP16_UnfusedOptimizer even for GPU ZeRO-1 with FP32 accumulation. Route only
    this exact configuration to DeepSpeed's existing BF16_Optimizer constructor.
    No site-packages file, optimizer math, dtype, collective or loss is changed.
    """
    if engine_module is None:
        import deepspeed
        import deepspeed.runtime.engine as engine_module
        if deepspeed.__version__ != '0.18.4':
            raise RuntimeError('resident route compatibility requires reviewed DeepSpeed 0.18.4')
    if torch_module is None:
        import torch as torch_module
    cls = engine_module.DeepSpeedEngine
    original = cls._configure_fp16_optimizer
    if getattr(original, '_distillschedule_resident_zero1_route', False):
        raise RuntimeError('nested resident initialization route scopes are forbidden')
    source = getattr(engine_module, '__file__', None)
    audit = dict(enabled=engine_module.BFLOAT16 == engine_module.DDP_BFLOAT16,
        api='DeepSpeedEngine._configure_fp16_optimizer -> _configure_bf16_optimizer',
        scope='initialization_only_gpu_zero1_bf16_compute_fp32_accumulation', routed_calls=0,
        source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else 'CPU_TEST_DOUBLE',
        shared_environment_files_modified=False, changes_optimizer_math=False)
    def route(self, optimizer, low_precision_dtype):
        if (self.zero_optimization_stage() == 1 and not self.zero_cpu_offload()
            and low_precision_dtype == torch_module.bfloat16
            and self.get_data_types() == (torch_module.bfloat16, torch_module.float32)):
            audit['routed_calls'] += 1
            result = self._configure_bf16_optimizer(optimizer)
            if type(result).__name__ == 'BF16_Optimizer':
                audit['unscaled_bf16_api_fields_initialized'] = initialize_native_bf16_scaling_api(result)
            return result
        return original(self, optimizer, low_precision_dtype)
    route._distillschedule_resident_zero1_route = True
    if audit['enabled']:
        cls._configure_fp16_optimizer = route
    try:
        yield audit
    finally:
        if audit['enabled']:
            changed = cls._configure_fp16_optimizer is not route
            cls._configure_fp16_optimizer = original
            if changed:
                raise RuntimeError('DeepSpeed optimizer route changed inside the guarded scope')


def build_resident_zero1_engine(
    *, module: Any, package: Any, component: str, lr: float,
    betas: tuple[float, float], eps: float, weight_decay: float,
    optimizer_dispatch_warmup: bool = False,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import deepspeed
    from runtime.critic_pool_checkpoint import restore_zero1_optimizer_partition_from_logical

    if optimizer_dispatch_warmup:
        raise ValueError('resident reference cannot enable optimizer dispatch optimization')
    if torch.distributed.get_world_size() != 8:
        raise ValueError('this registered resident reference is physical world8/B16 only')
    trainables = [p for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainables, lr=lr, betas=betas, eps=eps,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    config = resident_zero1_config()
    with gpu_zero1_bf16_route_compat() as route_audit:
        engine, _, _, _ = deepspeed.initialize(model=module, optimizer=optimizer,
            lr_scheduler=scheduler, model_parameters=trainables, config=config,
            dist_init_required=False)
    if engine.zero_optimization_stage() != 1 or getattr(engine.optimizer, 'partition_gradients', False):
        raise AssertionError('resident reference requires ZeRO-1, never ZeRO-2/3')
    if getattr(engine.optimizer, 'cpu_offload', False):
        raise AssertionError('resident reference optimizer unexpectedly uses CPU offload')
    accumulation = getattr(engine.optimizer, 'gradient_accumulation_dtype',
                           getattr(engine.optimizer, 'grad_acc_dtype', None))
    if accumulation != torch.float32:
        raise AssertionError('resident reference must accumulate logical B1 gradients in FP32')
    masters = zero1_master_partitions(engine)
    if any(p.dtype != torch.float32 or p.device.type != 'cuda' for p in masters):
        raise AssertionError('resident master partitions must remain FP32 on GPU')
    for name, p in engine.module.named_parameters():
        if p.device.type != 'cuda' or hasattr(p, 'device_mesh') or hasattr(p, 'ds_tensor'):
            raise AssertionError(f'{component}/{name}: parameters are not fully GPU resident')
        if p.dtype != torch.bfloat16:
            raise AssertionError(f'{component}/{name}: original FSDP BF16 compute contract changed')
    alias = not hasattr(engine.optimizer, 'single_partition_of_fp32_groups')
    if alias:
        # Receipt-backed, process-local restore ABI only. The original loader
        # writes these actual tensors; no optimizer method/algorithm is replaced.
        engine.optimizer.single_partition_of_fp32_groups = masters
    try:
        restore = restore_zero1_optimizer_partition_from_logical(engine, package,
            component=component, partition_rank=torch.distributed.get_rank(), dp_world_size=8)
    finally:
        if alias:
            del engine.optimizer.single_partition_of_fp32_groups
    # Registry only, not a Module child: registering engine here would recurse.
    object.__setattr__(module, '_distillschedule_resident_engine_ref', weakref.ref(engine))
    engine._distillschedule_resident_zero1 = True
    return engine, dict(restore, backend='FULL_PARAMETER_GPU_RESIDENT_ZERO1',
        optimizer_cpu_offload=False, parameter_offload=False, parameter_streaming=False,
        optimizer_implementation='PYTORCH_ADAMW', trainable_master_dtype='torch.float32',
        model_compute_dtype='torch.bfloat16', communication_dtype='torch.float32',
        gradient_accumulation_dtype='torch.float32', actual_optimizer_class=type(engine.optimizer).__name__,
        initialization_api_compatibility=route_audit,
        physical_b1_rows_per_rank=2, loss_scale_per_row=0.5, config=config,
        optimizer_dispatch_warmup={'enabled': False, 'disabled_reason': 'REFERENCE'})


def zero1_gradient_tensors(engine: Any) -> list[tuple[str, Any]]:
    """Actual reduced GPU partitions, not absent BF16 parameter.grad views."""
    import torch
    rows = []
    zero = engine.optimizer
    if type(zero).__name__ == 'BF16_Optimizer':
        groups = [(i, [t for t, active in zip(values, zero.fp32_groups_has_gradients[i], strict=True) if active])
                  for i, values in enumerate(zero.fp32_groups_gradients)]
    else:
        groups = zero.averaged_gradients.items()
    for group, values in groups:
        for index, tensor in enumerate(values or ()):
            if not torch.is_tensor(tensor) or tensor.device.type != 'cuda':
                raise AssertionError('ZeRO-1 gradient audit found non-GPU/missing data')
            rows.append((f'zero1_group{group}/gradient{index}', tensor))
    if not rows:
        raise AssertionError('ZeRO-1 pre-commit gradient partition audit is empty')
    return rows


def zero1_optimizer_tensors(engine: Any) -> list[tuple[str, Any]]:
    import torch
    zero = engine.optimizer
    rows = []
    for group, master in enumerate(zero1_master_partitions(engine)):
        if master.dtype != torch.float32 or master.device.type != 'cuda':
            raise AssertionError('ZeRO-1 master must remain FP32 on GPU')
        state = zero.optimizer.state[master]
        for key in ('exp_avg', 'exp_avg_sq'):
            tensor = state[key]
            if tensor.device.type != 'cuda' or tensor.dtype != torch.float32:
                raise AssertionError('ZeRO-1 Adam state was offloaded or changed precision')
            rows.append((f'zero1_group{group}/{key}', tensor))
    return rows


def local_ema_shadow(tensors: list[Any], *, partition_rank: int,
                     partition_numel: int, device: Any) -> Any:
    """Construction-only gather of the owning contiguous EMA slice."""
    import torch
    if not tensors or len({x.dtype for x in tensors}) != 1:
        raise ValueError('EMA requires a nonempty uniform original checkpoint dtype')
    out = torch.zeros(partition_numel, dtype=tensors[0].dtype, device=device)
    lower = partition_rank * partition_numel
    upper = lower + partition_numel
    offset = 0
    for tensor in tensors:
        flat = tensor.reshape(-1)
        start, stop = max(lower, offset), min(upper, offset + flat.numel())
        if stop > start:
            out[start-lower:stop-lower].copy_(flat[start-offset:stop-offset].to(device))
        offset += flat.numel()
    if lower >= offset:
        raise ValueError('EMA partition lies entirely outside logical state')
    return out


class ResidentZero1ShardEMA:
    """Original EMA recurrence over GPU FP32-master slices; no collectives."""
    @classmethod
    def load_from_logical(cls, module: Any, logical_root: Path, *,
                          device: str = 'cuda', nvtx_enabled: bool = False):
        import torch
        if device != 'cuda' or nvtx_enabled:
            raise ValueError('resident reference is GPU EMA and unprofiled only')
        engine = module._distillschedule_resident_engine_ref()
        if engine is None:
            raise RuntimeError('EMA lost its owning resident engine')
        candidates = [logical_root / name for name in ('ema', 'model_ema')]
        root = next((p for p in candidates if (p/'sparse_ema_manifest.json').is_file()), None)
        if root is None:
            raise FileNotFoundError('logical sparse EMA missing')
        metadata = json.loads((root/'sparse_ema_manifest.json').read_text())
        if metadata['schema_version'] != 'distillschedule.sparse_ema.v1':
            raise ValueError('EMA checkpoint schema changed')
        trainables = dict((n,p) for n,p in module.named_parameters() if p.requires_grad)
        loaded = {}
        for shard in metadata['shards']:
            loaded.update(torch.load(root/shard['file'], map_location='cpu', weights_only=True))
        normalized = {n[7:] if n.startswith('module.') and n[7:] in trainables else n: n
                      for n in metadata['tracked_names']}
        if set(normalized) != set(trainables):
            raise ValueError('EMA and logical trainable inventory differ')
        masters = zero1_master_partitions(engine)
        if len(masters) != 1:
            raise ValueError('registered logical restore requires one optimizer group')
        self = cls()
        self.source = masters[0]
        self.shadow = local_ema_shadow([loaded[normalized[n]] for n in trainables],
            partition_rank=torch.distributed.get_rank(), partition_numel=self.source.numel(),
            device=self.source.device)
        self.valid_numel = max(0, min(self.source.numel(), sum(p.numel() for p in trainables.values())
            - torch.distributed.get_rank()*self.source.numel()))
        self.tracked_count = len(trainables)
        self.device = 'cuda'
        self.decay = float(metadata['decay'])
        self.min_decay = 0.0
        self.update_after_step = int(metadata['update_after_step'])
        self.optimization_step = int(metadata['optimization_step'])
        self.cur_decay_value = metadata.get('cur_decay_value')
        self.last_transfer_bytes = 0
        return self

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step-self.update_after_step-1)
        return 0.0 if step <= 0 else max(self.min_decay, min((1+step)/(10+step), self.decay))

    def step(self) -> float:
        from .fsdp2_static import ema_update_reference
        self.optimization_step += 1
        self.cur_decay_value = self.get_decay(self.optimization_step)
        ema_update_reference(self.shadow, self.source.detach().to(self.shadow.dtype), self.cur_decay_value)
        self.last_transfer_bytes = 0
        return self.cur_decay_value

    def checksum(self) -> dict[str, Any]:
        value = self.shadow[:self.valid_numel].double()
        return dict(tracked_tensor_count=self.tracked_count, local_numel=self.valid_numel,
                    local_sum=float(value.sum()), local_norm=float(value.square().sum().sqrt()),
                    optimization_step=self.optimization_step, decay=self.cur_decay_value,
                    communication_per_update=False, device='cuda')
