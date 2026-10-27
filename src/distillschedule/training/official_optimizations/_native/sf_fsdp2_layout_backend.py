"""Measured implementation helper, relocated from r2_fsdp2_baselines/scripts/sf_fsdp2_layout_backend.py.

The public owner validates capabilities and lifecycle; this module does not
launch experiments or claim framework GPU acceptance. Source lock records all
original and relocated executable definitions.
"""
from contextlib import contextmanager
import hashlib
import inspect
import os
from pathlib import Path
import types
ROLES = ('generator', 'real_score', 'fake_score', 'text_encoder')
SCHEMA = 'dmd.official_sf_fsdp2_or_hsdp2_backend.v3'

def require(ok, why):
    if not ok:
        raise ValueError(why)

def role_blocks(module, role):
    """Explicit audited module boundaries; do not silently fall back to root-only."""
    require(role in ROLES, 'UNKNOWN_NATIVE_ROLE')
    if role == 'text_encoder':
        model = module.text_encoder
        expected = ('T5Encoder', 24, 4096)
    else:
        model = module.model
        expected = ('CausalWanModel', 30, 1536) if role == 'generator' else ('WanModel', 40, 5120) if role == 'real_score' else ('WanModel', 30, 1536)
    require((type(model).__name__, len(model.blocks), model.dim) == expected, 'UNEXPECTED_NATIVE_MODEL:' + role)
    blocks = list(model.blocks)
    require(len({id(b) for b in blocks}) == len(blocks), 'ALIASED_NATIVE_BLOCK')
    trainable = role in ('generator', 'fake_score')
    require(all((p.requires_grad == trainable for p in module.parameters())), 'TRAINABILITY_DRIFT:' + role)
    if trainable:
        require(sum((p.numel() for p in module.parameters())) == 1418996800, 'TRAINABLE_PARAMETER_COUNT')
    return blocks

def clip_grad_norm(module, max_norm, norm_type=2.0):
    import torch
    from torch.distributed.tensor import DTensor
    value = torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm, norm_type=norm_type)
    module._sf_fsdp2_receipt['clip_calls'] += 1
    return value.full_tensor() if isinstance(value, DTensor) else value

def shard_module(module, blocks, mesh, *, mixed_precision, role):
    """The same production primitive is exercised on real CPU FSDP2 in tests."""
    import torch
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import DTensor
    require(not any((isinstance(m, FSDPModule) for m in module.modules())), 'ALREADY_SHARDED')
    require(mesh.ndim in (1, 2) and mesh.size() > 1, 'SUPPORTED_SHARD_MESH_REQUIRED')
    require(all((p.dtype == torch.float32 for p in module.parameters())), 'FP32_MASTER_WEIGHTS_REQUIRED')
    require(all((not b.is_floating_point() or b.dtype == torch.float32 for b in module.buffers())), 'FP32_NATIVE_BUFFERS_REQUIRED')
    names = {id(m): n for n, m in module.named_modules()}
    require(len({id(b) for b in blocks}) == len(blocks) and all((id(b) in names for b in blocks)), 'INVALID_BLOCK_BOUNDARIES')
    members = [{id(p) for p in b.parameters()} for b in blocks]
    require(sum(map(len, members)) == len(set().union(*members)), 'OVERLAPPING_BLOCK_PARAMETERS')
    before = [(n, tuple(p.shape), p.dtype, p.requires_grad) for n, p in module.named_parameters()]
    group_numel = [sum((p.numel() for p in b.parameters())) for b in blocks]
    total = sum((p.numel() for p in module.parameters()))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16 if mixed_precision else None, reduce_dtype=torch.float32, output_dtype=None, cast_forward_inputs=False)
    for block in blocks:
        fully_shard(block, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    fully_shard(module, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    after = [(n, tuple(p.shape), p.dtype, p.requires_grad) for n, p in module.named_parameters()]
    require(before == after, 'PARAMETER_METADATA_CHANGED')
    require(all((isinstance(p, DTensor) and p.device_mesh == mesh for p in module.parameters())), 'PARAMETERS_NOT_ACTUALLY_SHARDED')
    rec = dict(schema=SCHEMA, role=role, backend='FSDP2', shard_world_size=mesh.shape[-1], replicate_world_size=mesh.shape[0] if mesh.ndim == 2 else 1, global_world_size=mesh.size(), device_type=mesh.device_type, block_paths=[names[id(b)] for b in blocks], fsdp_units=sum((isinstance(m, FSDPModule) for m in module.modules())), group_numel=group_numel + [total - sum(group_numel)], global_parameter_numel=total, local_parameter_numel=sum((p.to_local().numel() for p in module.parameters())), master_dtype='torch.float32', param_dtype='torch.bfloat16' if mixed_precision else None, reduce_dtype='torch.float32', buffer_dtype='torch.float32', cast_forward_inputs=False, reshard_after_forward=True, gradient_reduction='MEAN_HYBRID_SHARD_AND_REPLICATE' if mesh.ndim == 2 else 'MEAN_ON_FULL_SHARD_MESH', cpu_offload=False, clip_calls=0)
    module._sf_fsdp2_receipt = rec
    module.clip_grad_norm_ = types.MethodType(clip_grad_norm, module)
    return module

def full_cpu_parameters(module):
    import torch
    from torch.distributed.tensor import DTensor
    with torch.no_grad():
        for name, param in module.named_parameters():
            full = param.full_tensor() if isinstance(param, DTensor) else param
            yield (name, full.detach().float().cpu().clone())
            del full

class EMAFSDP2:
    """Same FP32 CPU EMA formula; one parameter gathered at a time, not a new EMA rule."""

    def __init__(self, module, decay=0.999):
        self.decay = decay
        self.shadow = dict(full_cpu_parameters(module))

    def update(self, module):
        for name, value in full_cpu_parameters(module):
            self.shadow[name].mul_(self.decay).add_(value, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

@contextmanager
def native_fsdp2_bindings(trainer_class, source_root, *, layout='full16'):
    """Scoped constructor binding; restoration on error; no upstream source edits."""
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    scope = trainer_class.__init__.__globals__
    original_wrap, original_ema = (scope['fsdp_wrap'], scope['EMA_FSDP'])
    source = Path(inspect.getsourcefile(original_wrap)).resolve()
    require(source == Path(source_root).resolve() / 'utils/distributed.py', 'NATIVE_FACTORY_SOURCE_DRIFT')
    records = []
    mesh = None

    def wrap(module, **kwargs):
        nonlocal mesh
        require(len(records) < 4, 'EXTRA_NATIVE_ROLE')
        require(kwargs.get('sharding_strategy') == 'full' and kwargs.get('wrap_strategy') == 'size' and (kwargs.get('mixed_precision') is True) and (not kwargs.get('cpu_offload', False)) and (set(kwargs) <= {'sharding_strategy', 'wrap_strategy', 'mixed_precision', 'cpu_offload'}), 'UNSUPPORTED_NATIVE_FSDP2_POLICY')
        require(dist.is_initialized() and dist.get_world_size() == 16, 'WORLD16_REQUIRED')
        os.environ['NCCL_CROSS_NIC'] = '1'
        if mesh is None:
            require(layout in ('full16', 'node-hsdp'), 'UNKNOWN_LAYOUT')
            mesh = init_device_mesh('cuda', (16,) if layout == 'full16' else (2, 8), mesh_dim_names=('fsdp',) if layout == 'full16' else ('replicate', 'shard'))
            if layout == 'full16':
                require(mesh.get_group() is dist.group.WORLD, 'UNEXPECTED_EXTRA_PROCESS_GROUP')
        trainer_class._sf_active_mesh = mesh
        role = ROLES[len(records)]
        blocks = role_blocks(module, role)
        out = shard_module(module, blocks, mesh, mixed_precision=True, role=role)
        if role == 'generator':
            from .sf_fsdp2_mutable_cache_abi import install_native_mutable_cache_abi
            handle, abi = install_native_mutable_cache_abi(out)
            out._sf_mutable_kv_handle = handle
            out._sf_fsdp2_receipt['mutable_kv_abi'] = abi
        records.append(out._sf_fsdp2_receipt)
        return out
    scope['fsdp_wrap'], scope['EMA_FSDP'] = (wrap, EMAFSDP2)
    try:
        yield dict(schema=SCHEMA, factory_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), calls=records, replacement_scope='FSDP_CONSTRUCTION_CLIP_EMA_AND_MUTABLE_KV_ABI')
        require([r['role'] for r in records] == list(ROLES), 'NATIVE_ROLE_WRAP_CENSUS')
    finally:
        require(scope['fsdp_wrap'] is wrap and scope['EMA_FSDP'] is EMAFSDP2, 'CONCURRENT_FACTORY_MUTATION')
        scope['fsdp_wrap'], scope['EMA_FSDP'] = (original_wrap, original_ema)
