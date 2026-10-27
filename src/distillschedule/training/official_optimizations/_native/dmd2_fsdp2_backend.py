"""Exact measured FSDP2 placement and classifier ABI; benchmark loop excluded."""
import ast
import copy
import hashlib

def require(condition, message):
    if not condition:
        raise ValueError('DMD2_FSDP2:' + message)

def shard_tree(module, mesh, *, role, synchronize=True):
    import torch
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import DTensor
    require(not any(isinstance(m, FSDPModule) for m in module.modules()), 'ALREADY_SHARDED')
    require(mesh.ndim == 1 and mesh.size() > 1, 'SHARD_MESH')
    # Match Accelerator.prepare_model's native device-placement step before NCCL.
    # This preserves original parameter values/dtypes and consumes no RNG.
    target = torch.device(mesh.device_type, torch.cuda.current_device()) if mesh.device_type == 'cuda' else torch.device(mesh.device_type)
    original_dtypes = [(n, p.dtype) for n, p in module.named_parameters()]
    module.to(target)
    require(original_dtypes == [(n, p.dtype) for n, p in module.named_parameters()], 'PLACEMENT_CHANGED_DTYPE')
    require(all(p.device == target for p in module.parameters()), 'PARAMETER_DEVICE_PLACEMENT')
    before = [(n, tuple(p.shape), p.dtype, p.requires_grad) for n, p in module.named_parameters()]
    if synchronize:
        # DDP synchronizes these same two role trees from rank0 on construction.
        with torch.no_grad():
            for tensor in list(module.parameters()) + list(module.buffers()):
                torch.distributed.broadcast(tensor, src=0, group=mesh.get_group())
    policy = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=None,
                                  output_dtype=None, cast_forward_inputs=False)
    names = {id(m): name for name, m in module.named_modules()}
    # Neither selected class contains the other. These are Diffusers' native
    # checkpoint execution boundaries, not a new checkpoint/recompute policy.
    blocks = [m for m in module.modules() if type(m).__name__ in
              ('ResnetBlock2D', 'Transformer2DModel')]
    ids = [{id(p) for p in m.parameters()} for m in blocks]
    require(sum(map(len, ids)) == len(set().union(*ids)) if ids else True,
            'OVERLAPPING_BLOCK_OWNERS')
    for block in blocks:
        fully_shard(block, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    # Frozen Teacher and the complete Fake UNet have their own root leftovers.
    unets = [m for m in module.modules() if type(m).__name__ == 'UNet2DConditionModel']
    for unet in unets:
        fully_shard(unet, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    if not isinstance(module, FSDPModule):
        fully_shard(module, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    after = [(n, tuple(p.shape), p.dtype, p.requires_grad) for n, p in module.named_parameters()]
    require(before == after, 'PARAMETER_METADATA_CHANGED')
    require(all(isinstance(p, DTensor) and p.device_mesh == mesh for p in module.parameters()),
            'UNSHARDED_PARAMETER')
    receipt = dict(backend='FSDP2', role=role, mesh_size=mesh.size(),
                   reshard_after_forward=True, cpu_offload=False,
                   mixed_precision_policy='PRESERVE_NATIVE_DTYPES_AND_AUTOCAST',
                   global_numel=sum(p.numel() for p in module.parameters()),
                   local_numel=sum(p.to_local().numel() for p in module.parameters()),
                   fsdp_units=sum(isinstance(m, FSDPModule) for m in module.modules()),
                   block_paths=[names[id(m)] for m in blocks])
    module._dmd2_fsdp2_receipt = receipt
    return receipt

def bind_classifier_call(module):
    """Route a direct native .forward through the existing FSDP2 module hooks.

    The native classifier math and arguments are exactly retained. This is
    applied to reference and optimized arms for the same observable census.
    """
    import inspect
    source = inspect.getsource(module.SDGuidance.compute_cls_logits)
    import textwrap
    tree = ast.parse(textwrap.dedent(source))
    original = copy.deepcopy(tree)
    sites = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'forward'
             and ast.unparse(node.func.value) == 'self.fake_unet']
    require(len(sites) == 1, 'CLASSIFIER_NATIVE_FORWARD_SITE')
    sites[0].func = sites[0].func.value
    restored = copy.deepcopy(tree)
    call = next(node for node in ast.walk(restored) if isinstance(node, ast.Call)
                and ast.unparse(node.func) == 'self.fake_unet')
    call.func = ast.Attribute(value=call.func, attr='forward', ctx=ast.Load())
    require(ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False),
            'CLASSIFIER_MATH_CHANGED')
    namespace = dict(module.SDGuidance.compute_cls_logits.__globals__)
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(module.SDGuidance), 'exec'), namespace)
    module.SDGuidance.compute_cls_logits = namespace['compute_cls_logits']
    return dict(scope='DIRECT_FORWARD_TO_MODULE_CALL_ONLY', reversible_native_ast=True,
                source_sha256=hashlib.sha256(source.encode()).hexdigest())
