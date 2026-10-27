"""Official Self-Forcing FSDP2 placement in the public native-cycle lifecycle.

The upstream Trainer's actual fwdbwd method and SelfForcingOwner's update loop
remain in charge. Parameters and Adam moments are real DTensors, serialized by
the common same-topology local-shard checkpoint ABI, not gathered replicas.
No GPU or quality qualification of this integration is implied by provenance.
"""
from copy import deepcopy
from contextlib import contextmanager
import json
import os

from .self_forcing_owner import SelfForcingOwner
from .self_forcing_source import require
from .official_profiles import resolve, verify_sources

PROFILE = 'self-forcing-wan-fsdp2-r6-v1'
BACKEND = 'fsdp2-native-bf16-v1'
SPEC_SCHEMA = 'distillschedule.native_sharded_owner.v1'


def requested(parameters):
    return 'optimization_profile' in parameters or parameters.get('native_role_backend') == BACKEND


def validate_environment():
    """Require the native launch policy without rewriting the process environment."""
    for name, value in resolve(PROFILE)['environment'].items():
        require(os.environ.get(name) == value,
                f'native FSDP2 requires explicit {name}={value} before process-group initialization')


def admit(parameters):
    import torch
    from .single_device import admit_fp32
    profile = parameters.get('optimization_profile')
    require(profile is None or profile == PROFILE, 'unknown official Self-Forcing optimization profile')
    require(parameters.get('native_role_backend', BACKEND) == BACKEND,
            'native FSDP2 profile backend changed')
    validate_environment()
    require(str(parameters.get('device', 'cpu')).startswith('cuda')
            and torch.distributed.is_initialized() and torch.distributed.get_world_size() == 16,
            'native FSDP2 profile requires real CUDA world16 ownership')
    device = admit_fp32(parameters, require=require, allow_replicated=True)
    require('A100' in torch.cuda.get_device_name(device)
            and str(torch.__version__).split('+')[0] == '2.10.0',
            'native FSDP2 profile requires source-qualified A100/Torch2.10')
    require(parameters.get('native_cache_lifetime', 'native-v1') == 'native-v1',
            'native FSDP2 owns the full early-C/post-commit-G cache lifecycle; no second release policy')
    verify_sources(PROFILE)
    return device


def validate_settings(args):
    require(args.mixed_precision is True and args.gradient_checkpointing is True
            and args.image_or_video_shape == [1, 21, 16, 46, 80]
            and args.dfake_gen_update_ratio == 5 and args.batch_size == 1
            and args.ema_weight == 0 and args.num_frame_per_block == 3
            and args.denoising_step_list == [1000, 750, 500, 250],
            'native FSDP2 profile shape/update/checkpoint/precision/EMA contract changed')


def place_roles(children, api):
    """Shard CPU-loaded weights before any full 14B role can be moved to CUDA."""
    validate_environment()
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from .official_optimizations.bound_source import SourceBoundHelpers
    helper = SourceBoundHelpers(api.source)
    backend = helper.module('sf_fsdp2_layout_backend')
    mesh = init_device_mesh('cuda', (16,), mesh_dim_names=('fsdp',))
    require(mesh.get_group() is dist.group.WORLD, 'native FSDP2 must use the declared world mesh')
    for name in ('generator', 'real_score', 'fake_score'):
        root = children[name]
        backend.shard_module(root, backend.role_blocks(root, name), mesh,
                             mixed_precision=True, role=name)
    abi = helper.module('sf_fsdp2_mutable_cache_abi')
    handle, record = abi.install_native_mutable_cache_abi(children['generator'])
    children['generator']._sf_mutable_kv_handle = handle
    children['generator']._sf_fsdp2_receipt['mutable_kv_abi'] = record
    return mesh


class SelfForcingFSDP2Owner(SelfForcingOwner):
    def __init__(self, model, args, source, negative_embedding, *, identity):
        self.optimization_profile = identity.get('optimization_profile', {}).get('id')
        self._optimized_runtime = None
        self._last_optimization_receipt = None
        self._collective_prepared = False
        self._role_forwards = dict(generator=0, real_score=0, fake_score=0)
        self._forward_handles = []
        self._reference_attention = None
        super().__init__(model, args, source, negative_embedding, identity=identity)

    @property
    def generator_optimizer(self):
        return self.optimizers['generator']

    @property
    def critic_optimizer(self):
        return self.optimizers['critic']

    def _validate_placement(self):
        validate_environment()
        import torch
        import torch.distributed as dist
        from torch.distributed.fsdp import FSDPModule
        from torch.distributed.tensor import DTensor
        validate_settings(self.args)
        if self.optimization_profile is not None:
            require(self.identity.get('optimization_profile') == resolve(PROFILE),
                    'native FSDP2 optimized runtime/source profile changed')
        require(dist.is_initialized() and dist.get_world_size() == 16
                and str(dist.get_backend()) == 'nccl' and self.device.type == 'cuda'
                and torch.cuda.current_device() == self.device.index,
                'native FSDP2 group or local device ownership changed')
        require(self.identity['placement'] == dict(device='cuda', layout='NATIVE_FSDP2_FULL16',
                distributed=True, amp=False) and self.dtype == self.model.dtype == torch.bfloat16
                and self.model.device == self.device and self.model.args is self.args,
                'native FSDP2 placement/precision identity changed')
        require(not torch.is_autocast_enabled('cuda'), 'native FSDP2 forbids outer AMP')
        for role in ('generator', 'real_score', 'fake_score'):
            root = getattr(self.model, role)
            require(isinstance(root, FSDPModule) and all(isinstance(p, DTensor)
                    and p.dtype == torch.float32 and p.device == self.device
                    and p.device_mesh.size() == 16 and p.device_mesh.mesh_dim_names == ('fsdp',)
                    for p in root.parameters()), 'native FSDP2 role/master/shard ownership changed: ' + role)
        require(all(not t.is_floating_point() or t.dtype == torch.float32
                    for t in (*self.model.parameters(), *self.model.buffers()))
                and all(t.device == self.device for t in self.runtime_tensors())
                and self.negative_embedding.dtype == torch.float32,
                'native FSDP2 master/buffer/condition placement changed')

    def _checkpoint_unconditional(self, negative_cpu):
        import torch
        return {'prompt_embeds': negative_cpu.repeat(self.args.batch_size, 1, 1).to(torch.bfloat16)}

    def _validate_unconditional(self):
        import torch
        expected = self._checkpoint_unconditional(self.negative_embedding)['prompt_embeds']
        require(set(self.unconditional_dict) == {'prompt_embeds'}
                and self.unconditional_dict['prompt_embeds'].device == self.device
                and torch.equal(self.unconditional_dict['prompt_embeds'], expected)
                and self.unconditional_dict['prompt_embeds'].dtype == torch.bfloat16,
                'native FSDP2 unconditional cache differs from exact BF16 encoded negative prompt')

    def _validate_checkpoint_tensors(self, value):
        import torch
        from torch.distributed.tensor import DTensor
        from torch.utils._pytree import tree_leaves
        from .tensor_state import layout
        mesh = next(self.model.generator.parameters()).device_mesh
        expected_mesh, expected_names = mesh.mesh.tolist(), list(mesh.mesh_dim_names)
        for tensor in tree_leaves(value):
            if not isinstance(tensor, torch.Tensor):
                continue
            require(type(tensor) in (torch.Tensor, DTensor) and tensor.layout == torch.strided
                    and not tensor.requires_grad and (tensor.device.type == 'cpu' or tensor.device == self.device),
                    'native FSDP2 checkpoint needs dense owned-device or CPU local shards')
            if isinstance(tensor, DTensor):
                info = layout(tensor)
                require(info['mesh'] == expected_mesh and info['mesh_dim_names'] == expected_names,
                        'native FSDP2 checkpoint shard topology changed')

    def __deepcopy__(self, memo):
        raise ValueError('native FSDP2 owner cannot be cloned for calibration')

    def admit_execution(self, mode, implementation):
        from distillschedule.open_workload.collective_owner import implementation as admit_implementation
        require(mode in ('reference', 'manual'), 'native FSDP2 owner auto/plan requires separate qualification')
        if mode == 'reference':
            require(not implementation, 'reference native FSDP2 cannot override its physical implementation')
        else:
            admit_implementation(implementation)

    def collective_execution_spec(self):
        self._validate_live()
        return dict(schema_version=SPEC_SCHEMA, backend='fsdp2', world_size=16, device_type='cuda',
                    shared_identity=dict(identity=self.identity, source=self.source.identity))

    def prepare_collective_execution(self):
        self._validate_live()
        require(not self._collective_prepared, 'native FSDP2 owner must be prepared exactly once')
        for role in self._role_forwards:
            def observe(module, args, role=role):
                self._role_forwards[role] += 1
            self._forward_handles.append(getattr(self.model, role).register_forward_pre_hook(observe))
        self._collective_prepared = True
        if self.optimization_profile is None:
            self._reference_attention = NativeAttentionScope(self.source)

    def validate_collective_execution(self):
        import torch.distributed as dist
        self._validate_live()
        roles = {name: dict(forward_calls=count, optimizer_commits=(
            self.generator_updates if name == 'generator' else self.critic_updates if name == 'fake_score' else 0))
                 for name, count in self._role_forwards.items()}
        value = dict(world_size=16, rank=dist.get_rank(), roles=roles,
            zero_gradients=True, bucket_storage_bound_bytes=0,
            storage_note='FSDP_EXECUTION_WEIGHTS_DRAINED_AT_BOUNDARY; NOT_ZERO_INFLIGHT_WORKSPACE',
            forward_counter_scope='CURRENT_PROCESS_NATIVE_ROLE_HOOKS',
            optimizer_counter_scope='PERSISTENT_NATIVE_TRAINING_CLOCK',
            precision='FSDP2_BF16_COMPUTE_FP32_MASTER_REDUCE', numerical_status='REVIEW', gpu='NOT_RUN')
        if self._last_optimization_receipt is not None:
            value['optimization_effects'] = deepcopy(self._last_optimization_receipt)
        return value

    def run(self, feeds, *, cycle):
        from .self_forcing_source import original
        from .official_optimizations.self_forcing import SelfForcingOptimizedRuntime
        require(self._collective_prepared, 'native FSDP2 must be prepared before its first update')
        if self.optimization_profile is not None and self._optimized_runtime is None:
            self._optimized_runtime = SelfForcingOptimizedRuntime(self, original(self.source.root))
        try:
            if self._optimized_runtime is not None:
                self._optimized_runtime.begin_cycle(cycle)
            value = super().run(feeds, cycle=cycle)
            if self._optimized_runtime is not None:
                value['optimization_effects'] = deepcopy(self._last_optimization_receipt)
            return value
        except BaseException:
            self._resident_trial_poisoned = True
            if self._optimized_runtime is not None:
                self._optimized_runtime.abort()
            raise

    def _before_pipeline_discard(self, cycle):
        if self._optimized_runtime is not None:
            self._last_optimization_receipt = json.loads(json.dumps(
                self._optimized_runtime.finish_cycle(cycle), allow_nan=False))

    def validate_checkpoint_progress(self, value, progress):
        cycle = value['native_step'] // self.ratio
        require(progress == dict(cycle=cycle, cursor=cycle * 16, versions={'native_cycle': cycle}),
                'native FSDP2 inner clocks and rank-strided Program progress disagree')

    @contextmanager
    def inference_scope(self):
        """Checkpoint evaluation uses native FA2, without training replay state."""
        self._validate_live()
        require(self._collective_prepared, 'prepare native ownership before checkpoint evaluation')
        if self._reference_attention is not None or self._optimized_runtime is not None:
            yield
        else:
            scope = NativeAttentionScope(self.source)
            try:
                yield
            finally:
                scope.close()

    def close(self):
        if self._optimized_runtime is not None:
            self._optimized_runtime.close()
        if self._reference_attention is not None:
            self._reference_attention.close()
        for handle in self._forward_handles:
            handle.remove()
        self._forward_handles.clear()
        handle = getattr(self.model.generator, '_sf_mutable_kv_handle', None)
        if handle is not None:
            handle.remove()


class NativeAttentionScope:
    """Reference native FA2, without any R6 scheduling/cache/kernel helpers."""
    def __init__(self, source):
        from .self_forcing_source import original
        from .official_optimizations.self_forcing import _ACTIVE_SOURCE_PATCH
        self.api = original(source.root)
        require(self.api.source is source, 'native attention source ownership changed')
        attention = source.import_module('wan.modules.attention')
        require(attention.FLASH_ATTN_2_AVAILABLE and not attention.FLASH_ATTN_3_AVAILABLE,
                'source-qualified native A100 FA2 reference required; no SDPA fallback')
        require(_ACTIVE_SOURCE_PATCH.acquire(blocking=False), 'another native SF owner is still active')
        self.saved = self.api.wan.flash_attention, self.api.causal.attention
        self.closed = False
        self.api.wan.flash_attention, self.api.causal.attention = attention.flash_attention, attention.attention

    def close(self):
        from .official_optimizations.self_forcing import _ACTIVE_SOURCE_PATCH
        if not self.closed:
            self.api.wan.flash_attention, self.api.causal.attention = self.saved
            self.closed = True
            _ACTIVE_SOURCE_PATCH.release()
