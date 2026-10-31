"""Source-bound SD1.5 FSDP2 reference in the common native-cycle lifecycle.

Only native placement and a direct-forward module-hook ABI are imported from
the measured baseline. The existing DMD2Owner still owns every loss, update,
RNG stream, scheduler and sample. Checkpoints are same-topology local shards.
"""
from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

from .dmd2_collective import DMD2CollectiveOwner
from .dmd2_owner import DMD2Owner
from .sd import require

BACKEND = 'fsdp2-native-sd15-v1'


def source_identity():
    """Source metadata verification; no CUDA or helper source execution."""
    root = Path(__file__).resolve().parent
    value = json.loads((root / 'dmd2_fsdp_source.json').read_text())
    payload = (root / value['target']).read_bytes()
    require(value['schema_version'] == 'distillschedule.dmd2_fsdp_reference_source.v1'
            and hashlib.sha256(payload).hexdigest() == value['integrated_sha256'],
            'FSDP2 backend source changed')
    from .official_profiles import _definition_sha256
    definitions = {node.name: _definition_sha256(node, set())
                   for node in ast.parse(payload).body if isinstance(node, ast.FunctionDef)}
    require(definitions == value['definitions'] and value['benchmark_loop_imported'] is False,
            'FSDP2 backend executable closure changed')
    return value


def shard_tree(module, mesh, *, role, synchronize=True):
    source_identity()
    from .official_optimizations._native.dmd2_fsdp2_backend import shard_tree as native
    return native(module, mesh, role=role, synchronize=synchronize)


def local_finite(value):
    import torch
    from torch.distributed.tensor import DTensor
    from torch.utils._pytree import tree_leaves
    return all(bool(torch.isfinite(t.to_local() if isinstance(t, DTensor) else t).all())
               for t in tree_leaves(value) if isinstance(t, torch.Tensor) and t.is_floating_point())


def admit(parameters, *, sdxl):
    """Fail closed on source-qualified ownership before loading any asset."""
    import torch
    import torch.distributed as dist
    require(not sdxl and 'optimization_profile' not in parameters,
            'FSDP2 reference is SD1.5-only and cannot enable the resident optimization bundle')
    require(str(parameters.get('device', 'cpu')).startswith('cuda')
            and dist.is_initialized() and dist.get_world_size() == 16,
            'FSDP2 reference requires actual CUDA world16 ownership')
    from .single_device import admit_fp32
    device = admit_fp32(parameters, require=require, allow_replicated=True)
    require('A100' in torch.cuda.get_device_name(device)
            and str(torch.__version__).split('+')[0] == '2.10.0',
            'FSDP2 reference requires source-qualified A100/Torch2.10')
    from .sd import runtime_identity
    runtime = runtime_identity()
    require(all(runtime[name] == expected for name, expected in
                dict(diffusers='0.37.1', transformers='4.51.3', peft='0.17.1').items()),
            'FSDP2 reference numerical dependencies changed')
    require(parameters.get('native_checkpoint_policy', 'coupled-v1') == 'coupled-v1',
            'FSDP2 reference does not admit singleton checkpoint overrides')
    source_identity()
    return device


def validate_settings(args):
    # Equal native work and precision to the existing optimized profile, but
    # NONE of its activation/cast/classifier packing or fused Adam mechanisms.
    from .official_optimizations.dmd2 import validate_settings as native_settings
    native_settings(args)


class ClassifierCallScope:
    """Route one owner's native direct .forward through its FSDP pre-hooks.

    The exact reversible measured AST transform is reused, but its temporary
    class is private: another owner or the pinned upstream class is untouched.
    """
    def __init__(self, guidance):
        source_identity()
        require(not getattr(guidance, '_dmd2_fsdp_classifier_scope', False),
                'FSDP2 classifier scope already installed')
        from .official_optimizations._native.dmd2_fsdp2_backend import bind_classifier_call
        class OwnerClass(type(guidance)):
            pass
        OwnerClass.compute_cls_logits = type(guidance).compute_cls_logits
        self.receipt = bind_classifier_call(SimpleNamespace(SDGuidance=OwnerClass))
        self.guidance, self.saved = guidance, vars(guidance).get('compute_cls_logits')
        self.present, self.closed = 'compute_cls_logits' in vars(guidance), False
        self.installed = MethodType(OwnerClass.compute_cls_logits, guidance)
        guidance.compute_cls_logits = self.installed
        guidance._dmd2_fsdp_classifier_scope = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.closed:
            return
        if self.present:
            self.guidance.compute_cls_logits = self.saved
        else:
            delattr(self.guidance, 'compute_cls_logits')
        delattr(self.guidance, '_dmd2_fsdp_classifier_scope')
        self.closed = True


def place_roles(model, api, mesh=None):
    """Real role sharding after all-rank construction admission, never DDP-on-FSDP."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from .distributed import gather
    require(type(model) is api.unified.SDUniModel and type(model.guidance_model) is api.guidance.SDGuidance,
            'FSDP2 role placement requires actual source-owned SD1.5 numerical classes')
    if mesh is None:
        mesh = init_device_mesh('cuda', (16,), mesh_dim_names=('fsdp',))
    require(mesh.get_group() is dist.group.WORLD, 'FSDP2 roles must use the original world group')
    # Native Trainer seeds seed+rank then DDP broadcasts both complete role
    # trees. Preserve that transition, including random classifier/dummy state.
    from .dmd2_owner import _tensor_identity
    teacher = {name: _tensor_identity(p) for name, p in model.guidance_model.real_unet.named_parameters()}
    rows = gather(teacher)
    require(all(row == rows[0] for row in rows), 'FSDP2 pretrained frozen Teacher differs across ranks')
    classifier = ClassifierCallScope(model.guidance_model)
    try:
        for name, root in (('generator', model.feedforward_model), ('guidance', model.guidance_model)):
            shard_tree(root, mesh, role=name)
        return classifier
    except BaseException:
        classifier.close()
        raise


class DMD2FSDP2Owner(DMD2CollectiveOwner):
    """Persistent native master/Adam shards; no clone/calibration fallback."""
    def __init__(self, model, args, source, uncond_embedding, *, identity, seed):
        self._collective = deepcopy(identity['native_collective'])
        self._role_replicas = None
        self._prepared = False
        self._forward_counts = dict(generator=0, guidance=0, fake=0, teacher=0, classifier=0)
        self._forward_handles = []
        DMD2Owner.__init__(self, model, args, source, uncond_embedding, identity=identity, seed=seed)

    def _validate_process_ownership(self):
        import torch
        import torch.distributed as dist
        from torch.distributed.fsdp import FSDPModule
        from torch.distributed.tensor import DTensor
        from .single_device import admit_fp32
        device = admit_fp32({'device': str(self.device)}, require=require, allow_replicated=True)
        require(device.type == 'cuda' and dist.is_initialized() and dist.get_world_size() == 16
                and device == self.device and self._collective == self.identity['native_collective']
                and self._collective['backend'] == BACKEND,
                'FSDP2 native source/topology/device ownership changed')
        require(self.identity['fsdp_source'] == source_identity(), 'FSDP2 native source identity changed')
        roots = (self.model.feedforward_model, self.model.guidance_model)
        if self._prepared:
            for root in roots:
                require(isinstance(root, FSDPModule)
                        and all(isinstance(p, DTensor) and p.device_mesh.size() == 16
                                and p.device_mesh.mesh_dim_names == ('fsdp',) and p.device == self.device
                                for p in root.parameters()), 'FSDP2 role root or shard ownership changed')
            scope = getattr(self.model, '_dmd2_classifier_call_scope', None)
            require(isinstance(scope, ClassifierCallScope) and not scope.closed
                    and scope.guidance is self.model.guidance_model
                    and self.model.guidance_model.compute_cls_logits is scope.installed
                    and getattr(self.model.guidance_model, '_dmd2_fsdp_classifier_scope', False),
                    'FSDP2 native classifier module-call scope changed')
        else:
            require(not any(isinstance(module, FSDPModule) for module in self.model.modules())
                    and not any(isinstance(p, DTensor) for p in self.model.parameters())
                    and not hasattr(self.model, '_dmd2_classifier_call_scope'),
                    'FSDP2 construction must remain collective-free before native preparation')
        require(not any(isinstance(m, torch.nn.parallel.DistributedDataParallel) for m in self.model.modules()),
                'FSDP2 owner cannot contain DDP wrappers')

    def _validate_precision(self):
        import torch
        validate_settings(self.args)
        require(not torch.is_autocast_enabled(self.device.type), 'FSDP2 reference forbids outer AMP')
        for kind in ('parameters', 'buffers'):
            for name, tensor in self._model_named(kind):
                dtype = (torch.bfloat16 if name.startswith('guidance_model.real_unet.') else
                         torch.float16 if name.startswith('vae.') else torch.float32)
                require(tensor.device == self.device and (not tensor.is_floating_point() or tensor.dtype == dtype),
                        'FSDP2 native precision exception changed: ' + name)
        require(all(t.device == self.device for t in self.runtime_tensors()),
                'FSDP2 runtime state left its owned device')
        for optimizer in self.optimizers.values():
            require(all(group.get('fused') in (None, False) for group in optimizer.param_groups),
                    'FSDP2 reference cannot enable fused AdamW')
            for parameter, state in optimizer.state.items():
                require(parameter.dtype == torch.float32 and all(state[key].dtype == torch.float32
                        and state[key].device == self.device for key in ('exp_avg', 'exp_avg_sq'))
                        and state['step'].dtype == torch.float32 and state['step'].device.type == 'cpu',
                        'FSDP2 native Adam master/moments/step changed')

    def _clip_grad_norm(self, parameters):
        import torch
        from torch.distributed.tensor import DTensor
        result = torch.nn.utils.clip_grad_norm_(parameters, self.args.max_grad_norm)
        return result.full_tensor() if isinstance(result, DTensor) else result

    def __deepcopy__(self, memo):
        raise ValueError('native FSDP2 owner cannot be cloned for calibration')

    def admit_execution(self, mode, implementation):
        from distillschedule.open_workload.collective_owner import implementation as admit_implementation
        require(mode in ('reference', 'manual'), 'native FSDP2 auto/plan requires separate qualification')
        if mode == 'reference':
            require(not implementation, 'FSDP2 reference cannot override physical implementation')
        else:
            admit_implementation(implementation)

    def collective_execution_spec(self):
        self._validate_live()
        return dict(schema_version='distillschedule.native_sharded_owner.v1', backend='fsdp2', world_size=16,
                    device_type='cuda', shared_identity=dict(identity=self.identity, source=self.source.identity))

    def execution_options(self):
        value = super().execution_options()
        value['identity']['scope'] = 'NATIVE_FSDP2_MIXED_REFERENCE_NO_LOCAL_OVERRIDES'
        return value

    def prepare_collective_execution(self):
        from .distributed import checked
        from .sd import original
        from .dmd2_owner import _hash_tensor
        def preflight():
            self._validate_live()
            require(not self._prepared, 'FSDP2 owner must be prepared exactly once')
            require(all(not optimizer.state for optimizer in self.optimizers.values()),
                    'FSDP2 preparation must precede the first Adam update')
            return original(self.source.root)
        api = checked(preflight, 'native FSDP2 preparation admission')
        try:
            # NativeOwnerLifecycle has already agreed construction, source,
            # layout and per-rank errors before entering these collectives.
            self.model._dmd2_classifier_call_scope = place_roles(self.model, api)
            named = dict(self._model_named('parameters'))
            for role, optimizer in self.optimizers.items():
                # Match the native baseline's empty-Adam parameter remap.
                # Keep group order/options, scheduler and optimizer objects.
                for group, names in zip(optimizer.param_groups, self.ownership[role], strict=True):
                    group['params'] = [named[name] for name in names]
            # Placement validates Teacher equality before native rank0 sync.
            # Immutable tensors are now checked as local shards, not replicas.
            self.frozen_hashes = {name: _hash_tensor(named[name]) for name in self.frozen_hashes}
            self._frozen_identity = self._frozen_tensors_identity()
            roots = dict(generator=self.model.feedforward_model, guidance=self.model.guidance_model,
                         fake=self.model.guidance_model.fake_unet, teacher=self.model.guidance_model.real_unet,
                         classifier=self.model.guidance_model.cls_pred_branch)
            for name, root in roots.items():
                def observe(module, args, name=name):
                    self._forward_counts[name] += 1
                self._forward_handles.append(root.register_forward_pre_hook(observe))
            self._prepared = True
            self._validate_live()
        except BaseException:
            self._resident_trial_poisoned = True
            self.close()
            raise

    def validate_collective_execution(self):
        import torch.distributed as dist
        self._validate_live()
        roles = {name: dict(forward_calls=count, optimizer_commits=(self.generator_updates if name == 'generator'
                          else self.guidance_updates if name == 'guidance' else 0))
                 for name, count in self._forward_counts.items()}
        return dict(world_size=16, rank=dist.get_rank(), roles=roles, zero_gradients=True,
            bucket_storage_bound_bytes=0, storage_note='FSDP_SHARDS; TRANSIENT_ALLGATHER_NOT_ZERO',
            forward_counter_scope='CURRENT_PROCESS_NATIVE_MODULE_HOOKS',
            optimizer_counter_scope='PERSISTENT_NATIVE_TRAINING_CLOCK',
            precision='NATIVE_MIXED_EXCEPTIONS_FP32_MASTER_REDUCE', numerical_status='NOT_RUN', gpu='NOT_RUN')

    def run(self, feeds, *, cycle):
        require(self._prepared, 'prepare FSDP2 collective ownership before native updates')
        try:
            return super().run(feeds, cycle=cycle)
        except BaseException:
            self._resident_trial_poisoned = True
            raise

    def _validate_checkpoint_tensors(self, value):
        import torch
        from torch.distributed.tensor import DTensor
        from torch.utils._pytree import tree_leaves
        from .tensor_state import layout
        mesh = next(self.model.feedforward_model.parameters()).device_mesh
        for t in tree_leaves(value):
            if not isinstance(t, torch.Tensor):
                continue
            require(type(t) in (torch.Tensor, DTensor) and t.layout == torch.strided and not t.requires_grad
                    and (t.device.type == 'cpu' or t.device == self.device),
                    'FSDP2 checkpoint requires dense owned-device or CPU shards')
            if isinstance(t, DTensor):
                info = layout(t)
                require(info['mesh'] == mesh.mesh.tolist() and info['mesh_dim_names'] == list(mesh.mesh_dim_names),
                        'FSDP2 checkpoint shard topology changed')
        require(local_finite(value), 'nonfinite FSDP2 checkpoint tensor')

    def close(self):
        for handle in self._forward_handles:
            handle.remove()
        self._forward_handles.clear()
        scope = getattr(self.model, '_dmd2_classifier_call_scope', None)
        if scope is not None:
            scope.close()
