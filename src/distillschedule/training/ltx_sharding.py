"""Model-owned native LTX block boundaries, not a loader or execution backend.

The public declaration stays Torch-free. Construction remains dense CPU FP32;
the existing FSDP lifecycle installs children, outer methods, Adam and shadows.
"""
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
import weakref

from .single_device import FLOW_FROZEN_TEACHER_EXECUTION, execution_policy_matches


CONSTRUCTION = 'cpu-fsdp2-native-blocks-v1'
EXECUTION = deepcopy(FLOW_FROZEN_TEACHER_EXECUTION)
ROLES = ('student', 'fake', 'teacher')
_REGISTERED_ARGUMENT_TYPES = {}


def _provider_source():
    from distillschedule.workloads.best_known.registry import file_sha256
    return dict(module=__name__, sha256=file_sha256(Path(__file__)))


def _register_native_arguments(source):
    """Let native FSDP find tensors in upstream block inputs and outputs.

    LTX returns frozen dataclasses, not Tensor tuples. Without their native
    pytree registration child groups cannot install pre-backward unshard hooks.
    Only the opt-in owner registers these classes; their values/math are intact.
    """
    from torch.utils._pytree import register_dataclass, tree_is_leaf, tree_flatten, tree_unflatten
    from .ltx_source import require
    api = source.import_module('ltx_core.model.transformer.transformer_args')
    for cls in (api.AdaLNTimestepEmbedding, api.TransformerArgs):
        values = {field.name: object() for field in fields(cls) if field.init}
        probe = cls(**values)
        if cls not in _REGISTERED_ARGUMENT_TYPES:
            require(tree_is_leaf(probe), 'native argument class is already registered by a foreign pytree owner')
            register_dataclass(cls)
            _REGISTERED_ARGUMENT_TYPES[cls] = tree_flatten(probe)[1]
        leaves, spec = tree_flatten(probe)
        require(spec == _REGISTERED_ARGUMENT_TYPES[cls] and len(leaves) == len(values)
                and all(leaf is value for leaf, value in zip(leaves, values.values())),
                'native argument pytree registration changed')
        restored = tree_unflatten(leaves, spec)
        require(type(restored) is cls and all(getattr(restored, name) is value for name, value in values.items()),
                'native argument pytree reconstruction changed')


class _NativeBlockConstruction:
    def __init__(self, parameters, factory, builder):
        self.parameters = deepcopy(parameters)
        self.factory, self.builder = factory, builder
        self.provider_source = _provider_source()
        self.started = self.attached = self.closed = False

    def validate_constructor(self, parameters):
        from .ltx_source import require
        require(self.started and not self.attached and not self.closed and parameters == self.parameters,
                'native block construction requires its live single-use provider')

    def construct(self, factory, parameters):
        from .ltx_source import require
        require(not self.started and not self.closed and factory is self.factory and parameters == self.parameters,
                'native block construction provider/factory/parameters changed')
        self.started = True
        try:
            return self.builder(parameters, construction=self)
        except BaseException:
            self.close()
            raise

    def attach(self, components):
        import torch
        from .ltx import LTXRole
        from .ltx_source import require
        from .tensor_state import ParameterShadows
        self.validate_constructor(self.parameters)
        require(components.sharded_construction is self and 'teacher' in components.immutable_modules,
                'native block construction returned a foreign provider or frozen root')
        roots, boundaries, native_owners = {}, {}, {}
        for role in ROLES:
            root = components.state[role]
            require(type(root) is LTXRole and root.device == torch.device('cpu') and root.compute_precision == 'fp32',
                    'native block construction requires actual CPU FP32 LTX roots')
            native_class = root.source.import_module('ltx_core.model.transformer.model').LTXModel
            require(type(root.native) is native_class and len(root.native.transformer_blocks) == root.architecture['num_layers']
                    and all(type(p) is torch.nn.Parameter and p.is_contiguous() for p in root.parameters()),
                    'native block construction requires dense actual native transformer blocks')
            roots[role] = root
            boundaries[role] = tuple('native.transformer_blocks.' + str(index)
                                     for index in range(len(root.native.transformer_blocks)))
            native_owners[role] = (root.source, root.native, root.native.transformer_blocks,
                                  tuple(root.native.transformer_blocks),
                                  {name: weakref.ref(p) for name, p in root.named_parameters()})
            _register_native_arguments(root.source)
        require(not any(p.requires_grad for p in roots['teacher'].parameters()), 'native Teacher must remain frozen')
        self.shadows = {'student_ema': 'student'} if 'student_ema' in components.state else {}
        if self.shadows:
            require(type(components.state['student_ema']) is dict,
                    'native block construction expects the existing dense full-parameter Student EMA')
            components.state['student_ema'] = ParameterShadows('student', components.state['student_ema'])
        self.components, self.roots, self.boundaries = components, roots, boundaries
        self.native_owners = native_owners
        components.metadata = {**components.metadata,
            'precision': 'FP32_NATIVE_CPU_BLOCK_FSDP2',
            'replication': 'EXPLICIT_EQUAL_RANK_LOCAL_B1_NATIVE_FSDP_MEAN',
            'scope': 'CPU_GLOO_MANUAL_DENSE_INITIALIZATION; NO_CUDA_FULL22B_OR_SHARD_FIRST_LOADING_CLAIM',
            'construction': dict(schema_version=CONSTRUCTION,
                provider_source=deepcopy(self.provider_source),
                shard_boundaries={role: list(paths) for role, paths in boundaries.items()},
                initialization='DENSE_CPU_CONSTRUCTOR_WEIGHTS_LORA_AND_RNG_UNCHANGED',
                media='FROZEN_RESIDENT', parameter_shadows=dict(self.shadows))}
        self.attached = True

    def bind_workload(self, components, workload):
        from .ltx_source import require
        from torch.distributed.fsdp import FSDPModule
        require(self.attached and not self.closed and self.components is components
                and components.sharded_construction is self
                and all(workload.state.get(role) is root and components.state.get(role) is root
                        for role, root in self.roots.items()),
                'native block construction lifetime/provider/root changed')
        require(_provider_source() == self.provider_source
                and components.metadata['construction']['provider_source'] == self.provider_source,
                'native block construction provider source identity changed')
        for role, root in self.roots.items():
            source, native, blocks, children, parameters = self.native_owners[role]
            require(root.source is source and root.native is native and native.transformer_blocks is blocks
                    and len(blocks) == len(children) and all(a is b for a, b in zip(blocks, children)),
                    'native block construction source/native/child identity changed')
            if not isinstance(root, FSDPModule):
                current = dict(root.named_parameters())
                require(current.keys() == parameters.keys() and all(current[name] is ref() for name, ref in parameters.items()),
                        'native block construction dense Parameter identity changed before installation')
            source.validate()
            _register_native_arguments(source)
        require(workload.replica_semantics is not None, 'native block construction requires distributed flow ownership')
        workload.replica_semantics = replace(workload.replica_semantics,
            forward_interface='registered_methods',
            forward_methods={role + '_forward': 'dmd_forward' for role in ROLES},
            frozen_module_bindings={'teacher_forward': 'teacher'},
            parameter_shadows=dict(self.shadows), shard_boundaries=deepcopy(self.boundaries))
        workload.identity_extra = {**workload.identity_extra,
            'native_ltx_block_construction': deepcopy(components.metadata['construction'])}
        return workload

    def close(self):
        self.closed = True
        self.components = None
        self.roots = {}
        self.native_owners = {}


def admit_construction(*, parameters, request, budget, factory, builder):
    from .distributed import checked, gather, rank_world
    from .ltx_source import require
    choices = gather(parameters.get('construction'))
    require(all(choice == choices[0] for choice in choices), 'ranks selected different native block construction')
    if choices[0] is None:
        return None
    def admit():
        import torch
        require(choices[0] == CONSTRUCTION and budget is None
                and parameters.get('compute_precision', 'fp32') == 'fp32'
                and parameters.get('device', 'cpu') == request.trainer.get('device', 'cpu') == 'cpu'
                and rank_world()[1] >= 2 and torch.distributed.get_backend() == 'gloo'
                and request.trainer.get('distributed', {}).get('backend') == 'gloo'
                and execution_policy_matches(request.execution, EXECUTION),
                'native block construction requires CPU/Gloo ranks and exact FP32 manual fsdp2_reshard; '
                'trials, CUDA, HSDP, immutable sharing and saved plans are unsupported')
    checked(admit, 'native LTX block construction request')
    return _NativeBlockConstruction(parameters, factory, builder)
