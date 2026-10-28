"""Replica training facts. No model names, execution choices, or winners.

A local loss must already represent an equal-size rank-local mean. The system
averages parameter gradients, not input/latent gradients, before updates.
`module_bindings` declares model call ownership. Its legacy default requires
ordinary module-forward interfaces; arbitrary private methods are safe only
after backward. Explicit direct method bindings can instead opt into native
FSDP method hooks (not DDP's module.__call__). Random streams remain
shared by logical call unless explicitly rank-namespaced. Neither field selects
a reducer implementation or bucket size. Arbitrary autograd.grad
on trainable parameters and unequal-rank batches need different contracts.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import inspect
from typing import Mapping
from distillschedule.core.contracts import require


def module_method_binding(module_key, method_name):
    """Bind one model-owned method through the existing Context callable ABI.

    Program inputs, in their declared order, become positional arguments;
    attributes become keyword arguments. Resolve from ctx.state on every call,
    including recomputation, so native FSDP hooks and restored owners are used.
    This is intentionally not a declaration attached to an arbitrary wrapper.
    """
    require(isinstance(module_key, str) and bool(module_key)
            and isinstance(method_name, str) and method_name.isidentifier()
            and not method_name.startswith('_'),
            'explicit root and public method name required', path='open.replica')
    def invoke(ctx):
        return getattr(ctx.state[module_key], method_name)(*ctx.inputs.values(), **ctx.attributes)
    return invoke


@dataclass(frozen=True)
class ReplicaSemantics:
    module_bindings: Mapping[str, str]
    backward_calls: tuple[str, ...]
    local_batch_size: int
    objective: str = 'equal_rank_mean'
    # Legacy constructors and serialized identities retain their original
    # module-forward/shared-call contract unless explicitly opted in.
    forward_interface: str = 'module_forward'
    random_streams: str = 'shared_logical_call'
    forward_methods: Mapping[str, str] = field(default_factory=dict)
    # Complete named-parameter tensor maps whose layout follows a model root.
    # This declares ownership only, never EMA math, decay, or update frequency.
    parameter_shadows: Mapping[str, str] = field(default_factory=dict)
    # Separately owned immutable roots may opt into native forward lifetimes.
    # They never acquire optimizer, gradient-reduction, or shadow ownership.
    frozen_module_bindings: Mapping[str, str] = field(default_factory=dict)

    @property
    def forward_module_bindings(self):
        return {**self.module_bindings, **self.frozen_module_bindings}

    def validate(self, workload):
        import torch
        require(self.objective == 'equal_rank_mean', 'unsupported replica objective', path='open.replica')
        require(self.forward_interface in ('module_forward', 'private_methods', 'registered_methods'),
                'unsupported replica forward interface', path='open.replica')
        require(self.random_streams in ('shared_logical_call', 'rank_independent_v1'),
                'unsupported replica random streams', path='open.replica')
        require(type(self.local_batch_size) is int and self.local_batch_size > 0,
                'positive equal local batch required', path='open.replica')
        require(self.module_bindings and all(isinstance(k, str) and isinstance(v, str)
                for k, v in self.module_bindings.items()), 'module forward bindings required', path='open.replica')
        calls = {c.id: c for c in workload.program.calls}
        require(self.backward_calls and len(set(self.backward_calls)) == len(self.backward_calls),
                'explicit unique parameter-backward calls required', path='open.replica')
        require(all(k in calls and calls[k].backward_of and calls[k].grad == 'grad'
                    for k in self.backward_calls), 'parameter backward must have provenance', path='open.replica')
        trainable = {k for k, v in workload.state.items() if isinstance(v, torch.nn.Module)
                     and any(p.requires_grad for p in v.parameters())}
        require(set(self.module_bindings.values()) == trainable,
                'all trainable modules need a forward interface', path='open.replica')
        require(isinstance(self.frozen_module_bindings, Mapping)
                and all(isinstance(k, str) and bool(k) and isinstance(v, str) and bool(v)
                        for k, v in self.frozen_module_bindings.items()),
                'frozen forward bindings must be a mapping of nonempty strings', path='open.replica')
        frozen = set(self.frozen_module_bindings.values())
        require(not (set(self.module_bindings) & set(self.frozen_module_bindings)) and not (trainable & frozen),
                'trainable and frozen forward ownership must be disjoint', path='open.replica')
        if frozen:
            require(self.forward_interface == 'registered_methods',
                    'frozen forward ownership requires registered_methods', path='open.replica')
            require(all(key in workload.state and isinstance(workload.state[key], torch.nn.Module)
                        and list(workload.state[key].parameters())
                        and all(not p.requires_grad for p in workload.state[key].parameters()) for key in frozen),
                    'frozen forward roots must be nonempty all-frozen modules', path='open.replica')
            require(frozen <= set(workload.immutable_modules),
                    'frozen forward roots require whole-root immutable declarations', path='open.replica')
        forwards = self.forward_module_bindings
        require(set(forwards) <= set(workload.bindings), 'unknown module binding', path='open.replica')
        require(isinstance(self.forward_methods, Mapping), 'method bindings must be a mapping', path='open.replica')
        if self.forward_interface == 'registered_methods':
            require(set(self.forward_methods) == set(forwards),
                    'exact registered forward method bindings required', path='open.replica')
            per_root = {}
            for binding, name in self.forward_methods.items():
                key = forwards[binding]
                expected = module_method_binding(key, name)
                fn = workload.bindings[binding]
                require(inspect.isfunction(fn) and fn.__code__ is expected.__code__
                        and inspect.getclosurevars(fn).nonlocals == inspect.getclosurevars(expected).nonlocals,
                        'forward method binding must invoke its declared root and method', path='open.replica')
                model = workload.state[key]
                method = getattr(model, name, None)
                require(inspect.ismethod(method) and method.__self__ is model
                        and (name == 'forward' or not hasattr(torch.nn.Module, name))
                        and method.__func__ is not torch.nn.Module.forward,
                        'registered forward must be a callable model-owned public method', path='open.replica')
                require(key not in per_root or per_root[key] == name,
                        'one outer forward method per root; nested methods must not be registered', path='open.replica')
                per_root[key] = name
        else:
            require(not self.forward_methods, 'method declarations require registered_methods', path='open.replica')
        for c in calls.values():
            require(not (set(c.writes) & frozen), 'frozen forward roots cannot be written', path='open.replica')
            if set(c.reads) & frozen:
                require(c.binding in self.frozen_module_bindings,
                        'frozen forward is missing its declared interface', path='open.replica')
            if c.binding in forwards:
                require(forwards[c.binding] in c.reads and not c.writes and not c.backward_of
                        and not ((set(c.reads) & frozen) - {forwards[c.binding]}),
                        'module-forward binding cannot update state', path='open.replica')
            elif c.grad == 'grad' and not c.writes and not c.backward_of:
                require(not (set(c.reads) & trainable),
                        'trainable forward is missing its module interface', path='open.replica')
        optimizers = {k for k, value in workload.state.items()
                      if isinstance(value, torch.optim.Optimizer)}
        require(all(not (set(calls[k].writes) & (trainable | optimizers)) for k in self.backward_calls),
                'parameter backward and model/optimizer update must be separate calls', path='open.replica')
        # Every optimizer parameter must belong to the registered models. No
        # mutation of hidden/external tensors can be given replica semantics.
        params = {id(p) for k in trainable for p in workload.state[k].parameters() if p.requires_grad}
        owned = []
        for obj in workload.state.values():
            if isinstance(obj, torch.optim.Optimizer):
                for group in obj.param_groups:
                    owned.extend(id(p) for p in group['params'])
        require(owned and len(set(owned)) == len(owned) and set(owned) == params,
                'optimizer ownership must exactly cover disjoint trainables', path='open.replica')
        require(not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training
                        for k in trainable | frozen for m in workload.state[k].modules()),
                'training BatchNorm is not an equal-global-batch reference', path='open.replica')
        from distillschedule.training.tensor_state import validate_parameter_shadows
        validate_parameter_shadows(workload.state, self.parameter_shadows, trainable, feeds=workload.feeds)
        return self

    def to_dict(self):
        return {'module_bindings': dict(self.module_bindings), 'backward_calls': list(self.backward_calls),
                'local_batch_size': self.local_batch_size, 'objective': self.objective,
                **({'forward_interface': self.forward_interface} if self.forward_interface != 'module_forward' else {}),
                **({'random_streams': self.random_streams} if self.random_streams != 'shared_logical_call' else {}),
                **({'forward_methods': dict(self.forward_methods)} if self.forward_methods else {}),
                **({'parameter_shadows': dict(self.parameter_shadows)} if self.parameter_shadows else {}),
                **({'frozen_module_bindings': dict(self.frozen_module_bindings)} if self.frozen_module_bindings else {})}


def ddp_completion_edges(program, semantics, context_modules=None):
    """Conservative reducer-context lifetimes derived from semantic forwards.

    The reference call order must already be executable. Add ONLY ordering
    constraints, never computation or a new update. These edges prevent search
    from starting a second forward before the first backward has completed.
    """
    ancestors = {}
    for call in program.calls:
        parents = set(call.inputs) & set(ancestors)
        ancestry = set(parents)
        for parent in parents: ancestry.update(ancestors[parent])
        ancestors[call.id] = ancestry
    active = {}; completed = {}; edges = set()
    forwards = semantics.forward_module_bindings
    for call in program.calls:
        module = forwards.get(call.binding)
        if context_modules is not None and module not in context_modules: module = None
        if module is not None:
            require(module not in active, 'native reference overwrites an outstanding forward', path='open.replica')
            if module in completed: edges.add((completed[module], call.id))
            if call.grad == 'grad': active[module] = call.id
            else: completed[module] = call.id
        if call.id in semantics.backward_calls:
            for module, forward in tuple(active.items()):
                if forward in ancestors[call.id]:
                    completed[module] = call.id
                    del active[module]
        require(not (set(call.writes) & set(active)),
                'native parameter update precedes backward completion', path='open.replica')
    require(not active, 'native forward has no declared backward completion', path='open.replica')
    return tuple(sorted(edges))


def execution_graph(program, semantics, reduction, context_modules=None):
    graph = program.compile()
    if reduction == 'post_backward': return graph
    require(reduction in ('ddp_overlap', 'fsdp_forward'), 'unknown reducer', path='open.replica')
    require(semantics.forward_interface != 'private_methods',
            'private methods require post_backward reduction, not forward wrappers', path='open.replica')
    require(semantics.forward_interface == 'module_forward' or reduction == 'fsdp_forward',
            'DDP requires module.__call__, not registered forward methods', path='open.replica')
    from distillschedule.core.graph import Dependency, ExecutionGraph
    edges = list(graph.dependencies)
    existing = {(edge.source, edge.target) for edge in edges}
    for source, target in ddp_completion_edges(program, semantics, context_modules):
        if (source, target) not in existing:
            edges.append(Dependency(source, target, 'reducer_context_completion'))
    return ExecutionGraph(graph.operations, tuple(edges),
        {**graph.metadata, 'execution_constraint': ('fsdp_forward_backward_context_v1'
            if reduction == 'fsdp_forward' and semantics.forward_interface == 'registered_methods'
            else 'ddp_forward_backward_context_v1')})
