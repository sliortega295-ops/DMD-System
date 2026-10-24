"""Replica training facts. No model names, execution choices, or winners.

A local loss must already represent an equal-size rank-local mean. The system
averages parameter gradients, not input/latent gradients, before updates.
`module_bindings` declares model call ownership. Its legacy default requires
ordinary module-forward interfaces; explicit private methods are safe only
after backward, never with forward-dependent wrappers. Random streams remain
shared by logical call unless explicitly rank-namespaced. Neither field selects
a reducer implementation or bucket size. Arbitrary autograd.grad
on trainable parameters and unequal-rank batches need different contracts.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
from distillschedule.core.contracts import require


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

    def validate(self, workload):
        import torch
        require(self.objective == 'equal_rank_mean', 'unsupported replica objective', path='open.replica')
        require(self.forward_interface in ('module_forward', 'private_methods'),
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
        require(set(self.module_bindings) <= set(workload.bindings), 'unknown module binding', path='open.replica')
        for c in calls.values():
            if c.binding in self.module_bindings:
                require(self.module_bindings[c.binding] in c.reads and not c.writes and not c.backward_of,
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
                        for k in trainable for m in workload.state[k].modules()),
                'training BatchNorm is not an equal-global-batch reference', path='open.replica')
        return self

    def to_dict(self):
        return {'module_bindings': dict(self.module_bindings), 'backward_calls': list(self.backward_calls),
                'local_batch_size': self.local_batch_size, 'objective': self.objective,
                **({'forward_interface': self.forward_interface} if self.forward_interface != 'module_forward' else {}),
                **({'random_streams': self.random_streams} if self.random_streams != 'shared_logical_call' else {})}


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
    for call in program.calls:
        module = semantics.module_bindings.get(call.binding)
        if context_modules is not None and module not in context_modules: module = None
        if module is not None:
            require(module not in active, 'DDP reference overwrites an outstanding forward', path='open.replica')
            if module in completed: edges.add((completed[module], call.id))
            if call.grad == 'grad': active[module] = call.id
            else: completed[module] = call.id
        if call.id in semantics.backward_calls:
            for module, forward in tuple(active.items()):
                if forward in ancestors[call.id]:
                    completed[module] = call.id
                    del active[module]
        require(not (set(call.writes) & set(active)),
                'DDP parameter update precedes backward completion', path='open.replica')
    require(not active, 'DDP forward has no declared parameter backward', path='open.replica')
    return tuple(sorted(edges))


def execution_graph(program, semantics, reduction, context_modules=None):
    graph = program.compile()
    if reduction == 'post_backward': return graph
    require(semantics.forward_interface == 'module_forward',
            'private methods require post_backward reduction, not forward wrappers', path='open.replica')
    require(reduction == 'ddp_overlap', 'unknown reducer', path='open.replica')
    from distillschedule.core.graph import Dependency, ExecutionGraph
    edges = list(graph.dependencies)
    existing = {(edge.source, edge.target) for edge in edges}
    for source, target in ddp_completion_edges(program, semantics, context_modules):
        if (source, target) not in existing:
            edges.append(Dependency(source, target, 'reducer_context_completion'))
    return ExecutionGraph(graph.operations, tuple(edges),
        {**graph.metadata, 'execution_constraint': 'ddp_forward_backward_context_v1'})
