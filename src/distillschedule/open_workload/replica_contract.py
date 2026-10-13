"""Replica training facts. No model names, execution choices, or winners.

A local loss must already represent an equal-size rank-local mean. The system
averages parameter gradients, not input/latent gradients, before updates.
`module_bindings` declares ordinary module-forward interfaces; it does not tell
which reduction implementation or bucket size to use. Arbitrary autograd.grad
on trainable parameters and unequal-rank batches need different contracts.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping
from distillschedule.core.contracts import require


@dataclass(frozen=True)
class ReplicaSemantics:
    module_bindings: Mapping[str, str]
    backward_calls: tuple[str, ...]
    local_batch_size: int
    objective: str = 'equal_rank_mean'

    def validate(self, workload):
        import torch
        require(self.objective == 'equal_rank_mean', 'unsupported replica objective', path='open.replica')
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
                'local_batch_size': self.local_batch_size, 'objective': self.objective}
