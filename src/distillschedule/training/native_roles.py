"""Persistent native role replicas, without changing the owner's update loop.

Canonical checkpoint names omit wrapper implementation details. The existing
common lifecycle still owns shards, RNG, data progress and atomic publication.
"""
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


class NativeRoleDDP(DistributedDataParallel):
    """Retain upstream role attributes used outside its original forward."""
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('module'), name)


def unwrap_role(module):
    return module.module if isinstance(module, NativeRoleDDP) else module


class RoleReplicas:
    def __init__(self, model, paths, device):
        self.model, self.paths, self.device = model, dict(paths), device
        self.original = {role: model.get_submodule(path) for role, path in self.paths.items()}
        if (not self.paths or len(set(self.paths.values())) != len(self.paths)
                or any(a.startswith(b + '.') for a in self.paths.values() for b in self.paths.values() if a != b)):
            raise ValueError('native role roots must be distinct, non-nested modules')
        parameter_ids = [{id(p) for p in root.parameters()} for root in self.original.values()]
        if sum(map(len, parameter_ids)) != len(set().union(*parameter_ids)):
            raise ValueError('native trainable roles may not alias parameters')
        self.wrappers, self.counters = {}, {role: dict(forward_calls=0) for role in self.paths}
        self.handles = []

    def prepare(self):
        if self.wrappers:
            raise ValueError('native role replicas must be installed exactly once')
        for role, path in self.paths.items():
            root = self.original[role]
            wrapper = NativeRoleDDP(root, process_group=dist.group.WORLD,
                device_ids=[self.device.index] if self.device.type == 'cuda' else None,
                broadcast_buffers=True, find_unused_parameters=False,
                gradient_as_bucket_view=True, bucket_cap_mb=128)
            parent, _, leaf = path.rpartition('.')
            setattr(self.model.get_submodule(parent), leaf, wrapper)
            self.wrappers[role] = wrapper
            def observe(module, args, row=self.counters[role]):
                row['forward_calls'] += 1
            self.handles.append(wrapper.register_forward_pre_hook(observe))
        self.validate()

    def validate(self):
        if set(self.wrappers) != set(self.paths):
            raise ValueError('native role replicas are not prepared')
        for role, path in self.paths.items():
            wrapper = self.wrappers[role]
            if (self.model.get_submodule(path) is not wrapper or type(wrapper) is not NativeRoleDDP
                    or wrapper.module is not self.original[role] or wrapper.process_group is not dist.group.WORLD
                    or wrapper.broadcast_buffers is not True or wrapper.find_unused_parameters is not False
                    or wrapper.gradient_as_bucket_view is not True or wrapper.bucket_bytes_cap != 128 * 1024**2):
                raise ValueError('native role wrapper, reducer or ownership changed')

    def canonical(self, name):
        for role in self.wrappers:
            prefix = self.paths[role] + '.module'
            if name == prefix or name.startswith(prefix + '.'):
                return self.paths[role] + name[len(prefix):]
        return name

    def named(self, kind):
        wrappers = {id(value) for value in self.wrappers.values()}
        for name, value in getattr(self.model, 'named_' + kind)():
            if kind != 'modules' or id(value) not in wrappers:
                yield self.canonical(name), value

    def state_dict(self):
        return {self.canonical(name): value for name, value in self.model.state_dict().items()}

    def load_state_dict(self, value):
        current = self.model.state_dict()
        names = {self.canonical(name): name for name in current}
        if set(names) != set(value) or len(names) != len(current):
            raise ValueError('native role checkpoint names changed')
        self.model.load_state_dict({names[name]: tensor for name, tensor in value.items()}, strict=True)
