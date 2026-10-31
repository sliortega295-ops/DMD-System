"""Independent scalar CPU workload. No Torch, CUDA, FSDP or training claims."""
from types import SimpleNamespace


class Unit:
    def __init__(self):
        self.group = SimpleNamespace(is_unsharded=False, _reshard_after_forward=True, reshard_after_backward=True)
        self.weights = [2., -3., 4.]
        self.materializations = 0

    def modules(self):
        return [self]

    def _get_fsdp_state(self):
        return SimpleNamespace(_fsdp_param_group=self.group)

    def set_reshard_after_forward(self, value, recurse=False):
        self.group._reshard_after_forward = value

    def set_reshard_after_backward(self, value, recurse=False):
        self.group.reshard_after_backward = value

    def unshard(self, async_op=False):
        self.materializations += 1
        self.local = self.weights.copy()
        self.group.is_unsharded = True

    def reshard(self):
        self.local = None
        self.group.is_unsharded = False


class Engine:
    def __init__(self):
        self.module = Unit()
        self.optimizer = SimpleNamespace(param_groups=[{"params": [self.module.weights]}])

    def __call__(self, x):
        unit = self.module
        if not unit.group.is_unsharded:
            unit.unshard()
        result = sum(w * v for w, v in zip(unit.local, x))
        if unit.group._reshard_after_forward:
            unit.reshard()
        return result


def baseline(engine, inputs):
    return [engine(x) for x in inputs], {"restored": True, "enabled": False}


execute = baseline
