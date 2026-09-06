from __future__ import annotations

from types import SimpleNamespace

from distillschedule.evidence.trajectory_state_sketch import (
    SCHEMA_VERSION,
    collect_numeric_state_sketch,
)


class _FakeTensor:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)
        self.shape = (len(values),)
        self.dtype = "torch.float32"
        self.device = "cpu"

    def detach(self) -> "_FakeTensor":
        return self

    def contiguous(self) -> "_FakeTensor":
        return self

    def view(self, _shape: int) -> "_FakeTensor":
        return self

    def index_select(self, _dimension: int, indices: "_FakeTensor") -> "_FakeTensor":
        return _FakeTensor([self.values[int(index)] for index in indices.values])

    def float(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def tolist(self) -> list[float]:
        return list(self.values)


class _FakeDistributed:
    def is_initialized(self) -> bool:
        return True

    def get_rank(self) -> int:
        return 0

    def get_world_size(self) -> int:
        return 1

    def all_gather_object(self, output: list[object], value: object) -> None:
        output[0] = value


def test_full_replica_sketch_exposes_numeric_coordinate_values() -> None:
    torch = SimpleNamespace(
        distributed=_FakeDistributed(),
        int64="torch.int64",
        tensor=lambda values, **_kwargs: _FakeTensor(list(values)),
    )
    sketch = collect_numeric_state_sketch(
        torch=torch,
        named_values=[("module._orig_mod.layer.weight", _FakeTensor([1.0, 2.0, 3.0]), None)],
        samples_per_tensor=16,
    )
    assert sketch["schema_version"] == SCHEMA_VERSION
    assert sketch["inventory_tensor_count"] == 1
    assert sketch["coordinate_count"] == 3
    assert sketch["coordinates"] == [
        {"tensor": "layer.weight", "flat_index": 0, "value_fp32": 1.0},
        {"tensor": "layer.weight", "flat_index": 1, "value_fp32": 2.0},
        {"tensor": "layer.weight", "flat_index": 2, "value_fp32": 3.0},
    ]
