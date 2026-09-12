from __future__ import annotations

from types import SimpleNamespace
import copy

import pytest

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


class _EvidenceGroupDistributed(_FakeDistributed):
    def __init__(self, group, *, corrupt=False, empty=False):
        self.group, self.corrupt, self.empty = group, corrupt, empty
        self.calls = []

    def get_rank(self, *, group=None):
        assert group is self.group
        return 0

    def get_world_size(self, *, group=None):
        assert group is self.group
        return 2

    def all_gather_object(self, output, value, *, group=None):
        assert group is self.group
        self.calls.append(group)
        output[0] = copy.deepcopy(value)
        output[1] = {"inventory_names": [], "rows": []} if self.empty else copy.deepcopy(value)
        for row in output[1]["rows"]:
            row["contributor_rank"] = 1
            if self.corrupt:
                row["values"]["0"] += 1.0


def _all_rank_sketch(*, corrupt=False, empty=False):
    group = object()
    dist = _EvidenceGroupDistributed(group, corrupt=corrupt, empty=empty)
    torch = SimpleNamespace(distributed=dist, int64="torch.int64",
                            tensor=lambda values, **_kwargs: _FakeTensor(list(values)))
    value = collect_numeric_state_sketch(torch=torch,
        named_values=[("module.layer.weight", _FakeTensor([1.0, 2.0, 3.0]), None)],
        object_collective_group=group, verify_all_replicas=True)
    assert dist.calls == [group]
    return value


def test_v2_keeps_gloo_transport_and_all_rank_replica_evidence():
    from distillschedule.evidence.trajectory_state_sketch import ALL_RANKS_SCHEMA_VERSION
    value = _all_rank_sketch()
    assert value["schema_version"] == ALL_RANKS_SCHEMA_VERSION
    assert value["tensors"][0]["contributor_ranks"] == [0, 1]
    assert value["replica_validation"] == "ALL_RANKS_CONTRIBUTE_SAMPLED_COORDINATES_MUST_MATCH"


def test_v2_rejects_replica_disagreement_and_empty_owner():
    with pytest.raises(AssertionError, match="disagree"):
        _all_rank_sketch(corrupt=True)
    with pytest.raises(AssertionError, match="inventory differs"):
        _all_rank_sketch(empty=True)


def test_v2_comparator_does_not_upgrade_v1_or_relax_thresholds():
    from distillschedule.evidence.trajectory_state_sketch import compare_numeric_state_sketches
    value = _all_rank_sketch()
    same = compare_numeric_state_sketches(value, value, cycle=1, component="generator_trainable")
    assert same.cosine == pytest.approx(1.0) and same.normalized_l2 == 0.0
    legacy = dict(value, schema_version=SCHEMA_VERSION)
    with pytest.raises(ValueError, match="schema"):
        compare_numeric_state_sketches(value, legacy, cycle=1, component="generator_trainable")


def test_nonprimary_replica_contributes_only_with_explicit_v2_policy():
    from distillschedule.evidence.trajectory_state_sketch import _local_samples
    torch = SimpleNamespace(int64="torch.int64", tensor=lambda values, **_kwargs: _FakeTensor(list(values)))
    args = dict(torch=torch, name="weight", value=_FakeTensor([1.0, 2.0]),
                layout_source=None, count=16, rank=1, primary_rank=0)
    assert _local_samples(**args)["values"] == {}
    assert _local_samples(**args, verify_all_replicas=True)["values"] == {"0": 1.0, "1": 2.0}
