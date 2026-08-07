import pytest

from runtime.zero_partition import zero_partition_valid_length


def test_zero_partition_valid_length_excludes_tail_padding() -> None:
    assert zero_partition_valid_length(logical_numel=5, partition_numel=3, rank=0) == 3
    assert zero_partition_valid_length(logical_numel=5, partition_numel=3, rank=1) == 2
    assert zero_partition_valid_length(logical_numel=5, partition_numel=3, rank=2) == 0


def test_even_zero_partition_has_no_padding() -> None:
    assert zero_partition_valid_length(logical_numel=6, partition_numel=3, rank=0) == 3
    assert zero_partition_valid_length(logical_numel=6, partition_numel=3, rank=1) == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"logical_numel": -1, "partition_numel": 1, "rank": 0},
        {"logical_numel": 1, "partition_numel": -1, "rank": 0},
        {"logical_numel": 1, "partition_numel": 1, "rank": -1},
    ],
)
def test_zero_partition_valid_length_rejects_negative_inputs(kwargs) -> None:
    with pytest.raises(ValueError):
        zero_partition_valid_length(**kwargs)
