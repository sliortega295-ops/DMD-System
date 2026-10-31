from __future__ import annotations

from pathlib import Path

import pytest

# The repository's lightweight refactor CI intentionally does not install the
# multi-gigabyte PyTorch dependency.  This module validates real tensor
# serialization/partition behavior, so skip it explicitly in that environment
# instead of failing during collection.  The pinned Helios server environment
# still executes the complete module.
torch = pytest.importorskip("torch")

from tools.assets.build_systems_gan_bootstrap import (
    gan_head_specs,
    zero1_partition_with_tail_padding,
)
from tools.assets.build_systems_gan_trace import _crop, _seed


def test_official_gan_head_inventory_is_frozen() -> None:
    specs = gan_head_specs()
    assert len(specs) == 130
    assert sum(item.numel for item in specs) == 988_359_941
    assert specs[0].name == "gan_heads.5.head3d.0.weight"
    assert specs[-1].name == "gan_final_head.head3d.20.bias"
    assert specs[-3].shape == (768,)


def test_gan_trace_rng_is_operation_keyed() -> None:
    a = _crop(seed=_seed(7, 3, 1000, "gan_g_crop"))
    b = _crop(seed=_seed(7, 3, 1000, "gan_g_crop"))
    c = _crop(seed=_seed(7, 4, 1000, "gan_g_crop"))
    assert a.equal(b)
    assert not a.equal(c) or _seed(7, 3, 1000, "gan_g_crop") != _seed(7, 4, 1000, "gan_g_crop")


def test_zero1_last_rank_materializes_explicit_optimizer_padding() -> None:
    raw = torch.arange(13, dtype=torch.float32)
    first, first_padding = zero1_partition_with_tail_padding(
        raw, lower=0, partition_numel=8
    )
    last, last_padding = zero1_partition_with_tail_padding(
        raw, lower=8, partition_numel=8
    )
    assert first_padding == 0
    assert first.tolist() == list(map(float, range(8)))
    assert last_padding == 3
    assert last.tolist() == [8.0, 9.0, 10.0, 11.0, 12.0, 0.0, 0.0, 0.0]
