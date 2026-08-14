from pathlib import Path

import torch

from runtime.sparse_ema import SparseShardEMA


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        self.lora = torch.nn.Parameter(torch.tensor([1.0, 2.0]))


def test_sparse_ema_bootstrap_is_exact_and_roundtrips(tmp_path: Path) -> None:
    module = _Tiny()
    ema = SparseShardEMA(
        module.named_parameters(),
        decay=0.99,
        update_after_step=750,
        base_model="tiny",
    )
    ema.optimization_step = 10
    with torch.no_grad():
        module.lora.copy_(torch.tensor([3.0, 4.0]))
    assert ema.step(module.parameters()) == 0.0
    assert ema.optimization_step == 11
    assert ema.compare_to_parameters(module.parameters())["matches_exact"]

    root = tmp_path / "ema"
    ema.save_pretrained(root, max_shard_bytes=16)
    loaded = SparseShardEMA.load_pretrained(module, root)
    assert loaded.optimization_step == 11
    assert loaded.compare_to_parameters(module.parameters())["matches_exact"]


def test_sparse_ema_accepts_exact_deepspeed_module_prefix_projection(
    tmp_path: Path,
) -> None:
    inner = _Tiny()
    wrapper = torch.nn.Module()
    wrapper.add_module("module", inner)
    ema = SparseShardEMA(
        wrapper.named_parameters(),
        decay=0.99,
        update_after_step=750,
        base_model="tiny",
    )
    ema.optimization_step = 10
    root = tmp_path / "wrapped_ema"
    ema.save_pretrained(root, max_shard_bytes=16)

    loaded = SparseShardEMA.load_pretrained(inner, root)

    assert loaded.names == ["module.lora"]
    assert loaded.indices == [1]
    assert loaded.optimization_step == 10
    assert loaded.compare_to_parameters(inner.parameters())["matches_exact"]
