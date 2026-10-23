"""Real CPU Torch checkpoint/autograd/RNG/Adam; only CUDA op discovery is substituted.

This is not a model/GPU numerical gate. Production replay receipt census is
asserted on the existing Torch 2.10 runtime, never inferred from newer Torch.
Set LIVEWAN_REQUIRE_TORCH210_CPU=1 for a fail-closed target-runtime CPU gate.
"""
import os
from types import SimpleNamespace


def test_real_c51_sac_checkpoint_backward_rng_optimizer_and_receipt(monkeypatch, record_property):
    import torch
    import torch.utils.checkpoint as checkpoint
    from distillschedule.runtime import livewan_score_fa3 as fa3
    assert not torch.cuda.is_initialized()
    pinned = torch.__version__.startswith("2.10.")
    assert os.environ.get("LIVEWAN_REQUIRE_TORCH210_CPU", "0") != "1" or pinned, (
        "The required production checkpoint/receipt census must run on existing Torch 2.10")
    counts = {"calls": 0}

    @torch.library.custom_op("livewan_c51_cpu::attention", mutates_args=())
    def attention(x: torch.Tensor) -> torch.Tensor:
        counts["calls"] += 1
        return x.sin()

    @attention.register_fake
    def _(x):
        return torch.empty_like(x)

    def setup(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    def backward(ctx, grad):
        return grad * ctx.saved_tensors[0].cos()

    attention.register_autograd(backward, setup_context=setup)
    op = torch.ops.livewan_c51_cpu.attention.default
    identity = dict(interface_sha256=fa3.INTERFACE_SHA256, operator=str(op),
                    evidence_scope="CPU_SUBSTITUTED_CUDA_OP_DISCOVERY_ONLY")
    monkeypatch.setattr(fa3, "discover_flash_forward", lambda _: (op, identity))
    monkeypatch.setattr(fa3, "attention_metadata", lambda t, args, kw: (((2, 19320, 12, 128), "CPU_METADATA"),) * 3)
    monkeypatch.setattr(fa3, "OP_NAME", op._schema.name)
    results = []
    for use_sac in (False, True):
        torch.manual_seed(2718)
        x = torch.linspace(-1, 1, 48, requires_grad=True)
        optimizer = torch.optim.AdamW([x], lr=1e-3)
        proxy = fa3.CriticFA3CheckpointProxy(checkpoint, torch, SimpleNamespace()) if use_sac else checkpoint
        before = counts["calls"]

        def block(x):
            return (attention(x) * torch.rand_like(x)).square()

        y0 = proxy.checkpoint(block, x, use_reentrant=False)
        y1 = proxy.checkpoint(block, x + .5, use_reentrant=False)
        (y0.sum() + y1.sum()).backward()
        optimizer.step()
        results.append((torch.stack((y0.detach(), y1.detach())), x.grad.clone(),
            torch.get_rng_state().clone(), x.detach().clone(),
            optimizer.state[x]["exp_avg"].clone(), optimizer.state[x]["exp_avg_sq"].clone()))
        actual = counts["calls"] - before
        assert actual == (2 if use_sac else 4)
        if use_sac:
            raw = proxy.contexts.receipt()
            assert raw["checkpoint_calls"] == 2 and raw["original_fa_calls"] == 2
            assert raw["contract_failures"] == 0
            assert raw["rng_preservation"] == "UNCHANGED_CHECKPOINT_DEFAULT_TRUE"
            assert len(proxy.contexts.checkpoints) == 2
            if pinned:
                # The actual production proxy/receipt validator, not a hand-
                # authored PASS. A newer Torch API is not evidence for 2.10.
                receipt = proxy.receipt(expected_calls=2)
                assert receipt["replay_fa_cache_reads"] == 2
                assert not fa3.runtime_receipt_errors(receipt, rank=8, expected_calls=2)
    assert all(torch.equal(a, b) for a, b in zip(*results))
    assert not torch.cuda.is_initialized()
    record_property("evidence_scope", "REAL_CPU_CHECKPOINT_NOT_GPU_MODEL_VALIDATION")
    record_property("torch_version", torch.__version__)
    record_property("production_210_receipt_census_verified", pinned)
    record_property("forward_gradient_rng_adam_state_equal", True)
