"""Execution tests for two independent, explicitly opt-in baseline mechanisms."""
from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest

from distillschedule.runtime.livewan_baseline_placement import (
    BaselinePlacement, apply_contract, check_collective_trace, from_plan,
)
from distillschedule.runtime.livewan_cpu_optimizer import CPUAdamW
from distillschedule.runtime.livewan_collective_padding import (
    StudentRolloutPadding, padding_call_active,
)


def torch_or_skip():
    try:
        import torch
        return torch
    except ImportError:
        raise unittest.SkipTest("Torch unavailable; validated remote environment is required")


def test_cpu_optimizer_and_padding_are_independent_default_off():
    value = {"metadata": {"execution_mode": "baseline"},
             "workload": {"shape": {"batch_size": 16}},
             "training_backends": {r: {"kind": "native"}
                                   for r in ("generator", "score", "teacher")}}
    before = copy.deepcopy(value)
    apply_contract(value, mode="baseline", name="legacy-hybrid", teacher_offload="off")
    assert value == before
    cpu = apply_contract(value, mode="baseline", name="full-resident",
                         teacher_offload="off", optimizer_device="cpu")
    assert cpu == from_plan(value)
    assert cpu.receipt()["all_model_weights_resident"]
    assert cpu.receipt()["optimizer_offload"] and not cpu.receipt()["pure_resident"]
    rows = [SimpleNamespace(logical_step=0, new_blocks=i) for i in (4, 5, 6)]
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "FAIL_FSDP_COLLECTIVE_TRACE"):
        check_collective_trace(BaselinePlacement("world16-fsdp"), rows)
    padded = BaselinePlacement("world16-fsdp", student_padding="max-noop")
    check_collective_trace(padded, rows)
    assert padded.receipt()["optimizer_device"] == "cuda"
    for kwargs in ({"name": "legacy-hybrid", "optimizer_device": "cpu"},
                   {"name": "full-resident", "student_padding": "max-noop"},
                   {"name": "world16-fsdp", "optimizer_device": "cpu"}):
        with unittest.TestCase().assertRaises(ValueError): BaselinePlacement(**kwargs)


def test_cpu_adam_multi_update_matches_original_cpu_foreach_formula():
    torch = torch_or_skip()
    torch.manual_seed(43)
    parameters = [torch.nn.Parameter(torch.randn(5, 7)), torch.nn.Parameter(torch.randn(9))]
    reference = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    kwargs = dict(lr=2e-6, betas=(0., .99), weight_decay=0., eps=1e-8, foreach=True)
    candidate = CPUAdamW(torch, parameters, **kwargs)
    candidate.materialize_state(step=0)
    control = torch.optim.AdamW(reference, **kwargs)
    for step in range(6):
        for p, q in zip(parameters, reference):
            gradient = torch.randn_like(p)
            p.grad = gradient.clone(); q.grad = gradient.clone()
        candidate.step(); control.step()
        for p, q in zip(parameters, reference):
            assert torch.equal(p, q), step
            for key in ("step", "exp_avg", "exp_avg_sq"):
                assert torch.equal(candidate.state[p][key], control.state[q][key]), (step, key)
        candidate.zero_grad(); control.zero_grad()
    receipt = candidate.receipt()
    assert receipt["steps"] == 6
    assert receipt["gradient_d2h_bytes"] == receipt["updated_parameter_h2d_bytes"]
    assert all(p.grad is None for p in parameters)


def test_cpu_optimizer_rejects_parameter_mutation_and_non_fp32():
    torch = torch_or_skip()
    p = torch.nn.Parameter(torch.ones(3))
    candidate = CPUAdamW(torch, [p], lr=1e-3)
    candidate.materialize_state()
    with torch.no_grad(): p.add_(1)
    p.grad = torch.ones_like(p)
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "changed outside"):
        candidate.step()
    with unittest.TestCase().assertRaisesRegex(ValueError, "FP32"):
        CPUAdamW(torch, [torch.nn.Parameter(p.to(torch.bfloat16))], lr=1e-3)


def test_padding_preserves_rng_cache_and_records_only_discarded_nograd_calls():
    torch = torch_or_skip()
    row = SimpleNamespace(logical_step=0, new_blocks=4, operation_key="step=0/rank=0")
    maximum = SimpleNamespace(logical_step=0, new_blocks=6)
    padding = StudentRolloutPadding(torch, [row, maximum])
    cache = SimpleNamespace(num_tokens=17, evictions=0)
    streamer = SimpleNamespace(schedule=[1., .75, .5, .25], noisy_context=False,
        model=object(), n_frames=21, n_world=9, cache=cache, ctx=torch.ones(3),
        ctx_lens=None, rope=object(), dtype=torch.float32, time_scale=1000.)
    recs = [[{"z": torch.ones(1, 2, 3), "block_start": i*3}] for i in range(4)]
    observed = []
    def forward(model, z, t, start, rope, context, lengths, **kwargs):
        assert padding_call_active() and not torch.is_grad_enabled()
        assert kwargs["kv"] is None
        observed.append(t)
        torch.rand(1)  # fork must restore even a future stochastic kernel.
        return z + t
    rng = torch.random.get_rng_state().clone()
    input_before = recs[-1][0]["z"].clone()
    padding.pad(blockcausal=SimpleNamespace(block_forward=forward),
                streamer=streamer, row=row, recs=recs)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.equal(recs[-1][0]["z"], input_before)
    assert (streamer.n_frames, streamer.n_world, cache.num_tokens, cache.evictions) == (21, 9, 17, 0)
    assert observed == [1., .75, .5, .25, 0.] * 2
    assert padding.receipt()["extra_physical_forward_calls"] == 10
    assert not padding_call_active()


def test_padding_exception_resets_context_and_rng():
    torch = torch_or_skip()
    row = SimpleNamespace(logical_step=0, new_blocks=4, operation_key="row")
    padding = StudentRolloutPadding(torch, [row, SimpleNamespace(logical_step=0, new_blocks=6)])
    streamer = SimpleNamespace(schedule=[1., .75, .5, .25], noisy_context=False,
        model=None, n_frames=21, n_world=9, cache=SimpleNamespace(num_tokens=17, evictions=0),
        ctx=None, ctx_lens=None, rope=None, dtype=torch.float32, time_scale=1000.)
    def fail(*args, **kwargs):
        torch.rand(1)
        raise RuntimeError("bounded failure")
    rng = torch.random.get_rng_state().clone()
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "bounded failure"):
        padding.pad(blockcausal=SimpleNamespace(block_forward=fail), streamer=streamer, row=row,
            recs=[[{"z": torch.ones(1), "block_start": 0}]]*4)
    assert not padding_call_active() and torch.equal(rng, torch.random.get_rng_state())


if __name__ == "__main__":
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
        for name, value in sorted(globals().items()) if name.startswith("test_"))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
