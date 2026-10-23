"""Real full-class CPU FP32 numerical proof, NOT native geometry or a trainer.

Student/Critic dim16, Teacher dim32, head16; 32x32-pixel fixture, three clean
world frames and one three-frame/four-step event. No meta/scalar replacements,
weights/downloads, optimizer, Program, checkpoint or GPU/performance claims.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import time

import numpy as np
import pytest
import torch

from distillschedule.reference import livewan_wan21 as provider


def original():
    return provider.load_cpu(backend="torch-sdpa-cpu", precision="float32")


@pytest.fixture(autouse=True)
def bounded_cpu_probe():
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    start = time.monotonic()
    previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("120s CPU probe limit")))
    signal.alarm(120)
    try:
        yield
        assert time.monotonic() - start < 120
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 2 * 1024**3
        assert not torch.cuda.is_initialized()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def tiny_config(teacher=False):
    return dict(model_type="t2v", patch_size=(1, 2, 2), text_len=512,
                in_dim=16, dim=32 if teacher else 16, ffn_dim=64 if teacher else 32,
                freq_dim=16, text_dim=4096, out_dim=16,
                num_heads=2 if teacher else 1, num_layers=1)


def fixture_model(seed, teacher=False):
    # All changes are TEST fixture weights. The complete original constructor
    # still zero-initializes head.head.weight and its AST/bytes are untouched.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = original().model.WanModel(**tiny_config(teacher)).float().eval()
    assert torch.count_nonzero(model.head.head.weight) == 0
    with torch.no_grad():
        model.head.head.weight.copy_(torch.randn(model.head.head.weight.shape,
            generator=torch.Generator().manual_seed(seed + 1000)) * .035)
    return model


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_snapshot():
    return torch.get_rng_state().clone(), random.getstate(), np.random.get_state()


def assert_rng_equal(before):
    assert torch.equal(before[0], torch.get_rng_state())
    assert before[1] == random.getstate()
    after = np.random.get_state()
    assert before[2][0] == after[0] and np.array_equal(before[2][1], after[1])
    assert before[2][2:] == after[2:]


def test_complete_original_constructor_forward_and_no_grad_teacher():
    api = original()
    before = rng_snapshot()
    model = fixture_model(101, teacher=True).requires_grad_(False)
    assert type(model) is api.model.WanModel
    assert len(model.blocks) == 1 and model.dim == 32 and model.num_heads == 2
    assert model.text_embedding[0].in_features == 4096 and model.text_len == 512
    assert isinstance(model.blocks[0], api.model.WanAttentionBlock)
    generator = torch.Generator().manual_seed(22)
    latent = torch.randn(16, 3, 4, 4, generator=generator)
    context = torch.randn(512, 4096, generator=generator)
    state = state_digest(model)
    with torch.no_grad():
        output = model([latent], torch.tensor([600.]), [context], seq_len=12)[0]
    assert output.shape == latent.shape and output.dtype is torch.float32
    assert torch.isfinite(output).all() and output.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters()) and state == state_digest(model)
    assert_rng_equal(before)


def test_real_streaming_prefix_recompute_dmd_and_critic_lora_backward(record_property, recwarn):
    api = original()
    before = rng_snapshot()
    student = fixture_model(120)
    teacher = fixture_model(121, teacher=True).requires_grad_(False)
    critic = fixture_model(122)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        lora, modules, numel = api.lora.inject_lora(critic, rank=32)
    assert modules == 6 and numel == 7168
    assert all(torch.count_nonzero(m.b) == 0 for m in critic.modules() if isinstance(m, api.lora.LoRALinear))
    # Separately identified nonzero TEST fixture LoRA state makes both A and B
    # gradients nonvacuous. This does not alter upstream LoRA initialization.
    local = torch.Generator().manual_seed(124)
    with torch.no_grad():
        for module in critic.modules():
            if isinstance(module, api.lora.LoRALinear):
                module.b.copy_(torch.randn(module.b.shape, generator=local) * .015)
    identities = {name: state_digest(model) for name, model in
                  (("student", student), ("teacher", teacher), ("critic", critic))}
    calls = []
    hook = student.patch_embedding.register_forward_hook(lambda *_: calls.append("patch"))
    head_outputs = []
    head_hook = student.head.head.register_forward_hook(
        lambda _module, _inputs, output: head_outputs.append(output.detach().clone()))
    inputs = torch.Generator().manual_seed(125)
    raw_pos = torch.randn(1, 512, 4096, generator=inputs)
    raw_neg = torch.randn(1, 512, 4096, generator=inputs)
    world = torch.randn(1, 16, 3, 4, 4, generator=inputs)
    lenses = torch.tensor([509])
    with torch.no_grad():
        student_ctx = student.text_embedding(raw_pos).detach()
        teacher_pos = teacher.text_embedding(raw_pos).detach()
        teacher_neg = teacher.text_embedding(raw_neg).detach()
        critic_ctx = critic.text_embedding(raw_pos).detach()
    streamer = api.stream.FewStepStreamer(student, width=32, height=32,
        max_frames=12, cache_frames=9, block_frames=3, num_steps=4,
        device="cpu", dtype=torch.float32, cuda_graphs=False)
    assert streamer.graphs is None and streamer.S == 4
    assert streamer.schedule == [1., .75, .5, .25] and streamer.sampler == "renoise"
    streamer.set_text(student_ctx, lenses)
    assert streamer.set_world(world) == 3
    assert len(calls) == 1 and streamer.cache.num_tokens == 12
    pinned_k = streamer.cache.k[:, :12].clone()
    pinned_v = streamer.cache.v[:, :12].clone()
    sampler = torch.Generator().manual_seed(126)
    sampler_initial = sampler.get_state().clone()
    outputs, blocks = streamer.rollout_record(1, generator=sampler)
    event, records = outputs[0], blocks[0]
    assert event.shape == (1, 16, 3, 4, 4) and torch.isfinite(event).all()
    assert not event.requires_grad and len(records) == 4 and len(calls) == 6
    assert len(head_outputs) == 6  # world, four noisy forwards, clean-KV rerun
    assert [r["t"] for r in records] == [1., .75, .5, .25]
    assert all(r["prefix"] == 12 and r["block_start"] == 3 and r["evictions"] == 0 for r in records)
    assert streamer.n_frames == 6 and streamer.n_world == 3 and streamer.cache.num_tokens == 24
    assert torch.equal(pinned_k, streamer.cache.k[:, :12]) and torch.equal(pinned_v, streamer.cache.v[:, :12])
    # Exactly one initial noise + three independent renoises, local RNG only.
    shadow = torch.Generator().set_state(sampler_initial)
    noises = [torch.randn(event.shape, generator=shadow) for _ in range(4)]
    assert torch.equal(noises[0], records[0]["z"])
    assert torch.equal(shadow.get_state(), sampler.get_state())
    assert_rng_equal(before)
    cache_k, cache_v = streamer.cache.k.clone(), streamer.cache.v.clone()
    prefix = api.blockcausal.BufferPrefixKV(streamer.cache)
    step = records[2]
    assert streamer.prefix_intact(step)
    with torch.no_grad():
        expected_v = api.blockcausal.block_forward(student, step["z"], step["t"], step["block_start"],
            streamer.rope, student_ctx, lenses, kv=prefix, dtype=torch.float32, prefix_upto=step["prefix"])
        expected_x0 = step["z"] - step["t"] * expected_v
    x0_hat = streamer.recompute_step(step, prefix)
    x0_hat.retain_grad()
    torch.testing.assert_close(x0_hat, expected_x0, rtol=0, atol=0)
    # Compare with the actual recorded rollout forward, not merely a second
    # recomputation. BufferPrefixKV reorders keys as [current;prefix] vs the
    # streamer's [prefix;current]; preserve the original math and FP32 tolerance.
    rollout_v = api.blockcausal._unpatchify(student, head_outputs[1 + step["step"]], 3, 2, 2)
    replay_x0 = step["z"] - step["t"] * rollout_v
    torch.testing.assert_close(x0_hat, replay_x0, rtol=2e-6, atol=2e-6)
    replay_max_error = float((x0_hat.detach() - replay_x0).abs().max())
    assert x0_hat.requires_grad and torch.equal(cache_k, streamer.cache.k) and torch.equal(cache_v, streamer.cache.v)
    # Actual Teacher CFG and fake velocities over the whole world+event clip.
    clean = torch.cat([world, event], dim=2).detach()
    score_t = torch.tensor([.61])
    score_noise = torch.randn(clean.shape, generator=inputs)
    z_t, _, _ = api.data.add_noise(clean, score_t, noise=score_noise)
    teacher_rope = api.core.make_rope_table(teacher, 2, 2, 12, "cpu")
    assert torch.equal(teacher_rope.table, streamer.rope.table)
    with torch.no_grad():
        real = api.dmd.cfg_velocity(teacher, z_t, score_t, teacher_rope,
            teacher_pos, teacher_neg, lenses, torch.float32, guidance=3.)
        fake = api.dmd.bidirectional_velocity(critic, z_t, score_t, streamer.rope,
            critic_ctx, lenses, torch.float32)
        grad, metrics = api.dmd.dmd_gradient(clean, score_t, z_t, real, fake)
    assert grad.shape == clean.shape and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert metrics["dmd_real_fake_gap"] > 0 and metrics["dmd_normalizer"] > 0
    event_grad = grad[:, :, 3:]
    generator_loss = api.dmd.dmd_surrogate(x0_hat, event_grad, scale=.75)
    generator_loss.backward()
    torch.testing.assert_close(x0_hat.grad, .75 * event_grad / event_grad.numel(), rtol=2e-5, atol=1e-8)
    ggrads = {n: p.grad for n, p in student.named_parameters() if p.grad is not None}
    assert ggrads and all(torch.isfinite(g).all() for g in ggrads.values())
    nonzero_g = [n for n, g in ggrads.items() if torch.count_nonzero(g)]
    assert all(any(n.startswith(prefix) for n in nonzero_g) for prefix in
               ("patch_embedding", "time_embedding", "blocks.0.self_attn", "blocks.0.ffn", "head.head"))
    assert all(p.grad is None for p in student.text_embedding.parameters())
    g_hashes = {n: tensor_digest(g) for n, g in ggrads.items()}
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in critic.parameters())
    critic_noise = torch.randn(clean.shape, generator=inputs)
    critic_loss = api.dmd.critic_loss(critic, clean, score_t, streamer.rope,
        critic_ctx, lenses, torch.float32, grad_checkpoint=True, noise=critic_noise)
    critic_loss.backward()
    assert len(lora) == 12 and all(p.grad is not None and torch.isfinite(p.grad).all()
                                 and torch.count_nonzero(p.grad) for p in lora)
    assert all(p.grad is None for p in critic.parameters() if not p.requires_grad)
    assert all(p.grad is None for p in teacher.parameters())
    assert g_hashes == {n: tensor_digest(g) for n, g in ggrads.items()}
    assert identities == {name: state_digest(model) for name, model in
                          (("student", student), ("teacher", teacher), ("critic", critic))}
    assert torch.equal(cache_k, streamer.cache.k) and torch.equal(cache_v, streamer.cache.v)
    # Real prefix invalidation and world pinning, no extra rollout or new seed.
    streamer.trim_to_window(0)
    assert not streamer.prefix_intact(step) and streamer.cache.num_tokens == 12 and streamer.cache.evictions == 1
    assert torch.equal(pinned_k, streamer.cache.k[:, :12]) and torch.equal(pinned_v, streamer.cache.v[:, :12])
    assert_rng_equal(before)
    assert not torch.cuda.is_initialized()
    hook.remove()
    head_hook.remove()
    report = dict(scope="FULL_ORIGINAL_CPU_FP32_TINY_FIXTURE_NOT_NATIVE_GEOMETRY", weights=identities,
        model_classes={name: type(model).__module__ + "." + type(model).__name__ for name, model in
                       (("student", student), ("teacher", teacher), ("critic", critic))},
        parameter_numel={name: sum(p.numel() for p in model.parameters()) for name, model in
                         (("student", student), ("teacher", teacher), ("critic", critic))},
        generator_grad_tensors=len(ggrads), generator_nonzero_grad_tensors=len(nonzero_g),
        generator_grad_l2=math.sqrt(sum(float(g.square().sum()) for g in ggrads.values())),
        critic_lora_modules=modules, critic_lora_numel=numel, critic_lora_nonzero_grad_tensors=len(lora),
        critic_grad_l2=math.sqrt(sum(float(p.grad.square().sum()) for p in lora)),
        generator_loss=float(generator_loss.detach()), critic_loss=float(critic_loss.detach()), dmd=metrics,
        four_step_records=len(records), world_prime=1, event_blocks=1,
        recorded_replay_max_abs_error=replay_max_error,
        cpu_threads=torch.get_num_threads(), cuda_initialized=torch.cuda.is_initialized(),
        global_rng_unchanged=True, local_sampler_exact_four_draws=True,
        frozen_teacher_base_unchanged=True, model_weights_unchanged=True,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in recwarn])
    record_property("original_cpu_probe", json.dumps(report, sort_keys=True))
    print("ORIGINAL_CPU_PROBE=" + json.dumps(report, sort_keys=True))
    if os.environ.get("DISTILLSCHEDULE_W4_PROBE_REPORT"):
        Path(os.environ["DISTILLSCHEDULE_W4_PROBE_REPORT"]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def dense_oracle(q, k, v, q_lens=None, k_lens=None, causal=False, q_scale=None, softmax_scale=None):
    b, lq, nq, c = q.shape
    lk, nk = k.shape[1:3]
    qq = q.double().transpose(1, 2) * (q_scale if q_scale is not None else 1.)
    kk = k.double().transpose(1, 2).repeat_interleave(nq // nk, dim=1)
    vv = v.double().transpose(1, 2).repeat_interleave(nq // nk, dim=1)
    scores = (qq @ kk.transpose(-1, -2)) * (softmax_scale if softmax_scale is not None else c**-.5)
    keep = torch.ones(b, 1, lq, lk, dtype=torch.bool)
    if k_lens is not None:
        keep &= torch.arange(lk).view(1, 1, 1, lk) < torch.as_tensor(k_lens).view(b, 1, 1, 1)
    if causal:
        keep &= torch.arange(lk).view(1, 1, 1, lk) <= torch.arange(lq).view(1, 1, lq, 1) + lk - lq
    weights = torch.nan_to_num(scores.masked_fill(~keep, -torch.inf).softmax(-1))
    output = (weights @ vv).transpose(1, 2)
    if q_lens is not None:
        output = output * (torch.arange(lq).view(1, lq, 1, 1) < torch.as_tensor(q_lens).view(b, 1, 1, 1))
    return output.float()


@pytest.mark.parametrize("lq,lk,nq,nk,options", [
    (3, 3, 2, 2, {}), (3, 5, 2, 2, dict(k_lens=[4, 2], q_lens=[3, 1])),
    (3, 5, 4, 2, dict(causal=True, k_lens=[4, 0], q_lens=[3, 2], q_scale=1.2, softmax_scale=.31)),
    (5, 3, 2, 2, dict(causal=True)), (3, 5, 2, 2, dict(q_lens=[0, 2])),
    (3, 3, 2, 1, dict(causal=True, k_lens=[0, 2], q_lens=[3, 2], q_scale=-.75, softmax_scale=0.)),
])
def test_original_sdpa_masks_gqa_scales_dtype_and_backward_match_dense_oracle(lq, lk, nq, nk, options):
    api = original()
    generator = torch.Generator().manual_seed(314)
    values = [torch.randn(shape, generator=generator).requires_grad_()
              for shape in ((2, lq, nq, 4), (2, lk, nk, 4), (2, lk, nk, 3))]
    copies = [value.detach().clone().requires_grad_() for value in values]
    before = torch.get_rng_state().clone()
    flags = [torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.flash_sdp_enabled(),
             torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.cudnn_sdp_enabled()]
    actual = api.attention.flash_attention(*values, **options)
    expected = dense_oracle(*copies, **options)
    assert actual.dtype is torch.float32 and actual.shape == (2, lq, nq, 3)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=1e-6)
    probe = torch.linspace(-.4, .7, actual.numel()).reshape_as(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    for value, copy in zip(values, copies):
        assert torch.isfinite(value.grad).all()
        torch.testing.assert_close(value.grad, copy.grad, rtol=6e-5, atol=2e-6)
    assert torch.equal(before, torch.get_rng_state())
    assert flags == [torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.flash_sdp_enabled(),
                     torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.cudnn_sdp_enabled()]


@pytest.mark.parametrize("options", [dict(dtype=torch.bfloat16), dict(dropout_p=.1), dict(window_size=(1, 1)),
    dict(version=2), dict(deterministic=True), dict(causal=1), dict(q_scale=float("nan")),
    dict(softmax_scale=float("inf")), dict(q_scale=torch.tensor(1.)), dict(q_lens=[1.5]),
    dict(q_lens=[True]), dict(q_lens=1), dict(k_lens=[-1]), dict(k_lens=[4]), dict(q_lens=[1, 2])])
def test_cpu_bridge_rejects_unsupported_options_and_invalid_lengths(options):
    value = torch.ones(1, 3, 1, 4)
    with pytest.raises(ValueError, match="CPU attention"):
        original().attention.flash_attention(value, value, value, **options)


@pytest.mark.parametrize("bad", ["bf16", "meta", "nan", "empty", "batch", "head", "value_length", "budget"])
def test_cpu_bridge_rejects_invalid_tensor_geometry_and_over_budget_attention(bad):
    q = k = v = torch.ones(1, 3, 1, 4)
    if bad == "bf16": q = q.bfloat16()
    elif bad == "meta": q = torch.ones(1, 3, 1, 4, device="meta")
    elif bad == "nan": q = q.clone(); q[0, 0, 0, 0] = torch.nan
    elif bad == "empty": q = q[:, :0]
    elif bad == "batch": k = k.expand(2, -1, -1, -1)
    elif bad == "head": k = k.expand(-1, -1, 2, -1)
    elif bad == "value_length": v = v[:, :2]
    else: q = k = v = torch.ones(1, 4097, 1, 2)
    with pytest.raises(ValueError, match="CPU attention"):
        original().attention.flash_attention(q, k, v)


def test_cpu_autocast_is_not_a_silent_precision_workaround():
    value = torch.ones(1, 3, 1, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="autocast"):
            original().attention.flash_attention(value, value, value)
