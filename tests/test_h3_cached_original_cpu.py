"""Actual original tiny H3/LoRA/DMD2/Adam proof, never a pretrained/W16 claim.

The architecture is the archived original test_train_dmd2 tiny architecture.
The algorithm remains 8step/G1F1/LoRA64; only the discriminator is explicitly
tiny. The GAN-start=2 fixture is labeled, and never changes canonical start=500.
No model doubles, fake tensors, numerical replacements or download payloads.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from safetensors.torch import save_file
import torch

from distillschedule.reference import diffsynth_h3_cached as provider


def original():
    return provider.load_cpu(backend="torch-dense-cpu")


def tiny_architecture():
    return dict(num_layers=1, token_refiner_num_layers=1, hidden_size=24,
                num_attention_heads=2, attention_head_dim=12, ffn_hidden_size=48,
                timestep_input_dim=8, time_embed_hidden_size=24, time_embed_dim=12,
                adaln_out_features=432, final_adaln_out_features=48, rope_inv_freq_len=2)


def explicit_g1f1_config(**fixture_overrides):
    # Original DMD2Config defaults F5. Every qualified G1F1 caller must be explicit.
    return original().dmd2.DMD2Config(**{
        **dict(num_inference_steps=8, critic_updates=1, lora_rank=64, lora_alpha=64,
               video_shift=2.22, audio_shift=2.22, audio_loss_weight=0.5,
               teacher_cfg_scale=1.0, adversarial_enabled=True, gan_start_step=500,
               hidden_channels=8, num_layers=2, gradient_checkpointing=True),
        **fixture_overrides,
    })


def sample(index=0, reference=True):
    data = dict(input_latents=torch.randn(1, 24, 2, 4, 4) + index,
                audio_input_latents=torch.randn(2, 32, 3) - index,
                prompt_embeds=torch.randn(3, 5120) + index / 10)
    if reference:
        data.update(ref_video_latents=torch.randn(1, 24, 3, 4, 4) + index / 3,
                    ref_audio_latents=torch.randn(2, 32, 5) - index / 3)
    return data


def real(inputs):
    return inputs["input_latents"], inputs["audio_input_latents"]


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray) and left.dtype == right.dtype
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            equal_tree(a, b)
    else:
        assert type(left) is type(right) and left == right


def setup_trainer(config=None, model=None, observe=False):
    api = original()
    config = config or explicit_g1f1_config()
    api.training.seed_all(123)
    adapter = api.dmd2.H3Adapter(model if model is not None else api.model.MiniMaxH3DiT(**tiny_architecture()), config)
    discriminator = api.dmd2.VelocityDiscriminator(config)
    optimizers, parameters = api.training.make_optimizers(adapter, discriminator, config)
    events = dict(predict=[], backward=[], commits=[], gradient_roles=[])
    if observe:
        predict = adapter.predict
        def observed_predict(latents, sigmas, inputs, checkpoint=False):
            role = adapter.layers[0][1].role
            values = predict(latents, sigmas, inputs, checkpoint)
            events["predict"].append(dict(role=role, grad=torch.is_grad_enabled(),
                                          output_grad=any(x.requires_grad for x in values),
                                          trainable=any(p.requires_grad for p in adapter.role_parameters(role))
                                          if role != "teacher" else False))
            return values
        adapter.predict = observed_predict
        for role, optimizer in optimizers.items():
            def before_step(opt, args, kwargs, role=role):
                other = "critic" if role == "student" else "student"
                assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in parameters[role])
                assert all(p.grad is None for p in parameters[other])
                assert all(p.grad is None for n, p in adapter.named_parameters() if ".adapters." not in n)
                events["commits"].append(role)
                events["gradient_roles"].append({
                    "role": role, "active": sum(p.grad is not None for p in parameters[role]),
                    "heads": sum(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in discriminator.parameters()),
                })
            optimizer.register_step_pre_hook(before_step)

    def step(role, loss):
        if observe:
            loss.register_hook(lambda gradient: events["backward"].append(role))
        api.training.optimizer_step(role, loss, optimizers, parameters, config)

    return api.dmd2.DMD2Trainer(adapter, discriminator, config, step), optimizers, events


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)
    original().training.seed_all(123)
    yield
    assert not torch.cuda.is_initialized()


def test_original_defaults_are_retained_but_g1f1_is_explicit():
    api = original()
    assert api.dmd2.DMD2Config().critic_updates == 5
    assert api.dmd2.DMD2Config().gan_start_step == 500
    config = explicit_g1f1_config()
    expected = provider.source_identity()["qualified_protocol"]
    for key, value in expected.items():
        if key != "ema_updates":
            assert getattr(config, key) == value
    assert not hasattr(api.dmd2.DMD2Trainer, "ema")
    model = api.model.MiniMaxH3DiT(**tiny_architecture())
    assert len(model.blocks) == 1 and model.condition_proj.in_features == 5120
    assert model.video_patch_proj.in_features == 96 and model.audio_patch_proj.in_features == 32


def test_actual_original_g1f1_update_census_and_immutable_references():
    api = original()
    trainer, optimizers, events = setup_trainer(observe=True)
    raw = sample()
    raw_before = copy.deepcopy(raw)
    inputs, negative = api.dmd2.prepare_batch(raw)
    inputs_before = copy.deepcopy(inputs)
    frozen = {k: p.detach().clone() for k, p in trainer.adapter.named_parameters() if ".adapters." not in k}
    adapters = api.training.adapter_state(trainer.adapter)
    heads = copy.deepcopy(trainer.discriminator.state_dict())
    result = trainer.update(real(inputs), inputs, negative)
    for role, count, grad in [("student", 9, 1), ("fake_score", 2, 1), ("teacher", 1, 0)]:
        calls = [x for x in events["predict"] if x["role"] == role]
        assert len(calls) == count
        assert sum(x["output_grad"] for x in calls) == grad
    assert events["backward"] == events["commits"] == ["critic", "student"]
    assert events["gradient_roles"][0]["heads"] == 0
    assert (result["generator_step"], result["critic_step"], result["gan_active"]) == (1, 1, False)
    for name, value in frozen.items():
        equal_tree(value, dict(trainer.adapter.named_parameters())[name])
    equal_tree(raw_before, raw)
    equal_tree(inputs_before, inputs)
    equal_tree(heads, trainer.discriminator.state_dict())
    after = api.training.adapter_state(trainer.adapter)
    for role in ("student", "fake_score"):
        assert any(not torch.equal(value, after[key]) for key, value in adapters.items() if f".{role}." in key)
        params = trainer.adapter.role_parameters(role)
        assert params and all(p.shape[0] == 64 if ".A." in key else True
                              for key, p in trainer.adapter.named_parameters() if f".adapters.{role}." in key)
        optimizer = optimizers["student" if role == "student" else "critic"]
        assert all(optimizer.state[p]["step"].item() == 1 for p in params)
        assert any(torch.count_nonzero(optimizer.state[p]["exp_avg"]) for p in params)
    print(json.dumps(dict(proof="actual_original_g1f1", census=dict(student=9, fake_score=2, teacher=1,
                         student_grad=1, fake_score_grad=1), backward=events["backward"], ema_updates=0,
                         config=asdict(trainer.config), metrics=result), sort_keys=True))


def test_actual_original_gan_transition_and_head_adam(monkeypatch):
    api = original()
    config = explicit_g1f1_config(gan_start_step=2)  # Explicit transition fixture, NOT canonical 500.
    trainer, optimizers, events = setup_trainer(config, observe=True)
    before = copy.deepcopy(trainer.discriminator.state_dict())
    original_grad = torch.autograd.grad
    observed_grad = []
    def track_grad(outputs, inputs, *args, **kwargs):
        result = original_grad(outputs, inputs, *args, **kwargs)
        observed_grad.append(dict(leaf_inputs=all(x.is_leaf and x.requires_grad for x in inputs),
                                  finite=all(bool(torch.isfinite(x).all()) for x in result),
                                  nonzero=all(bool(torch.count_nonzero(x)) for x in result)))
        return result
    monkeypatch.setattr(torch.autograd, "grad", track_grad)
    results = []
    for index in range(3):
        inputs, negative = api.dmd2.prepare_batch(sample(index))
        results.append(trainer.update(real(inputs), inputs, negative))
        if index < 2:
            equal_tree(before, trainer.discriminator.state_dict())
            assert all(p not in optimizers["critic"].state for p in trainer.discriminator.parameters())
    assert [x["gan_active"] for x in results] == [False, False, True]
    assert events["backward"] == events["commits"] == ["critic", "student"] * 3
    assert len(observed_grad) == 1 and all(observed_grad[0].values())
    assert events["gradient_roles"][4]["heads"] > 0
    for parameter in trainer.discriminator.parameters():
        state = optimizers["critic"].state[parameter]
        assert state["step"].item() == 1
        assert torch.isfinite(state["exp_avg"]).all() and torch.count_nonzero(state["exp_avg"])
    assert any(not torch.equal(before[k], value) for k, value in trainer.discriminator.state_dict().items())
    assert api.dmd2.DMD2Config().gan_start_step == 500
    print(json.dumps(dict(proof="GAN_START_2_TRANSITION_FIXTURE", active=[x["gan_active"] for x in results],
                         original_input_autograd=observed_grad, metrics=results,
                         canonical_gan_start_step=500), sort_keys=True))


def test_actual_model_gan_sign_and_exact_single_noising_chain():
    api = original()
    trainer, _, _ = setup_trainer(explicit_g1f1_config(gan_start_step=2))
    inputs, _ = api.dmd2.prepare_batch(sample())
    trainer.adapter.select("fake_score")
    trainer.discriminator.requires_grad_(False)
    generated = tuple(x.detach().requires_grad_() for x in real(inputs))
    noise = tuple(torch.randn_like(x) for x in generated)
    sigmas = (torch.tensor(0.3712345), torch.tensor(0.6287654))
    noisy = tuple(api.dmd2.add_noise(x, s, n) for x, s, n in zip(generated, sigmas, noise))
    logits = trainer.discriminator(trainer.adapter.predict(noisy, sigmas, inputs))
    direct_loss = trainer.config.generator_loss_weight * api.dmd2.modality_sum(
        [torch.nn.functional.softplus(x).mean() for x in logits], trainer.config)
    direct = torch.autograd.grad(direct_loss, generated)
    leaves = tuple(x.detach().requires_grad_() for x in noisy)
    logits = trainer.discriminator(trainer.adapter.predict(leaves, sigmas, inputs))
    loss = trainer.config.generator_loss_weight * api.dmd2.modality_sum(
        [torch.nn.functional.softplus(x).mean() for x in logits], trainer.config)
    gradients = torch.autograd.grad(loss, leaves)
    injected = sum(api.dmd2.exact_gradient_surrogate(api.dmd2.add_noise(p, s, n), g)
                   for p, s, n, g in zip(generated, sigmas, noise, gradients))
    for wanted, actual, leaf_gradient, sigma in zip(direct, torch.autograd.grad(injected, generated), gradients, sigmas):
        torch.testing.assert_close(wanted, actual, rtol=0, atol=0)
        torch.testing.assert_close(actual, (1 - sigma) * leaf_gradient, rtol=0, atol=0)
        assert torch.count_nonzero(actual)
    # Actual model/head logits, original real-negative / fake-positive convention.
    real_logits = tuple(x.detach().requires_grad_() for x in logits)
    fake_logits = tuple((x.detach() + 0.1).requires_grad_() for x in logits)
    critic = api.dmd2.modality_sum([torch.nn.functional.softplus(r).mean() +
                                  torch.nn.functional.softplus(-f).mean()
                                  for r, f in zip(real_logits, fake_logits)], trainer.config)
    signs = torch.autograd.grad(critic, real_logits + fake_logits)
    assert all(torch.all(x > 0) for x in signs[:2]) and all(torch.all(x < 0) for x in signs[2:])


def test_original_continuous_sigma_native_schedule_and_flow_sign():
    api = original()
    trainer, _, _ = setup_trainer()
    inputs, _ = api.dmd2.prepare_batch(sample())
    state = torch.get_rng_state()
    base = torch.randn(()).sigmoid()
    expected = (2.22 * base / (1 + 1.22 * base)).clamp(0.02, 0.98)
    torch.set_rng_state(state)
    sigmas, noise = trainer.score_noise(real(inputs))
    equal_tree(sigmas[0], expected)
    equal_tree(sigmas[1], expected)
    assert not torch.equal(sigmas[0] * 1000, (sigmas[0] * 1000).round())
    for schedule in trainer.schedule:
        assert len(schedule) == 9 and schedule[0] == 1 and schedule[-1] == 0
        assert torch.all(schedule[:-1] > schedule[1:])
    for clean, n, sigma in zip(real(inputs), noise, sigmas):
        noisy = api.dmd2.add_noise(clean, sigma, n)
        torch.testing.assert_close(api.dmd2.prediction_to_x0(noisy, sigma, n - clean), clean)
    # Actual native clamp behavior, with a deliberately narrow fixture interval.
    narrow, _, _ = setup_trainer(explicit_g1f1_config(sigma_min=0.49, sigma_max=0.51))
    values = [float(narrow.score_noise(real(inputs))[0][0]) for _ in range(32)]
    assert min(values) == float(torch.tensor(0.49)) and max(values) == float(torch.tensor(0.51))


def test_previous_current_continuous_rope_stereo_and_different_negative_text():
    api = original()
    raw = sample()
    raw["negative_prompt_embeds"] = torch.randn(1, 5120)
    positive, negative = api.dmd2.prepare_batch(raw, 3.0)
    assert len(positive["packed"]["text_pos"]) == 3 and len(negative["packed"]["text_pos"]) == 1
    equal_tree(positive["ref_visual_anchor"], negative["ref_visual_anchor"])
    equal_tree(positive["ref_audio_anchor"], negative["ref_audio_anchor"])
    packed = positive["packed"]
    positions = packed["img_position_ids"][0]
    ref_visual = packed["img_pos"][:12]
    target_visual = packed["img_pos"][12:]
    assert len(target_visual) == 8
    torch.testing.assert_close(positions[ref_visual, 0].unique(sorted=True),
                               torch.tensor([3, 3 + 5/3, 3 + 25/3], dtype=torch.float64))
    assert positions[target_visual[0], 0] == 18  # 3 text + complete previous span 15.
    assert positions[target_visual[4], 0] == 18 + 5/3
    audio_positions = positions[packed["audio_pos"], 0]
    equal_tree(audio_positions[:5], audio_positions[5:10])
    equal_tree(audio_positions[10:13], audio_positions[13:])
    expected_audio = raw["ref_audio_latents"].permute(0, 2, 1).reshape(10, 32)
    equal_tree(positive["ref_audio_anchor"], expected_audio)
    packed_raw = {**raw, "ref_video_latents": None, "ref_audio_latents": None,
                  "ref_visual_anchor": positive["ref_visual_anchor"],
                  "ref_audio_anchor": positive["ref_audio_anchor"], "packed": packed}
    recovered, _ = api.dmd2.prepare_batch(packed_raw)
    equal_tree(recovered, positive)


def test_actual_model_fn_returns_generated_only_negative_velocity_and_clean_reference_times():
    api = original()
    model = api.model.MiniMaxH3DiT(**tiny_architecture())
    inputs, _ = api.dmd2.prepare_batch(sample())
    captured = {}
    def before(module, args, kwargs):
        captured["kwargs"] = copy.deepcopy(kwargs)
    def after(module, args, output):
        captured["rows"] = tuple(x.detach().clone() for x in output)
    h1 = model.register_forward_pre_hook(before, with_kwargs=True)
    h2 = model.register_forward_hook(after)
    try:
        result = api.packing.model_fn_minimax_h3(model, *real(inputs),
            timestep_video=torch.tensor(412.3456), timestep_audio=torch.tensor(628.7654), **inputs)
    finally:
        h1.remove()
        h2.remove()
    kwargs = captured["kwargs"]
    times = kwargs["unique_timesteps"][kwargs["inverse_indices"]]
    packed = inputs["packed"]
    assert torch.all(times[packed["img_pos"][:12]] == 1) and torch.all(times[packed["audio_pos"][:10]] == 1)
    torch.testing.assert_close(times[packed["img_pos"][12:]], torch.full((8,), 1 - 0.4123456))
    torch.testing.assert_close(times[packed["audio_pos"][10:]], torch.full((6,), 1 - 0.6287654))
    equal_tree(api.model.patchify_video(result[0]), -captured["rows"][0][12:])
    equal_tree(api.model.pack_audio(result[1]), -captured["rows"][1][10:])
    assert result[0].shape == (1, 24, 2, 4, 4) and result[1].shape == (2, 32, 3)


@pytest.mark.parametrize("key", ["denoise_mask_video", "denoise_mask_audio", "padding_mask", "attention_mask", "control_rows"])
def test_original_parser_rejects_unsupported_masks_before_model(key):
    with pytest.raises(ValueError, match="unsupported"):
        original().dmd2.prepare_batch({**sample(), key: torch.ones(1)})


def test_original_parser_rejects_dirty_previous_references_and_invalid_shapes():
    api = original()
    raw = sample()
    positive, _ = api.dmd2.prepare_batch(raw)
    data = {**raw, "ref_video_latents": None, "ref_audio_latents": None,
            "ref_visual_anchor": positive["ref_visual_anchor"], "ref_audio_anchor": positive["ref_audio_anchor"],
            "packed": positive["packed"], "imgvid_cond_noise_aug": 0.5}
    with pytest.raises(ValueError, match="already noise-augmented"):
        api.dmd2.prepare_batch(data)
    for key, value in [("input_latents", torch.randn(1, 23, 2, 4, 4)),
                       ("audio_input_latents", torch.randn(1, 32, 3)),
                       ("prompt_embeds", torch.randn(3, 5119)),
                       ("input_latents_video", torch.randn(1, 24, 2, 4, 4))]:
        with pytest.raises(ValueError):
            api.dmd2.prepare_batch({**raw, key: value})
    with pytest.raises(ValueError, match="negative"):
        api.dmd2.prepare_batch(raw, 3.0)


def restart_worker(mode, root):
    """Each invocation owns actual original construction in its own process."""
    torch.set_num_threads(1)
    api = original()
    root = Path(root)
    teacher, caches = root / "teacher", root / "caches"
    if mode == "A":
        teacher.mkdir(parents=True)
        caches.mkdir()
        api.training.seed_all(921)
        model = api.model.MiniMaxH3DiT(**tiny_architecture())
        save_file(model.state_dict(), str(teacher / "model.safetensors"))
        (teacher / "config.json").write_text(json.dumps(tiny_architecture()))
        for index in range(3):
            torch.save(sample(index), caches / f"sample-{index}.pth")
    config = explicit_g1f1_config(gan_start_step=2)  # GAN transition fixture only.
    model, architecture, teacher_identity = api.training.load_teacher(teacher, "cpu", torch.float32)
    trainer, optimizers, events = setup_trainer(config, model=model, observe=True)
    frozen = {k: p.detach().clone() for k, p in trainer.adapter.named_parameters() if ".adapters." not in k}
    dataset = api.training.CacheDataset(caches, config.teacher_cfg_scale)
    assert len(dataset) == 3
    content = [hashlib.sha256(path.read_bytes()).hexdigest() for path in dataset.files]
    assert len(set(content)) == 3
    order = list(api.cache.NodeLocalSampler(len(dataset), 0, 1, len(dataset), seed=418))
    metadata = dict(format_version=1, dmd=asdict(config), teacher=teacher_identity,
                    architecture=architecture, world_size=1, local_world_size=1,
                    data=dataset.identity(), seed=777, attention="torch", torch_version=str(torch.__version__),
                    source_manifest_sha256=provider.source_identity()["manifest_sha256"],
                    precision="float32_cpu_tiny_fixture", cache_content_sha256=content,
                    fixture="GAN_START_2_EXPLICIT_NOT_CANONICAL_500")
    checkpoint = root / "step-2"
    api.training.seed_all(777)
    cursor = dict(epoch=0, offset=0)
    if mode == "A":
        for index in order[:2]:
            inputs, negative = dataset[index]
            trainer.update(real(inputs), inputs, negative)
            cursor["offset"] += 1
        api.training.save_checkpoint(checkpoint, trainer, optimizers, metadata, cursor.copy())
    else:
        with pytest.raises(ValueError, match="complete"):
            api.training.load_checkpoint(root / "missing", trainer, optimizers, metadata)
        with pytest.raises(ValueError, match="dmd"):
            api.training.load_checkpoint(checkpoint, trainer, optimizers, {**metadata, "dmd": {}})
        # This is deliberately the exact trusted original local pickle loader.
        cursor = api.training.load_checkpoint(checkpoint, trainer, optimizers, metadata)
        assert cursor == dict(epoch=0, offset=2)
    assert trainer.generator_step == trainer.critic_step == 2
    next_index = order[cursor["offset"]]
    inputs, negative = dataset[next_index]
    input_before = copy.deepcopy(inputs)
    metrics = trainer.update(real(inputs), inputs, negative)
    cursor["offset"] += 1
    equal_tree(input_before, inputs)
    for name, value in frozen.items():
        equal_tree(value, dict(trainer.adapter.named_parameters())[name])
    state = dict(adapters=api.training.adapter_state(trainer.adapter),
                 heads=trainer.discriminator.state_dict(),
                 optimizers={name: opt.state_dict() for name, opt in optimizers.items()},
                 rng=api.training.rng_state(), cursor=cursor, metrics=metrics,
                 generator_step=trainer.generator_step, critic_step=trainer.critic_step,
                 schedule=trainer.schedule, data=dataset.identity(), content_sha256=content,
                 order=order, next_index=next_index,
                 roles=[(name, layer.role) for name, layer in trainer.adapter.layers],
                 requires_grad={name: p.requires_grad for name, p in trainer.adapter.named_parameters()},
                 head_requires_grad=[p.requires_grad for p in trainer.discriminator.parameters()])
    torch.save(dict(pid=os.getpid(), state=state), root / f"result-{mode}.pt")
    print(json.dumps(dict(process=mode, pid=os.getpid(), fixture="GAN_START_2", cache_count=len(dataset),
                         next_index=next_index, cursor=cursor, metrics=metrics,
                         lora_tensors=len(state["adapters"]), head_tensors=len(state["heads"]),
                         canonical_gan_start_step=api.dmd2.DMD2Config().gan_start_step), sort_keys=True))


def test_fresh_process_original_checkpoint_exact_continuation(tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for mode in ("A", "B"):
        result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), mode, str(tmp_path)],
                                env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        print(result.stdout.strip())
    # Only our own just-created proof artifacts are unpickled, never remote data.
    first = torch.load(tmp_path / "result-A.pt", map_location="cpu", weights_only=False)
    second = torch.load(tmp_path / "result-B.pt", map_location="cpu", weights_only=False)
    assert first["pid"] != second["pid"] != os.getpid()
    equal_tree(first["state"], second["state"])
    state = second["state"]
    assert state["cursor"] == dict(epoch=0, offset=3) and state["metrics"]["gan_active"]
    assert state["generator_step"] == state["critic_step"] == 3
    for name, opt in state["optimizers"].items():
        assert opt["state"] and all("exp_avg" in value and "exp_avg_sq" in value for value in opt["state"].values())
        clocks = {int(value["step"].item()) for value in opt["state"].values()}
        assert clocks == ({3} if name == "student" else {1, 3})


if __name__ == "__main__":
    restart_worker(sys.argv[1], sys.argv[2])
