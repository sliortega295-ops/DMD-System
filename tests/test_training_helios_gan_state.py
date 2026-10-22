"""Original GAN helper/head on tiny CPU construction, not Helios model parity.

No CUDA shims, new loss, update solver or public W3 training route. The head
definition and trainer name-binding/reset blocks are compiled from the pinned
original source; all optimizer commits run through the real helper/adapter.
"""
import ast
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter
from distillschedule.training.checkpoint import (
    pack_state, unpack_state, rng_state, save_checkpoint, restore_checkpoint,
)
from distillschedule.training.native_state import engine_state
from distillschedule.workloads.best_known.registry import repository_root, file_sha256
from test_training_end_to_end import equal


SOURCES = {
    "reference/train_helios.py": "f1dd9c3076e737aa1e578cc47e50c14ba8fa4a7623141dd7d8258c36e4534993",
    "source/helios/utils/utils_helios_post.py": "e3ff9e5f8bf025ee740e436bc0c209c972f1e56eb0236b968fe6bd8284107004",
    "source/helios/modules/transformer_helios.py": "5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886",
}
IDENTITY = {"scope": "ORIGINAL_GAN_HELPER_EXACT_HEAD_TINY_CPU_NOT_PRETRAINED_OR_PUBLIC_W3"}


def _code(node, path):
    return compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec")


@pytest.fixture(scope="module")
def original():
    from distillschedule.strategies.wan_capture import native_primitives
    root = repository_root() / "third_party/helios_compat"
    assert all(file_sha256(root / name) == sha for name, sha in SOURCES.items())
    post = native_primitives().post
    assert Path(post.__file__).resolve() == root / "source/helios/utils/utils_helios_post.py"
    assert Path(post._critic_loss.__code__.co_filename).resolve() == Path(post.__file__).resolve()
    path = root / "source/helios/modules/transformer_helios.py"
    head = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == "Discriminator3DHead")
    namespace = {"nn": nn}
    exec(_code(head, path), namespace)
    head_type = namespace["Discriminator3DHead"]
    assert Path(head_type.forward.__code__.co_filename) == path
    trainer = root / "reference/train_helios.py"
    nodes = list(ast.walk(ast.parse(trainer.read_text())))
    bind = [n for n in nodes if isinstance(n, ast.If) and any(
        isinstance(a, ast.Assign) and any(isinstance(k, ast.Name) and k.id == "gan_critic_trainable_params"
                                         for k in a.targets) for a in n.body)]
    reset = [n for n in nodes if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
             and n.test.id == "USE_GAN" and any(isinstance(a, ast.Assign)
             and isinstance(a.value, ast.Constant) and a.value.value is True
             and any(isinstance(k, ast.Attribute) and k.attr == "requires_grad" for k in a.targets)
             for a in ast.walk(n))]
    assert len(bind) == len(reset) == 1
    return SimpleNamespace(post=post, Head=head_type, bind=_code(bind[0], trainer),
                           reset=_code(reset[0], trainer))


def _owner(original, seed=57):
    class TinyConstruction(nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = nn.Conv3d(16, 16, 1).requires_grad_(False)
            self.lora = nn.Conv3d(16, 16, 1, bias=False)
            self.gan_head = original.Head(16, 32)
            self.register_buffer("flow_scale", torch.tensor(.9))
            # Fresh owners reconstruct the SAME immutable base, independently
            # of initialization of the optimizer-owned tensors.
            with torch.no_grad():
                self.frozen.weight.fill_(.02)
                self.frozen.bias.fill_(.01)

        def forward(self, hidden_states, gan_mode=False, **kwargs):
            flow = self.flow_scale * self.frozen(hidden_states) + self.lora(hidden_states)
            return (flow, self.gan_head(flow)) if gan_mode else (flow,)

    torch.manual_seed(seed)
    model = TinyConstruction()
    binding = {"args": SimpleNamespace(training_config=SimpleNamespace(is_use_gan=True)),
               "real_score_model": model, "critic_trainable_modules": ["gan_head"]}
    exec(original.bind, binding)
    base = binding["gan_base_critic_trainable_params"]
    extra = binding["gan_extra_critic_trainable_params"]
    assert base == {"lora.weight"}
    assert base | extra == binding["gan_critic_trainable_params"]
    named = dict(model.named_parameters())
    # Bind the original name sets, not current flags or a later substring scan.
    optimizer = torch.optim.AdamW([named[n] for n in named if n in base | extra], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.7)
    engine = FSDP2EngineAdapter(model, optimizer, scheduler, physical_step=0, global_batch=1)
    return SimpleNamespace(model=model, engine=engine, binding=binding, base=base, extra=extra,
                           frozen={"frozen.weight", "frozen.bias"})


def _state(owner, separate):
    from distillschedule.training.helios_gan_state import critic_state
    return critic_state(owner.engine, gan_base_critic_trainable_params=owner.base,
                        gan_extra_critic_trainable_params=owner.extra,
                        frozen_parameter_names=owner.frozen, is_separate_gan_grad=separate)


def _update(original, owner, separate, checkpoint_probe=None):
    from runtime.gan_rolephase_official import HeliosAcceleratorAdapter
    # Execute the exact original full-trainer reset before the next real helper.
    exec(original.reset, {**owner.binding, "USE_GAN": True})
    calls = []
    engine = owner.engine
    backward, step = engine.backward, engine.step

    def observed_backward(loss):
        calls.append(("backward", engine._gradient_accumulation_boundary))
        backward(loss)
        if checkpoint_probe is not None:
            checkpoint_probe(engine._gradient_accumulation_boundary)

    def observed_step():
        calls.append(("commit", engine._gradient_accumulation_boundary))
        return step()

    engine.backward, engine.step = observed_backward, observed_step
    try:
        shape = (1, 16, 9, 16, 16)
        prompt = torch.randn(1, 2, 4)
        loss, logs = original.post._critic_loss(
            args=SimpleNamespace(training_config=SimpleNamespace()),
            critic_accelerator=HeliosAcceleratorAdapter(torch.device("cpu"), engine),
            fake_score_model=owner.model, transformer=None, scheduler=None,
            noise=torch.randn(shape), prompt_embeds=prompt, is_gan_low_vram_mode=True,
            is_use_gan=True, is_separate_gan_grad=separate,
            gan_base_critic_trainable_params=owner.base, gan_extra_critic_trainable_params=owner.extra,
            gan_vae_latents=torch.randn(shape), gan_prompt_embeds=prompt,
            precomputed_generated_image_or_video=torch.randn(shape),
            r1_weight=.01, r2_weight=.01, global_step=1000)
    finally:
        engine.backward, engine.step = backward, step
    assert calls == [("backward", False), ("backward", True), ("commit", True)]
    assert logs["critic_optimizer_committed"] is True
    assert all(logs[k] > 0 for k in ("denoising_loss", "gan_D_loss", "r1_loss", "r2_loss"))
    assert all(p.grad is None for p in owner.model.parameters())
    assert torch.cuda.is_initialized() is False
    return loss.detach(), logs, calls


def _assert_moments(owner, steps):
    named = dict(owner.model.named_parameters())
    assert set(owner.engine.optimizer.state) == {named[n] for n in owner.base | owner.extra}
    for name in owner.base | owner.extra:
        slots = owner.engine.optimizer.state[named[name]]
        assert slots["step"].item() == steps, name
        for field in ("exp_avg", "exp_avg_sq"):
            assert slots[field].shape == named[name].shape, name
            assert torch.count_nonzero(slots[field]) > 0, (name, field)
    assert owner.engine.global_steps == owner.engine.global_samples == owner.engine.step_calls == steps
    assert owner.engine.lr_scheduler.last_epoch == steps


@pytest.mark.parametrize("separate", [True, False])
def test_actual_original_helper_characterization_and_generic_guard(original, tmp_path, separate):
    owner = _owner(original)
    state = engine_state({"critic": owner.engine})
    rejections = []

    def checkpoint_probe(boundary):
        with pytest.raises(ValueError, match="outstanding backward/accumulation|unconsumed gradients"):
            save_checkpoint(tmp_path / "incomplete", state, identity=IDENTITY,
                            cycle=0, cursor=0, versions={"critic": 0})
        rejections.append(boundary)

    _update(original, owner, separate, checkpoint_probe)
    assert rejections == [False, True] and not (tmp_path / "incomplete").exists()
    _assert_moments(owner, 1)
    named = dict(owner.model.named_parameters())
    assert all(named[n].requires_grad is (not separate) for n in owner.base)
    assert all(named[n].requires_grad is True for n in owner.extra)
    assert all(named[n].requires_grad is False for n in owner.frozen)
    fresh = _owner(original, seed=999)
    before, random_before = pack_state(engine_state({"critic": fresh.engine})), rng_state()
    if separate:
        with pytest.raises(ValueError, match="native trainability changed"):
            unpack_state(pack_state(state), engine_state({"critic": fresh.engine}))
        assert equal(before, pack_state(engine_state({"critic": fresh.engine})))
        assert equal(random_before, rng_state())
    else:
        unpack_state(pack_state(state), engine_state({"critic": fresh.engine}))
        assert equal(pack_state(state), pack_state(engine_state({"critic": fresh.engine})))


@pytest.mark.parametrize("separate", [True, False])
def test_original_gan_checkpoint_fresh_owner_next_update_exact(original, tmp_path, separate):
    control = _owner(original)
    state = _state(control, separate)
    mid_update_rejections = []

    def checkpoint_probe(boundary):
        with pytest.raises(ValueError, match="illegal critic|outstanding backward/accumulation|unconsumed gradients"):
            save_checkpoint(tmp_path / "mid-update", state, identity=IDENTITY,
                            cycle=0, cursor=0, versions={"critic": 0})
        mid_update_rejections.append(boundary)

    _update(original, control, separate, checkpoint_probe)
    assert mid_update_rejections == [False, True]
    _assert_moments(control, 1)
    control.model.frozen.eval()
    control.model.gan_head.eval()
    control.model.flow_scale.fill_(.83)
    random.seed(773)
    checkpoint = save_checkpoint(tmp_path / "complete", state, identity=IDENTITY,
                                 cycle=1, cursor=1, versions={"critic": 1})
    saved, saved_rng = pack_state(state), rng_state()
    weights = saved["critic.weights"]["value"]
    assert set(weights["tensors"]) == control.base | control.extra
    assert weights["phase"] == ("separate_gan_complete" if separate else "ready")
    loss, logs, _ = _update(original, control, separate)
    expected, expected_rng = pack_state(state), rng_state()
    expected_python_draw = random.random()

    # New owner has genuinely different live trainables. Restore must not
    # replace Parameters, their hooks, or attached zero-gradient views.
    fresh = _owner(original, seed=999)
    fresh_state = _state(fresh, separate)
    parameters = dict(fresh.model.named_parameters())
    hook_calls = []
    hooks = {n: p.register_hook(lambda g, name=n: hook_calls.append(name) or g)
             for n, p in parameters.items() if n in fresh.base | fresh.extra}
    for name in fresh.base | fresh.extra:
        parameters[name].grad = torch.zeros_like(parameters[name])
    grad_ids = {n: id(p.grad) for n, p in parameters.items()}
    hook_ids = {n: tuple(p._backward_hooks.items()) for n, p in parameters.items() if n in hooks}
    before_frozen = {n: parameters[n].detach().clone() for n in fresh.frozen}
    forward_calls = []
    forward_handle = fresh.model.gan_head.register_forward_hook(lambda *a: forward_calls.append(True))
    try:
        torch.randn(17); random.random()
        restore_checkpoint(checkpoint, fresh_state, identity=IDENTITY)
        assert equal(saved, pack_state(fresh_state))
        assert equal(saved_rng, rng_state())
        assert all(p is dict(fresh.model.named_parameters())[n] for n, p in parameters.items())
        assert grad_ids == {n: id(p.grad) for n, p in parameters.items()}
        assert hook_ids == {n: tuple(p._backward_hooks.items()) for n, p in parameters.items() if n in hooks}
        assert all(torch.equal(parameters[n], value) and parameters[n].requires_grad is False
                   for n, value in before_frozen.items())
        resumed_loss, resumed_logs, _ = _update(original, fresh, separate)
        _assert_moments(fresh, 2)
        assert equal(loss, resumed_loss) and equal(logs, resumed_logs)
        assert equal(expected, pack_state(fresh_state))
        assert equal(expected_rng, rng_state())
        assert random.random() == expected_python_draw
        assert set(hook_calls) == fresh.base | fresh.extra
        assert forward_calls == [True]
        assert forward_handle.id in fresh.model.gan_head._forward_hooks
        (tmp_path / "proof.json").write_text(json.dumps({
            "scope": IDENTITY["scope"], "source_sha256": SOURCES,
            "head_import": "EXACT_CLASS_AST_NOT_FULL_HELIOS_TRANSFORMER_IMPORT",
            "trainer_name_binding_and_pre_critic_reset": "EXACT_ORIGINAL_AST_BLOCKS",
            "execution": "SINGLE_CPU_REAL_FSDP2_ADAPTER_NO_DISTRIBUTED_GROUP",
            "separate_grad": separate, "saved_phase": weights["phase"],
            "real_next_helper_calls": [["backward", False], ["backward", True], ["commit", True]],
            "actual_both_backward_boundaries_refuse_checkpoint_publication": mid_update_rejections,
            "next_losses": {k: logs[k] for k in ("denoising_loss", "gan_D_loss", "r1_loss", "r2_loss")},
            "owned_named_moments": {n: {"step": int(fresh.engine.optimizer.state[parameters[n]]["step"]),
                "exp_avg_nonzero": int(torch.count_nonzero(fresh.engine.optimizer.state[parameters[n]]["exp_avg"])),
                "exp_avg_sq_nonzero": int(torch.count_nonzero(fresh.engine.optimizer.state[parameters[n]]["exp_avg_sq"]))}
                for n in sorted(fresh.base | fresh.extra)},
            "live_parameter_hooks_and_zero_gradient_views_preserved_on_restore": True,
            "frozen_flags_and_reconstructed_bytes_preserved": True,
            "weights_buffers_modes_named_adam_scheduler_clocks_exact_next_update": True,
            "python_torch_numpy_rng_exact": True, "cuda_initialized": torch.cuda.is_initialized(),
            "unimplemented": ["full Helios model/forward/pretrained weights", "GAN Generator/DMD rollout",
                "public fresh W3 training", "complete-cycle expected counter/data integration", "GPU/NCCL/world16",
                "performance", "quality", "cross-device numerical parity"],
        }, indent=2) + "\n")
    finally:
        for handle in hooks.values():
            handle.remove()
        forward_handle.remove()


@pytest.mark.parametrize("separate", [True, False])
def test_explicit_binding_after_update_and_original_ready_reset_phase(original, tmp_path, separate):
    owner = _owner(original)
    _update(original, owner, separate)
    state = _state(owner, separate)  # Binding must not drop disabled base names.
    assert set(pack_state(state)["critic.weights"]["value"]["tensors"]) == owner.base | owner.extra
    exec(original.reset, {**owner.binding, "USE_GAN": True})
    checkpoint = save_checkpoint(tmp_path / "ready", state, identity=IDENTITY,
                                 cycle=1, cursor=1, versions={"critic": 1})
    assert pack_state(state)["critic.weights"]["value"]["phase"] == "ready"
    fresh = _owner(original, seed=999)
    fresh_state = _state(fresh, separate)
    restore_checkpoint(checkpoint, fresh_state, identity=IDENTITY)
    assert equal(pack_state(state), pack_state(fresh_state))
    _assert_moments(fresh, 1)
    assert all(p.requires_grad is (n not in fresh.frozen) for n, p in fresh.model.named_parameters())


def test_actual_nonzero_bootstrap_keeps_absolute_and_local_commit_clocks_distinct(original, tmp_path):
    owner = _owner(original)
    _update(original, owner, True)
    _update(original, owner, True)
    # Real already-committed state, rebound exactly as the existing adapter's
    # physical bootstrap constructor does. No clock correction or fake commit.
    previous = owner.engine
    owner.engine = FSDP2EngineAdapter(owner.model, previous.optimizer, previous.lr_scheduler,
        physical_step=previous.global_steps, global_batch=previous.train_batch_size())
    assert owner.engine.global_steps == 2 and owner.engine.step_calls == 0
    state = _state(owner, True)
    with pytest.raises(ValueError, match="completed critic update"):
        save_checkpoint(tmp_path / "before-local-commit", state, identity=IDENTITY,
                        cycle=0, cursor=0, versions={"critic": 2})
    _update(original, owner, True)
    clocks = pack_state(state)["critic.clocks"]["value"]["counters"]
    assert clocks == {"global_steps": 3, "global_samples": 3, "_step_calls": 1}
    checkpoint = save_checkpoint(tmp_path / "bootstrapped", state, identity=IDENTITY,
                                 cycle=1, cursor=1, versions={"critic": 3})
    fresh = _owner(original, seed=999)
    fresh_state = _state(fresh, True)
    restore_checkpoint(checkpoint, fresh_state, identity=IDENTITY)
    assert equal(pack_state(state), pack_state(fresh_state))
    assert fresh.engine.global_steps == 3 and fresh.engine.step_calls == 1
    assert all(slots["step"].item() == 3 for slots in fresh.engine.optimizer.state.values())


@pytest.mark.parametrize("case", ["base_unknown", "base_missing", "extra_missing", "frozen_missing",
    "extra_is_frozen", "overlap", "duplicate", "bare_string", "policy_not_bool"])
def test_wrong_original_name_binding_or_policy_rejected_without_mutation(original, case):
    from distillschedule.training.helios_gan_state import critic_state
    owner = _owner(original)
    arguments = dict(gan_base_critic_trainable_params=owner.base,
                     gan_extra_critic_trainable_params=owner.extra,
                     frozen_parameter_names=owner.frozen, is_separate_gan_grad=True)
    if case == "base_unknown":
        arguments["gan_base_critic_trainable_params"] = {"not_an_original_parameter"}
    elif case == "base_missing":
        arguments["gan_base_critic_trainable_params"] = set()
    elif case == "extra_missing":
        arguments["gan_extra_critic_trainable_params"] = owner.extra - {next(iter(owner.extra))}
    elif case == "frozen_missing":
        arguments["frozen_parameter_names"] = {"frozen.weight"}
    elif case == "extra_is_frozen":
        arguments["gan_extra_critic_trainable_params"] = owner.extra | owner.frozen
        arguments["frozen_parameter_names"] = set()
    elif case == "overlap":
        arguments["gan_extra_critic_trainable_params"] = owner.extra | owner.base
    elif case == "duplicate":
        arguments["gan_base_critic_trainable_params"] = ["lora.weight", "lora.weight"]
    elif case == "bare_string":
        arguments["gan_base_critic_trainable_params"] = "lora.weight"
    else:
        arguments["is_separate_gan_grad"] = 1
    before = pack_state(engine_state({"critic": owner.engine}))
    before_rng = rng_state()
    with pytest.raises(ValueError, match="original Helios GAN critic state"):
        critic_state(owner.engine, **arguments)
    assert equal(before, pack_state(engine_state({"critic": owner.engine})))
    assert equal(before_rng, rng_state())


@pytest.mark.parametrize("case", ["initial", "accumulation", "active_backward", "pending_gradient",
    "denoising_mask", "generator_mask", "frozen_enabled", "joint_separate_mask"])
def test_publication_only_after_quiescent_original_critic_commit(original, tmp_path, case):
    owner = _owner(original)
    separate = case != "joint_separate_mask"
    state = _state(owner, separate)
    if case != "initial":
        _update(original, owner, separate)
    named = dict(owner.model.named_parameters())
    if case == "accumulation":
        owner.engine.set_gradient_accumulation_boundary(is_boundary=False)
    elif case == "active_backward":
        owner.engine._active_backward = True
    elif case == "pending_gradient":
        named["lora.weight"].grad = torch.ones_like(named["lora.weight"])
    elif case == "denoising_mask":
        for name in owner.base:
            named[name].requires_grad = True
        for name in owner.extra:
            named[name].requires_grad = False
    elif case == "generator_mask":
        for name in owner.base | owner.extra:
            named[name].requires_grad = False
    elif case == "frozen_enabled":
        named["frozen.weight"].requires_grad = True
    elif case == "joint_separate_mask":
        named["lora.weight"].requires_grad = False
    before, before_rng = pack_state(engine_state({"critic": owner.engine})), rng_state()
    with pytest.raises(ValueError, match="completed critic update|outstanding backward|unconsumed gradients|illegal critic"):
        save_checkpoint(tmp_path / "forbidden", state, identity=IDENTITY,
                        cycle=1, cursor=1, versions={"critic": 1})
    assert not (tmp_path / "forbidden").exists()
    assert list(tmp_path.iterdir()) == []
    assert equal(before, pack_state(engine_state({"critic": owner.engine})))
    assert equal(before_rng, rng_state())


def _rewrite_payload(checkpoint, payload):
    """Rehash deliberately malformed data to exercise semantic, not hash gates."""
    torch.save(payload, checkpoint / "state.pt")
    path = checkpoint / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["state.pt"] = file_sha256(checkpoint / "state.pt")
    path.write_text(json.dumps(manifest))
    (checkpoint / "_SUCCESS").write_text(file_sha256(path) + "\n")


@pytest.mark.parametrize("case,match", [
    ("phase_unknown", "phase/mask"), ("phase_ready_mismatch", "phase/mask"),
    ("base_enabled_mismatch", "phase/mask"), ("extra_disabled", "illegal critic"),
    ("frozen_enabled", "illegal critic"), ("flag_nonbool", "extra-state type"),
    ("flag_missing", "module keys"), ("mode_nonbool", "extra-state type"),
    ("mode_missing", "module keys"), ("policy_toggle", "policy changed"),
    ("policy_nonbool", "extra-state type"), ("policy_base_name", "policy changed"),
    ("policy_extra_name", "policy changed"), ("policy_frozen_name", "policy changed"),
    ("policy_source", "policy changed"), ("policy_schema", "policy changed"),
    ("tensor_missing", "module keys"), ("buffer_dtype", "shape/dtype"),
    ("raw_moment_dtype", "shape/dtype"), ("raw_moment_shape", "shape/dtype"),
    ("optimizer_names", "ownership"), ("clock_boundary", "complete update boundary"),
    ("clock_negative", "complete update boundary"), ("clock_policy", "complete update boundary"),
    ("clock_initial", "completed critic update"), ("clock_no_commit", "completed critic update"),
    ("clock_commits_ahead", "completed critic update"), ("clock_samples", "completed critic update"),
    ("clock_policy_bool", "extra-state type"),
    ("extra_field", "phase state fields"),
])
def test_malformed_checkpoint_rejected_before_weights_flags_hooks_or_rng_mutation(original, tmp_path, case, match):
    owner = _owner(original)
    state = _state(owner, True)
    _update(original, owner, True)
    checkpoint = save_checkpoint(tmp_path / "bad", state, identity=IDENTITY,
                                 cycle=1, cursor=1, versions={"critic": 1})
    payload = torch.load(checkpoint / "state.pt", weights_only=True)
    weights = payload["state"]["critic.weights"]["value"]
    if case == "phase_unknown":
        weights["phase"] = "denoising_only"
    elif case == "phase_ready_mismatch":
        weights["phase"] = "ready"
    elif case == "base_enabled_mismatch":
        weights["requires_grad"]["lora.weight"] = True
    elif case == "extra_disabled":
        weights["requires_grad"][sorted(owner.extra)[0]] = False
    elif case == "frozen_enabled":
        weights["requires_grad"]["frozen.weight"] = True
    elif case == "flag_nonbool":
        weights["requires_grad"]["lora.weight"] = 0
    elif case == "flag_missing":
        weights["requires_grad"].pop("lora.weight")
    elif case == "mode_nonbool":
        weights["training"][""] = 1
    elif case == "mode_missing":
        weights["training"].pop("gan_head")
    elif case == "policy_toggle":
        weights["policy"]["is_separate_gan_grad"] = False
    elif case == "policy_nonbool":
        weights["policy"]["is_separate_gan_grad"] = 1
    elif case in {"policy_base_name", "policy_extra_name", "policy_frozen_name"}:
        weights["policy"][case.split("_")[1]][0] = "not_the_original_name"
    elif case == "policy_source":
        weights["policy"]["source"]["source/helios/utils/utils_helios_post.py"] = "0" * 64
    elif case == "policy_schema":
        weights["policy"]["schema"] = "unbound_dynamic_requires_grad"
    elif case == "tensor_missing":
        weights["tensors"].pop("lora.weight")
    elif case == "buffer_dtype":
        weights["buffers"]["flow_scale"] = weights["buffers"]["flow_scale"].double()
    elif case in {"raw_moment_dtype", "raw_moment_shape"}:
        adam = payload["state"]["critic.optimizer"]["value"]["optimizer"]
        first = next(iter(adam["state"].values()))
        first["exp_avg"] = first["exp_avg"].double() if case.endswith("dtype") else first["exp_avg"].flatten()[:1]
    elif case == "optimizer_names":
        payload["state"]["critic.optimizer"]["value"]["parameters"][0].reverse()
    elif case == "clock_boundary":
        payload["state"]["critic.clocks"]["value"]["accumulation_boundary"] = False
    elif case == "clock_negative":
        payload["state"]["critic.clocks"]["value"]["counters"]["global_steps"] = -1
    elif case == "clock_policy":
        payload["state"]["critic.clocks"]["value"]["policy"]["gradient_clip"] = 10.
    elif case == "clock_initial":
        payload["state"]["critic.clocks"]["value"]["counters"] = {
            "global_steps": 0, "global_samples": 0, "_step_calls": 0}
    elif case == "clock_no_commit":
        payload["state"]["critic.clocks"]["value"]["counters"]["_step_calls"] = 0
    elif case == "clock_commits_ahead":
        payload["state"]["critic.clocks"]["value"]["counters"]["_step_calls"] = 2
    elif case == "clock_samples":
        payload["state"]["critic.clocks"]["value"]["counters"]["global_samples"] = 0
    elif case == "clock_policy_bool":
        payload["state"]["critic.clocks"]["value"]["policy"]["global_batch"] = True
    else:
        weights["unbound_flag_policy"] = True
    _rewrite_payload(checkpoint, payload)
    fresh = _owner(original, seed=999)
    fresh_state = _state(fresh, True)
    parameters = dict(fresh.model.named_parameters())
    handle = parameters["lora.weight"].register_hook(lambda grad: grad)
    parameters["lora.weight"].grad = torch.zeros_like(parameters["lora.weight"])
    gradient = parameters["lora.weight"].grad
    before, before_rng = pack_state(fresh_state), rng_state()
    try:
        with pytest.raises(ValueError, match=match):
            restore_checkpoint(checkpoint, fresh_state, identity=IDENTITY)
        assert equal(before, pack_state(fresh_state))
        assert equal(before_rng, rng_state())
        assert all(p is dict(fresh.model.named_parameters())[n] for n, p in parameters.items())
        assert parameters["lora.weight"].grad is gradient
        assert handle.id in parameters["lora.weight"]._backward_hooks
    finally:
        handle.remove()


@pytest.mark.parametrize("case", ["pending_gradient", "accumulation", "active_backward"])
def test_restore_refuses_a_live_incomplete_receiver(original, tmp_path, case):
    owner = _owner(original)
    state = _state(owner, True)
    _update(original, owner, True)
    checkpoint = save_checkpoint(tmp_path / "complete", state, identity=IDENTITY,
                                 cycle=1, cursor=1, versions={"critic": 1})
    fresh = _owner(original, seed=999)
    fresh_state = _state(fresh, True)
    if case == "pending_gradient":
        parameter = dict(fresh.model.named_parameters())["lora.weight"]
        parameter.grad = torch.ones_like(parameter)
    elif case == "accumulation":
        fresh.engine.set_gradient_accumulation_boundary(is_boundary=False)
    else:
        fresh.engine._active_backward = True
    before, before_rng = pack_state(fresh_state), rng_state()
    with pytest.raises(ValueError, match="outstanding backward/accumulation|unconsumed gradients"):
        restore_checkpoint(checkpoint, fresh_state, identity=IDENTITY)
    assert equal(before, pack_state(fresh_state)) and equal(before_rng, rng_state())


@pytest.mark.parametrize("after_binding", [False, True])
def test_actual_original_source_drift_is_rejected_without_touching_vendor(original, tmp_path, monkeypatch, after_binding):
    import distillschedule.training.helios_gan_state as gan_state
    root = repository_root() / "third_party/helios_compat"
    for name in SOURCES:
        target = tmp_path / "third_party/helios_compat" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    monkeypatch.setattr(gan_state, "repository_root", lambda: tmp_path)
    owner = _owner(original)
    state = _state(owner, True) if after_binding else None
    changed = tmp_path / "third_party/helios_compat/source/helios/utils/utils_helios_post.py"
    changed.write_text(changed.read_text() + "\n# deliberate isolated drift\n")
    before, before_rng = pack_state(engine_state({"critic": owner.engine})), rng_state()
    with pytest.raises(ValueError, match="source changed"):
        pack_state(state) if after_binding else _state(owner, True)
    assert equal(before, pack_state(engine_state({"critic": owner.engine})))
    assert equal(before_rng, rng_state())
