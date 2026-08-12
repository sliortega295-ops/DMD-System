import pytest

from runtime.routeb_full5 import (
    RouteBCandidateBTopology,
    RouteBFull5Contract,
    RouteBFull5State,
    RouteBPhase,
)


def _load_candidate_a_executor():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "distillschedule"
        / "probe_routeb_full5_candidate_a.py"
    )
    spec = importlib.util.spec_from_file_location("routeb_candidate_a_executor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _enter_critic(contract: RouteBFull5Contract) -> RouteBFull5State:
    state = RouteBFull5State(contract)
    state.begin_step10a()
    state.finish_step10a(generator_before=2, generator_after=3)
    state.finish_transition(
        generator_version=3,
        critic_version=10,
        ema_step=10,
        target_dp_world_size=contract.critic_dp_world_size,
    )
    return state


def _run(contract: RouteBFull5Contract) -> RouteBFull5State:
    state = _enter_critic(contract)
    for micro_wave in range(contract.physical_critic_micro_waves):
        version = contract.critic_version_for_micro_wave(micro_wave)
        state.consume_micro_wave(micro_wave, observed_critic_version=version)
        if (
            contract.micro_index_in_update(micro_wave) + 1
            == contract.critic_micro_waves_per_update
        ):
            update = contract.update_for_critic_micro_wave(micro_wave)
            state.commit_optimizer(
                update, critic_before=10 + update, critic_after=11 + update
            )
            state.commit_ema(
                update,
                ema_before=10 + update,
                ema_after=11 + update,
                generator_version=3,
            )
    state.assert_flushed()
    return state


def test_candidate_a_is_matched_batch_gas2_full5() -> None:
    contract = RouteBFull5Contract.candidate_a()
    assert contract.critic_dp_world_size == 8
    assert contract.generator_pool_width == 8
    assert contract.global_critic_batch == 16
    assert contract.critic_micro_waves_per_update == 2
    assert contract.generator_production_rounds_per_update == 2
    assert contract.physical_critic_micro_waves == 10
    assert contract.physical_generator_rounds == 10
    assert contract.logical_samples == 80
    assert [contract.critic_version_for_micro_wave(i) for i in range(10)] == [
        10,
        10,
        11,
        11,
        12,
        12,
        13,
        13,
        14,
        14,
    ]
    assert [contract.source_trace_rank(r, 3) for r in range(2)] == [3, 11]
    assert _run(contract).phase is RouteBPhase.FLUSHED


def test_candidate_b_keeps_production_and_critic_axes_distinct() -> None:
    contract = RouteBFull5Contract.candidate_b()
    assert contract.critic_dp_world_size == 12
    assert contract.generator_pool_width == 4
    assert contract.global_critic_batch == 12
    assert contract.critic_micro_waves_per_update == 1
    assert contract.generator_production_rounds_per_update == 3
    assert contract.physical_critic_micro_waves == 5
    assert contract.physical_generator_rounds == 15
    assert contract.logical_samples == 60
    assert [contract.source_trace_rank(r, 2) for r in range(3)] == [2, 6, 10]
    assert _run(contract).phase is RouteBPhase.FLUSHED


def test_candidate_b_topology_is_node_local_four_to_twelve() -> None:
    topology = RouteBCandidateBTopology(RouteBFull5Contract.candidate_b())
    assert topology.critics_for_generator(0) == (2, 3, 4)
    assert topology.critics_for_generator(1) == (5, 6, 7)
    assert topology.critics_for_generator(8) == (10, 11, 12)
    assert topology.critics_for_generator(9) == (13, 14, 15)
    assert topology.producer_for_critic(2) == (0, 0)
    assert topology.producer_for_critic(4) == (0, 2)
    assert topology.producer_for_critic(15) == (9, 2)
    assert len(topology.edges) == 12
    assert all(g // 8 == c // 8 for g, c, _ in topology.edges)


def test_gas2_cannot_commit_after_one_micro_wave() -> None:
    state = _enter_critic(RouteBFull5Contract.candidate_a())
    state.consume_micro_wave(0, observed_critic_version=10)
    with pytest.raises(RuntimeError, match="requires 2"):
        state.commit_optimizer(0, critic_before=10, critic_after=11)


def test_next_update_cannot_consume_before_commit() -> None:
    state = _enter_critic(RouteBFull5Contract.candidate_a())
    state.consume_micro_wave(0, observed_critic_version=10)
    state.consume_micro_wave(1, observed_critic_version=10)
    with pytest.raises(ValueError, match="expected C11"):
        state.consume_micro_wave(2, observed_critic_version=10)


def test_ema_cannot_cross_optimizer_barrier() -> None:
    state = _enter_critic(RouteBFull5Contract.candidate_a())
    with pytest.raises(RuntimeError, match="must follow"):
        state.commit_ema(
            0, ema_before=10, ema_after=11, generator_version=3
        )


def test_official_clip_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="clipping"):
        RouteBFull5Contract(
            name="bad",
            generator_ranks=tuple(range(8)),
            critic_ranks=tuple(range(8, 16)),
            micro_batch_per_critic_rank=1,
            gradient_accumulation_steps=2,
            gradient_clipping=10.0,
        )


def test_candidate_a_teacher_uses_critic_capable_physical_score() -> None:
    module = _load_candidate_a_executor()
    assert module._step10a_model_roles(0) == ("generator", "generator")
    assert module._step10a_model_roles(8) == ("critic", "student")
    assert module._step10a_model_roles(12) == ("critic", "teacher")
    with pytest.raises(ValueError, match="outside Candidate-A"):
        module._step10a_model_roles(16)


def test_candidate_a_reactivates_critic_adapter_after_teacher_mode() -> None:
    import torch

    module = _load_candidate_a_executor()

    class _Toggle(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora = torch.nn.Parameter(torch.ones(3), requires_grad=False)
            self._distillschedule_has_adapter = True

        def enable_adapters(self) -> None:
            self.lora.requires_grad_(True)

    model = _Toggle()
    assert module._reactivate_critic_adapter(model) == {
        "trainable_tensor_count": 1,
        "trainable_numel": 3,
    }
    assert model.lora.requires_grad


def test_candidate_a_uses_host_control_plane_for_role_skew(monkeypatch) -> None:
    module = _load_candidate_a_executor()
    calls = []

    def fake_new_group(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(module.dist, "new_group", fake_new_group)
    topology = module.CriticPoolTopology(
        world_size=16,
        generator_ranks=module.G_RANKS,
        student_ranks=module.C_RANKS,
    )
    module._create_groups(topology)

    control = calls[0]
    assert control["ranks"] == list(range(16))
    assert control["backend"] == "gloo"
    assert control["timeout"].total_seconds() == 2 * 60 * 60
    assert calls[1]["backend"] == "nccl"
    assert calls[2]["backend"] == "nccl"


def test_candidate_a_control_barrier_does_not_bind_cuda_device(monkeypatch) -> None:
    module = _load_candidate_a_executor()
    calls = []
    group = object()
    monkeypatch.setattr(module.dist, "barrier", lambda **kwargs: calls.append(kwargs))

    module._control_barrier(group)

    assert calls == [{"group": group}]


def test_candidate_a_checkpoint_uses_one_world_aligned_save_call_per_rank(
    monkeypatch, tmp_path
) -> None:
    from types import SimpleNamespace

    module = _load_candidate_a_executor()
    barriers = []
    rng_ranks = []
    monkeypatch.setattr(
        module, "_control_barrier", lambda group: barriers.append(group)
    )
    monkeypatch.setattr(
        module, "_save_rng", lambda path, *, rank: rng_ranks.append(rank)
    )

    class _Engine:
        def __init__(self) -> None:
            self.calls = []

        def save_checkpoint(self, *args, **kwargs) -> None:
            self.calls.append((args, kwargs))

    args = SimpleNamespace(output_dir=tmp_path, timing_mode=False)
    root = tmp_path / "checkpoint-15"
    root.mkdir()
    (root / "critic").mkdir()
    generator = _Engine()
    critic = _Engine()

    module._save_checkpoint(
        args,
        rank=1,
        generator_engine=generator,
        critic_engine=None,
        ema=None,
        control_group="control",
        device=None,
    )
    module._save_checkpoint(
        args,
        rank=8,
        generator_engine=None,
        critic_engine=critic,
        ema=None,
        control_group="control",
        device=None,
    )

    assert len(generator.calls) == len(critic.calls) == 1
    assert generator.calls[0][0] == (str(root),)
    assert critic.calls[0][0] == (str(root / "critic"),)
    assert generator.calls[0][1]["tag"] == critic.calls[0][1]["tag"]
    assert generator.calls[0][1]["tag"] == "pytorch_model"
    assert rng_ranks == [1, 8]
    assert barriers == ["control"] * 6


def test_candidate_a_transition_checkpoint_freezes_g3_c10_ema10(
    monkeypatch, tmp_path
) -> None:
    from types import SimpleNamespace

    module = _load_candidate_a_executor()
    barriers = []
    monkeypatch.setattr(
        module, "_control_barrier", lambda group: barriers.append(group)
    )
    monkeypatch.setattr(module, "_save_rng", lambda path, *, rank: None)
    monkeypatch.setattr(
        module,
        "_checkpoint_inventory",
        lambda root: {
            "file_count_before_manifest": 39,
            "bytes_before_manifest": 123,
            "generator_optimizer_shards": 8,
            "critic_optimizer_shards": 8,
            "rng_files": 16,
            "sparse_ema": True,
        },
    )

    class _Engine:
        def __init__(self) -> None:
            self.calls = []

        def save_checkpoint(self, *args, **kwargs) -> None:
            self.calls.append((args, kwargs))

    class _EMA:
        optimization_step = 10

        def save_pretrained(self, path) -> None:
            path.mkdir(parents=True)

    args = SimpleNamespace(output_dir=tmp_path, timing_mode=False)
    generator = _Engine()
    identities = {
        "logical_manifest_hash": "logical",
        "execution_trace_bundle_hash": "trace",
        "routeb_trace_binding_hash": "binding",
    }
    module._save_checkpoint(
        args,
        rank=0,
        generator_engine=generator,
        critic_engine=None,
        ema=_EMA(),
        control_group="control",
        device=None,
        root_name="checkpoint-transition-g3-c10-ema10",
        logical_global_step=10,
        generator_version=3,
        critic_version=10,
        ema_optimization_step=10,
        transition_identities=identities,
    )

    root = tmp_path / "checkpoint-transition-g3-c10-ema10"
    manifest = module.json.loads((root / "transition_manifest.json").read_text())
    assert manifest["logical_global_step"] == 10
    assert manifest["generator_version"] == 3
    assert manifest["critic_version"] == 10
    assert manifest["ema_optimization_step"] == 10
    assert manifest["identities"] == identities
    assert generator.calls[0][1]["client_state"] == {
        "logical_global_step": 10,
        "generator_version": 3,
    }
    assert barriers == ["control"] * 4


def test_candidate_a_transition_manifest_is_identity_bound(tmp_path) -> None:
    module = _load_candidate_a_executor()
    root = tmp_path / "transition"
    root.mkdir()
    identities = {
        "logical_manifest_hash": "logical",
        "execution_trace_bundle_hash": "trace",
        "routeb_trace_binding_hash": "binding",
    }
    manifest = {
        "schema_version": "distillschedule.routeb_transition_checkpoint.v1",
        "logical_global_step": 10,
        "generator_version": 3,
        "critic_version": 10,
        "ema_optimization_step": 10,
        "identities": identities,
        "inventory": {},
    }
    module._atomic_json(root / "transition_manifest.json", manifest)

    loaded = module._transition_manifest(root, identities=identities)
    assert loaded["manifest"] == manifest
    assert len(loaded["manifest_sha256"]) == 64
    with pytest.raises(ValueError, match="identity changed"):
        module._transition_manifest(
            root,
            identities={**identities, "execution_trace_bundle_hash": "wrong"},
        )


def test_candidate_a_critic_timing_binds_passed_correctness(tmp_path) -> None:
    import hashlib
    import json
    from types import SimpleNamespace

    module = _load_candidate_a_executor()
    transition_sha = "a" * 64
    path = tmp_path / "correctness.json"
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "errors": [],
                "reference_run": "sequential-s1",
                "candidate_run": "pipeline-p1",
                "contract_transition_manifest_sha256": transition_sha,
            }
        )
    )
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    args = SimpleNamespace(
        timing_mode=True,
        resume_transition_checkpoint=tmp_path / "transition",
        timing_correctness_evaluation=path,
        expected_timing_correctness_sha256=observed,
        expected_transition_manifest_sha256=transition_sha,
    )

    binding = module._timing_correctness_binding(args)

    assert binding == {
        "schema_version": "distillschedule.routeb_timing_correctness_binding.v1",
        "path": str(path.resolve()),
        "sha256": observed,
        "status": "PASS",
        "reference_run": "sequential-s1",
        "candidate_run": "pipeline-p1",
        "transition_manifest_sha256": transition_sha,
    }
    args.expected_timing_correctness_sha256 = "b" * 64
    with pytest.raises(ValueError, match="SHA-256 changed"):
        module._timing_correctness_binding(args)


def test_candidate_a_low_overhead_timing_omits_heavy_state_hashes(
    monkeypatch,
) -> None:
    module = _load_candidate_a_executor()

    def forbidden(*args, **kwargs):
        raise AssertionError("correctness-only state hash executed")

    monkeypatch.setattr(module, "trainable_sha256", forbidden)
    monkeypatch.setattr(module, "trainable_checksum", forbidden)
    monkeypatch.setattr(module, "optimizer_state_checksum", forbidden)

    assert (
        module._optional_trainable_sha(object(), timing_mode=True)
        == module.TIMING_OMITTED
    )
    assert (
        module._optional_trainable_checksum(object(), timing_mode=True)
        == module.TIMING_OMITTED
    )
    assert (
        module._optional_optimizer_checksum(object(), timing_mode=True)
        == module.TIMING_OMITTED
    )
