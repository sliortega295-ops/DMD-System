"""CPU construction dispatch, not CUDA/FSDP/native-capacity qualification.

The entire original P0 constructor executes from its source AST. Only expensive
model creation, CUDA transfer/telemetry and sharding are event-only boundaries;
the logical FP32 restore uses its unchanged production loader and real tensors.
Startup executes the real public lifecycle with explicit synthetic asset/trace
and distributed boundaries. No synthetic successful native receipt is emitted.
"""
import ast
from contextlib import nullcontext
from dataclasses import fields, replace
from hashlib import sha256
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
from distillschedule.training import helios_lifecycle as lifecycle

ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py"
NATIVE_POLICY = "source_native_amplified_v1"
CPU_FIRST = "CPU_RESTORE_THEN_BLOCKWISE_FSDP_DEVICE_SHARD"
LEGACY = "FULL_DEVICE_MATERIALIZE_THEN_BLOCKWISE_FSDP_SHARD"


def _original_function(name, namespace):
    """Compile a whole original function, not a reimplementation/sliced branch."""
    function = next(n for n in ast.parse(P0.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", level=0,
        names=[ast.alias(name="annotations")]), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(P0), "exec"), namespace)
    return namespace[name]


def _startup(tmp_path, monkeypatch, *, gan=False, native=True):
    from distillschedule.training import distributed, helios_worker as worker
    from distillschedule.training import helios_gan_training as gan_training
    from distillschedule.training import helios_gan_inputs as gan_inputs
    from distillschedule.training import helios_inputs as inputs
    from distillschedule.training.config import TrainingConfig
    from distillschedule.runtime.helios_native_history import descriptor
    from distillschedule.runtime.helios_gan_native_history import descriptor as gan_descriptor
    from runtime.gst_pool_contract import NegativePromptAsset

    events = []
    native_policy = NATIVE_POLICY if native else None
    task = (gan_inputs.GANCohortTask if gan else inputs.HeliosCohortTask)(0, native_policy=native_policy)
    kind = ((gan_inputs.NATIVE_KIND if native else gan_inputs.KIND) if gan
            else (inputs.NATIVE_KIND if native else inputs.KIND))
    flags = {name.replace("-", "_"): False for name in worker._OFF}
    if gan:
        flags.update(gan_training.effective_flags())
    values = dict(flags, mesh_policy="world16-fsdp", rollout_batch_size=1,
        pack_real_score_pair=False, modulation_chunk_tokens=2048 if gan else 512,
        selective_flash_checkpoint_stride=2 if gan else 0,
        generator_nograd_persistent_unshard_flush_allocator_cache=True)
    policy = StaticFSDP2P0Policy(**{f.name: values[f.name] for f in fields(StaticFSDP2P0Policy) if f.name in values})
    # Asset filenames/trace data are synthetic boundaries, not model payloads.
    roles = ["generator_model", "score_model", "generator_precast", "score_precast",
             "negative_prompt", "logical_checkpoint", *(["gan_config"] if gan else [])]
    assets = {"assets": {role: {"path": str(tmp_path / role)} for role in roles}}
    stream = dict(schema_version="distillschedule." + kind, stream_sha256="s" * 64,
        identity=dict(source_fingerprint="f" * 64, request=task.request.to_dict(),
                      **({"native_history_policy": native_policy} if native else {})),
        cohorts=[dict(task=task.to_dict(), trace_bundle=str(tmp_path / "trace"), trace_hash="t" * 64)])
    schema = ("distillschedule.helios_gan_native_history_training_identity.v1" if gan else
              "distillschedule.helios_native_history_training_identity.v1") if native else (
              "distillschedule.helios_gan_training_identity.v1" if gan else "distillschedule.helios_training_identity.v1")
    selected = gan_training.descriptor(native_policy) if gan else dict(mesh="world16-fsdp")
    if native and not gan:
        selected["native_history"] = descriptor()
    identity = dict(schema_version=schema, source_head="6bbd29075dd6f79ee0085705b189bb5a04b940fc",
        source_fingerprint="f" * 64, stream_sha256=stream["stream_sha256"], bootstrap_logical_hash="l" * 64,
        bootstrap_trace_hash="t" * 64, **{"execution_policy" if gan else "reference_policy": selected})
    if native:
        identity["gan_native_history" if gan else "native_history"] = gan_descriptor() if gan else descriptor()
    config = TrainingConfig.from_dict(dict(schema_version="distillschedule.training.v1",
        model=dict(workload="W3" if gan else "W2"),
        algorithm=dict(name="original_helios_gan" if gan else "original_helios_nogan"),
        data=dict(input_kind=kind, prepared=str(tmp_path / "stream.json")),
        execution=dict(adapter="native", mode="manual" if gan else "reference", task="task.json", assets="assets.json"),
        trainer=dict(cycles=2, checkpoint_every=1, output_dir=str(tmp_path / "run"))), source=tmp_path / "training.json")
    binding = dict(config=config, stream=stream, assets=assets, identity=identity, portable_reference=True)
    if gan:
        binding.update(gan_stream=True, metadata_group=object(), implementation=gan_training.implementation())
    args = NS(**dict(values, variant="gan" if gan else "no-gan", timing_mode=True,
        profile_mode=False, profile_diagnostics=False, validate_only=False, operation_source=False,
        critic_only_step=None, attention_backend="local-fa2" if gan else "sdpa", ema_device="cuda",
        generator_residual_storage="cuda_out_of_place", critic_residual_storage="cuda_out_of_place",
        _planner_gan_manual_bundle=gan, rollout_backend="native-fsdp2-batch",
        generator_root=tmp_path / "generator_model", score_root=tmp_path / "score_model",
        precast_generator_root=tmp_path / "generator_precast", precast_score_root=tmp_path / "score_precast",
        negative_prompt=tmp_path / "negative_prompt", logical_package=tmp_path / "logical_checkpoint",
        gan_config=tmp_path / "gan_config", trace_bundle=tmp_path / "trace",
        expected_logical_hash="l" * 64, expected_trace_hash="t" * 64))
    for key in ("LOCAL_KERNELS", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        monkeypatch.delenv(key, raising=False)
    if gan:
        # This synthetic W3 startup owns explicit CLI controls, not an embedding
        # W4/benchmark kernel environment. Isolation is fixture-local only; the
        # unchanged native environment gate is exercised below after injection.
        for key in tuple(os.environ):
            if key.startswith("DISTILLSCHEDULE_") and key != "DISTILLSCHEDULE_REPOSITORY_ROOT":
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("DISTILLSCHEDULE_HELIOS_PLANNER_MODE", "off")
    monkeypatch.setattr(lifecycle, "_plane", lambda _: nullcontext())
    monkeypatch.setattr(distributed, "checked", lambda fn, label: (events.append(label), fn())[1])
    monkeypatch.setattr(distributed, "gather", lambda row: (events.append("peer_identity_agreement"), [row] * 16)[1])
    monkeypatch.setattr(distributed, "admit_run_controls", lambda _: events.append("run_controls_admitted"))
    owner = gan_training.assets_io if gan else worker
    monkeypatch.setattr(owner, "immutable_assets", lambda _: dict(evidence_kind="SYNTHETIC_TEST_ONLY_ASSET_BOUNDARY"))
    monkeypatch.setattr(owner, "load_bootstrap" if gan else "load_reference_bootstrap",
                        lambda *a, **k: (NS(manifest_hash="l" * 64), {}))
    loader = gan_inputs if gan else lifecycle
    monkeypatch.setattr(loader, "load_stream", lambda *a, **k: stream)
    monkeypatch.setattr(loader, "load_cohort", lambda *a, **k: NS(task=task, logical_steps=[]))
    monkeypatch.setattr(NegativePromptAsset, "load", lambda _: NS(embedding="SYNTHETIC_TEST_ONLY"))
    dist = NS(get_world_size=lambda: 16, new_group=lambda **k: object())

    def admit():
        lifecycle.admit_original_rank(torch=torch, dist=dist, rank=0, args=args, policy=policy, seed_runtime=False)
    return NS(binding=binding, args=args, policy=policy, admit=admit, events=events)


@pytest.mark.parametrize("native", [False, True])
def test_synthetic_w3_startup_carries_actual_producer_request_without_default_inference(tmp_path, monkeypatch, native):
    from distillschedule.training.helios_gan_inputs import GANCohortTask, task_for_stream
    startup = _startup(tmp_path, monkeypatch, gan=True, native=native)
    stream = startup.binding["stream"]
    task = GANCohortTask.from_dict(stream["cohorts"][0]["task"])
    assert stream["identity"]["request"] == task.request.to_dict()
    assert task_for_stream(stream, task.index) == task
    missing = {**stream, "identity": {k: v for k, v in stream["identity"].items() if k != "request"}}
    with pytest.raises(KeyError, match="request"):
        task_for_stream(missing, task.index)
    assert stream["identity"]["request"] == task.request.to_dict()


def _constructor(tmp_path, *, args, policy, role="critic", failure=None):
    from runtime.critic_pool_checkpoint import LogicalCheckpointPackage, load_logical_master_into_trainables
    events, observed = [], {}
    payload = torch.arange(1, 81, dtype=torch.float32).reshape(40, 2) / 7
    payload_path = tmp_path / (role + ".fp32")
    payload_path.write_bytes(payload.numpy().tobytes())
    names = [f"blocks.{i}.lora" for i in range(40)]
    package = LogicalCheckpointPackage(tmp_path, {"components": {role: dict(parameter_version=2 if role == "generator" else 10,
        parameter_names=names, logical_numel=80,
        payloads=dict(master=dict(dtype="torch.float32", path=payload_path.name,
            bytes=payload_path.stat().st_size, sha256=sha256(payload_path.read_bytes()).hexdigest())),
        records={name: dict(offset_elements=2*i, numel=2, shape=[2]) for i, name in enumerate(names)})}})

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(40)])
            for block in self.blocks:
                block.register_parameter("frozen", torch.nn.Parameter(torch.randn(2).bfloat16(), requires_grad=False))
                block.register_parameter("lora", torch.nn.Parameter(torch.randn(2), requires_grad=True))
            self.register_buffer("input_values", torch.tensor([[1, 3], [2, 4]], dtype=torch.bfloat16).t())

        def to(self, device):
            # Never move tensors, pretend to allocate CUDA, or monkeypatch nn.Module.to.
            events.append(("event_only.whole_model_to", str(device)))
            if failure == "transfer":
                raise RuntimeError("original transfer boundary failure")
            return self

    def snapshot(model):
        return {name: (t.detach().clone(), t.dtype, t.requires_grad, t.stride(), t.storage_offset(), t.data_ptr())
                for name, t in [*model.named_parameters(), *model.named_buffers()]}

    def build(**kwargs):
        events.append(("constructor", kwargs))
        model = Tiny()
        observed.update(before=snapshot(model), constructor_rng=torch.get_rng_state().clone())
        return model, {"evidence_kind": "SYNTHETIC_TEST_ONLY_MODEL_BOUNDARY"}

    def restore(model, package, **kwargs):
        events.append(("logical_restore", kwargs))
        result = load_logical_master_into_trainables(model, package, **kwargs)
        observed.update(after_restore=snapshot(model), restore_rng=torch.get_rng_state().clone())
        if failure == "restore":
            raise RuntimeError("original logical restore failure")
        return result

    def shard(model, **kwargs):
        events.append(("event_only.same_existing_sharder", kwargs))
        observed.update(at_shard=snapshot(model), shard_rng=torch.get_rng_state().clone())
        if failure == "sharder":
            raise RuntimeError("original sharder boundary failure")
        return {"evidence_kind": "SYNTHETIC_TEST_ONLY_SHARDER_BOUNDARY; CUDA_FSDP_NOT_RUN"}

    def event(*a, **k):
        return {"evidence_kind": "SYNTHETIC_TEST_ONLY_DISABLED_OPTIMIZATION_BOUNDARY"}

    scope = dict(build_helios_pool_model=build, load_logical_master_into_trainables=restore,
        suspend_transformers_zero3_init=nullcontext, dist=NS(get_rank=lambda: 0), WORLD_SIZE=16,
        apply_blockwise_fsdp2_policy=shard, trim_host_allocator=lambda: False, gc=NS(collect=lambda: None),
        base=NS(_memory=lambda _: dict(evidence_kind="CPU_ONLY_NOT_GPU_MEMORY")),
        torch=NS(cuda=NS(synchronize=lambda _: None, empty_cache=lambda: None)))
    function = next(n for n in ast.parse(P0.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_build_static_model")
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.startswith(("configure_", "install_", "warmup_")):
            scope[node.func.id] = event
    run = _original_function("_build_static_model", scope)
    return NS(run=lambda: run(args=args, role=role, package=package, device="cuda:EVENT_ONLY", policy=policy),
              events=events, observed=observed, payload=payload)


@pytest.mark.parametrize("gan", [False, True], ids=["W2", "W3"])
@pytest.mark.parametrize("role", ["generator", "critic"])
def test_entire_native_history_constructor_skips_only_eager_transfer(tmp_path, monkeypatch, gan, role):
    startup = _startup(tmp_path, monkeypatch, gan=gan)
    with lifecycle.original_training_scope(startup.binding):
        startup.admit()
        case = _constructor(tmp_path, args=startup.args, policy=startup.policy, role=role)
        model, receipt = case.run()
    assert [event[0] for event in case.events] == ["constructor", "logical_restore", "event_only.same_existing_sharder"]
    assert receipt["construction_order"] == CPU_FIRST and receipt["construction_only_compatibility"] is True
    assert receipt["logical_restore"]["optimizer_materialized"] is False
    assert receipt["logical_restore"]["scheduler_materialized"] is False
    kwargs = case.events[-1][1]
    assert kwargs == dict(mesh_policy="world16-fsdp", world_size=16,
                         **({"separate_gan_heads": True} if gan and role == "critic" else {}))
    before, restored, at_shard = (case.observed[key] for key in ("before", "after_restore", "at_shard"))
    for name, initial in before.items():
        assert initial[1:] == restored[name][1:] == at_shard[name][1:]
        expected = case.payload[int(name.split(".")[1])] if name.endswith(".lora") else initial[0]
        assert torch.equal(restored[name][0], expected) and torch.equal(at_shard[name][0], expected)
    assert torch.equal(case.observed["constructor_rng"], case.observed["restore_rng"])
    assert torch.equal(case.observed["restore_rng"], case.observed["shard_rng"])
    assert all(t.device.type == "cpu" for t in model.parameters()) and not torch.cuda.is_initialized()


@pytest.mark.parametrize("context", ["none", "unadmitted_native", "normal_public", "normal_public_w3", "node_hsdp"])
def test_legacy_world16_and_existing_hsdp_dispatch_remain(tmp_path, monkeypatch, context):
    startup = _startup(tmp_path, monkeypatch, gan=context == "normal_public_w3",
                       native=context not in {"normal_public", "normal_public_w3"})
    policy = replace(startup.policy, mesh_policy="node-hsdp") if context == "node_hsdp" else startup.policy
    with lifecycle.original_training_scope(startup.binding) if context != "none" else nullcontext():
        if context in {"normal_public", "normal_public_w3"}:
            startup.admit()
        case = _constructor(tmp_path, args=startup.args, policy=policy)
        _, receipt = case.run()
    expected = ["constructor", "logical_restore", "event_only.same_existing_sharder"]
    if context != "node_hsdp":
        expected.insert(2, "event_only.whole_model_to")
    assert [e[0] for e in case.events] == expected
    assert receipt["construction_order"] == (CPU_FIRST if context == "node_hsdp" else LEGACY)


@pytest.mark.parametrize("gan", [False, True], ids=["W2", "W3"])
def test_capability_requires_completed_original_startup_and_unchanged_identity(tmp_path, monkeypatch, gan):
    startup = _startup(tmp_path, monkeypatch, gan=gan)
    with lifecycle.original_training_scope(startup.binding):
        assert not lifecycle.native_history_cpu_first_construction(policy=startup.policy)
        startup.admit()
        assert startup.events[-1] == "peer_identity_agreement"
        assert lifecycle.native_history_cpu_first_construction(policy=startup.policy)
        with pytest.raises(ValueError, match="construction.*changed"):
            lifecycle.native_history_cpu_first_construction(policy=replace(startup.policy,
                optimizer_dispatch_warmup=not startup.policy.optimizer_dispatch_warmup))
        startup.binding["identity"]["source_head"] = "0" * 40
        with pytest.raises(ValueError, match="construction.*changed"):
            lifecycle.native_history_cpu_first_construction(policy=startup.policy)
    assert lifecycle.current_training() is None


@pytest.mark.parametrize("gan", [False, True], ids=["W2", "W3"])
@pytest.mark.parametrize("change", ["typed_task", "identity_schema", "descriptor", "physical_policy"])
def test_invalid_native_identity_never_grants_capability(tmp_path, monkeypatch, gan, change):
    startup = _startup(tmp_path, monkeypatch, gan=gan)
    if change == "typed_task":
        startup.binding["stream"]["cohorts"][0]["task"]["native_policy"] = "unbound"
    elif change == "identity_schema":
        startup.binding["identity"]["schema_version"] = "untrusted_string_hint"
    elif change == "descriptor":
        startup.binding["identity"]["gan_native_history" if gan else "native_history"]["checkpoint_blocks"] = 39
    else:
        startup.args.attention_backend = "unbound"
    with lifecycle.original_training_scope(startup.binding):
        with pytest.raises(ValueError):
            startup.admit()
        assert not lifecycle.native_history_cpu_first_construction(policy=startup.policy)


@pytest.mark.parametrize("gan", [False, True], ids=["W2", "W3"])
def test_peer_rejection_cannot_publish_construction_capability(tmp_path, monkeypatch, gan):
    from distillschedule.training import distributed
    startup = _startup(tmp_path, monkeypatch, gan=gan)
    monkeypatch.setattr(distributed, "gather", lambda row: [row] * 16 if set(row) == {"stop_after_cycle"}
                        else [row, {"foreign_identity": True}])
    with lifecycle.original_training_scope(startup.binding):
        with pytest.raises(ValueError, match="rank source/stream|GAN rank source/stream"):
            startup.admit()
        assert not lifecycle.native_history_cpu_first_construction(policy=startup.policy)


@pytest.mark.parametrize("key", ["DISTILLSCHEDULE_FLASH_ATTN_KERNEL", "DISTILLSCHEDULE_SCORE_FA3_KERNEL",
                                  "DISTILLSCHEDULE_UNKNOWN_CONSTRUCTION_CONTROL"])
def test_w3_original_environment_gate_rejects_controls_injected_after_fixture(tmp_path, monkeypatch, key):
    startup = _startup(tmp_path, monkeypatch, gan=True)
    monkeypatch.setenv(key, "SYNTHETIC_TEST_ONLY_UNBOUND_CONTROL")
    with lifecycle.original_training_scope(startup.binding):
        with pytest.raises(ValueError, match="unbound GAN environment control: " + key):
            startup.admit()
        assert "_native_history_construction" not in startup.binding
        assert not lifecycle.native_history_cpu_first_construction(policy=startup.policy)


@pytest.mark.parametrize("gan", [False, True], ids=["W2", "W3"])
def test_seeded_tiny_model_and_rng_match_legacy_at_existing_sharder(tmp_path, monkeypatch, gan):
    import random
    startup = _startup(tmp_path, monkeypatch, gan=gan)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1701)
        python_rng = random.getstate()
        legacy = _constructor(tmp_path, args=startup.args, policy=startup.policy)
        legacy.run()
        torch.random.default_generator.manual_seed(1701)
        with lifecycle.original_training_scope(startup.binding):
            startup.admit()
            candidate = _constructor(tmp_path, args=startup.args, policy=startup.policy)
            candidate.run()
        assert random.getstate() == python_rng
        for name, old in legacy.observed["at_shard"].items():
            new = candidate.observed["at_shard"][name]
            assert torch.equal(old[0], new[0]) and old[1:5] == new[1:5]
        assert torch.equal(legacy.observed["constructor_rng"], candidate.observed["constructor_rng"])
        assert torch.equal(legacy.observed["shard_rng"], candidate.observed["shard_rng"])


def test_new_capability_helper_failure_propagates_and_hsdp_does_not_consult_it(tmp_path, monkeypatch):
    startup = _startup(tmp_path, monkeypatch)
    def fail(**kwargs):
        raise RuntimeError("original lifecycle boundary failure")
    monkeypatch.setattr(lifecycle, "native_history_cpu_first_construction", fail)
    case = _constructor(tmp_path, args=startup.args, policy=startup.policy)
    with pytest.raises(RuntimeError, match="original lifecycle boundary failure"):
        case.run()
    assert [e[0] for e in case.events] == ["constructor", "logical_restore"]
    case = _constructor(tmp_path, args=startup.args, policy=replace(startup.policy, mesh_policy="node-hsdp"))
    assert case.run()[1]["construction_order"] == CPU_FIRST


def test_untyped_context_hint_cannot_grant_construction_capability(tmp_path, monkeypatch):
    startup = _startup(tmp_path, monkeypatch)
    startup.binding["_native_history_construction"] = True
    with lifecycle.original_training_scope(startup.binding):
        with pytest.raises(ValueError, match="construction.*changed"):
            lifecycle.native_history_cpu_first_construction(policy=startup.policy)


@pytest.mark.parametrize("failure", ["restore", "sharder", "transfer"])
def test_original_constructor_failures_propagate(tmp_path, monkeypatch, failure):
    startup = _startup(tmp_path, monkeypatch, native=False)
    case = _constructor(tmp_path, args=startup.args, policy=startup.policy, failure=failure)
    with pytest.raises(RuntimeError, match="original .* failure"):
        case.run()
    if failure == "restore":
        assert [e[0] for e in case.events] == ["constructor", "logical_restore"]


def test_original_build_order_and_optimizer_restore_are_unchanged():
    events = []
    def build(**kwargs):
        events.append(("build_complete", kwargs["role"])); return kwargs["role"], {}
    def engine(**kwargs):
        events.append(("optimizer_restore", kwargs["component"])); return None, {}
    run = _original_function("_build_engines", dict(_build_static_model=build, build_fsdp2_engine=engine,
        dist=NS(barrier=lambda: events.append(("barrier", None)))))
    run(args=None, package=None, device=None, policy=StaticFSDP2P0Policy())
    assert events == [("build_complete", "generator"), ("barrier", None), ("build_complete", "critic"),
                      ("barrier", None), ("optimizer_restore", "generator"), ("optimizer_restore", "critic")]
