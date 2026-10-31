"""Persistent W3 lifecycle over original P0 owners and the full manual bundle.

The original Program cycle, rollout, DMD/GAN loss, Adam, constant LR and EMA
remain the execution owners. This module owns only admission, fresh inputs,
native state publication and complete-cycle continuation. GPU acceptance is a
separate gate; CPU/source support never promotes the bundle's REVIEW status.
"""
import json
import os
from pathlib import Path
import re
import sys

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint, repository_root
from . import distributed, helios_gan_assets as assets_io, helios_gan_inputs as inputs
from .native_presets import HELIOS_GAN_STREAM as PRESET_ID

SCHEMA = "distillschedule.original_helios_gan_training.v1"
TOPOLOGY = {"mesh": list(range(16)), "mesh_dim_names": ["fsdp"], "device_type": "cuda",
            "placements": [{"kind": "shard", "dim": 0}]}


def implementation():
    from distillschedule.search.helios_implementation import HeliosImplementation
    from distillschedule.search.helios_gan_bundle import HeliosGANBundle
    return HeliosImplementation("gan", "world16-fsdp", 2048, 5, True, HeliosGANBundle.manual())


def expected_versions(completed):
    require(type(completed) is int and completed >= 0, "nonnegative complete cycle required", path="helios.gan.training")
    return {"generator": 2 + completed, "critic": 10 + 5 * completed, "ema": 800 + completed}


def logical_versions(completed):
    value = expected_versions(completed)
    return {**value, "generator": value["generator"] + 198, "critic": value["critic"] + 990}


def effective_flags():
    from .helios_worker import _OFF
    from distillschedule.search.helios_safe_bundle import FLAG_ENV
    enabled = set(FLAG_ENV) | {"all_role_lora_scale1_add", "optimizer_dispatch_warmup",
        "generator_nograd_residual_fusion", "exact_grad_residual_fusion", "exact_history_current_fusion",
        "generator_nograd_cross_kv_cache", "generator_nograd_persistent_unshard",
        "elide_critic_scalar_logs", "elide_official_generator_free_memory"}
    return {name.replace("-", "_"): name.replace("-", "_") in enabled for name in _OFF}


def checkpoint_cpu_policy(config):
    from . import helios_gan_checkpoint_cpu
    return helios_gan_checkpoint_cpu if getattr(config,"execution",{}).get("preset") == helios_gan_checkpoint_cpu.PRESET_ID else None


def descriptor(native_policy=None):
    root = repository_root()
    owners = tuple("src/distillschedule/" + name for name in (
        "training/helios_gan_training.py", "training/helios_gan_inputs.py", "training/helios_gan_assets.py",
        "runtime/helios_gan_manual/stream.py", "runtime/helios_gan_manual/runtime.py",
        "runtime/helios_gan_manual/down_replay.py", "runtime/helios_gan_manual/backend_library.py",
        "training/helios_lifecycle.py")) + (
        "scripts/distillschedule/probe_static_fsdp2_full5_world16.py",
        "scripts/distillschedule/probe_static_fsdp2_p0_world16.py",
        "scripts/distillschedule/probe_static_fsdp2_p0_world16_hardened.py")
    if native_policy is not None:
        from distillschedule.runtime.helios_gan_native_history import descriptor as history_descriptor, require_task
        require_task(inputs.GANCohortTask(0, native_policy=native_policy))
        owners += ("src/distillschedule/runtime/helios_gan_native_history.py",
                   "src/distillschedule/runtime/helios_native_history.py",
                   "src/distillschedule/runtime/fsdp2_variable_forward.py",
                   "src/distillschedule/runtime/fsdp2_nograd_persistence.py",
                   "src/distillschedule/runtime/helios_planner_cycle.py")
    value = {"schema_version": "distillschedule.helios_gan_stream_policy.v1", "preset": PRESET_ID,
        "implementation": implementation().assignment, "candidate_bundle": implementation().candidate_bundle.cost_identity(),
        "current_integration_sha256": {name: file_sha256(root / name) for name in owners},
        "source_policy": "EXPLICIT_CURRENT_INTEGRATION_NOT_FROZEN_TWO_CYCLE_IDENTITY",
        "flags": effective_flags(), "topology": TOPOLOGY, "attention": "pinned-sm80-fa3",
        "checkpoint": "native_local_shards_same_topology_complete_cycle", "scheduler": "original_native_constant_unbounded",
        "bootstrap": "explicit_systems_only_G2_C10_EMA800_NOT_OFFICIAL_CHECKPOINT1000",
        "numerical_status": "REVIEW", "gpu_execution": "NOT_RUN"}
    if native_policy is not None:
        value.update(schema_version="distillschedule.helios_gan_native_history_stream_policy.v1",
                     native_history=history_descriptor())
    return value


def validate_environment(environment=None):
    from distillschedule.search.helios_implementation import current_implementation
    env = os.environ if environment is None else environment
    require(current_implementation(env) is None, "GAN stream cannot inherit a fixed benchmark plan", path="helios.gan.training")
    require(not env.get("PYTORCH_CUDA_ALLOC_CONF") and not env.get("PYTORCH_ALLOC_CONF"),
            "GAN stream allocator environment differs", path="helios.gan.training")
    # CLI arguments, not ambient historical flags, own the new invocation.
    for key, value in env.items():
        if key.startswith("DISTILLSCHEDULE_") and value and key not in {"DISTILLSCHEDULE_REPOSITORY_ROOT"}:
            require(key == "DISTILLSCHEDULE_HELIOS_PLANNER_MODE" and value == "off",
                    "unbound GAN environment control: " + key, path="helios.gan.training")


def bind_effective(*, args, policy, native_policy=None):
    validate_environment()
    require(args.variant == "gan" and args.timing_mode and not args.profile_mode and not args.profile_diagnostics
            and not args.validate_only and args.critic_only_step is None and not args.operation_source,
            "complete unprofiled original GAN helpers required", path="helios.gan.training")
    for name, expected in effective_flags().items():
        require(getattr(args, name, None) is expected and (not hasattr(policy, name) or getattr(policy, name) is expected),
                "GAN stream effective flag differs: " + name, path="helios.gan.training")
    require(args.mesh_policy == policy.mesh_policy == "world16-fsdp"
            and args.modulation_chunk_tokens == policy.modulation_chunk_tokens == 2048
            and args.rollout_batch_size == policy.rollout_batch_size == 1
            and args.rollout_backend == policy.rollout_backend == "native-fsdp2-batch"
            and args.selective_flash_checkpoint_stride == policy.selective_flash_checkpoint_stride == 2
            and args.attention_backend == "local-fa2" and args.ema_device == "cuda"
            and args.generator_residual_storage == args.critic_residual_storage == "cuda_out_of_place"
            and args.generator_nograd_persistent_unshard_flush_allocator_cache is True
            and policy.generator_nograd_persistent_unshard_flush_allocator_cache is True
            and args._planner_gan_manual_bundle is True,
            "GAN full world16/C2048/p1/window5/FA3/head policy changed", path="helios.gan.training")
    from .helios_lifecycle import current_training
    binding = current_training()
    candidate = None if binding is None else checkpoint_cpu_policy(binding["config"])
    return descriptor(native_policy) if candidate is None else candidate.descriptor()


def native_argv(assets, stream, output, logical_hash):
    from .helios_worker import reference_argv
    # Reuse the complete original parser surface, but supply only the explicitly
    # admitted six local assets and first fresh trace (never load_profile paths).
    projected = assets_io._reference(assets)
    args = reference_argv(projected, output, stream=stream, logical_hash=logical_hash)
    for option, value in (("--variant", "gan"), ("--run-id", "original-helios-gan-stream-v1"),
                          ("--modulation-chunk-tokens", "2048"), ("--attention-backend", "local-fa2"),
                          ("--selective-flash-checkpoint-stride", "2")):
        args[args.index(option) + 1] = value
    for name, enabled in effective_flags().items():
        if enabled:
            args[args.index("--no-" + name.replace("_", "-"))] = "--" + name.replace("_", "-")
    return args + ["--gan-config", assets["assets"]["gan_config"]["path"]]


def admit_training(config, task_value, assets_value, paths):
    from distillschedule.search.helios_request import OriginalHeliosRequest
    inputs.GANCohortTask(0, request=OriginalHeliosRequest.from_dict(task_value))
    assets = assets_io.asset_config(json.loads(Path(paths["assets"]).read_text()))
    require(assets == assets_value, "GAN assets changed during admission", path="helios.gan.training")
    observations = assets_io.verify_pins(assets)
    logical, _ = assets_io.load_bootstrap(assets, verify_payloads=False)
    stream = inputs.load_stream(config.data["prepared"], cycles=config.trainer["cycles"], expected_kind=config.data["input_kind"])
    require(stream["identity"]["request"] == task_value, "GAN stream/task request differs", path="helios.gan.training")
    require(stream["identity"]["training_config_sha256"] == assets["assets"]["gan_config"]["sha256"],
            "GAN stream and runtime config identities differ", path="helios.gan.training")
    resume = config.trainer.get("resume_from")
    if resume:
        path = Path(resume)
        require((path / "manifest.json").is_file() and (path / "_SUCCESS").is_file()
                and (path / "_SUCCESS").read_text().strip() == file_sha256(path / "manifest.json"),
                "complete native GAN resume checkpoint required", path="helios.gan.training")
    candidate = checkpoint_cpu_policy(config)
    extra = {}
    if candidate is not None:
        candidate.validate_config(config)
        require("native_history_policy" in stream["identity"], "W3 checkpoint CPU requires original native history", path="helios.gan.training")
        extra = dict(selected=candidate.descriptor(), checkpoint_input_capacity=candidate.load_capacity(config))
    return {"task": task_value, "assets": assets, "stream": stream,
            "selected": descriptor(stream["identity"].get("native_history_policy")),
            "implementation": implementation(), "plan": None, "asset_observations": observations,
            "bootstrap": {"logical_manifest_hash": logical.manifest_hash},
            "argv": native_argv(assets, stream, paths["output"], logical.manifest_hash),
            "validation_scope": "CPU_NATIVE_GAN_SOURCE_ASSET_STREAM; GPU_KERNEL_NUMERICAL_CAPACITY_NOT_RUN", **extra}


def _checkpoint_policy(identity):
    from . import helios_gan_checkpoint_cpu as candidate
    if (identity.get("schema_version") == candidate.IDENTITY or "checkpoint_input_capacity" in identity
            or identity.get("execution_policy",{}).get("preset") == candidate.PRESET_ID):
        candidate.validate_identity(identity)
        return {"schema_version":candidate.CHECKPOINT, "identity_sha256":fingerprint(identity)}
    if "gan_native_history" in identity:
        from distillschedule.runtime.helios_gan_native_history import descriptor as history_descriptor
        require(identity["gan_native_history"] == history_descriptor()
                and identity["execution_policy"].get("native_history") == history_descriptor(),
                "GAN native history checkpoint capability changed", path="helios.gan.training")
        return {"schema_version": "distillschedule.helios_gan_native_history_checkpoint_policy.v1",
                "identity_sha256": fingerprint(identity)}
    return {"schema_version": "distillschedule.helios_gan_stream_checkpoint_policy.v1", "identity_sha256": fingerprint(identity)}


def validate_progress(header, binding):
    from distillschedule.data.helios_epoch_inputs import validate_cursor
    validate_cursor(header, binding["stream"])
    cycle = header["cycle"]
    count = len(binding["stream"]["cohorts"])
    within_stream = type(cycle) is int and (cycle <= count if binding.get("native_output") is not None else cycle < count)
    require(type(cycle) is int and 0 < cycle == header["cursor"] and within_stream
            and header["versions"] == expected_versions(cycle) and header["policy"] == binding["checkpoint_policy"],
            "GAN restart requires an incomplete-stream complete G1/C5/EMA1 boundary", path="helios.gan.training")


def admit_original_rank(*, torch, dist, rank, args, policy, seed_runtime=True):
    from .helios_lifecycle import current_training, _plane, seed_original_runtime, admit_stop_control, stop_cycle
    from .helios_lifecycle import _bind_native_history_construction
    binding = current_training()
    require(binding is not None and binding.get("gan_stream") is True and dist.get_world_size() == 16
            and binding.get("metadata_group") is not None, "owned original GAN world16/Gloo scope required", path="helios.gan.training")
    with _plane(binding):
        def local():
            effective = bind_effective(args=args, policy=policy,
                                       native_policy=binding["stream"]["identity"].get("native_history_policy"))
            require(binding["identity"]["execution_policy"] == effective, "GAN source/policy binding changed", path="helios.gan.training")
            stream = inputs.load_stream(binding["config"].data["prepared"], cycles=binding["config"].trainer["cycles"],
                                        verify_rank=rank, expected_kind=binding["config"].data["input_kind"])
            require(stream["stream_sha256"] == binding["identity"]["stream_sha256"], "GAN stream changed", path="helios.gan.training")
            frozen = assets_io.immutable_assets(binding["assets"])
            logical, _ = assets_io.load_bootstrap(binding["assets"], verify_payloads=False)
            first = stream["cohorts"][0]
            rows = binding["assets"]["assets"]
            require(all(Path(getattr(args, option)).resolve() == Path(rows[role]["path"]).resolve() for role, option in (
                ("generator_model", "generator_root"), ("score_model", "score_root"),
                ("generator_precast", "precast_generator_root"), ("score_precast", "precast_score_root"),
                ("negative_prompt", "negative_prompt"), ("logical_checkpoint", "logical_package"), ("gan_config", "gan_config")))
                and args.expected_logical_hash == logical.manifest_hash == binding["identity"]["bootstrap_logical_hash"]
                and args.expected_trace_hash == first["trace_hash"]
                and Path(args.trace_bundle).resolve() == Path(first["trace_bundle"]).resolve(),
                "GAN original bootstrap/local asset binding changed", path="helios.gan.training")
            from runtime.gst_pool_contract import NegativePromptAsset
            from distillschedule.runtime.fsdp2_p0 import validate_rollout_entries
            trace = inputs.load_cohort(inputs.task_for_stream(stream, 0), first["trace_bundle"], expected_hash=first["trace_hash"])
            for step in trace.logical_steps:
                validate_rollout_entries([trace.load_critic_entry(rank=rank, step=step, device="cpu")])
            identity = {**binding["identity"], "verified_immutable_assets": frozen}
            resume = binding["config"].trainer.get("resume_from")
            if resume:
                saved = json.loads((Path(resume) / "manifest.json").read_text()).get("identity", {}).get("native_owner_layout")
                validate_layout_record(saved); identity["native_owner_layout"] = saved
            return stream, identity, {"logical": logical, "trace": trace, "request":trace.task.request,
                "negative": NegativePromptAsset.load(rows["negative_prompt"]["path"])}
        stream, identity, startup = distributed.checked(local, "Helios GAN native input/asset admission")
        binding.update(stream=stream, identity=identity, startup=startup,
                       checkpoint_policy=_checkpoint_policy(identity))
        candidate = checkpoint_cpu_policy(binding["config"])
        if candidate is not None:
            binding["effective_configuration"] = identity["execution_policy"]
            distributed.checked(lambda: candidate.admit_startup(binding, rank=rank),
                                "W3 checkpoint input extra host capacity before construction")
        distributed.admit_run_controls(binding["config"])
        admit_stop_control(binding["config"])
        require(all(item == identity for item in distributed.gather(identity)), "GAN rank source/stream differs", path="helios.gan.training")
        if binding["config"].trainer.get("resume_from"):
            from .checkpoint import read_manifest
            header = distributed.checked(lambda: read_manifest(binding["config"].trainer["resume_from"], identity=identity),
                                         "Helios GAN restart metadata")
            validate_progress(header, binding)
            if binding.get("native_output") is None:
                stop_cycle(binding["config"], start=header["cycle"])
        _bind_native_history_construction(policy=policy)
    if seed_runtime:
        seed_original_runtime(torch=torch, rank=rank)


def owner_layout(generator, critic):
    from torch.distributed.tensor import DTensor
    from .tensor_state import layout
    from .native_state import _named
    rows = {}
    for role, engine in (("generator", generator), ("critic", critic)):
        _, parameters, _ = _named(engine)
        require(parameters, "missing native GAN trainables", path="helios.gan.training")
        rows[role] = {}
        for name, tensor in parameters.items():
            require(isinstance(tensor, DTensor), "GAN stream requires original native DTensor owners", path="helios.gan.training")
            info = layout(tensor)
            require(all(info[k] == v for k, v in TOPOLOGY.items()) and info["dtype"] == "torch.float32",
                    "GAN native FP32 world16 owner layout changed", path="helios.gan.training")
            rows[role][name] = {k:v for k,v in info.items() if k != "local_shape"}
    return {"topology": TOPOLOGY, "trainables_sha256": fingerprint(rows)}


def validate_layout_record(value):
    require(isinstance(value, dict) and set(value) == {"topology", "trainables_sha256"} and value["topology"] == TOPOLOGY
            and isinstance(value["trainables_sha256"], str) and re.fullmatch("[0-9a-f]{64}", value["trainables_sha256"]),
            "GAN checkpoint native owner layout missing/changed", path="helios.gan.training")


def executed_effects(build, before):
    """Fresh dynamic exits retain the original strict mechanism installation gate.

    The old effects_between literal 27-rollout-call receipt remains unchanged.
    Here the original observer checks the actual new trace's exact call census;
    fixed C graphs still have exactly 400 named frozen-down saves/reuses/cycle.
    Every other enabled mechanism must have a positive actual per-cycle delta,
    no fallback, and matching issued/completed/replay counters.
    """
    after = implementation().candidate_bundle.effect_snapshot(build)
    require(set(before) == set(after), "GAN stream effect inventory changed", path="helios.gan.training")
    delta = {key: after[key] - before[key] for key in before}
    require(all(type(v) is int and v > 0 for v in delta.values())
            and delta["down_original"] == delta["down_replay"] == 400,
            "full native GAN stream mechanisms missing/stale", path="helios.gan.training")
    for role in ("generator", "critic"):
        require(len({delta[role + "." + k] for k in ("lora_b_gemm_selected_invocation_count", "lora_b_gemm_selected_attempt_count",
            "lora_b_gemm_completed_invocation_count", "lora_b_gemm_kernel_call_count")}) == 1
            and delta[role + ".lora_b_gemm_selected_invocation_count"] ==
                delta[role + ".lora_b_gemm_selected_grad_invocation_count"] + delta[role + ".lora_b_gemm_selected_nograd_invocation_count"]
            and delta[role + ".checkpoint_scope_execution_count"] == delta[role + ".checkpoint_scope_restore_count"],
            "GAN stream LoRA execution/replay counter mismatch", path="helios.gan.training")
    require(delta["critic_checkpoint.selective_flash_original_save_count"] ==
            delta["critic_checkpoint.selective_flash_recompute_reuse_count"]
            and delta["norm.backward_call_count"] == delta["norm.backward_kernel_call_count"],
            "GAN stream FA3/norm replay counter mismatch", path="helios.gan.training")
    return {"status": "PASS_EXECUTED_FRESH_GAN_BUNDLE", "call_deltas": delta,
            "scope": "ACTUAL_DYNAMIC_COHORT_COUNTERS_NOT_NUMERICAL_PARITY", "numerical_status": "REVIEW"}


def validate_boundary(state, generator, critic, trace):
    from .native_state import _named
    from distillschedule.runtime.helios_planner_cycle import _initial_adapter_state
    trace.validate_boundary()
    for engine in (generator, critic):
        require(_initial_adapter_state(engine.module) is True
                and all(p.requires_grad for p in _named(engine)[1].values()),
                "GAN checkpoint requires joint-gradient, enabled-adapter boundary", path="helios.gan.training")
    for view in state.values():
        validate = getattr(view, "validate_checkpoint_boundary", None)
        if callable(validate):
            validate()


def validate_output_runtime(core, result):
    """Output restores state, but must not start or pretend to finish training."""
    require(isinstance(result, dict) and result.get("status") == "EXPORTED"
            and result.get("completed_cycles") == 0
            and result.get("output_only") == dict(cycles=0, capture=0, optimizer_updates=0, ema_updates=0, merge_count=0)
            and core.started_at is core.completed is core.cohort is core.current is None
            and core.calls == 0 and not core.rows and not core.stores
            and core.state["combined_context_count"] == 0 and not core.state["audit_errors"],
            "UNTOUCHED_OUTPUT_ONLY_GAN_STREAM", path="helios.gan.training")


def run_native_cycles(*, p0, args, policy, result, output, rank, device, logical,
                      generator_engine, critic_engine, ema, diagnostics, negative, cfg, generator_free_memory):
    from .helios_lifecycle import current_training, _plane
    binding = current_training()
    require(binding is not None and binding.get("gan_core") is not None
            and binding.get("gan_stream") is True and binding.get("metadata_group") is not None,
            "installed full native GAN stream runtime required", path="helios.gan.training")
    require(cfg is not None and cfg.training_config.is_separate_gan_grad is False,
            "original joint GAN gradient owner required", path="helios.gan.training")
    with _plane(binding):
        if "native_history_policy" in binding.get("stream", {}).get("identity", {}):
            from distillschedule.runtime.helios_gan_native_history import admit_engine, descriptor as history_descriptor
            def native_history_capability():
                require(binding["identity"].get("gan_native_history") == history_descriptor(),
                        "native GAN history identity changed", path="helios.gan.training")
                if checkpoint_cpu_policy(binding["config"]) is not None:
                    from distillschedule.runtime.helios_gan_checkpoint_cpu import admit_engine as admit_capacity
                    return {k:v for k,v in admit_capacity(generator_engine).items() if k != "callables"}
                return admit_engine(generator_engine)
            capability = distributed.checked(native_history_capability, "Helios GAN complete-block native history owner")
            result["native_history"] = {"policy": history_descriptor(), "actual_capability": capability}
        layout = distributed.checked(lambda: owner_layout(generator_engine, critic_engine), "Helios GAN native owner layout")
        saved = binding["identity"].get("native_owner_layout")
        distributed.checked(lambda: require(not binding["config"].trainer.get("resume_from") or saved == layout,
            "rebuilt native GAN owner layout differs", path="helios.gan.training"), "Helios GAN restored layout")
        binding["identity"] = {**binding["identity"], "native_owner_layout": layout}
        binding["checkpoint_policy"] = _checkpoint_policy(binding["identity"])
        require(all(row == binding["identity"] for row in distributed.gather(binding["identity"])),
                "GAN rank owner layouts differ", path="helios.gan.training")
    return execute_cycles(p0=p0, args=args, policy=policy, result=result, rank=rank, device=device,
        generator=generator_engine, critic=critic_engine, ema=ema, diagnostics=diagnostics, negative=negative,
        cfg=cfg, generator_free_memory=generator_free_memory, binding=binding,
        progress=lambda value: p0.base._atomic_json(output, value))


def execute_cycles(*, p0, args, policy, result, rank, device, generator, critic, ema,
                   diagnostics, negative, cfg, generator_free_memory, binding, progress=None):
    from distillschedule.runtime.helios_planner_cycle import _Observer, _prepare_cycle, _versions
    from .helios_lifecycle import _plane, admit_stop_control, stop_cycle
    from .native_state import engine_state
    from .checkpoint import read_manifest, restore_checkpoint, save_checkpoint
    require(_versions(generator, critic, ema) == expected_versions(0), "native GAN bootstrap clocks changed", path="helios.gan.training")
    state = engine_state({"generator": generator, "critic": critic}, ema=ema)
    core, impl = binding["gan_core"], implementation()
    resume = binding["config"].trainer.get("resume_from")
    start = 0
    with _plane(binding):
        end = admit_stop_control(binding["config"])
        if resume:
            header = distributed.checked(lambda: read_manifest(resume, identity=binding["identity"]), "Helios GAN restart metadata")
            validate_progress(header, binding)
            if binding.get("native_output") is None:
                stop_cycle(binding["config"], start=header["cycle"])
            restored = restore_checkpoint(resume, state, identity=binding["identity"])
            start = restored["cycle"]
            require(_versions(generator, critic, ema) == expected_versions(start), "restored GAN clocks differ", path="helios.gan.training")
        if binding.get("native_output") is not None:
            require(resume is not None, "GAN output requires complete checkpoint", path="helios.gan.training")
            from .helios_lifecycle import publish_native_output
            return publish_native_output(args=args, generator=generator, critic=critic, ema=ema, binding=binding,
                                         manifest=header, result=result)
    stream = binding["stream"]
    result.update(schema_version=SCHEMA, status="RUNNING", mode="manual", preset=PRESET_ID,
        same_process=True, initializations=1, cycles=[], source_cohort_policy="fresh_step_qualified_original_GAN_inputs",
        checkpoint_identity=binding["identity"], stream_sha256=stream["stream_sha256"], resume_cycle=start,
        initial_versions=expected_versions(start), initial_curriculum_versions=logical_versions(start), latest_checkpoint=resume,
        numerical_status="REVIEW", native_GPU_continuation="NOT_RUN", capacity_validation="NOT_RUN",
        formal_timing_eligible=False, claimable_speedup=False, generation="UNSUPPORTED", export="UNSUPPORTED")
    if "native_history_policy" in stream["identity"]:
        result.update(schema_version="distillschedule.original_helios_gan_native_history_training.v1",
                      source_history_policy=stream["identity"]["native_history_policy"])
    candidate = checkpoint_cpu_policy(binding["config"])
    if candidate is not None:
        result.update(schema_version="distillschedule.original_helios_gan_checkpoint_cpu_training.v1",preset=candidate.PRESET_ID,
            checkpoint_input_cpu=dict(policy=candidate.descriptor(),capacity=binding["identity"]["checkpoint_input_capacity"],
                startup=binding["checkpoint_input_startup"],executions=binding.setdefault("checkpoint_input_executions",[])))
    from distillschedule.data.helios_epoch_inputs import receipt as input_receipt
    result.update(input_receipt(stream, cycle=start))
    cadence = binding["config"].trainer.get("checkpoint_every", 0)
    for index in range(start, end):
        entry, task = stream["cohorts"][index], inputs.task_for_stream(stream, index)
        trace = observer = prepared = None
        try:
            with _plane(binding):
                trace = distributed.checked(lambda: inputs.load_cohort(task, entry["trace_bundle"], expected_hash=entry["trace_hash"]),
                                            "Helios GAN fresh cohort")
                observed = {"cohort": entry, "versions": _versions(generator, critic, ema)}
                require(all(v == observed for v in distributed.gather(observed)), "rank GAN cohort/version mismatch", path="helios.gan.training")
                distributed.checked(lambda: core.begin_cycle(index, trace), "Helios GAN fresh named-down qualification")
                before = distributed.checked(lambda: impl.candidate_bundle.effect_snapshot(result["build"]), "Helios GAN pre-cycle bundle census")
            observer = _Observer(torch=p0.torch, device=device, rank=rank, impl=impl, generator=generator,
                critic=critic, ema=ema, mode="native", version_offsets={"generator":198, "critic":990, "ema":0})
            prepared, graph, order = _prepare_cycle(observer=observer, p0=p0, args=args, policy=policy,
                steps=task.source_steps, trace=trace, negative=negative, cfg=cfg,
                diagnostics=diagnostics, generator_free_memory=generator_free_memory)
            with observer.installed():
                prepared.run()
            with _plane(binding):
                validation = distributed.checked(observer.finish, "Helios GAN actual call/version census")
                def boundary():
                    core.cycle_boundary(index + 1)
                    validate_boundary(state, generator, critic, trace)
                    versions = _versions(generator, critic, ema)
                    require(versions == expected_versions(index + 1), "GAN complete-cycle clocks changed", path="helios.gan.training")
                    return versions
                versions = distributed.checked(boundary, "Helios GAN complete-cycle cache/adapter/state boundary")
                effects = distributed.checked(lambda: executed_effects(result["build"], before), "Helios GAN full executed stream bundle")
            result["cycles"].append({"cycle":index + 1, "cohort_index":index, "source_steps":list(task.source_steps),
                "trace_hash":entry["trace_hash"], "semantic_validation":validation, "dispatch":prepared.receipt(),
                "selected_order":list(order), "actions":observer.actions, "final_versions":versions,
                "curriculum_versions":logical_versions(index+1), "candidate_bundle":effects,
                "peak_live_rollouts":observer.peak_live, "retained_rollouts":len(observer.live), "status":"PASS"})
            result.update(status="PARTIAL_ORIGINAL_HELIOS_GAN_TRAINING", completed_cycles=index + 1,
                          next_cohort=index + 1, final_versions=versions)
            result.update(input_receipt(stream, cycle=index + 1))
            saved = None
            if cadence and ((index + 1) % cadence == 0 or index + 1 == len(stream["cohorts"])):
                path = Path(binding["config"].trainer["output_dir"]) / "checkpoints" / f"cycle-{index+1:06d}"
                with _plane(binding):
                    saved = save_checkpoint(path, state, identity=binding["identity"], cycle=index + 1, cursor=index + 1,
                                            versions=versions, policy=binding["checkpoint_policy"])
                result["latest_checkpoint"] = str(saved)
            if "stop_after_cycle" in binding["config"].trainer and index + 1 == end:
                require(saved is not None, "stop_after_cycle requires a published complete checkpoint boundary",
                        path="helios.gan.training")
            if progress:
                with _plane(binding):
                    distributed.checked(lambda: progress(result), "Helios GAN rank progress")
        except BaseException:
            core.abort()
            raise
        finally:
            if trace is not None:
                trace.close()
            del trace, observer, prepared
    core.finish()
    result.update(status=("STOPPED_ORIGINAL_HELIOS_GAN_AT_CHECKPOINT" if end < len(stream["cohorts"])
                          else "COMPLETED_ORIGINAL_HELIOS_GAN_TRAINING"), semantics_status="PASS",
                  completed_cycles=end, final_versions=_versions(generator, critic, ema),
                  training_segment=dict(horizon_cycles=len(stream["cohorts"]), start_cycle=start,
                      requested_stop_cycle=end, completed_cycle=end, executed_cycles=end-start,
                      complete_horizon=end == len(stream["cohorts"])))
    return result


def run_training(config, *, admit, native_output=None):
    from .helios_worker import _backend
    from .helios_lifecycle import original_training_scope, admit_original_output
    require(native_output is None or config.trainer.get("resume_from") and not config.trainer.get("checkpoint_every"),
            "GAN output requires complete checkpoint and no training saves", path="helios.gan.training")
    with distributed.world16_reference_process_group("W3") as ownership:
        _, source, _, admitted = distributed.checked(admit, "Helios GAN source/config/input admission")
        distributed.checked(lambda: require(not source["identity"]["dirty"], "execution requires clean committed source", path="helios.gan.training"),
                            "Helios GAN source cleanliness")
        distributed.checked(validate_environment, "Helios GAN environment")
        backend = distributed.checked(_backend, "Helios GAN original imports")
        identity = {"schema_version":"distillschedule.helios_gan_training_identity.v1",
            "source_head":source["identity"]["head"], "source_fingerprint":admitted["stream"]["identity"]["source_fingerprint"],
            "stream_sha256":admitted["stream"]["stream_sha256"], "assets":admitted["assets"]["assets"],
            "execution_policy":admitted["selected"], "bootstrap_logical_hash":admitted["bootstrap"]["logical_manifest_hash"]}
        from distillschedule.data.helios_epoch_inputs import identity_binding
        identity.update(identity_binding(admitted["stream"]))
        if "native_history_policy" in admitted["stream"]["identity"]:
            from distillschedule.runtime.helios_gan_native_history import descriptor as history_descriptor
            identity.update(schema_version="distillschedule.helios_gan_native_history_training_identity.v1",
                            gan_native_history=history_descriptor())
        candidate = checkpoint_cpu_policy(config)
        if candidate is not None:
            identity.update(schema_version=candidate.IDENTITY,checkpoint_input_capacity=admitted["checkpoint_input_capacity"])
        binding = {"config":config, "stream":admitted["stream"], "assets":admitted["assets"], "identity":identity,
                   "gan_stream":True, "portable_reference":True, "implementation":admitted["implementation"], **ownership}
        if native_output is not None:
            binding["native_output"] = native_output
        old = sys.argv
        try:
            sys.argv = ["original-helios-gan-stream-v1", *admitted["argv"]]
            with original_training_scope(binding):
                admit_original_output(rank=int(os.environ["RANK"]))
                if native_output is not None:
                    from .helios_lifecycle import _plane
                    with _plane(binding):
                        native_output.admit()
                return backend.main()
        finally:
            sys.argv = old
