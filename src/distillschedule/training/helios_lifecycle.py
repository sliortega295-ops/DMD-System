"""Opt-in lifecycle around the original W2 setup and one-cycle helpers.

No model, loss, optimizer or EMA recurrence is implemented here. The original
P0 worker constructs/restores those owners once. This seam selects fresh inputs
between complete cycles and serializes actual owned state through native_state.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import timedelta
import json
import os
from pathlib import Path
import random

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import fingerprint
from . import distributed
from .helios_inputs import HeliosCohortTask, load_cohort, load_stream

SCHEMA = "distillschedule.original_helios_training.v1"
_ACTIVE = ContextVar("original_helios_training", default=None)


def current_training():
    return _ACTIVE.get()


@dataclass(frozen=True)
class _NativeHistoryConstruction:
    identity_sha256: str
    policy: object


def _construction_identity(binding):
    return fingerprint({"identity": binding["identity"],
        "input_kind": binding["config"].data["input_kind"],
        "stream_sha256": binding["stream"]["stream_sha256"]})


def _bind_native_history_construction(*, policy):
    """Mint only after the original collective source/input/startup admission."""
    binding = current_training()
    stream = binding["stream"]
    if "native_history_policy" not in stream["identity"]:
        return
    from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
    from distillschedule.strategies.wan_task import NATIVE_HISTORY_POLICY
    if binding.get("gan_stream") is True:
        from .helios_gan_inputs import GANCohortTask as Task, NATIVE_KIND, NATIVE_SCHEMA
        from distillschedule.runtime.helios_gan_native_history import descriptor
        schema = "distillschedule.helios_gan_native_history_training_identity.v1"
        capability, selected = "gan_native_history", "execution_policy"
    else:
        from .helios_inputs import HeliosCohortTask as Task, NATIVE_KIND, NATIVE_SCHEMA
        from . import helios_native_safe_stream, helios_checkpoint_cpu
        from .helios_worker import _safe_policy
        combined = _safe_policy(binding["config"])
        if combined in (helios_native_safe_stream, helios_checkpoint_cpu):
            combined.validate_config(binding["config"])
            descriptor = combined.history_descriptor
            schema, capability, selected = combined.IDENTITY, "native_history", "execution_policy"
        else:
            from distillschedule.runtime.helios_native_history import descriptor
            schema = "distillschedule.helios_native_history_training_identity.v1"
            capability, selected = "native_history", "reference_policy"
    task = Task.from_dict(stream["cohorts"][0]["task"])
    identity = binding["identity"]
    mesh = "node-hsdp" if schema in ("distillschedule.helios_native_safe_stream_identity.v1",
        "distillschedule.helios_checkpoint_hidden_cpu_identity.v1") else "world16-fsdp"
    require(type(policy) is StaticFSDP2P0Policy and policy.mesh_policy == mesh
            and binding.get("portable_reference") is True
            and binding["config"].data["input_kind"] == NATIVE_KIND
            and stream["schema_version"] == NATIVE_SCHEMA
            and task.native_policy == stream["identity"]["native_history_policy"] == NATIVE_HISTORY_POLICY
            and identity["schema_version"] == schema
            and identity.get(capability) == identity[selected].get("native_history") == descriptor()
            and identity["stream_sha256"] == stream["stream_sha256"]
            and "verified_immutable_assets" in identity,
            "native history construction identity/policy changed", path="helios.training")
    if schema == "distillschedule.helios_checkpoint_hidden_cpu_identity.v1":
        from .helios_checkpoint_cpu import require_binding
        require_binding()
    binding["_native_history_construction"] = _NativeHistoryConstruction(_construction_identity(binding), policy)


def native_history_cpu_first_construction(*, policy):
    """Construction capability, not a new model/sharder or a benchmark flag."""
    binding = current_training()
    capability = None if binding is None else binding.get("_native_history_construction")
    if capability is None:
        return False
    require(type(capability) is _NativeHistoryConstruction and capability.policy == policy
            and policy.mesh_policy == ("node-hsdp" if binding["identity"]["schema_version"] in (
                "distillschedule.helios_native_safe_stream_identity.v1",
                "distillschedule.helios_checkpoint_hidden_cpu_identity.v1") else "world16-fsdp")
            and capability.identity_sha256 == _construction_identity(binding),
            "native history construction identity/policy changed", path="helios.training")
    return True


@contextmanager
def original_training_scope(binding):
    require(current_training() is None, "nested original training scope", path="helios.training")
    token = _ACTIVE.set(binding)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def expected_versions(completed):
    return {"generator": 2 + completed, "critic": 10 + 5 * completed, "ema": 10 + 5 * completed}


def stop_cycle(config, *, start=0):
    """Execution control only: never shorten the identity-bound prepared stream."""
    trainer = config.trainer
    horizon = trainer["cycles"]
    stop = trainer.get("stop_after_cycle", horizon)
    if "stop_after_cycle" in trainer:
        every = trainer.get("checkpoint_every", 0)
        require(type(stop) is int and 1 <= stop <= horizon and type(every) is int and every > 0
                and (stop == horizon or stop % every == 0),
                "trainer.stop_after_cycle must be a complete checkpoint boundary within the unchanged horizon",
                path="helios.training")
    require(start < stop, "stop_after_cycle must be after the completed resume boundary", path="helios.training")
    return stop


def admit_stop_control(config):
    """Every rank must choose the same last collective before constructing owners."""
    stop = distributed.checked(lambda: stop_cycle(config), "Helios stop checkpoint boundary admission")
    controls = {"stop_after_cycle": stop}
    require(all(row == controls for row in distributed.gather(controls)),
            "ranks disagree on Helios stop_after_cycle", path="helios.training")
    return stop


def _plane(binding):
    if binding.get("metadata_group") is None:
        return nullcontext()
    from distillschedule.open_workload.replicated import control_plane
    return control_plane(binding["metadata_group"])


def owns_original_process_group():
    binding = current_training()
    return binding is not None and binding.get("owns_process_group") is True


def admit_original_output(*, rank):
    """Collectively admit an unused rank destination BEFORE entering P0."""
    binding = current_training()
    require(binding is not None, "original training scope missing", path="helios.training")
    def local():
        output = Path(binding["config"].trainer["output_dir"]) / "ranks" / f"rank_{rank:02d}.json"
        require(not os.path.lexists(output),
                "new output directory required; old rank receipt is preserved", path="helios.training")
        return str(output.resolve())
    with _plane(binding):
        binding["owned_rank_output"] = distributed.checked(local, "Helios original output receipt admission")


def owns_original_rank_output(output):
    """Snapshot failure-write ownership at P0 entry, not after owned progress.

    Default benchmark publication is unchanged. An opt-in invocation must have
    admitted this destination, and may not replace a receipt present on entry.
    This is not an exclusive filesystem reservation against concurrent writers.
    """
    binding = current_training()
    return binding is None or (binding.get("owned_rank_output") == str(Path(output).resolve())
                               and not os.path.lexists(output))


def adopt_original_process_group(*, dist):
    """Only adopt the exact fresh group owned by the public training wrapper."""
    if not owns_original_process_group():
        return False
    binding = current_training()
    require(dist.is_initialized() and dist.group.WORLD is binding.get("process_group")
            and dist.get_world_size() == 16 and str(dist.get_backend()) == "nccl"
            and binding.get("metadata_group") is not None,
            "original training process-group ownership changed", path="helios.training")
    return True


def checked_original_runtime(fn, label):
    binding = current_training()
    if binding is None or binding.get("metadata_group") is None:
        return fn()
    with _plane(binding):
        return distributed.checked(fn, label)


def admit_reference_startup(*, torch, dist, rank, args, policy):
    """Portable opt-in only: admit before any legacy logical/trace payload IO."""
    binding = current_training()
    if binding is None or not binding.get("portable_reference"):
        return None
    admit_original_rank(torch=torch, dist=dist, rank=rank, args=args, policy=policy, seed_runtime=False)
    return binding["startup"]


def admit_original_rank(*, torch, dist, rank, args, policy, seed_runtime=True):
    """Called by original main BEFORE model construction, after trace admission."""
    binding = current_training()
    if binding is None:
        return
    if binding.get("gan_stream") is True:
        from .helios_gan_training import admit_original_rank as admit_gan
        return admit_gan(torch=torch, dist=dist, rank=rank, args=args, policy=policy, seed_runtime=seed_runtime)
    require(dist.get_world_size() == 16,
            "reference original world16 helpers required", path="helios.training")
    if binding.get("metadata_group") is None:
        binding["metadata_group"] = dist.new_group(backend="gloo", timeout=timedelta(minutes=15))
    with _plane(binding):
        def local_admission():
            from distillschedule.search.helios_implementation import current_implementation
            from .helios_worker import _OFF, immutable_assets, load_reference_bootstrap
            require(args.variant == "no-gan" and args.timing_mode
                    and not args.profile_mode and not args.profile_diagnostics,
                    "reference original world16 helpers required", path="helios.training")
            implementation = binding.get("implementation")
            effective = None
            if implementation is not None:
                from .helios_worker import _safe_policy
                safe = _safe_policy(binding["config"])
                require(binding.get("portable_reference") and implementation == safe.implementation()
                        and binding["identity"].get("execution_policy") == safe.descriptor(),
                        "safe stream implementation/source binding changed", path="helios.training")
                safe.validate_environment(require_kernel=True)
                effective = safe.bind_effective(args=args, policy=policy)
            else:
                require(policy.mesh_policy == "world16-fsdp" and policy.rollout_batch_size == 1
                        and not policy.pack_real_score_pair and policy.modulation_chunk_tokens == 512
                        and not policy.generator_nograd_persistent_unshard and args.attention_backend == "sdpa"
                        and policy.selective_flash_checkpoint_stride == args.selective_flash_checkpoint_stride == 0
                        and args.ema_device == "cuda"
                        and args.generator_residual_storage == args.critic_residual_storage == "cuda_out_of_place",
                        "reference physical policy changed", path="helios.training")
                require(all(getattr(args, name.replace("-", "_")) is False for name in _OFF),
                        "reference cannot enable an unbound optimization", path="helios.training")
                require(current_implementation() is None,
                        "reference cannot inherit a generated plan environment", path="helios.training")
                require(not os.environ.get("LOCAL_KERNELS") and not os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
                        and not os.environ.get("PYTORCH_ALLOC_CONF"),
                        "reference attention/allocator environment differs", path="helios.training")
            output = Path(binding["config"].trainer["output_dir"])
            require(not os.path.lexists(output / "ranks" / f"rank_{rank:02d}.json"),
                    "new output directory required; old rank receipt is preserved", path="helios.training")
            stream = load_stream(binding["config"].data["prepared"],
                                 cycles=binding["config"].trainer["cycles"], verify_rank=rank,
                                 expected_kind=binding["config"].data["input_kind"])
            require(stream["stream_sha256"] == binding["identity"]["stream_sha256"],
                    "stream changed during worker admission", path="helios.training")
            identity = {**binding["identity"], "verified_immutable_assets": immutable_assets(binding["assets"])}
            if effective is not None:
                identity["effective_configuration"] = effective
            startup = None
            if binding.get("portable_reference"):
                from runtime.gst_pool_contract import NegativePromptAsset
                from distillschedule.runtime.fsdp2_p0 import validate_rollout_entries, chunked
                logical, _ = load_reference_bootstrap(binding["assets"], verify_payloads=False)
                first = stream["cohorts"][0]
                rows = binding["assets"]["assets"]
                require(all(Path(getattr(args, option)).resolve() == Path(rows[role]["path"]).resolve()
                    for role, option in (("generator_model","generator_root"), ("score_model","score_root"),
                        ("generator_precast","precast_generator_root"), ("score_precast","precast_score_root"),
                        ("negative_prompt","negative_prompt"))),
                    "direct reference model/negative asset binding changed", path="helios.training")
                require(args.expected_logical_hash == logical.manifest_hash == identity["bootstrap_logical_hash"]
                        and args.expected_trace_hash == first["trace_hash"] == identity["bootstrap_trace_hash"]
                        and args.logical_package.resolve() == Path(binding["assets"]["assets"]["logical_checkpoint"]["path"]).resolve()
                        and args.trace_bundle.resolve() == Path(first["trace_bundle"]).resolve(),
                        "direct reference bootstrap/startup binding changed", path="helios.training")
                trace = load_cohort(HeliosCohortTask.from_dict(first["task"]), first["trace_bundle"], expected_hash=first["trace_hash"])
                controls = [trace.load_critic_entry(rank=rank, step=step, device="cpu") for step in trace.logical_steps]
                for group in chunked(controls, policy.rollout_batch_size):
                    validate_rollout_entries(group)
                negative = NegativePromptAsset.load(binding["assets"]["assets"]["negative_prompt"]["path"])
                startup = {"logical":logical, "trace":trace, "request":trace.task.request, "negative":negative}
            if implementation is not None and binding["config"].trainer.get("resume_from"):
                # Metadata only, never deserialize model-sized state here.
                # read_manifest below authenticates the complete identity and
                # publication. Actual rebuilt native layouts must match again
                # before restore, not merely trust this saved descriptor.
                path = Path(binding["config"].trainer["resume_from"]) / "manifest.json"
                saved_layout = json.loads(path.read_text()).get("identity", {}).get("native_owner_layout")
                safe.validate_layout_record(saved_layout)
                identity["native_owner_layout"] = saved_layout
            checkpoint_policy = _checkpoint_policy(identity, optimized=implementation is not None)
            return stream, identity, checkpoint_policy, startup, effective

        stream, identity, checkpoint_policy, startup, effective = distributed.checked(
            local_admission, "Helios reference input/asset admission")
        # Publish only collectively admitted, actually verified identities;
        # neither identity agreement nor restart may use unverified asset pins.
        binding.update(stream=stream, identity=identity, checkpoint_policy=checkpoint_policy)
        if effective is not None:
            binding["effective_configuration"] = effective
        if startup is not None:
            binding["startup"] = startup
        distributed.admit_run_controls(binding["config"])
        admit_stop_control(binding["config"])
        peers = distributed.gather({"identity": binding["identity"], "policy": binding["checkpoint_policy"]})
        require(all(item == peers[0] for item in peers), "rank source/stream/reference differs", path="helios.training")
        resume = binding["config"].trainer.get("resume_from")
        if resume:
            from .checkpoint import read_manifest
            header = distributed.checked(lambda: read_manifest(resume, identity=binding["identity"]),
                                         "Helios restart metadata")
            validate_progress(header, binding)
            if binding.get("native_output") is None:
                stop_cycle(binding["config"], start=header["cycle"])
        from . import helios_checkpoint_cpu as checkpoint_cpu
        if binding["identity"].get("schema_version") == checkpoint_cpu.IDENTITY:
            binding["checkpoint_input_startup"] = distributed.checked(
                lambda: checkpoint_cpu.admit_startup(binding, rank=rank), "Helios checkpoint input extra host capacity before construction")
        _bind_native_history_construction(policy=policy)
    if seed_runtime:
        seed_original_runtime(torch=torch, rank=rank)


def seed_original_runtime(*, torch, rank):
    """Keep the original seed keys and original post-dependency setup point."""
    binding = current_training()
    require(binding is not None, "original training scope missing", path="helios.training")
    # New input sampling is already explicit in the stream. The runtime stream
    # is separate and rank-qualified; restart later restores its exact state.
    seed = int(fingerprint({"stream": binding["stream"]["stream_sha256"], "rank": rank,
                            "scope": "original_helios_runtime.v1"})[:16], 16) % (2**63)
    random.seed(seed); torch.random.default_generator.manual_seed(seed); torch.cuda.manual_seed(seed)
    binding["rank_seed"] = seed


def validate_progress(header, binding):
    from distillschedule.data.helios_epoch_inputs import validate_cursor
    validate_cursor(header, binding["stream"])
    count = len(binding["stream"]["cohorts"])
    cycle = header["cycle"]
    within_stream = type(cycle) is int and (cycle <= count if binding.get("native_output") is not None else cycle < count)
    require(header["policy"] == binding["checkpoint_policy"] and type(cycle) is int
            and 0 < cycle == header["cursor"] and within_stream and header["versions"] == expected_versions(cycle),
            "restart must be an incomplete-stream complete G1/C5/EMA5 boundary", path="helios.training")


def _checkpoint_policy(identity, *, optimized):
    from . import helios_native_safe_stream as combined, helios_checkpoint_cpu as checkpoint_cpu
    if identity.get("schema_version") == checkpoint_cpu.IDENTITY:
        checkpoint_cpu.validate_identity(identity, optimized=optimized)
        return {"schema_version":checkpoint_cpu.CHECKPOINT, "identity_sha256":fingerprint(identity)}
    if identity.get("schema_version") == combined.IDENTITY:
        combined.validate_identity(identity, optimized=optimized)
        return {"schema_version":combined.CHECKPOINT, "identity_sha256":fingerprint(identity)}
    if "native_history" in identity:
        from distillschedule.runtime.helios_native_history import descriptor
        require(not optimized and identity["native_history"] == descriptor(),
                "native history checkpoint phase policy changed", path="helios.training")
        return {"schema_version":"distillschedule.helios_native_history_checkpoint_policy.v1",
                "identity_sha256":fingerprint(identity)}
    return {"schema_version": ("distillschedule.helios_safe_stream_checkpoint_policy.v1" if optimized
                               else "distillschedule.helios_training_checkpoint_policy.v1"),
            "identity_sha256": fingerprint(identity)}


def publish_native_output(*, args, generator, critic, ema, binding, manifest, result):
    """Shared post-restore output seam; each algorithm owns its version gate."""
    from types import SimpleNamespace
    from distillschedule.runtime.helios_planner_cycle import _versions
    require(binding.get("native_output") is not None and binding["config"].trainer.get("resume_from"),
            "native output requires a complete checkpoint", path="helios.training")
    view = SimpleNamespace(generator=generator, args=args, request=binding["stream"]["identity"]["request"])
    exported = binding["native_output"].publish(view, manifest=manifest, identity=binding["identity"])
    result.update(schema_version="distillschedule.helios_native_output.v1", status="EXPORTED",
        inference_export=exported, completed_cycles=0, resume_cycle=manifest["cycle"],
        final_versions=_versions(generator, critic, ema), numerical_status="NOT_RUN",
        formal_timing_eligible=False, claimable_speedup=False,
        output_only=dict(cycles=0, capture=0, optimizer_updates=0, ema_updates=0, merge_count=0))
    return result


def execute_cycles(*, p0, args, policy, generator, critic, ema, diagnostics, negative,
                   rank, device, binding, result, progress=None):
    """One original preparation/dispatch per fresh cohort; persistent owners."""
    from distillschedule.runtime.helios_planner_cycle import _Observer, _prepare_cycle, _versions
    from distillschedule.search.helios_implementation import HeliosImplementation
    from .native_state import engine_state
    from .checkpoint import read_manifest, restore_checkpoint, save_checkpoint
    implementation = binding.get("implementation") or HeliosImplementation("no-gan", "world16-fsdp", 512, 1, False)
    optimized = implementation.candidate_bundle.enabled
    if optimized:
        from .helios_worker import _safe_policy
        safe = _safe_policy(binding["config"])
        require(implementation == safe.implementation() and binding.get("effective_configuration") is not None,
                "safe stream preconstruction physical binding missing", path="helios.training")
    state = engine_state({"generator": generator, "critic": critic}, ema=ema)
    native_history = "native_history_policy" in binding["stream"]["identity"]
    if native_history:
        if optimized:
            from distillschedule.runtime.helios_native_safe_history import admit_engine
            descriptor = safe.history_descriptor
            safe.validate_identity(binding["identity"], optimized=True)
        else:
            from distillschedule.runtime.helios_native_history import admit_engine, descriptor
        require(binding["identity"].get("native_history") == descriptor(),
                "native history identity/phase policy differs", path="helios.training")
        with _plane(binding):
            capability = distributed.checked(lambda: admit_engine(generator), "W2 native history whole-block phase admission")
            require(all(row == capability for row in distributed.gather(capability)),
                    "rank native history FSDP/checkpoint topology differs", path="helios.training")
        result["native_history"] = {"policy":descriptor(), "actual_capability":capability}
    require(_versions(generator, critic, ema) == expected_versions(0),
            "original bootstrap G/C/EMA clocks changed", path="helios.training")
    start = 0
    resume = binding["config"].trainer.get("resume_from")
    with _plane(binding):
        end = admit_stop_control(binding["config"])
        if resume:
            header = distributed.checked(lambda: read_manifest(resume, identity=binding["identity"]),
                                         "Helios restart metadata")
            validate_progress(header, binding)
            if binding.get("native_output") is None:
                stop_cycle(binding["config"], start=header["cycle"])
                # Native update helpers lazily import Diffusers/HTTPX/Rich;
                # Rich may draw Python RNG while initializing its style IDs.
                # Complete that import before restoring the training RNG, not
                # inside the first resumed rollout. No model or update runs.
                import importlib
                importlib.import_module("helios.utils.utils_helios_post")
            restored = restore_checkpoint(resume, state, identity=binding["identity"])
            start = restored["cycle"]
            require(_versions(generator, critic, ema) == restored["versions"],
                    "restored original clocks differ from checkpoint", path="helios.training")
            del restored
        if binding.get("native_output") is not None:
            require(resume is not None, "native output requires a complete checkpoint", path="helios.training")
            return publish_native_output(args=args, generator=generator, critic=critic, ema=ema, binding=binding,
                                         manifest=header, result=result)
    # A value object only for _prepare_cycle's graph/call census. No generated
    # plan is selected, no winner inherited, and receipt_binding is not bypassed
    # for a saved plan: those combinations are rejected by public admission.
    if optimized:
        # Validate actual installation before the first cycle, including after
        # restore. Kernel/replay caches are reconstructed by original setup,
        # never loaded from checkpoint or substituted with a reference policy.
        with _plane(binding):
            distributed.checked(lambda: implementation.candidate_bundle.effect_snapshot(result.get("build", {})),
                                "Helios safe stream installation census")
    stream = binding["stream"]
    result.update(schema_version=SCHEMA, status="RUNNING", mode="reference", cycles=[],
        bootstrap_contract=result.pop("contract", None), source_cohort_policy="fresh_step_qualified_original_inputs",
        bootstrap_initial=result.pop("initial", None),
        stream_sha256=stream["stream_sha256"], runtime_seed=binding.get("rank_seed"),
        resume_cycle=start, checkpoint_identity=binding["identity"], latest_checkpoint=resume,
        initializations=1, same_process=True, initial_versions=expected_versions(start),
        numerical_status="NOT_RUN", formal_timing_eligible=False, claimable_speedup=False,
        generation="UNSUPPORTED", export="UNSUPPORTED")
    from distillschedule.data.helios_epoch_inputs import receipt as input_receipt
    result.update(input_receipt(stream, cycle=start))
    if native_history:
        result.update(schema_version="distillschedule.original_helios_native_history_training.v1",
                      source_history_policy=stream["identity"]["native_history_policy"],
                      native_input_kind=binding["config"].data["input_kind"])
    if optimized:
        result.update(schema_version=("distillschedule.original_helios_native_safe_stream.v1" if native_history
                                      else "distillschedule.original_helios_safe_stream.v1"), mode="manual",
            candidate_id=implementation.id, preset=safe.PRESET_ID,
            effective_configuration=binding["effective_configuration"],
            numerical_status="REVIEW", gpu_kernel_numerical_parity="NOT_RUN",
            capacity_validation="NOT_RUN; source bridge does not establish native setup/checkpoint/restore reserve")
    from . import helios_checkpoint_cpu as checkpoint_cpu
    if binding["identity"].get("schema_version") == checkpoint_cpu.IDENTITY:
        result.update(schema_version="distillschedule.original_helios_native_checkpoint_cpu_training.v1",
            checkpoint_input_cpu=dict(policy=checkpoint_cpu.descriptor(), capacity=binding["identity"]["checkpoint_input_capacity"],
                startup=binding.get("checkpoint_input_startup"), target_torch_2_10="NOT_RUN", native_GO=False,
                actions=binding.setdefault("checkpoint_input_executions", [])))
    cadence = binding["config"].trainer.get("checkpoint_every", 0)
    for index in range(start, end):
        entry = stream["cohorts"][index]
        task = HeliosCohortTask.from_dict(entry["task"])
        with _plane(binding):
            trace = distributed.checked(lambda: load_cohort(task, entry["trace_bundle"], expected_hash=entry["trace_hash"]),
                                        "Helios fresh cohort binding")
            observed = {"cohort": entry, "versions": _versions(generator, critic, ema)}
            require(all(item == observed for item in distributed.gather(observed)),
                    "rank cohort/version mismatch", path="helios.training")
            before = (distributed.checked(lambda: implementation.candidate_bundle.effect_snapshot(result.get("build", {})),
                      "Helios safe stream pre-cycle census") if optimized else None)
        observer = _Observer(torch=p0.torch, device=device, rank=rank, impl=implementation,
            generator=generator, critic=critic, ema=ema, mode="native",
            version_offsets={"generator": 0, "critic": 0, "ema": 0})
        prepared, graph, order = _prepare_cycle(observer=observer, p0=p0, args=args, policy=policy,
            steps=task.source_steps, trace=trace, negative=negative, cfg=None, diagnostics=diagnostics,
            generator_free_memory={})
        with observer.installed():
            prepared.run()
        with _plane(binding):
            validation = distributed.checked(observer.finish, "Helios original cycle validation")
            def boundary():
                versions = _versions(generator, critic, ema)
                require(versions == expected_versions(index + 1) == graph.metadata["final_versions"],
                        "original complete-cycle versions changed", path="helios.training")
                for value in state.values():
                    validate = getattr(value, "validate_checkpoint_boundary", None)
                    if callable(validate):
                        validate()
                return versions
            versions = distributed.checked(boundary, "Helios complete-cycle boundary")
            effects = (distributed.checked(lambda: implementation.candidate_bundle.executed_effects(
                result.get("build", {}), before), "Helios safe stream executed bundle census") if optimized else None)
        row = {"cycle": index + 1, "cohort_index": index, "source_steps": list(task.source_steps),
               "trace_hash": entry["trace_hash"], "semantic_validation": validation,
               "dispatch": prepared.receipt(), "selected_order": list(order),
               "actions": observer.actions, "final_versions": versions, "status": "PASS"}
        if native_history:
            row["native_history"] = {"task":task.to_dict(), "rank_control_validation":trace.rank_control_validation,
                                    "phases":[action["native_history_phase"] for action in observer.actions
                                              if "native_history_phase" in action]}
        if effects is not None:
            row.update(candidate_bundle=effects, rollout_lifetimes=observer.lifetimes,
                       peak_live_rollouts=observer.peak_live, retained_rollouts=len(observer.live))
        result["cycles"].append(row)
        result.update(status="PARTIAL_ORIGINAL_HELIOS_TRAINING", final_versions=versions,
                      completed_cycles=index + 1, next_cohort=index + 1)
        result.update(input_receipt(stream, cycle=index + 1))
        saved = None
        if cadence and ((index + 1) % cadence == 0 or index + 1 == len(stream["cohorts"])):
            path = Path(binding["config"].trainer["output_dir"]) / "checkpoints" / f"cycle-{index+1:06d}"
            with _plane(binding):
                saved = save_checkpoint(path, state, identity=binding["identity"], cycle=index + 1, cursor=index + 1,
                                        versions=versions, policy=binding["checkpoint_policy"])
            result["latest_checkpoint"] = str(saved)
        if "stop_after_cycle" in binding["config"].trainer and index + 1 == end:
            # save_checkpoint returns only after all ranks see the durable
            # manifest, success marker and required shards. Never stop on an
            # earlier checkpoint, or a failed/unpublished save.
            require(saved is not None, "stop_after_cycle requires a published complete checkpoint boundary",
                    path="helios.training")
        if progress is not None:
            with _plane(binding):
                distributed.checked(lambda: progress(result), "Helios rank progress receipt")
        del observer, prepared, trace
    result.update(status=("STOPPED_ORIGINAL_HELIOS_AT_CHECKPOINT" if end < len(stream["cohorts"])
                          else "COMPLETED_ORIGINAL_HELIOS_TRAINING"), semantics_status="PASS",
                  final_versions=_versions(generator, critic, ema), completed_cycles=end,
                  training_segment=dict(horizon_cycles=len(stream["cohorts"]), start_cycle=start,
                      requested_stop_cycle=end, completed_cycle=end, executed_cycles=end-start,
                      complete_horizon=end == len(stream["cohorts"])))
    return result


def run_training_cycles(*, p0, args, policy, result, output, rank, device, logical,
                        generator_engine, critic_engine, ema, diagnostics, negative, cfg=None,
                        generator_free_memory=None):
    binding = current_training()
    require(binding is not None and binding.get("metadata_group") is not None,
            "original training preflight missing", path="helios.training")
    if binding.get("gan_stream") is True:
        from .helios_gan_training import run_native_cycles
        return run_native_cycles(p0=p0, args=args, policy=policy, result=result, output=output, rank=rank,
            device=device, logical=logical, generator_engine=generator_engine, critic_engine=critic_engine,
            ema=ema, diagnostics=diagnostics, negative=negative, cfg=cfg, generator_free_memory=generator_free_memory)
    require(all(int(logical.component(key)["optimizer_step"]) == expected_versions(0)[key]
                for key in ("generator", "critic")), "logical original bootstrap changed", path="helios.training")
    if binding.get("implementation") is not None:
        from .helios_safe_stream import owner_layout
        with _plane(binding):
            observed = distributed.checked(
                lambda: owner_layout(generator_engine, critic_engine), "Helios safe stream native owner layout")
            saved = binding["identity"].get("native_owner_layout")
            distributed.checked(lambda: require(not binding["config"].trainer.get("resume_from") or saved == observed,
                "rebuilt safe stream native owner layout differs from checkpoint", path="helios.training"),
                "Helios safe stream restored layout admission")
            binding["identity"] = {**binding["identity"], "native_owner_layout": observed}
            binding["checkpoint_policy"] = _checkpoint_policy(binding["identity"], optimized=True)
            require(all(row == binding["identity"] for row in distributed.gather(binding["identity"])),
                    "rank safe stream native owner layout differs", path="helios.training")
            result["native_owner_layout"] = observed
    return execute_cycles(p0=p0, args=args, policy=policy, result=result, rank=rank, device=device,
        generator=generator_engine, critic=critic_engine, ema=ema, diagnostics=diagnostics, negative=negative,
        binding=binding, progress=lambda value: p0.base._atomic_json(output, value))
