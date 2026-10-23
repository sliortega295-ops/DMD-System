"""Opt-in lifecycle around the original W2 setup and one-cycle helpers.

No model, loss, optimizer or EMA recurrence is implemented here. The original
P0 worker constructs/restores those owners once. This seam selects fresh inputs
between complete cycles and serializes actual owned state through native_state.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from datetime import timedelta
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
                                 cycles=binding["config"].trainer["cycles"], verify_rank=rank)
            require(stream["stream_sha256"] == binding["identity"]["stream_sha256"],
                    "stream changed during worker admission", path="helios.training")
            identity = {**binding["identity"], "verified_immutable_assets": immutable_assets(binding["assets"])}
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
                trace = load_cohort(HeliosCohortTask(0), first["trace_bundle"], expected_hash=first["trace_hash"])
                controls = [trace.load_critic_entry(rank=rank, step=step, device="cpu") for step in trace.logical_steps]
                for group in chunked(controls, policy.rollout_batch_size):
                    validate_rollout_entries(group)
                negative = NegativePromptAsset.load(binding["assets"]["assets"]["negative_prompt"]["path"])
                startup = {"logical":logical, "trace":trace, "request":trace.task.request, "negative":negative}
            checkpoint_policy = {"schema_version": "distillschedule.helios_training_checkpoint_policy.v1",
                                 "identity_sha256": fingerprint(identity)}
            return stream, identity, checkpoint_policy, startup

        stream, identity, checkpoint_policy, startup = distributed.checked(
            local_admission, "Helios reference input/asset admission")
        # Publish only collectively admitted, actually verified identities;
        # neither identity agreement nor restart may use unverified asset pins.
        binding.update(stream=stream, identity=identity, checkpoint_policy=checkpoint_policy)
        if startup is not None:
            binding["startup"] = startup
        distributed.admit_run_controls(binding["config"])
        peers = distributed.gather({"identity": binding["identity"], "policy": binding["checkpoint_policy"]})
        require(all(item == peers[0] for item in peers), "rank source/stream/reference differs", path="helios.training")
        resume = binding["config"].trainer.get("resume_from")
        if resume:
            from .checkpoint import read_manifest
            header = distributed.checked(lambda: read_manifest(resume, identity=binding["identity"]),
                                         "Helios restart metadata")
            validate_progress(header, binding)
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
    count = len(binding["stream"]["cohorts"])
    cycle = header["cycle"]
    require(header["policy"] == binding["checkpoint_policy"] and type(cycle) is int
            and 0 < cycle == header["cursor"] < count and header["versions"] == expected_versions(cycle),
            "restart must be an incomplete-stream complete G1/C5/EMA5 boundary", path="helios.training")


def execute_cycles(*, p0, args, policy, generator, critic, ema, diagnostics, negative,
                   rank, device, binding, result, progress=None):
    """One original preparation/dispatch per fresh cohort; persistent owners."""
    from distillschedule.runtime.helios_planner_cycle import _Observer, _prepare_cycle, _versions
    from distillschedule.search.helios_implementation import HeliosImplementation
    from .native_state import engine_state
    from .checkpoint import read_manifest, restore_checkpoint, save_checkpoint
    state = engine_state({"generator": generator, "critic": critic}, ema=ema)
    require(_versions(generator, critic, ema) == expected_versions(0),
            "original bootstrap G/C/EMA clocks changed", path="helios.training")
    start = 0
    resume = binding["config"].trainer.get("resume_from")
    with _plane(binding):
        if resume:
            header = distributed.checked(lambda: read_manifest(resume, identity=binding["identity"]),
                                         "Helios restart metadata")
            validate_progress(header, binding)
            restored = restore_checkpoint(resume, state, identity=binding["identity"])
            start = restored["cycle"]
            require(_versions(generator, critic, ema) == restored["versions"],
                    "restored original clocks differ from checkpoint", path="helios.training")
            del restored
    # A value object only for _prepare_cycle's graph/call census. No generated
    # plan is selected, no winner inherited, and receipt_binding is not bypassed
    # for a saved plan: those combinations are rejected by public admission.
    implementation = HeliosImplementation("no-gan", "world16-fsdp", 512, 1, False)
    stream = binding["stream"]
    result.update(schema_version=SCHEMA, status="RUNNING", mode="reference", cycles=[],
        bootstrap_contract=result.pop("contract", None), source_cohort_policy="fresh_step_qualified_original_inputs",
        bootstrap_initial=result.pop("initial", None),
        stream_sha256=stream["stream_sha256"], runtime_seed=binding.get("rank_seed"),
        resume_cycle=start, checkpoint_identity=binding["identity"], latest_checkpoint=resume,
        initializations=1, same_process=True, initial_versions=expected_versions(start),
        numerical_status="NOT_RUN", formal_timing_eligible=False, claimable_speedup=False,
        generation="UNSUPPORTED", export="UNSUPPORTED")
    cadence = binding["config"].trainer.get("checkpoint_every", 0)
    for index in range(start, len(stream["cohorts"])):
        task = HeliosCohortTask(index)
        entry = stream["cohorts"][index]
        with _plane(binding):
            trace = distributed.checked(lambda: load_cohort(task, entry["trace_bundle"], expected_hash=entry["trace_hash"]),
                                        "Helios fresh cohort binding")
            observed = {"cohort": entry, "versions": _versions(generator, critic, ema)}
            require(all(item == observed for item in distributed.gather(observed)),
                    "rank cohort/version mismatch", path="helios.training")
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
        row = {"cycle": index + 1, "cohort_index": index, "source_steps": list(task.source_steps),
               "trace_hash": entry["trace_hash"], "semantic_validation": validation,
               "dispatch": prepared.receipt(), "selected_order": list(order),
               "actions": observer.actions, "final_versions": versions, "status": "PASS"}
        result["cycles"].append(row)
        result.update(status="PARTIAL_ORIGINAL_HELIOS_TRAINING", final_versions=versions,
                      completed_cycles=index + 1, next_cohort=index + 1)
        if cadence and ((index + 1) % cadence == 0 or index + 1 == len(stream["cohorts"])):
            path = Path(binding["config"].trainer["output_dir"]) / "checkpoints" / f"cycle-{index+1:06d}"
            with _plane(binding):
                saved = save_checkpoint(path, state, identity=binding["identity"], cycle=index + 1, cursor=index + 1,
                                        versions=versions, policy=binding["checkpoint_policy"])
            result["latest_checkpoint"] = str(saved)
        if progress is not None:
            with _plane(binding):
                distributed.checked(lambda: progress(result), "Helios rank progress receipt")
        del observer, prepared, trace
    result.update(status="COMPLETED_ORIGINAL_HELIOS_TRAINING", semantics_status="PASS",
                  final_versions=_versions(generator, critic, ema), completed_cycles=len(stream["cohorts"]))
    return result


def run_training_cycles(*, p0, args, policy, result, output, rank, device, logical,
                        generator_engine, critic_engine, ema, diagnostics, negative):
    binding = current_training()
    require(binding is not None and binding.get("metadata_group") is not None,
            "original training preflight missing", path="helios.training")
    require(all(int(logical.component(key)["optimizer_step"]) == expected_versions(0)[key]
                for key in ("generator", "critic")), "logical original bootstrap changed", path="helios.training")
    return execute_cycles(p0=p0, args=args, policy=policy, result=result, rank=rank, device=device,
        generator=generator_engine, critic=critic_engine, ema=ema, diagnostics=diagnostics, negative=negative,
        binding=binding, progress=lambda value: p0.base._atomic_json(output, value))
