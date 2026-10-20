"""Default-off H100 C2 orchestration of the *actual* 8fc logical-B1 helpers.

No model, optimizer, loss, fake executor, or duration model is implemented here.
The caller constructs its native engines once. This adapter calls the registered
two-slot helpers twice and measures the second continuous cycle. Installation
requires the separate documented hook and a NEW launcher/auditor/source gate;
the old single-Full5 launcher and cost bridge deliberately do not admit this.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import time

MODE_ENV = "DISTILLSCHEDULE_HELIOS_H100_TWO_CYCLE_MODE"
SCHEMA = "distillschedule.helios_h100_two_cycle_rank.v1"
PARENT_HEAD = "8fc70e8ce9f6e0243604812120119c9f4c078e55"
WORKER = "scripts/distillschedule/probe_static_fsdp2_p0_world16.py"
WORKER_SHA = "e9c794287c8b2ad3dca8c9a51e666898d498e671ca950c9f615f6eb24ef7412c"
AUDITOR = "scripts/distillschedule/summarize_helios_hardware_p0.py"
AUDITOR_SHA = "ef8629fcaebf8714da9fa571f348fa9b18bf3b7162e865edfa91a5666acd61b6"
HELPERS = {
    "_run_logical_generator_update_p0": "7f3ba32af905695d1219f127a84702b5d9c3dc0f1630ea6d022eecb41735d803",
    "_precompute_logical_critic_rollouts": "78fc66561b3123b7c57985b03ad3aefb3b086dc9db1c14d7fe9a6dcad966ee8a",
    "_run_logical_critic_update_p0": "0abd912af3e208a18019ec3d77afd7e260a83f54f999c20bf3ef489ab45752a4",
}
POLICIES = {
    "best": "9723dd6a01f3f4cde059a24a1f5dd66b12f7cad19af3229bbdeb3b5299c937ec",
    "h100-nograd-retain": "02de1da3c92e7107b645abd8bd9bb613c065c68899187f4aacd7cb78fdc581ca",
    "h100-nograd-retain-nosync": "c512b34a1cf2a8f4380ae04a95ee3392019915cfdd67eefbf73a8a6f6e18d064",
}
STEPS = (10, 11, 12, 13, 14)
INITIAL = {"generator": 2, "critic": 10, "ema": 10}
ARG_POLICY_FIELDS = {"attention_backend", "generator_nograd_lora_fused_epilogue", "all_role_lora_scale1_add",
                     "generator_residual_storage", "critic_residual_storage"}

# The only permitted edit to the pinned math worker. Also emitted as a patch.
HOOK = '''        # H100_C2_HOOK_BEGIN: default off; installed source is a new cohort.
        from distillschedule.runtime.h100_helios_two_cycle import requested_mode
        if requested_mode() != "off":
            from distillschedule.runtime.h100_helios_two_cycle import run_two_cycles
            run_two_cycles(
                p0=sys.modules[__name__], args=args, policy=policy, result=result,
                rank=rank, device=device, logical=logical, trace=trace,
                negative=negative, cfg=cfg, generator_engine=generator_engine,
                critic_engine=critic_engine, ema=ema, diagnostics=diagnostics,
                generator_free_memory=generator_free_memory,
            )
            base._atomic_json(output, result)
            dist.barrier()
            return 0
        # H100_C2_HOOK_END
'''


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def requested_mode(environment=None):
    mode = (os.environ if environment is None else environment).get(MODE_ENV, "off")
    _require(mode in ("off", "correctness", "timing"), "unregistered H100 two-cycle mode")
    return mode


def inspect_helper_source(p0):
    """CPU/source inspection; accepts only original worker plus the exact hook."""
    path = Path(p0.__file__).resolve()
    text = path.read_text()
    _require(text.count(HOOK) in (0, 1), "duplicate H100 C2 hook")
    _require(hashlib.sha256(text.replace(HOOK, "").encode()).hexdigest() == WORKER_SHA,
             "H100 C2 requires the exact 8fc worker plus only the registered hook")
    for name, expected in HELPERS.items():
        function = getattr(p0, name)
        _require(Path(inspect.getsourcefile(function)).resolve() == path and
                 hashlib.sha256(inspect.getsource(function).encode()).hexdigest() == expected,
                 f"native logical-B1 helper changed: {name}")
    root = path.parents[2]
    pins = {AUDITOR: AUDITOR_SHA,
        "src/distillschedule/runtime/hardware_profile.py": "9d71603c3171ddc6a0277435d3ed5e4290240508755d876fc62b702f4371235d",
        "src/distillschedule/runtime/fsdp2_p0.py": "e706090caddb0e7e337cc7f9e57485785832563cba6de049f0f2d9201649831c",
        "src/distillschedule/runtime/fsdp2_static.py": "caed2c8f0d5109372e8631d0aeff7671d7a96076d4346f71b16d83055ac857bd",
        "scripts/distillschedule/probe_static_fsdp2_p0_world16_hardened.py": "65f7f85ccd12b6b6a779af2d7acae5e19a3641e760f79cf61a022afed2adc0c7",
        "runtime/static_world16_trace.py": "4f99c2b20466f06fd2d34ef48276e2c8e4a6ba5e0d47b10ffc1c0bfc6cc18a04",
        "runtime/gst_pool_helios.py": "01ed9704226b074b32e131111d08d52dcdf61c90247ca6902b63211215374efb",
        "configs/distillschedule/best_known/helios14b_nogan_best.json": "00716145197fa0ff9b0fce2557b732ef88d66864a1068d463bbf2d07f2c5783c",
        "configs/distillschedule/experiments/helios_h100_nograd_retain_v1.json": "e2d7aa8243c3d00afce3def178b43b00b281c6846e12bcdc43777630b57bde15",
        "configs/distillschedule/experiments/helios_h100_nograd_retain_nosync_v1.json": "ee8206d157ab9c3ffd8ea527cb1c1c4a34b4f94964121f8bdf3bae8f266b9f50"}
    for relative, expected in pins.items():
        _require(hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, f"native dependency changed: {relative}")
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True).strip()
    _require(not dirty, "native source must be clean; installing this hook needs a newly registered revision")
    return {"parent_native_head": PARENT_HEAD, "actual_source_head": head,
            "worker_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "helper_sha256": dict(HELPERS), "dependency_sha256": pins, "hook_present": HOOK in text,
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _load_native(p0):
    path = Path(p0.__file__).resolve().parents[2] / AUDITOR
    spec = importlib.util.spec_from_file_location("h100_c2_original_cpu_auditor", path)
    _require(spec is not None and spec.loader is not None, "native auditor unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _versions(generator, critic, ema):
    values = dict(generator=generator.global_steps, critic=critic.global_steps, ema=ema.optimization_step)
    _require(all(type(v) is int and v >= 0 for v in values.values()), "actual engine versions must be integers")
    return values


def _objects(generator, critic, ema):
    return {"generator": [id(generator), id(generator.module), id(generator.optimizer), id(generator.lr_scheduler)],
            "critic": [id(critic), id(critic.module), id(critic.optimizer), id(critic.lr_scheduler)], "ema": id(ema)}


@contextmanager
def _census(generator, critic, ema):
    """Count actual native method calls, not an inferred expected inventory."""
    counts = {}
    with ExitStack() as stack:
        for role, engine in (("generator", generator), ("critic", critic)):
            for suffix, obj, name in (("backward", engine, "backward"), ("commit", engine, "step"),
                                      ("optimizer", engine.optimizer, "step"), ("scheduler", engine.lr_scheduler, "step")):
                key = role + "_" + suffix
                original, owned = getattr(obj, name), name in vars(obj)
                counts[key] = 0
                def counted(*args, _key=key, _original=original, **kwargs):
                    counts[_key] += 1
                    return _original(*args, **kwargs)
                setattr(obj, name, counted)
                stack.callback(setattr if owned else delattr, obj, name, *([original] if owned else []))
        original, owned = ema.step, "step" in vars(ema)
        counts["ema"] = 0
        def counted_ema(*args, **kwargs):
            counts["ema"] += 1
            return original(*args, **kwargs)
        ema.step = counted_ema
        stack.callback(setattr if owned else delattr, ema, "step", *([original] if owned else []))
        yield counts


def _validate_setup(*, p0, args, policy, result, rank, logical, trace, generator, critic, ema):
    mode = requested_mode()
    _require(mode != "off", "H100 two-cycle adapter is default-off")
    _require(args.variant == "no-gan" and args.parameter_placement == "fsdp2" and args.ema_device == "cuda"
             and args.profile_mode is False and args.operation_source is False and args.critic_only_step is None
             and args.timing_mode is (mode == "timing") and getattr(args, "_official_role_swap", None) is None,
             "H100 C2 only supports unprofiled No-GAN native FSDP2 with CUDA EMA")
    _require(type(rank) is int and rank in range(8) and result["rank"] == rank and result["world_size"] == 8,
             "H100 C2 requires actual eight-rank ownership")
    hardware = p0.hardware_profile("h100-8-b16")
    slots = list(hardware.logical_slots(rank))
    _require(result["hardware_profile"] == {"name": "h100-8-b16", "physical_world_size": 8,
        "logical_global_batch": 16, "logical_b1_microbatches_per_rank": 2, "logical_slots": slots,
        "loss_scale_per_logical_b1": .5, "rank_mapping": "slot = physical_rank + microbatch * physical_world_size"},
        "native hardware/independent B1 slot mapping changed")
    source = inspect_helper_source(p0)
    native = _load_native(p0)
    selected = [arm for arm, sha in POLICIES.items() if _digest(result["p0_policy"]) == sha]
    _require(len(selected) == 1, "unsupported H100 manual policy; no automatic FA3/recompute substitution")
    arm = selected[0]
    frozen = native.freeze_policy(Path(p0.__file__).resolve().parents[2], variant="no-gan", arm=arm)
    _require(frozen["policy"] == result["p0_policy"] and
             frozen["contract"] == json.loads(json.dumps(result["contract"])), "native frozen contract changed")
    for name, value in frozen["policy"].items():
        actual = getattr(args if name in ARG_POLICY_FIELDS else policy, name)
        _require(type(actual) is type(value) and actual == value, f"effective native policy changed: {name}")
    _require(trace.bundle_hash == frozen["trace_hash"] and logical.manifest_hash == frozen["logical_hash"]
             and tuple(trace.logical_steps) == STEPS, "immutable trace/checkpoint changed")
    _require(_versions(generator, critic, ema) == INITIAL and
             all(logical.component(k)["optimizer_step"] == INITIAL[k] for k in ("generator", "critic")),
             "C2 must start from the actual registered G2/C10/EMA10 checkpoint")
    _require(generator.module is not critic.module, "Generator and Score cannot alias")
    backend = result["attention_backend"]
    _require(backend["status"] == "PASS" and backend["requested"] == backend["runtime"] == "local-fa2"
             and backend["extension_sha256"] == native.FLASH_SHA, "actual local-FA2 receipt required; no FA3 claim")
    controls = native.trace_controls(args.trace_bundle, "no-gan", frozen["trace_hash"])
    _require(set(controls) == set(range(16)), "all sixteen source slots required")
    _require(trace.receipt()["critic_entries_still_cached"] == 0, "initial hardened trace cache is not empty")
    return mode, hardware, slots, source, native, frozen, controls


def _audit_cycle(row, *, rank, slots, native, frozen, controls):
    # The original auditor consumes JSON receipts (stage-count object keys are
    # strings). Normalize only after the native timer, never change model data.
    row = json.loads(json.dumps(row, allow_nan=False))
    _require(type(row["wall_seconds"]) in (float, int) and math.isfinite(row["wall_seconds"])
             and row["wall_seconds"] > 0 and row["stop_unix_ns"] > row["start_unix_ns"], "missing actual cycle timer")
    before, after, errors = row["initial_versions"], row["final_versions"], []
    expected = {key: before[key] + (1 if key == "generator" else 5) for key in INITIAL}
    _require(after == expected, "actual per-cycle G1/C5/EMA5 versions changed")
    _require(row["census"] == {"generator_backward": 2, "generator_commit": 1, "generator_optimizer": 1,
        "generator_scheduler": 1, "critic_backward": 10, "critic_commit": 5, "critic_optimizer": 5,
        "critic_scheduler": 5, "ema": 5}, "actual native backward/optimizer/scheduler/EMA census changed")
    for role, aggregates in (("generator", [row["generator"]]), ("critic", row["critic"])):
        _require(len(aggregates) == (1 if role == "generator" else 5), "missing native update records")
        score_key = "physical_score_forwards" if role == "generator" else "score_forwards"
        for offset, aggregate in enumerate(aggregates):
            _require(aggregate["source_ranks"] == slots and aggregate["logical_b1_microbatch_count"] == 2
                     and aggregate[score_key] == (6 if role == "generator" else 2), "native two-B1 aggregate changed")
            children = aggregate["microbatches"]
            _require(len(children) == 2, "missing native logical B1 child")
            for index, (child, slot) in enumerate(zip(children, slots, strict=True)):
                step = STEPS[0] if role == "generator" else STEPS[offset]
                _require(child["source_rank"] == child["logical_slot"] == slot and child["physical_rank"] == rank
                         and child["logical_step"] == step and child["optimizer_committed"] is bool(index)
                         and child[f"physical_{role}_step_after"] == before[role] + offset + index
                         and child[score_key] == (3 if role == "generator" else 1), "native child identity/outer commit changed")
                if role == "generator":
                    _require(child["logical_score_requests"] == 3 and child["real_score_pair_packed"] is False, "Score packing changed")
                    native._check_rollout(child["rollout"], controls[slot], variant="no-gan", slot=slot,
                                          step=step, role="generator", label=f"G/{slot}", errors=errors)
    batches = row["rollout_batches"]
    _require([(b["logical_slot"], b["logical_steps"], b["physical_rank"]) for b in batches] ==
             [(slot, [step], rank) for slot in slots for step in STEPS], "native rollout slot/step order changed")
    for batch in batches:
        native._check_rollout(batch, controls[batch["logical_slot"]], variant="no-gan", slot=batch["logical_slot"],
                              step=batch["logical_steps"][0], role="critic", label="C rollout", errors=errors)
    native._audit_persistence(row, slots=slots, policy=frozen["policy"], label="cycle", errors=errors)
    _require(not errors, "; ".join(errors))
    _require([x["logical_step"] for x in row["ema"]] == list(STEPS) and
             [x["optimization_step"] for x in row["ema"]] == list(range(before["ema"] + 1, after["ema"] + 1))
             and all(type(x["decay"]) in (float, int) and math.isfinite(x["decay"]) for x in row["ema"]), "EMA recurrence changed")
    nosync = frozen["policy"]["elide_explicit_rollout_sync"]
    expected_fence = {"generator_update_empty_cache_count": 0 if nosync else 2,
                      "generator_update_sync_count": 0 if nosync else 2,
                      "critic_rollout_sync_count": 0 if nosync else 10}
    _require(row["rollout_fences"] == expected_fence, "actual rollout fences differ from selected arm")
    _require(row["trace_cache"]["critic_entries_still_cached"] == 0, "cycle left unconsumed trace inputs")


def run_two_cycles(*, p0, args, policy, result, rank, device, logical, trace, negative, cfg,
                   generator_engine, critic_engine, ema, diagnostics, generator_free_memory):
    """Called once AFTER native model/optimizer/EMA construction, before Full5.

    All added source/asset parsing, receipt audits, state probes and serialization
    are outside both native timers. Required lightweight method counters and the
    original helpers' existing mode-specific diagnostics remain inside. No
    initialization, load, reseed, or training-state reset occurs in this adapter.
    """
    mode, hardware, slots, source, native, frozen, controls = _validate_setup(p0=p0, args=args, policy=policy,
        result=result, rank=rank, logical=logical, trace=trace, generator=generator_engine, critic=critic_engine, ema=ema)
    identities = _objects(generator_engine, critic_engine, ema)
    result.update(schema_version=SCHEMA, mode=mode, source_contract=source, arm=frozen["arm"],
        native_full5_contract=json.loads(json.dumps(result["contract"])),
        contract={"continuous_cycles": 2, "physical_world_size": 8, "logical_global_batch": 16,
                  "physical_microbatch": 1, "loss_scale_per_logical_b1": .5,
                  "source_steps_per_cycle": list(STEPS), "initial_versions": dict(INITIAL),
                  "final_versions": {"generator": 4, "critic": 20, "ema": 20},
                  "updates": {"generator": 2, "critic": 10, "ema": 10}},
        same_process=True, initializations=1, initial_versions=dict(INITIAL), initial_physical_versions=dict(INITIAL),
        version_mapping={"logical_minus_physical": {key: 0 for key in INITIAL}}, cycles=[], measured_cycle=2,
        source_cohort_policy="fixed_source_cohort_replay", source_steps=list(STEPS),
        timer_scope="NATIVE_CYCLE_WITH_REQUIRED_RECEIPT_COUNTERS_OUTSIDE_SETUP_AND_AUDITS",
        numerical_status="REVIEW_NOT_FULL_STATE_PARITY", claimable_speedup=False,
        launcher_auditor_gate="NOT_IMPLEMENTED_NEW_C2_SOURCE_ADMISSION_REQUIRED")
    previous = dict(INITIAL)
    for cycle in (1, 2):
        _require(_objects(generator_engine, critic_engine, ema) == identities, "engine/optimizer/EMA object replaced")
        _require(_versions(generator_engine, critic_engine, ema) == previous, "C1 to C2 state continuation changed before model work")
        row = {"cycle": cycle, "measured": cycle == 2, "status": "RUNNING", "logical_slots": slots,
               "initial_versions": _versions(generator_engine, critic_engine, ema), "object_identity": identities,
               "critic": [], "ema": [], "source_identity": dict(result["identities"])}
        row["initial_physical_versions"] = dict(row["initial_versions"])
        result["cycles"].append(row)
        common = dict(rank=rank, hardware=hardware, device=device, trace=trace,
                      collect_diagnostics=(not args.timing_mode) and args.profile_diagnostics, diagnostics=diagnostics)
        with _census(generator_engine, critic_engine, ema) as counters:
            p0.dist.barrier()
            p0.torch.cuda.synchronize(device)
            row["start_unix_ns"] = time.time_ns()
            started = time.perf_counter()
            row["generator"] = p0._run_logical_generator_update_p0(args=args, **common,
                generator_engine=generator_engine, critic_engine=critic_engine, cfg=cfg, negative=negative,
                policy=policy, score_pack_parity_check=not args.timing_mode, generator_free_memory=generator_free_memory)
            outputs, batches, persistence = p0._precompute_logical_critic_rollouts(**common, steps=STEPS,
                generator_engine=generator_engine, policy=policy, rollout_parity_check=not args.timing_mode)
            for step in STEPS:
                row["critic"].append(p0._run_logical_critic_update_p0(args=args, **common, step=step,
                    rollout_outputs=outputs, critic_engine=critic_engine, cfg=cfg, policy=policy))
                decay = ema.step()
                diagnostics.add_bytes("ema_shadow", int(ema.last_transfer_bytes), direction="GPU", source="fsdp2_ema_local_shard")
                row["ema"].append({"logical_step": step, "decay": decay, "optimization_step": int(ema.optimization_step)})
            # Preserve the incumbent's within-cycle lifetime; never carry stale
            # C1 generated tensors into C2. Do not reload or clear model state.
            outputs.clear()
            p0.torch.cuda.synchronize(device)
            row["wall_seconds"] = time.perf_counter() - started
            row["stop_unix_ns"] = time.time_ns()
        row.update(final_versions=_versions(generator_engine, critic_engine, ema), census=dict(counters),
            rollout_batches=batches, generator_nograd_persistent_unshard=persistence, trace_cache=trace.receipt(),
            outer_boundary_fences={"pre_barrier": 1, "pre_cuda_sync": 1, "post_cuda_sync": 1})
        row["final_physical_versions"] = dict(row["final_versions"])
        children = row["generator"]["microbatches"]
        for rollout in [c["rollout"] for c in children] + batches:
            _require(type(rollout["synchronize_after_rollout"]) is bool, "missing actual rollout sync receipt")
        _require(all(type(c["rollout"]["empty_cache_before_rollout"]) is bool for c in children), "missing actual allocator fence receipt")
        row["rollout_fences"] = {"generator_update_empty_cache_count": sum(c["rollout"]["empty_cache_before_rollout"] for c in children),
            "generator_update_sync_count": sum(c["rollout"]["synchronize_after_rollout"] for c in children),
            "critic_rollout_sync_count": sum(b["synchronize_after_rollout"] for b in batches)}
        _require(_objects(generator_engine, critic_engine, ema) == identities, "cycle replaced initialized state")
        _audit_cycle(row, rank=rank, slots=slots, native=native, frozen=frozen, controls=controls)
        if mode == "correctness":
            row["ema_checksum_outside_timer"] = ema.checksum()
        previous = dict(row["final_versions"])
        row["status"] = "PASS_NATIVE_PROTOCOL"
    final = _versions(generator_engine, critic_engine, ema)
    _require(final == {"generator": 4, "critic": 20, "ema": 20}, "continuous two-cycle final versions changed")
    result.update(status="PASS", final_versions=final, final_physical_versions=dict(final),
        updates={k: final[k] - INITIAL[k] for k in INITIAL},
        cycle_wall_seconds={str(r["cycle"]): r["wall_seconds"] for r in result["cycles"]},
        final={"generator_physical_step": final["generator"], "critic_physical_step": final["critic"],
               "generator_logical_version": final["generator"], "critic_logical_version": final["critic"], "ema_step": final["ema"]})
    return result
