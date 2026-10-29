"""Opt-in actual native W6 finite continuation, not the CPU reference trainer.

H3ARSession constructs the original DMD2Trainer/H3Adapter/AlternatingAdamW and
ZeRO-3 engine once. This owner supplies explicit inputs and complete-cycle state;
it never implements losses, optimizers, a sampler or a replacement backend.
"""
from copy import deepcopy
import json
import math
import os
import importlib.metadata
from pathlib import Path
import time

from distillschedule.core.contracts import require
from distillschedule.strategies.h3_ar_assets import H3ARAssets, checked
from distillschedule.strategies.h3_ar_task import H3ARTask, H3ARImplementation, native_config
from distillschedule.strategies.h3_ar_planning import validate_policy, policy_for_node
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint, repository_root, write_new_json
from . import distributed
from .h3_ar_inputs import KIND, EPOCH_KIND, EPOCH_POLICY, H3ARInputStream, input_metadata, record, source_identity
from .h3_ar_state import NativeCheckpoint, Progress

SCHEMA = "distillschedule.h3_ar_continuation.v1"
POLICY = "distillschedule.h3_ar_native_boundary.v1"
TOPOLOGY = dict(world_size=16, backend="nccl", node_groups=[list(range(8)), list(range(8, 16))])
DEEPSPEED_ENGINE_SHA256 = "ef3b4797809453f4941eaa769c9cd4e397e3b52085131e8916990ea78d4f0f6c"


def descriptor(input_kind=KIND):
    result = dict(schema_version=SCHEMA, owner="ORIGINAL_H3ARSession_DMD2Trainer_AlternatingAdamW_ZeRO3",
        input_kind=KIND, task=H3ARTask().to_dict(), topology=deepcopy(TOPOLOGY), native_config=native_config(),
        horizon=dict(min=2, max=100), updates=dict(student=1, fake=1, ema=0),
        checkpoint="actual_engine_save_checkpoint_load_checkpoint_same_topology",
        transient="attention replay/FIFO/prefetch/results reconstructed cold; pinned buffers process-local",
        numerical_status="REVIEW", gpu_continuation="NOT_RUN", decoded_AV="NOT_IMPLEMENTED", claimable_speedup=False)
    require(input_kind in (KIND, EPOCH_KIND), "unsupported W6 descriptor input kind", path="h3_ar.training")
    if input_kind == EPOCH_KIND:
        result.update(input_kind=EPOCH_KIND, input_policy=EPOCH_POLICY)
    return result


def _source_layout():
    from distillschedule.evidence.source_layout import reviewed_imports
    root = repository_root()
    for row in reviewed_imports(root):
        require(file_sha256(root/row["target"]) == row["integrated_sha256"],
                "reviewed source-layout identity changed: "+row["target"], path="h3_ar.training")
    return file_sha256(root/"docs/workloads/source_layout.lock.json")


def admit_training(config, task_value, assets_value, paths):
    from .native_presets import H3_AR_STREAM, preset_implementation
    task = H3ARTask.from_dict(task_value)
    assets = H3ARAssets.from_dict(assets_value)
    source_layout = _source_layout()
    assets_validation = assets.validate()
    inputs = input_metadata(config.data["prepared"], cycles=config.trainer["cycles"], seed=config.trainer["seed"], verify_files=True)
    require(config.data["input_kind"] == inputs["input_kind"],
            "W6 declared input kind differs from actual prepared policy", path="h3_ar.training")
    reference = dict(path=str(Path(config.data["prepared"]).resolve()), sha256=file_sha256(Path(config.data["prepared"])))
    require(assets.cache_manifest == reference, "assets.cache_manifest must bind the actual fresh prepared stream, not a private historical cache",
            path="h3_ar.training")
    policy = json.loads(paths["capacity_policy"].read_text())
    validate_policy(policy)
    if config.execution.get("mode") == "reference":
        require(config.execution.get("implementation") == "zero3-native" and "preset" not in config.execution,
                "fresh W6 reference requires explicit zero3-native; never a fallback", path="h3_ar.training")
        impl = H3ARImplementation("zero3-native")
    else:
        impl = preset_implementation("W6", H3_AR_STREAM)
    return dict(task=task.to_dict(), assets=assets.to_dict(), selected=impl.to_dict(), inputs=inputs,
        capacity_policy=policy, source_layout_sha256=source_layout, asset_admission=assets_validation,
        lifecycle=descriptor(inputs["input_kind"]), native_worker="distillschedule.training.h3_ar_training:run_training",
        validation_scope="SOURCE_LAYOUT_ASSETS_FINITE_INPUT_BYTES_AND_CAPACITY_SCHEMA_ONLY; NATIVE_CUDA_CONTINUATION_NOT_RUN")


def source_files():
    root = Path(__file__).parents[1]
    paths = set(root.glob("strategies/h3_ar_*.py")) | set(root.glob("training/h3_ar_*.py"))
    paths |= {root/"training/checkpoint.py", root/"training/distributed.py", root/"training/native.py", root/"training/native_presets.py"}
    return {str(p.relative_to(root)): file_sha256(p) for p in sorted(paths)}


def training_identity(head, admitted, config, *, runtime=None):
    return dict(schema_version=SCHEMA, source_head=head, source_files=source_files(),
        source_layout_sha256=admitted["source_layout_sha256"], assets_sha256=fingerprint(admitted["assets"]),
        input_sha256=admitted["inputs"]["input_sha256"], horizon=config.trainer["cycles"], seed=config.trainer["seed"],
        checkpoint_every=config.trainer["checkpoint_every"], implementation=admitted["selected"],
        topology=deepcopy(TOPOLOGY), capacity_policy=deepcopy(admitted["capacity_policy"]), runtime=runtime, contract=descriptor(admitted["inputs"]["input_kind"]))


def checkpoint_policy(identity):
    return dict(schema_version=POLICY, identity_sha256=fingerprint(identity), boundary="drained_complete_G1_F1_EMA0",
        persistent="both LoRAs/inactive discriminator/native FP32 master Adam/native engine and trainer clocks/rank RNG/fresh cursor",
        frozen="33B base reconstructed from exact admitted assets; exclude_frozen_parameters=True",
        scheduler="NONE_original_fixed_AlternatingAdamW_learning_rates", transient="RECONSTRUCT_COLD")


def validate_restart(header, identity, *, output_only=False):
    cycle = header.get("cycle")
    require(type(output_only) is bool, "W6 output-only control must be explicit bool", path="h3_ar.training")
    require(type(cycle) is int and 0 < cycle <= identity["horizon"]
            and (output_only or cycle < identity["horizon"]) and header.get("cursor") == cycle
            and header.get("versions") == dict(student=cycle, fake=cycle, ema=0)
            and all(type(v) is int for v in header["versions"].values())
            and header.get("policy") == checkpoint_policy(identity) and "native_checkpoint" in header,
            "W6 restart must be a complete nonterminal native cycle inside the identical immutable horizon", path="h3_ar.training")


def validate_joint(rows, *, cycle, inputs):
    require(len(rows) == 16 and [r.get("rank") for r in rows] == list(range(16)),
            "all sixteen native W6 rank boundaries required", path="h3_ar.training")
    for rank, row in enumerate(rows):
        expected = record(inputs, cycle, rank)
        require(row.get("cycle") == cycle and row.get("versions") == dict(student=cycle, fake=cycle, ema=0)
                and row.get("native_global_steps") == 2*cycle and row.get("input_identity") == source_identity(expected),
                "W6 rank fresh source/cursor/native clock mismatch", path="h3_ar.training")


def execute_segment(session, *, stop, output, identity, capacity_check, clock=time.perf_counter_ns, native_output=None):
    """Dispatch unchanged native updates; only process-local elapsed work is timed."""
    from .checkpoint import save_checkpoint
    torch = session.torch
    cursor = session.input_provider
    native = distributed.checked(lambda: NativeCheckpoint(session), "W6 native parameter/ordered optimizer layout")
    native.bind_writers()
    state = {"native_progress": Progress(session, cursor, native)}
    if getattr(session, "resume_from", None):
        from .checkpoint import restore_checkpoint
        restore_checkpoint(session.resume_from, state, identity=identity, native_checkpoint=native)
    if native_output is not None:
        require(bool(getattr(session,"resume_from",None)), "W6 output requires actual complete restoration", path="h3_ar.training")
        distributed.checked(native.validate_checkpoint_boundary,"W6 restored output-only native boundary")
        distributed.checked(capacity_check,"W6 restored output-only capacity")
        native_output.publish(session,native,identity=identity,capacity_check=capacity_check)
        return []
    records = []
    require(cursor.completed < stop <= cursor.cycles, "W6 stop must be after the restored boundary", path="h3_ar.training")
    for cycle in range(cursor.completed+1, stop+1):
        prepared = distributed.checked(lambda: session.prepare(cycle), "W6 original fresh native update preparation")
        distributed.gather(True)
        torch.cuda.synchronize(session.device)
        start = clock()
        prepared.run()
        torch.cuda.synchronize(session.device)
        elapsed = clock()-start
        checks = distributed.checked(lambda: session.finish(cycle), "W6 original native census/cache/Adam/clock checks")
        def finite():
            require(all(type(v) not in (float, int) or math.isfinite(v) for v in checks["metrics"].values()),
                    "W6 nonfinite actual native metric", path="h3_ar.training")
        distributed.checked(finite, "W6 finite native metrics")
        memory = distributed.checked(capacity_check, "W6 complete-cycle capacity/reserve/OOM")
        joint = distributed.gather(dict(rank=session.rank, cycle=cycle, versions=session.versions(),
            native_global_steps=session.engine.global_steps, input_identity=session.input_identity))
        distributed.checked(lambda: validate_joint(joint, cycle=cycle, inputs=cursor.document), "W6 joint complete fresh rank cohort")
        row = dict(cycle=cycle, segment_origin=session.segment_origin, process_local_cycle=session.local_completed,
            duration_ns=elapsed, timing_scope="NATIVE_UPDATE_ONLY_NOT_A_MATCHED_PERFORMANCE_CLAIM",
            input_identity=session.input_identity, prepared=prepared.receipt(), memory=memory, **checks)
        save = bool(identity["checkpoint_every"]) and (cycle % identity["checkpoint_every"] == 0 or cycle == cursor.cycles)
        if save or cycle == stop:
            distributed.checked(session.drain_for_checkpoint, "W6 native transient publication drain")
            distributed.checked(native.validate_checkpoint_boundary, "W6 actual native drained complete boundary")
            row["state"] = distributed.checked(lambda: session.state_metadata(cycle), "W6 native FP32 master/Adam/RNG evidence")
        if save:
            destination = Path(output)/"checkpoints"/f"cycle-{cycle:06d}"
            save_checkpoint(destination, state, identity=identity, cycle=cycle, cursor=cycle, versions=session.versions(),
                policy=checkpoint_policy(identity), native_checkpoint=native, pre_publication_guard=capacity_check)
            row["checkpoint"] = str(destination)
        records.append(row)
        write_new_json(Path(output)/"cycles"/f"rank-{session.rank:05d}-cycle-{cycle:06d}.json", row)
    return records


def run_training(config, *, admit, native_output=None):
    """Enter on existing torchrun ranks; no launcher, GPU reservation or download."""
    from .native import _options
    from distillschedule.strategies import h3_ar_worker as worker
    _options(config, "W6")
    implementation = H3ARImplementation("zero3-native" if config.execution["mode"] == "reference" else "zero3-ap")
    allocator = ("expandable_segments:True,pinned_use_cuda_host_register:True,pinned_num_register_threads:4"
                 if implementation.optimized else "expandable_segments:True")
    require(os.environ.get("DIFFSYNTH_ATTENTION_IMPLEMENTATION") == "torch"
            and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == allocator
            and os.environ.get("PYTORCH_ALLOC_CONF", allocator) == allocator,
            "explicit original native attention/allocator environment required before CUDA initialization", path="h3_ar.training")
    with distributed.world16_reference_process_group("W6 native continuation"):
        import torch
        distributed.admit_run_controls(config)
        _, source, paths, admitted = distributed.checked(admit, "W6 source/layout/assets/input/capacity admission before model imports")
        require(not source["identity"]["dirty"], "native W6 execution requires committed source", path="h3_ar.training")
        rank, local = worker.topology(os.environ)
        policy = admitted["capacity_policy"]
        assets = H3ARAssets.from_dict(admitted["assets"])
        host_entry = distributed.checked(lambda: worker.host_admission(policy, node_rank=rank//8), "W6 original host startup reserve")
        assets.install_import_paths()
        startup_path = checked(assets.startup_gpu_info[str(rank//8)])
        startup = json.loads(startup_path.read_text())
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == ",".join(startup["ordered_gpu_uuids"]),
                "W6 visible devices differ from explicit fresh startup GPU identities", path="h3_ar.training")
        from accelerate import Accelerator, DeepSpeedPlugin
        from accelerate.utils import DistributedType
        import accelerate.utils.environment as accelerate_environment
        assets.helper("native_accelerate_gpu_info_cache_r1").install(accelerate_environment, startup_path)
        runtime = distributed.checked(lambda: worker.runtime_identity(torch), "W6 validated native software/precision identity")
        def engine_source():
            path = Path(importlib.metadata.distribution("deepspeed").locate_file("deepspeed/runtime/engine.py"))
            require(runtime["packages"]["deepspeed"] == "0.18.4" and file_sha256(path) == DEEPSPEED_ENGINE_SHA256,
                    "W6 native checkpoint API requires the exact admitted DeepSpeed0.18.4 engine source", path="h3_ar.training")
            return dict(path=str(path), sha256=DEEPSPEED_ENGINE_SHA256)
        runtime["deepspeed_engine"] = distributed.checked(engine_source, "W6 native engine checkpoint API source admission")
        identity = training_identity(source["identity"]["head"], admitted, config, runtime=runtime)
        require(all(row == identity for row in distributed.gather(identity)), "W6 rank source/input/runtime identities differ", path="h3_ar.training")
        output_only = native_output is not None
        require(all(row==output_only for row in distributed.gather(output_only)), "W6 ranks disagree on output-only mode", path="h3_ar.training")
        if output_only:
            require(bool(config.trainer.get("resume_from")), "W6 output requires a complete native checkpoint", path="h3_ar.training")
            native_output.admit(policy)
        stop = config.trainer.get("stop_after_cycle", config.trainer["cycles"])
        require(all(row == stop for row in distributed.gather(stop)), "W6 rank stop boundaries differ", path="h3_ar.training")
        from .livewan_training import _reserve_output
        output = Path(paths["output"])
        _reserve_output(output)
        def capacity():
            free, _ = torch.cuda.mem_get_info(local)
            prop = torch.cuda.get_device_properties(local)
            resolved = policy_for_node(policy, rank//8)
            require(resolved["gpu_limit_bytes"]+resolved["gpu_reserve_bytes"] <= prop.total_memory,
                    "W6 device cannot admit declared capacity/reserve", path="h3_ar.training")
            host = worker.observe_host_memory(policy, node_rank=rank//8)
            require(worker.host_capacity_signature(host) == host_entry["host_capacity"], "W6 host capacity changed", path="h3_ar.training")
            memory = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(local),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(local), driver_free_bytes=free, **host,
                oom_event_delta={k: host["cgroup_events"][k]-v for k, v in host_entry["initial_events"].items()})
            require(worker.capacity_ok(memory, policy, node_rank=rank//8), "W6 actual GPU/host/reserve/OOM gate failed", path="h3_ar.training")
            return memory
        distributed.checked(capacity, "W6 pre-model resource admission")
        header = None
        if config.trainer.get("resume_from"):
            from .checkpoint import read_checkpoint, read_manifest
            header = distributed.checked(lambda: read_manifest(config.trainer["resume_from"], identity=identity), "W6 all-rank restart metadata")
            distributed.checked(lambda: validate_restart(header, identity,output_only=output_only), "W6 fixed-horizon restart boundary")
            distributed.checked(lambda: read_checkpoint(config.trainer["resume_from"], identity=identity), "W6 every native/common shard hash before model allocation")
        completed = header["cycle"] if header else 0
        if output_only:
            from .h3_ar_export import OutputInputStream
            cursor_type = OutputInputStream
        else:
            cursor_type = H3ARInputStream
        cursor = distributed.checked(lambda:cursor_type(config.data["prepared"], rank=rank,
            cycles=config.trainer["cycles"], seed=config.trainer["seed"], completed=completed),"W6 native input cursor construction")
        try:
            native_source = distributed.checked(lambda: assets.source_module("h3_dmd2"), "W6 admitted original deterministic parser import")
            distributed.checked(lambda: cursor.stage_initial(H3ARTask.from_dict(admitted["task"]), assets.cache_manifest,
                native_source, rank, torch), "W6 all-rank next original parser result before model allocation")
            distributed.checked(capacity, "W6 prepared AV input staging capacity")
            plugin = DeepSpeedPlugin(gradient_accumulation_steps=1, zero_stage=3, offload_optimizer_device="none",
                offload_param_device="none", zero3_init_flag=True, zero3_save_16bit_model=False)
            accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=1, deepspeed_plugin=plugin)
            require(accelerator.distributed_type == DistributedType.DEEPSPEED, "original ZeRO3 plugin is not active", path="h3_ar.training")
            assets.helper("native_ptxas_fd_isolation_r1").install(enabled=True)
            from distillschedule.strategies.h3_ar_session import H3ARSession
            session = distributed.checked(lambda: H3ARSession(H3ARTask.from_dict(admitted["task"]), assets, implementation,
                rank=rank, local_rank=local, torch=torch, accelerator=accelerator, input_provider=cursor,
                seed=config.trainer["seed"]), "W6 original persistent native session construction")
            session.resume_from = config.trainer.get("resume_from")
            distributed.checked(capacity, "W6 constructed native model capacity")
            records = execute_segment(session, stop=stop, output=output, identity=identity, capacity_check=capacity,native_output=native_output)
            def final_audit():
                assets.audit_imports()
                require(source_files() == identity["source_files"] and _source_layout() == identity["source_layout_sha256"]
                    and worker.runtime_identity(torch) == {k: v for k, v in runtime.items() if k != "deepspeed_engine"}
                    and engine_source() == runtime["deepspeed_engine"], "W6 source/runtime changed during native segment", path="h3_ar.training")
            distributed.checked(final_audit,"W6 final source/runtime audit")
            distributed.checked(capacity, "W6 final actual native resource gate")
            distributed.checked(lambda:write_new_json(output/"continuation"/f"rank-{rank:05d}.json", dict(schema_version=SCHEMA,
                status="EXPORTED" if output_only else "COMPLETED" if stop == cursor.cycles else "PARTIAL", identity=identity,
                completed_cycle=cursor.completed, segment_origin=session.segment_origin, segment_cycles=len(records),
                native_global_steps=session.engine.global_steps, versions=session.versions(), build=session.build,
                native_GPU_continuation="OUTPUT_ONLY_NO_UPDATES" if output_only else "EXECUTED_NOT_NUMERICALLY_QUALIFIED",
                decoded_AV="SEPARATE_DISTILLCTL_DECODE" if output_only else "NOT_IMPLEMENTED", claimable_speedup=False,
                **(dict(inference_export=native_output.result) if output_only else {}))),"W6 final rank receipt")
            return 0
        except BaseException as exc:
            if output_only:
                native_output.invalidate()
            write_new_json(output/"continuation"/f"rank-{rank:05d}.failed.json", dict(schema_version=SCHEMA, status="FAIL",
                identity=identity, error=type(exc).__name__+": "+str(exc), last_locally_completed_cycle=cursor.completed,
                native_GPU_continuation="UNQUALIFIED_PARTIAL_FAILURE", claimable_speedup=False))
            raise
