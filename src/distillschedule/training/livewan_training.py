"""Finite ORIGINAL W4 reference lifecycle, explicitly separate from C51.

The existing baseline owns all math and actual commits. This module constructs
its owners once and calls its one-cycle helper unchanged. It is not a solver,
pipeline-drain implementation, benchmark writer, or numerical qualification.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import importlib.util
import math
import os
from pathlib import Path
import random
import sys
import uuid

from distillschedule.data.livewan_inputs import KIND, SCHEMA as INPUT_SCHEMA, CONTRACT, _rows
from distillschedule.search.livewan_request import LiveWanRequest
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint, SHA256_RE
from . import distributed
from .livewan_reference_assets import (
    ReferenceAssets, asset_config, admit_runtime, admit_native_layouts, document, environment_snapshot, _environment, require,
)

SCHEMA = "distillschedule.livewan_reference_training.v1"
POLICY = "distillschedule.livewan_reference_checkpoint.v1"
CAPACITY = "distillschedule.livewan_reference_capacity.v1"
TOPOLOGY = {"world_size": 16, "backend": "nccl", "node_groups": [list(range(8)), list(range(8, 16))]}
REFERENCE = {"schema_version": SCHEMA, "mode": "original-reference", "generator": "world16_fsdp2_keep_unsharded",
             "teacher": "world16_fsdp2_reshard", "critic": "replicated_fp32_lora32_bf16_base",
             "global_batch": 16, "frames": 81, "updates": {"generator": 1, "critic": 5, "ema": 0},
             "sigma": CONTRACT["sigma"], "optimizer": "original_foreach_AdamW", "optional_optimizations": "FORBIDDEN"}


def versions(cycle):
    return {"generator": cycle, "critic": 5 * cycle, "ema": 0}


def _input_metadata(path, *, cycles, seed, source):
    """No tensor/model imports here. Actual InputLoader repeats full admission."""
    path = Path(path).resolve()
    sha = file_sha256(path)
    value = document(path)
    unsigned = {k: v for k, v in value.items() if k != "input_sha256"}
    require(value.get("schema_version") == INPUT_SCHEMA and value.get("input_sha256") == fingerprint(unsigned)
            and value.get("status") == "ADMITTED_FINITE_NATIVE_INPUTS" and value.get("contract") == CONTRACT
            and value.get("cycles") == cycles and type(value["cycles"]) is int
            and value.get("seed") == seed and type(value["seed"]) is int,
            "fresh native input schema/hash/horizon/seed/shape changed")
    original = value.get("source", {})
    require(original.get("root") == source["root"] and original.get("head") == source["head"]
            and original.get("files") and all(source["files"].get(k) == v for k, v in original["files"].items()),
            "prepared inputs use another original source")
    records = value.get("records")
    require(type(records) is list and len(records) == 80 * cycles
            and all(type(r) is dict and r.get("dataset_index") == i for i, r in enumerate(records)),
            "finite original record inventory changed")
    for key in ("path", "sha256", "record_id", "latent_sha256"):
        require(len({r[key] for r in records}) == len(records), "replayed/duplicate native input inventory")
    require(value.get("rows") == _rows(records, cycles, seed), "native input operation/row/seed identity changed")
    require(path.is_file() and file_sha256(path) == sha, "prepared input changed during admission")
    return value


def capacity_policy(value):
    from distillschedule.search.resources import validate_capacity_policy
    require(type(value) is dict and set(value) == {"schema_version", "gpu_metric", "memory_limits", "host_limits",
            "gpu_reserve_bytes", "host_reserve_bytes"} and value["schema_version"] == CAPACITY,
            "explicit supported reference capacity policy required")
    validate_capacity_policy(value)
    return deepcopy(value)


def training_identity(head, assets, inputs, config, capacity):
    return {"schema_version": SCHEMA, "source_head": head, "assets_sha256": fingerprint(assets),
            "input_sha256": inputs["input_sha256"], "reference": deepcopy(REFERENCE),
            "horizon": config.trainer["cycles"], "seed": config.trainer["seed"],
            "checkpoint_every": config.trainer["checkpoint_every"], "topology": deepcopy(TOPOLOGY),
            "capacity_policy": deepcopy(capacity)}


def checkpoint_policy(identity):
    return {"schema_version": POLICY, "identity_sha256": fingerprint(identity), "boundary": "complete_G1_C5_EMA0",
            "scratch": "reconstructed_streamer_KV_reset_by_next_original_prepare_rollout"}


def validate_progress(header, *, identity, target):
    cycle = header.get("cycle")
    require(type(cycle) is int and 0 < cycle < target and type(header.get("cursor")) is int
            and header["cursor"] == 5 * cycle and header.get("versions") == versions(cycle)
            and all(type(v) is int for v in header["versions"].values())
            and header.get("policy") == checkpoint_policy(identity),
            "restart requires an incomplete-target complete Gk/C5k/EMA0 boundary and exact finite cursor")


def resume_metadata(path, *, identity, target, torch_version, verify_payloads=False):
    from .checkpoint import SCHEMA as CHECKPOINT
    root = Path(path).resolve()
    require((root / "manifest.json").is_file() and (root / "_SUCCESS").is_file()
            and (root / "_SUCCESS").read_text().strip() == file_sha256(root / "manifest.json"),
            "complete versioned reference checkpoint required")
    header = document(root / "manifest.json")
    require(set(header) == {"schema_version", "kind", "identity", "cycle", "cursor", "versions", "policy", "topology", "torch", "files"}
            and header.get("schema_version") == CHECKPOINT and header.get("kind") == "RESUMABLE_COMPLETE_CYCLE"
            and header.get("identity") == identity and header.get("topology") == TOPOLOGY
            and header.get("torch") == torch_version, "reference checkpoint source/config/input/runtime/topology changed")
    validate_progress(header, identity=identity, target=target)
    files = header.get("files")
    require(type(files) is dict and set(files) == {f"rank-{r:05d}.pt" for r in range(16)}
            and all(type(sha) is str and SHA256_RE.fullmatch(sha) and (root / name).is_file()
                    and (not verify_payloads or file_sha256(root / name) == sha) for name, sha in files.items()),
            "reference checkpoint all-rank shards missing/hash changed")
    return header


def admit_reference(config, task_value, assets_value, paths):
    request = LiveWanRequest.from_dict(task_value)
    require(request == LiveWanRequest(), "reference supports exact original 81 frames only")
    assets = asset_config(assets_value)
    capacity = capacity_policy(document(paths["capacity_policy"]))
    inputs = _input_metadata(config.data["prepared"], cycles=config.trainer["cycles"], seed=config.trainer["seed"],
                             source=assets["sources"]["livewan"])
    # Reject inherited optimization/profiling knobs even in a dry-run; never
    # silently sanitize the user's environment before a reference dispatch.
    _environment(environment_snapshot(os.environ))
    if config.trainer.get("resume_from"):
        from distillschedule.workloads.best_known.registry import git_head, repository_root
        identity = training_identity(git_head(repository_root()), assets, inputs, config, capacity)
        resume_metadata(config.trainer["resume_from"], identity=identity, target=config.trainer["cycles"],
                        torch_version=assets["runtime"]["dependency_versions"]["torch"])
    return {"task": request.to_dict(), "assets": assets, "inputs": inputs, "plan": None,
            "selected": deepcopy(REFERENCE), "capacity_policy": capacity, "hardware": deepcopy(TOPOLOGY),
            "lifecycle": {"schema_version": SCHEMA, "input_kind": KIND, "same_topology_complete_cycle_resume": True,
                          "numerical_cross_reference": "NOT_RUN", "optimized_pipeline_restart": "NOT_IMPLEMENTED"},
            "validation_scope": "LAZY_SOURCE_CONFIG_INPUT_METADATA_ONLY; WEIGHT_BYTES_RUNTIME_CAPACITY_AND_CUDA_MATH_NOT_RUN"}


def collective_admission(config, admit):
    result = distributed.checked(admit, "W4 reference source/config/input metadata admission")
    _, source, _, admitted = result
    distributed.checked(lambda: require(source["identity"]["dirty"] is False,
        "execution requires a clean committed source checkout"), "W4 reference source cleanliness")
    identity = training_identity(source["identity"]["head"], admitted["assets"], admitted["inputs"], config, admitted["capacity_policy"])
    distributed.admit_run_controls(config)
    require(all(row == identity for row in distributed.gather(identity)), "ranks disagree on W4 immutable reference/input/policy identity")
    return result, identity


class CursorLoader:
    """One actual successful native load per row, never seek-as-consumption."""
    def __init__(self, loader, cursor):
        self.loader, self.cursor, self.inputs = loader, cursor, loader.inputs
        self.consumed = []

    def row(self, step, rank):
        return distributed.checked(lambda: self.inputs.row(step, rank), "W4 original operation row admission")

    def load(self, row):
        def local():
            require(row == self.cursor.peek(), "original baseline requested a repeated/skipped rank-local row")
            actual_row, sample = next(self.cursor)  # actual InputLoader.load, exactly once
            require(actual_row == row, "actual consumed row differs from baseline request")
            self.consumed.append({"operation_key": row.operation_key, "record_id": row.record_id,
                                  "sample_sha256": row.sample_sha256, "prompt_idx": row.prompt_idx,
                                  "seeds": dict(row.seeds)})
            return sample
        return distributed.checked(local, "W4 original native input consumption")

    @property
    def negative(self):
        return distributed.checked(lambda: self.loader.negative, "W4 original negative context admission")

    def context_from_trace(self, row, *, device):
        return self.loader.context_from_trace(row, device=device)


class _CursorState:
    def __init__(self, cursor, header):
        self.cursor, self.header = cursor, header

    def state_dict(self):
        return self.cursor.state_dict()

    def validate_state_dict(self, value):
        self.cursor.validate_state_dict(value)
        require(self.header is None or value["cursor"] == self.header["cursor"],
                "saved rank-local data cursor disagrees with checkpoint progress")

    def load_state_dict(self, value):
        self.validate_state_dict(value)
        self.cursor.load_state_dict(value)

    def validate_checkpoint_boundary(self):
        self.cursor.validate_checkpoint_boundary()


class _VersionsState:
    """Serialization of the actual commit dictionary, not invented clocks."""
    def __init__(self, actual, cursor, header):
        self.actual, self.cursor, self.header = actual, cursor, header

    def state_dict(self):
        return dict(self.actual)

    def validate_state_dict(self, value):
        expected = self.header["versions"] if self.header is not None else self.actual
        require(type(value) is dict and value == expected and all(type(v) is int for v in value.values()),
                "saved actual commit versions disagree with complete checkpoint")

    def load_state_dict(self, value):
        self.validate_state_dict(value)
        self.actual.clear(); self.actual.update(value)

    def validate_checkpoint_boundary(self):
        require(self.cursor.cursor % 5 == 0 and self.actual == versions(self.cursor.cursor // 5),
                "actual commit versions and successfully consumed input cursor disagree")


def quiesce(bundle):
    """Expose the original sharded owners after keep-unsharded rollout scratch.

    Clear consumed gradients on BOTH true module and optimizer owners. FSDP2
    swaps exposed views while the original optimizer retains its DTensors.
    This boundary-only reshard is not a new update or a replacement Parameter.
    """
    for module, optimizer in ((bundle.student, bundle.generator_optimizer), (bundle.critic, bundle.critic_optimizer)):
        optimizer.zero_grad(set_to_none=True)
        module.zero_grad(set_to_none=True)
    import torch
    device = next(bundle.student.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    for model in (bundle.student, bundle.teacher):
        for module in model.modules():
            reshard = getattr(module, "reshard", None)
            if callable(reshard):
                reshard()
    for module, optimizer in ((bundle.student, bundle.generator_optimizer), (bundle.critic, bundle.critic_optimizer)):
        optimizer.zero_grad(set_to_none=True)
        module.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _backend(admitted):
    """Import exact original modules; never patch math or install a kernel."""
    sources = admitted["assets"]["sources"]
    for name in ("wan21", "livewan"):
        path = sources[name]["root"]
        if path not in sys.path:
            sys.path.insert(0, path)
    _original_origins(sources)  # reject a preloaded foreign package BEFORE import
    for name in ("wan.configs", "wan.modules.model", "wanstreamer.fsdp", "wanstreamer.stream", "wanstreamer.lora"):
        importlib.import_module(name)
    _original_origins(sources)
    attention = sys.modules["wan.modules.attention"]
    require(attention.FLASH_ATTN_2_AVAILABLE and not attention.FLASH_ATTN_3_AVAILABLE
            and importlib.util.find_spec("flash_attn_interface") is None,
            "reference requires original FlashAttention2, not an inherited alternative/patch")
    from distillschedule.runtime import livewan_crossscale_worker as worker
    require(worker._operation_calibrator is None, "reference cannot inherit a benchmark operation calibrator")
    return worker


def _original_origins(sources):
    result = {}
    for name, module in tuple(sys.modules.items()):
        project = "livewan" if name == "wanstreamer" or name.startswith("wanstreamer.") else (
            "wan21" if name == "wan" or name.startswith("wan.") else None)
        if project is None:
            continue
        source = sources[project]
        path = Path(getattr(module, "__file__", "")).resolve()
        require(path.is_relative_to(source["root"]) and source["files"].get(str(path.relative_to(source["root"]))) == file_sha256(path),
                "actual original import origin/hash differs: " + name)
        result[name] = {"path": str(path), "sha256": source["files"][str(path.relative_to(source["root"]))]}
    return result


def _meta_layouts(worker, *, WanModel, config):
    """Actual original constructors, meta tensors ONLY; no alternative model."""
    import torch
    result = {}
    for size in ("1.3B", "14B"):
        with torch.device("meta"):
            model = worker._wan_model(WanModel, size=size, config=config)
        state = model.state_dict()
        require(state and all(t.is_meta for t in state.values()), "original shape admission must not allocate physical model tensors")
        result[size] = {name: list(t.shape) for name, t in state.items()}
        del state, model
    return result


def _admit_model_layout(owner, worker):
    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel
    return admit_native_layouts(owner._document, _meta_layouts(worker, WanModel=WanModel, config=WAN_CONFIGS["t2v-1.3B"]))


def _rng_identity():
    from .checkpoint import rng_state
    value = rng_state()
    result = {"python": fingerprint(repr(value["python"])), "torch": hashlib.sha256(value["torch"].numpy().tobytes()).hexdigest()}
    if "cuda" in value:
        result["cuda"] = hashlib.sha256(value["cuda"]["state"].cpu().numpy().tobytes()).hexdigest()
    if "numpy" in value:
        result["numpy"] = fingerprint(value["numpy"])
    return result


def _capacity(torch, observer, policy, *, rank, initial=None):
    """Unchanged common host/driver limits plus real allocator/OOM boundaries."""
    observer.sample()
    require(not observer.errors and observer.rows, "common capacity admission failed: " + "; ".join(observer.errors))
    row = observer.rows[-1]
    for gpu in row["gpus"]:
        reserve = policy["gpu_reserve_bytes"]
        require(gpu["free_bytes"] >= reserve, "observed driver GPU reserve below explicit policy")
        if initial is None:
            require(gpu["free_bytes"] >= policy["memory_limits"][str(rank // 8 * 8 + gpu["device"])]+reserve,
                    "available GPU memory cannot admit the explicit reference allocation budget/reserve")
    require(torch.cuda.max_memory_allocated() <= policy["memory_limits"][str(rank)],
            "actual allocator peak exceeded reference capacity policy")
    events = {k: row["cgroup_events"][k] for k in ("oom", "oom_kill")}
    require(initial is None or events == initial, "reference cgroup OOM counters changed")
    return events


def _reserve_output(output):
    """All-rank shared output ownership, before any rank may publish failure."""
    token = distributed.gather(uuid.uuid4().hex if distributed.rank_world()[0] == 0 else None)[0]
    def create():
        output.mkdir(parents=True, exist_ok=False)
        (output / ".reference-owner").write_text(token + "\n")
    distributed.primary(create, "W4 reference exclusive output reservation")
    distributed.checked(lambda: require((output / ".reference-owner").read_text().strip() == token,
        "reference output must be shared and owned by this invocation"), "W4 reference output visibility")
    return token


def _receipt(output, token, value):
    require((output / ".reference-owner").read_text().strip() == token, "reference output ownership changed")
    from distillschedule.runtime.livewan_crossscale_worker import atomic_json
    path = output / "ranks" / f"rank-{distributed.rank_world()[0]:05d}.json"
    previous = None
    if os.path.lexists(path):
        previous = document(path)
        require(previous.get("invocation_token") == token, "unowned entry-time rank receipt is preserved")
    value = {**value, "invocation_token": token}
    if value.get("status") == "FAIL" and previous is not None and previous.get("status") in {"PARTIAL", "COMPLETED"}:
        value["last_complete_progress"] = previous
    atomic_json(path, value)


def _cycle_metrics(evidence, consumed, actual, cycle):
    require(actual == versions(cycle) and len(consumed) == 5 and len({r["operation_key"] for r in consumed}) == 5,
            "original baseline commit/input census changed")
    require(not evidence.output_tensors and not evidence.output_fingerprints
            and len(evidence.optimizer_diagnostics) == 6 and len(evidence.gradient_sync) == 5
            and evidence.logical_score_calls.get("teacher/nograd/plain") == 1
            and evidence.logical_score_calls.get("critic/nograd/plain") == 1
            and evidence.logical_score_calls.get("critic/grad/checkpoint") == 5,
            "original reference call/commit evidence changed")
    losses, norms = evidence.loss_fingerprints, evidence.optimizer_diagnostics
    require(sum(r["kind"] == "critic" for r in losses) == 5 and sum(r["kind"] == "generator" for r in losses) == 3
            and all(r.get("finite") is True and all(math.isfinite(r[k]) for k in ("mean", "norm", "max_abs")) for r in losses)
            and all(math.isfinite(r["clip_input_norm"]) for r in norms), "native loss/gradient evidence incomplete/nonfinite")
    return {"cycle": cycle, "versions": dict(actual), "inputs": deepcopy(consumed), "losses": losses,
            "optimizer_norms": norms, "logical_score_calls": dict(evidence.logical_score_calls),
            "rng_sha256": _rng_identity(), "claimable_speedup": False}


def reference_process_group():
    return distributed.world16_reference_process_group("W4 reference")


def run_reference(config, *, admit):
    """Own a fresh physical execution; all local preflight fails collectively."""
    with reference_process_group():
        result, identity = collective_admission(config, admit)
        _, source, paths, admitted = result
        output = Path(paths["output"])
        token = _reserve_output(output)
        try:
            return _run(config, admitted, identity, output, token)
        except Exception as exc:
            # A preflight failure has already been exchanged by checked(). Each
            # admitted owner writes only its own receipt; old evidence is never
            # acquired or replaced when output reservation itself failed.
            error = type(exc).__name__ + ": " + str(exc)
            distributed.checked(lambda: _receipt(output, token, {"schema_version": SCHEMA, "status": "FAIL",
                "rank": distributed.rank_world()[0], "identity": identity, "error": error,
                "gpu_execution": "PARTIAL_OR_NOT_RUN", "numerical_cross_reference": "NOT_RUN", "claimable_speedup": False}),
                "W4 reference failure receipt")
            raise


def _run(config, admitted, identity, output, token):
    import torch
    from .checkpoint import save_checkpoint, restore_checkpoint
    from .livewan_inputs import NativeTeacherCursor
    from .livewan_state import reference_model_bundle_state
    from distillschedule.search.resources import ResourceObserver
    rank = distributed.rank_world()[0]
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    runtime = distributed.checked(lambda: admit_runtime(admitted["assets"]), "W4 reference dependency/environment/backend controls")
    observer = ResourceObserver(output / "capacity" / f"rank-{rank:05d}.json", "original-W4-reference", rank // 8,
                                capacity_policy=admitted["capacity_policy"])
    initial_oom = distributed.checked(lambda: _capacity(torch, observer, admitted["capacity_policy"], rank=rank),
                                      "W4 reference pre-payload physical capacity/reserve admission")
    owner = distributed.checked(lambda: ReferenceAssets.load(admitted["assets"]), "W4 reference all frozen/Stage1 payload admission")
    resume = config.trainer.get("resume_from")
    header = distributed.checked(lambda: resume_metadata(resume, identity=identity, target=config.trainer["cycles"],
        torch_version=torch.__version__, verify_payloads=True) if resume else None, "W4 reference all-rank checkpoint bytes")
    from distillschedule.runtime import livewan_crossscale_worker as native_worker
    loader = distributed.checked(lambda: native_worker.InputLoader(torch=torch, input_kind=KIND, prepared=config.data["prepared"]),
                                 "W4 actual InputLoader full source/record/prompt admission")
    distributed.checked(lambda: require(loader.inputs.input_sha256 == identity["input_sha256"]
        and loader.inputs.cycles == config.trainer["cycles"], "actual InputLoader identity/horizon differs"),
        "W4 actual InputLoader immutable identity")
    cursor = NativeTeacherCursor(loader, rank=rank)
    consuming = CursorLoader(loader, cursor)
    distributed.checked(lambda: _capacity(torch, observer, admitted["capacity_policy"], rank=rank, initial=initial_oom),
                        "W4 reference pre-model physical capacity/reserve admission")
    worker = distributed.checked(lambda: _backend(admitted), "W4 original import origin/FlashAttention2 admission")
    distributed.checked(lambda: admit_runtime(admitted["assets"]), "W4 effective controls unchanged across original imports")
    origins = distributed.checked(lambda: _original_origins(admitted["assets"]["sources"]), "W4 actual original runtime origins")
    native_layouts = distributed.checked(lambda: _admit_model_layout(owner, worker), "W4 complete original checkpoint key/shape admission")
    from distillschedule.runtime.livewan_geometry import install_geometry
    distributed.checked(lambda: install_geometry(LiveWanRequest(), plan_sha256=fingerprint(identity)), "W4 immutable reference geometry")
    from distillschedule.runtime.livewan_cuda_setup import materialize_blas_handles
    distributed.checked(lambda: materialize_blas_handles(torch, device), "W4 original CUDA library initialization")
    # Setup imports may draw RNG; seed immediately before the original builder.
    random.seed(config.trainer["seed"] + rank)
    torch.manual_seed(config.trainer["seed"] + rank)
    import numpy as np
    np.random.seed((config.trainer["seed"] + rank) % 2**32)
    bundle = worker.build_models(torch=torch, plan=owner.native_plan(), bootstrap=owner,
                                 mode="baseline", rank=rank, device=device)
    distributed.checked(owner.unchanged, "W4 assets unchanged across original model reads")
    streamers = worker._build_streamers(torch, student=bundle.student, count=1, device=device)
    score_context = worker.build_score_context(torch, bundle=bundle, loader=consuming, device=device)
    distributed.checked(lambda: quiesce(bundle), "W4 original initial owner boundary")
    actual = owner.versions
    state = distributed.checked(lambda: reference_model_bundle_state(bundle,
        source_root=admitted["assets"]["sources"]["livewan"]["root"], versions=actual,
        restore_versions=header["versions"] if header else None), "W4 all-role named native state ownership")
    state.update({"data.cursor": _CursorState(cursor, header), "commits": _VersionsState(actual, cursor, header)})
    completed = header["cycle"] if header else 0
    if resume:
        restored = restore_checkpoint(resume, state, identity=identity)
        validate_progress(restored, identity=identity, target=config.trainer["cycles"])
    def boundary():
        for value in state.values():
            check = getattr(value, "validate_checkpoint_boundary", None)
            if check:
                check()
    distributed.checked(boundary, "W4 initial/restored quiescent data/role boundary")
    metrics = []
    for cycle in range(completed, config.trainer["cycles"]):
        require(actual == versions(cycle) and cursor.cursor == 5 * cycle, "native outer cycle skipped or repeated")
        consuming.consumed.clear()
        evidence = worker.RunEvidence()
        # The exact existing baseline call order, losses, Adam and reductions.
        worker._run_baseline(torch=torch, rank=rank, trace=consuming, loader=consuming, bundle=bundle,
            streamers=streamers, score_context=score_context, run_kind="trajectory", evidence=evidence,
            versions=actual, device=device, step_offset=5 * cycle)
        distributed.checked(lambda: quiesce(bundle), "W4 consumed-gradient true-owner cleanup")
        distributed.checked(boundary, "W4 complete-cycle role/data quiescence")
        observed = distributed.gather(dict(actual))
        require(all(v == versions(cycle + 1) for v in observed), "actual native commit versions disagree across ranks")
        metric = distributed.checked(lambda: _cycle_metrics(evidence, consuming.consumed, actual, cycle + 1),
                                     "W4 bounded actual loss/norm/input evidence")
        distributed.checked(lambda: _capacity(torch, observer, admitted["capacity_policy"], rank=rank, initial=initial_oom),
                            "W4 reference complete-cycle capacity/reserve/OOM admission")
        metrics.append(metric)
        cadence = config.trainer["checkpoint_every"]
        if cadence and (cycle + 1) % cadence == 0:
            save_checkpoint(output / "checkpoints" / f"cycle-{cycle + 1:06d}", state, identity=identity,
                cycle=cycle + 1, cursor=cursor.cursor, versions=actual, policy=checkpoint_policy(identity))
        distributed.checked(lambda: _receipt(output, token, {"schema_version": SCHEMA,
            "status": "COMPLETED" if cycle + 1 == config.trainer["cycles"] else "PARTIAL", "rank": rank,
            "identity": identity, "completed_cycles": cycle + 1, "next_input_cursor": cursor.cursor,
            "versions": dict(actual), "metrics": metrics, "capacity_samples": observer.rows,
            "runtime": runtime, "original_runtime_origins": origins,
            "native_parameter_layout_sha256": native_layouts,
            "numerical_cross_reference": "NOT_RUN", "claimable_speedup": False,
            "input_quality": admitted["inputs"]["evidence_kind"], "pipeline_restart": "NOT_IMPLEMENTED"}),
            "W4 reference owned progress publication")
    distributed.checked(lambda: require(cursor.cursor == cursor.inputs.steps,
        "reference did not exhaust the exact finite native horizon"), "W4 finite complete input exhaustion")
    return 0
