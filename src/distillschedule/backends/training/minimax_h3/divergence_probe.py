"""Default-off, bounded first-Fake numerical diagnosis, never admission/timing.

Only tiny fixed-coordinate device snapshots are retained at native sites. All
tensor-to-host reads occur before cycle 1 or after the existing stop fence.
Device gathers perturb execution: even a passing diagnostic is NOT native time.
This module imports no Torch, installs no RNG hooks and changes no model values.
"""
from __future__ import annotations

import hashlib
import copy
import json
import math
import os
from pathlib import Path
import re

from distillschedule.evidence.trajectory_state_sketch import canonical_state_name, deterministic_flat_indices, _layout

SCHEMA = "distillschedule.h3_first_fake_divergence.v1"
BLOCKS = (0, 10, 49)
SAMPLES = 16
MAX_ROWS = 192
MAX_INPUT_TENSORS = 24
PARAMETERS = (
    "token_refiner.refiner_blocks.0.ff.net.0.proj.lora_A.student.weight",
    "token_refiner.refiner_blocks.0.ff.net.0.proj.lora_B.student.weight",
    "transformer_blocks.10.attn.to_q.lora_A.fake.weight",
    "transformer_blocks.10.attn.to_q.lora_B.fake.weight",
    "transformer_blocks.10.ff.net.0.proj.lora_A.fake.weight",
    "transformer_blocks.10.ff.net.0.proj.lora_B.fake.weight",
)
ARMS = {"d2A": "joint-h3v1-d2-n128", "d2B": "joint-h3v1-d2-n128", "d1A": "joint-h3v1-d1-n128"}
_ACTIVE = None
PLAN_SCHEMA = "distillschedule.h3_first_fake_divergence_plan.v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def helper_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def bind_plan(receipt, arm, *, raw_repeat=False, cache_roundtrip=False):
    """Explicit opt-in after ordinary candidate lowering; no assets/math edits."""
    _require(arm in ARMS and "h3_divergence" not in receipt, "invalid/already-bound diagnostic plan")
    _require(type(raw_repeat) is bool, "raw-repeat opt-in must be a plain bool")
    _require(type(cache_roundtrip) is bool and not (raw_repeat and cache_roundtrip), "exclusive bounded diagnostics required")
    value = copy.deepcopy(receipt)
    value["h3_divergence"] = dict(schema_version=PLAN_SCHEMA, arm=arm, source_head=value["integration_head"],
                                  helper_sha256=helper_sha256())
    for node in value["nodes"]:
        node["environment"].update(H3_PLANNER_DIVERGENCE="1", H3_PLANNER_DIVERGENCE_ARM=arm)
    if raw_repeat:
        from .raw_backward_probe import CONTRACT
        value["h3_divergence"]["raw_repeat"] = dict(CONTRACT)
        for node in value["nodes"]:
            node["environment"]["H3_PLANNER_RAW_BACKWARD_REPEAT"] = "1"
    if cache_roundtrip:
        from .cache_roundtrip_probe import source_contract
        value["h3_divergence"]["cache_roundtrip"] = source_contract()
        for node in value["nodes"]:
            node["environment"]["H3_PLANNER_CACHE_ROUNDTRIP"] = "1"
    is_diagnostic_plan(value)
    return value


def is_diagnostic_plan(receipt):
    marker = receipt.get("h3_divergence")
    nodes = receipt.get("nodes", [])
    flags = [n.get("environment", {}).get("H3_PLANNER_DIVERGENCE", "0") for n in nodes]
    if marker is None and not any(f != "0" for f in flags):
        _require(not any(n.get("environment", {}).get("H3_PLANNER_CACHE_ROUNDTRIP", "0") != "0" for n in nodes), "orphan roundtrip diagnostic")
        _require(not any(n.get("environment", {}).get("H3_PLANNER_RAW_BACKWARD_REPEAT", "0") != "0" for n in nodes), "orphan raw-repeat diagnostic")
        _require(not any(n.get("environment", {}).get("H3_PLANNER_DIVERGENCE_ARM") for n in nodes),
                 "orphan diagnostic arm")
        return False
    _require(isinstance(marker, dict) and set(marker) in ({"schema_version", "arm", "source_head", "helper_sha256"}, {"schema_version", "arm", "source_head", "helper_sha256", "raw_repeat"}, {"schema_version", "arm", "source_head", "helper_sha256", "cache_roundtrip"}),
             "diagnostic plan requires exact explicit binding")
    from .raw_backward_probe import CONTRACT as RAW
    raw = "raw_repeat" in marker
    _require(not raw or marker["raw_repeat"] == RAW, "raw-repeat contract changed")
    _require(all(n.get("environment", {}).get("H3_PLANNER_RAW_BACKWARD_REPEAT", "0") == ("1" if raw else "0") for n in nodes), "raw-repeat node/plan binding differs")
    from .cache_roundtrip_probe import source_contract
    roundtrip = "cache_roundtrip" in marker
    _require(not roundtrip or marker["cache_roundtrip"] == source_contract(), "roundtrip source/contract changed")
    _require(all(n.get("environment", {}).get("H3_PLANNER_CACHE_ROUNDTRIP", "0") == ("1" if roundtrip else "0") for n in nodes), "roundtrip node/plan binding differs")
    arm = marker["arm"]
    _require(marker["schema_version"] == PLAN_SCHEMA and arm in ARMS and marker["source_head"] == receipt.get("integration_head")
             and marker["helper_sha256"] == helper_sha256() and receipt.get("profile_id") == "minimax_h3_grid1000_best"
             and receipt.get("mode") == "correctness" and receipt.get("native_mode") == "correctness"
             and receipt.get("search_candidate", {}).get("id") == ARMS[arm] and flags == ["1", "1"]
             and [n.get("node_rank") for n in nodes] == [0, 1],
             "diagnostic plan source/helper/arm/mode mismatch")
    for node in nodes:
        _require(node["environment"].get("H3_PLANNER_DIVERGENCE_ARM") == arm, "mixed diagnostic arms")
        _require(node["environment"].get("DISTILLSCHEDULE_RUN_ID") == receipt["run_id"]
                 and node["environment"].get("DISTILLSCHEDULE_RUN_ROOT") ==
                 str(Path(receipt["asset_config"]["run_root"]) / receipt["run_id"]),
                 "diagnostic node run identity differs")
        from_environment(torch=None, rank=node["node_rank"] * 8, source_head=marker["source_head"],
                         environ=dict(node["environment"], EXPECTED_INTEGRATION_HEAD=marker["source_head"]))
    return True


def collect_diagnostic(receipt):
    """Check native contracts and captures, but write NO canonical summaries."""
    from distillschedule.search.h3_native import read_native_receipts, summarize_native_two_cycle, _rng
    from distillschedule.workloads.best_known.registry import write_new_json, repository_root, git_head
    from distillschedule.search.execution import validated_profile
    _require(is_diagnostic_plan(receipt), "not a diagnostic plan")
    source = repository_root()
    validated_profile(receipt, root=source)
    _require(git_head(source) == receipt["integration_head"], "diagnostic collector source changed")
    root = Path(receipt["asset_config"]["run_root"]) / receipt["run_id"]
    _require(not (root / "native-summary.json").exists() and not (root / "bound-summary.json").exists(),
             "diagnostic run cannot own canonical summaries")
    rows, refs = read_native_receipts(root)
    captures = []
    for rank, raw in enumerate(rows):
        ref = raw.get("divergence_diagnostic")
        path = root / "execution/divergence" / f"rank_{rank:03d}.json"
        _require(isinstance(ref, dict) and Path(ref.get("path", "")).resolve() == path.resolve()
                 and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == ref.get("sha256"),
                 "missing/mixed/unbound diagnostic capture")
        capture = json.loads(path.read_text())
        _require(capture.get("schema_version") == SCHEMA and capture.get("rank") == rank
                 and capture.get("integration_head") == receipt["integration_head"]
                 and capture.get("arm") == receipt["h3_divergence"]["arm"]
                 and capture.get("helper_sha256") == receipt["h3_divergence"]["helper_sha256"]
                 and capture.get("run_id") == receipt["run_id"] and capture.get("candidate_id") == receipt["search_candidate"]["id"]
                 and capture.get("status") == "CAPTURED_BOUNDED_DIAGNOSTIC_NOT_NUMERICAL_PASS"
                 and capture.get("numerical_admission") is False and capture.get("formal_timing_eligible") is False,
                 "diagnostic rank/source/run/scope mismatch")
        _require(_rng(raw, raw, rank, receipt["integration_head"]) == "PASS", "diagnostic native RNG contract incomplete")
        _require(capture.get("fa3_counts") == {f"{stage}/{block}": 1 for stage in ("original", "replay", "backward") for block in BLOCKS},
                 "diagnostic FA3 captures incomplete")
        samples = capture.get("rows", [])
        _require(0 < len(samples) <= MAX_ROWS and len({(r["boundary"], r["name"]) for r in samples}) == len(samples),
                 "diagnostic capture rows duplicate/unbounded/missing")
        for sample in samples:
            if "values_fp32" in sample:
                values = sample["values_fp32"]
                _require(0 < len(values) <= SAMPLES and len(values) == len(sample["local_flat_indices"])
                         and all(type(v) in (int, float) and math.isfinite(v) for v in values)
                         and sample.get("sample_sha256") == _digest(values), "diagnostic sample invalid")
        losses = [s["values_fp32"] for s in samples if s["boundary"] == "first_fake_loss" and s["name"] == "loss"]
        _require(losses == [raw["cycles"][0]["losses"]["fake"][:1]], "capture loss differs from native first Fake")
        _require(capture.get("rng", {}).get("initial") == raw["cycles"][0]["rng_before"]
                 and all(capture.get("rng", {}).get(f"cycle{i}_{edge}") == cycle.get(f"rng_{edge}")
                         for i, cycle in enumerate(raw["cycles"], 1) for edge in ("before", "after")),
                 "capture RNG differs from native boundaries")
        for cycle in raw.get("cycles", []):
            _require("native_wall_seconds" not in cycle and "gpu_completion_elapsed_s" not in cycle,
                     "diagnostic run retained misleading native durations")
        if "cache_roundtrip" in receipt["h3_divergence"]:
            audit = capture.get("cache_roundtrip_audit", {})
            _require(audit.get("contract") == receipt["h3_divergence"]["cache_roundtrip"]
                and audit.get("status") == "MATCHED_FINGERPRINTS_NOT_NUMERICAL_ADMISSION"
                and audit.get("pairs") == 1200 and len(audit.get("cycles", [])) == 2
                and all(c.get("pairs") == 600 and c.get("mismatches") == 0 for c in audit["cycles"])
                and audit.get("selftest", {}).get("status") == "PASS_GPU_KNOWN_BITS_AND_STRIDES"
                and audit.get("numerical_admission") is False and audit.get("formal_timing_eligible") is False,
                "roundtrip diagnostic missing/source/census/mismatch")
        captures.append(dict(path=str(path), sha256=ref["sha256"]))
    summarize_native_two_cycle(rows, expected_head=receipt["integration_head"],
                               candidate_id=receipt["search_candidate"]["id"], _diagnostic_contracts_only=True)
    write_new_json(root / "diagnostic-collection.json", dict(
        schema_version="distillschedule.h3_first_fake_divergence_collection.v1",
        status="DIAGNOSTIC_CAPTURE_COMPLETE_NOT_ADMISSION", source_head=receipt["integration_head"],
        diagnostic_plan=receipt["h3_divergence"], native_evidence=refs, capture_evidence=captures,
        native_timing="NOT_RUN", numerical_admission=False, formal_timing_eligible=False))
    return 0


def from_environment(*, torch, rank, source_head, environ=None):
    env = os.environ if environ is None else environ
    flag = env.get("H3_PLANNER_DIVERGENCE", "0")
    _require(flag in ("0", "1"), "H3 divergence requires explicit 0/1")
    raw = env.get("H3_PLANNER_RAW_BACKWARD_REPEAT", "0")
    _require(raw in ("0", "1") and (raw == "0" or flag == "1"), "raw-repeat requires the bound divergence diagnostic")
    roundtrip = env.get("H3_PLANNER_CACHE_ROUNDTRIP", "0")
    _require(roundtrip in ("0", "1") and (roundtrip == "0" or (flag == "1" and raw == "0")), "roundtrip requires exclusive bound divergence diagnostic")
    if flag == "0":
        _require(not env.get("H3_PLANNER_DIVERGENCE_ARM"), "disabled divergence has an arm")
        return None
    arm = env.get("H3_PLANNER_DIVERGENCE_ARM")
    _require(arm in ARMS and env.get("DISTILLSCHEDULE_SEARCH_CANDIDATE_ID") == ARMS[arm],
             "divergence arm/candidate mismatch")
    _require(re.fullmatch(r"[0-9a-f]{40}", source_head or "") is not None
             and env.get("EXPECTED_INTEGRATION_HEAD") == source_head,
             "divergence requires exact source identity")
    _require(source_head != "07fcc0ba1ec28b810b50cac92ec41c1a5232d3e1", "historical source is immutable")
    _require(env.get("DISTILLSCHEDULE_RUN_KIND") == "correctness"
             and env.get("H3_PLANNER_TWO_CYCLE") == "1"
             and env.get("H3_PLANNER_CLOCK") == "host_fenced"
             and env.get("H3_PLANNER_CALIBRATION", "0") == "0"
             and not env.get("H3_PLANNER_ROLLOUT_WINDOW"),
             "divergence is correctness-only; never calibration/timing/action admission")
    root, run_id = Path(env.get("DISTILLSCHEDULE_RUN_ROOT", "")), env.get("DISTILLSCHEDULE_RUN_ID", "")
    _require(root.is_absolute() and re.fullmatch(r"[A-Za-z0-9._-]{1,128}", run_id) is not None
             and run_id not in (".", "..") and type(rank) is int and 0 <= rank < 16,
             "divergence run identity/rank invalid")
    result = FirstFakeProbe(torch=torch, rank=rank, source_head=source_head,
                          run_root=root, run_id=run_id, arm=arm, candidate_id=ARMS[arm])
    result.raw_repeat_enabled = raw == "1"
    result.raw_repeat_result = None
    result.roundtrip_enabled = roundtrip == "1"
    return result


def capture_raw_backward(torch, runtime, block, inputs, primary, scale):
    probe = _ACTIVE
    if probe is None or not getattr(probe, 'raw_repeat_enabled', False) or not probe.active:
        return
    from .raw_backward_probe import CONTRACT, run
    if probe.rank == CONTRACT['rank'] and block == CONTRACT['block']:
        _require(probe.raw_repeat_result is None, 'raw backward probe repeated beyond one actual graph')
        probe.raw_repeat_result = run(torch, runtime.raw_backward, inputs, primary, scale)


def capture_fa3(stage, block, values):
    if _ACTIVE is not None and getattr(_ACTIVE, 'roundtrip', None) is not None:
        _ACTIVE.roundtrip.capture(stage, block, values)
    if _ACTIVE is not None and _ACTIVE.active and block in BLOCKS:
        _ACTIVE.fa3(stage, block, values)


class FirstFakeProbe:
    def __init__(self, *, torch, rank, source_head, run_root, run_id, arm, candidate_id):
        self.torch, self.rank, self.source_head = torch, rank, source_head
        self.run_root, self.run_id, self.arm, self.candidate_id = Path(run_root), run_id, arm, candidate_id
        self.pending, self.rows, self.keys = [], [], set()
        self.cycle, self.fake_index, self.active = 0, 0, False
        self.counts, self.rng, self.optimizer_groups = {}, {}, {}
        self.roundtrip = None

    def queue(self, boundary, name, value):
        """Metadata plus <=16 device elements; no full flatten/contiguous copy."""
        key = (boundary, name)
        _require(key not in self.keys and len(self.keys) < MAX_ROWS, "duplicate/unbounded divergence sample")
        self.keys.add(key)
        metadata = dict(boundary=boundary, name=name)
        if not self.torch.is_tensor(value):
            _require(type(value) in (int, float) and math.isfinite(value), "unsupported diagnostic scalar")
            self.pending.append((dict(metadata, representation="PYTHON_SCALAR", local_flat_indices=[0]), [float(value)]))
            return
        local = value.detach().to_local() if hasattr(value, "to_local") else value.detach()
        shape, numel = tuple(local.shape), int(local.numel())
        _require(numel > 0, "empty divergence local tensor")
        # A/B, gradient and Adam moments use the SAME parameter coordinates so
        # update direction can be inspected; scalar Adam step naturally has [0].
        coordinate_name = name.split("/", 1)[0] if name.startswith(("token_refiner.", "transformer_blocks.")) else name
        indices = deterministic_flat_indices(coordinate_name, numel, count=SAMPLES)
        if local.is_contiguous():
            sample = local.view(-1).index_select(0, self.torch.tensor(indices, dtype=self.torch.int64, device=local.device))
        else:
            # Advanced indexing gathers only the chosen coordinates from strided inputs.
            coords = []
            for axis in range(len(shape)):
                stride = math.prod(shape[axis + 1:])
                coords.append(self.torch.tensor([(i // stride) % shape[axis] for i in indices],
                                               dtype=self.torch.int64, device=local.device))
            sample = local[tuple(coords)]
        metadata.update(local_shape=list(shape), stored_shape=list(value.shape), dtype=str(local.dtype),
                        local_flat_indices=list(indices), representation="DTENSOR_LOCAL_SHARD" if hasattr(value, "to_local") else "LOCAL_TENSOR")
        metadata["coordinate_identity"] = coordinate_name
        if hasattr(value, "placements"):
            _, _, offset, _ = _layout(torch=self.torch, logical_tensor=value, local_tensor=local)
            metadata.update(placements=[str(x) for x in value.placements], mesh_shape=list(value.device_mesh.shape),
                            mesh_ranks=value.device_mesh.mesh.tolist(), global_offset=list(offset))
        self.pending.append((metadata, sample.detach()))

    def flush(self, *, outside_cycle):
        _require(outside_cycle and not self.active, "host reads require an existing cycle stop fence")
        for metadata, sample in self.pending:
            values = sample if isinstance(sample, list) else sample.float().cpu().tolist()
            _require(all(math.isfinite(v) for v in values), "nonfinite divergence sample")
            self.rows.append(dict(metadata, values_fp32=values, sample_sha256=_digest(values),
                                  sampled_l2=math.hypot(*values), sampled_peak_abs=max(map(abs, values))))
        self.pending.clear()

    def state(self, trainer, boundary, *, fake_only=False):
        for name, value in self.parameters.items():
            fake = ".fake." in name
            if fake_only and not fake:
                continue
            optimizer = trainer.fake_optimizer if fake else trainer.optimizer
            self.queue(boundary, name, value)
            stored = optimizer.state.get(value, {})
            if not stored:
                key = (boundary, name + "/optimizer")
                _require(key not in self.keys, "duplicate empty optimizer state")
                self.keys.add(key)
                self.rows.append(dict(boundary=boundary, name=key[1], status="NOT_INITIALIZED_NATIVE_LAZY_STATE"))
            else:
                _require({"step", "exp_avg", "exp_avg_sq"} <= set(stored), "native Adam state incomplete")
                for kind in ("step", "exp_avg", "exp_avg_sq"):
                    self.queue(boundary, name + "/" + kind, stored[kind])

    def initial(self, trainer, rng):
        global _ACTIVE
        _require(_ACTIVE is None, "divergence probe already installed")
        names = [(canonical_state_name(n), p) for n, p in trainer.shared_model.denoiser_module().named_parameters()]
        _require(len({n for n, _ in names}) == len(names), "duplicate native state names")
        selected = {n: p for n, p in names if n in PARAMETERS}
        _require(set(selected) == set(PARAMETERS), "required bounded LoRA samples missing")
        self.parameters = {n: selected[n] for n in PARAMETERS}
        self.rng["initial"] = dict(rng)
        self.fa3_registration = trainer.fa3_replay_cache_registration.receipt()
        self.state(trainer, "initial")
        for role, optimizer in (("student", trainer.optimizer), ("fake", trainer.fake_optimizer)):
            self.optimizer_groups[role] = [
                {k: g[k] for k in ("lr", "eps", "betas", "weight_decay", "foreach", "fused") if k in g}
                for g in optimizer.param_groups]
        self.flush(outside_cycle=True)
        if getattr(self, "roundtrip_enabled", False):
            from .cache_roundtrip_probe import RoundTripProbe, gpu_selftest
            self.roundtrip = RoundTripProbe(self.torch)
            self.roundtrip.selftest = gpu_selftest(self.torch, next(iter(self.parameters.values())).device)
        _ACTIVE = self

    def begin_cycle(self, cycle, rng):
        _require(cycle in (1, 2) and cycle == self.cycle + 1 and not self.active, "divergence cycle order")
        self.cycle, self.fake_index = cycle, 0
        self.rng[f"cycle{cycle}_before"] = dict(rng)
        if self.roundtrip is not None:
            self.roundtrip.begin(cycle)

    def before_fake_group(self, trainer, group):
        self.active = self.cycle == 1 and self.fake_index == 0
        self.fake_index += 1
        if not self.active:
            return
        _require(len(group) == 1, "divergence requires native B1/accum1 first Fake")
        self.state(trainer, "pre_first_fake")
        item, inputs = group[0], []
        for field in ("condition", "generated", "noises", "renoised", "sigmas"):
            value = getattr(item, field)
            pairs = sorted(value.items()) if isinstance(value, dict) else enumerate(value)
            for key, tensor in pairs:
                _require(self.torch.is_tensor(tensor), "unsupported first-Fake input tree")
                inputs.append((f"input/{field}/{key}", tensor))
        _require(0 < len(inputs) <= MAX_INPUT_TENSORS, "first-Fake inputs exceed fixed budget")
        for name, tensor in inputs:
            self.queue("first_fake_input", name, tensor)
        self.input_metadata = dict(latent_shape=item.latent_shape, cache_key=item.cache_key)

    def loss(self, value):
        if self.active:
            self.queue("first_fake_loss", "loss", value)

    def gradients(self, boundary, trainer):
        if not self.active:
            return
        for name, value in self.parameters.items():
            if ".fake." in name:
                _require(value.grad is not None, "first-Fake gradient not materialized")
                self.queue(boundary, name + "/grad", value.grad)

    def after_clip(self, trainer, norm):
        if self.active:
            self.queue("clip", "total_norm_before_clip", norm)
            self.queue("clip", "max_grad_norm", float(trainer.max_grad_norm))
            self.gradients("post_clip", trainer)
            self.state(trainer, "pre_commit", fake_only=True)

    def after_step(self, trainer):
        if self.active:
            self.state(trainer, "post_commit", fake_only=True)

    def after_fake_group(self):
        self.active = False

    def fa3(self, stage, block, values):
        _require(stage in ("original", "replay", "backward"), "unknown FA3 capture stage")
        key = f"{stage}/{block}"
        _require(key not in self.counts, "repeated first-Fake FA3 block capture")
        self.counts[key] = 1
        names = ("dq", "dk", "dv") if stage == "backward" else ("out", "lse")
        _require(len(values) == len(names), "FA3 diagnostic tensor arity changed")
        for name, value in zip(names, values):
            self.queue(f"fa3_{stage}", f"fa3/block{block}/{name}", value)

    def after_cycle(self, cycle, rng):
        _require(cycle == self.cycle and self.fake_index == 5 and not self.active, "first-Fake cycle incomplete")
        self.rng[f"cycle{cycle}_after"] = dict(rng)
        self.flush(outside_cycle=True)
        if self.roundtrip is not None:
            result = self.roundtrip.finish_cycle(outside_cycle=True)
            from distillschedule.workloads.best_known.registry import write_new_json
            path = self.run_root / "execution/roundtrip" / f"cycle_{cycle}" / f"rank_{self.rank:03d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            write_new_json(path, dict(result, source_head=self.source_head, rank=self.rank, run_id=self.run_id,
                numerical_admission=False, formal_timing_eligible=False))
            import torch.distributed as dist
            bad = self.torch.tensor(int(result["mismatches"] > 0), device=next(iter(self.parameters.values())).device)
            if dist.is_initialized():
                dist.all_reduce(bad, op=dist.ReduceOp.MAX)
            _require(int(bad.cpu()) == 0, "ROUNDTRIP_PAYLOAD_FINGERPRINT_MISMATCH_SAVED_NO_NUMERICAL_ADMISSION")

    def finish(self):
        _require(self.cycle == 2 and not self.pending and not self.active, "diagnostic capture unfinished")
        _require(self.counts == {f"{stage}/{block}": 1 for stage in ("original", "replay", "backward") for block in BLOCKS},
                 "first-Fake FA3 coverage incomplete")
        for boundary in ("initial", "pre_first_fake", "first_fake_input", "first_fake_loss", "post_backward_after_fsdp",
                         "post_explicit_sp_sync_pre_clip", "clip", "post_clip", "pre_commit", "post_commit"):
            _require(any(r["boundary"] == boundary for r in self.rows), f"missing divergence boundary {boundary}")
        path = self.run_root / "execution/divergence" / f"rank_{self.rank:03d}.json"
        payload = dict(schema_version=SCHEMA, status="CAPTURED_BOUNDED_DIAGNOSTIC_NOT_NUMERICAL_PASS",
                       integration_head=self.source_head, rank=self.rank, run_id=self.run_id, arm=self.arm,
                       candidate_id=self.candidate_id, rows=self.rows, rng=self.rng, fa3_counts=self.counts,
                       helper_sha256=helper_sha256(),
                       optimizer_groups=self.optimizer_groups, first_fake_input_metadata=self.input_metadata,
                       fa3_registration=self.fa3_registration,
                       pre_reduce_parameter_gradients="NOT_OBSERVED_FSDP_REDUCTION_LIVES_INSIDE_BACKWARD",
                       native_timing="NOT_RUN_DIAGNOSTIC_DEVICE_GATHERS_PERTURB_EXECUTION",
                       host_value_reads="ONLY_BEFORE_CYCLE1_OR_AFTER_EXISTING_CYCLE_STOP_FENCES",
                       claim_boundary="SAMPLED_DIFFERENCE_CAN_PROVE_INEQUALITY_NOT_FULL_TENSOR_EQUALITY",
                       formal_timing_eligible=False, numerical_admission=False,
                       limits=dict(blocks=list(BLOCKS), parameter_count=len(PARAMETERS), samples=SAMPLES,
                                   max_rows=MAX_ROWS, max_input_tensors=MAX_INPUT_TENSORS))
        if self.roundtrip is not None:
            payload["cache_roundtrip_audit"] = self.roundtrip.receipt()
            payload["claim_boundary"] += "; full cache payload fingerprints match is not a full bitwise/trajectory equivalence proof"
        if getattr(self, 'raw_repeat_enabled', False):
            from .raw_backward_probe import CONTRACT, finalize
            _require((self.raw_repeat_result is not None) == (self.rank == CONTRACT['rank']), 'raw-repeat actual owner/count missing')
            payload['same_input_raw_backward'] = (finalize(self.raw_repeat_result) if self.raw_repeat_result is not None else
                dict(status='NOT_APPLICABLE', owner_rank=CONTRACT['rank'], actual_extra_calls=0))
            payload['claim_boundary'] += '; raw backward comparisons cover full gradient outputs of one actual first-Fake block'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    status=payload["status"], native_timing=payload["native_timing"], numerical_admission=False)

    def close(self):
        global _ACTIVE
        if _ACTIVE is self:
            _ACTIVE = None
