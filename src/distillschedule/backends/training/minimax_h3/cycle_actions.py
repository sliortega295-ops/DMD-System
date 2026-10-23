"""Opt-in H3 lowering to the common prepared runner, not another trainer.

An update is atomic through the native optimizer/LR step. Prepared rollouts
retain their original objects until their own consumer. Failures are one-shot:
a diagnostic failure cannot roll back an already completed optimizer update.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy
import hashlib
import pickle
import random
import re
import sys
from time import perf_counter_ns

from distillschedule.core.enums import OperationKind
from distillschedule.runtime.cycle_plan import build_cycle_graph, rollout_window_order
from distillschedule.runtime.role_runner import StaticRoleRunner
from distillschedule.runtime.version_sync import VersionLedger
from distillschedule.search.contracts import require


def versions(updates):
    return {"generator": updates["student"], "critic": updates["fake"], "ema": updates["ema"]}


def prepare_cycle(trainer, *, samples, grad_accum_iters, current_iter, rollout_window, trace):
    require(grad_accum_iters == 1 and trainer.fake_update_ratio == 5,
            "H3 prepared cycle requires native B1 Student1/Fake5")
    observer = trainer.planner_cycle_observer
    initial = dict(generator=current_iter, critic=5 * current_iter, ema=0)
    require(versions(observer.updates) == initial, "H3 native initial updates differ from plan")
    graph = build_cycle_graph(source_steps=tuple(5 * current_iter + i for i in range(5)),
                              initial_versions=initial, ema_after=())
    # These are logical Fake-step IDs, not H3's native outer iteration number.
    order = rollout_window_order(graph, max_live_rollouts=rollout_window)

    def calibration(ctx):
        collector = getattr(observer, "collector", None)
        return (collector.stage(current_iter+1, ctx.operation.operation_id)
                if collector is not None else nullcontext())

    def student(ctx):
        with trace.scope(ctx, trainer), calibration(ctx), trainer._nvtx("h3/student_step"):
            return trainer._student_step(samples, grad_accum_iters, current_iter)

    def prepare(ctx):
        with trace.scope(ctx, trainer), calibration(ctx) as sample, trainer._nvtx("h3/critic_prepare_one"):
            result = trainer._prepare_one_fake_group(samples, grad_accum_iters, ctx.operation.metadata["fake_index"])
        if sample is not None:
            sample.record_output(result)  # Storage metadata outside calibrated range.
        return result

    def update(ctx):
        with trace.scope(ctx, trainer), calibration(ctx), trainer._nvtx("h3/critic_update_F"):
            return trainer._apply_one_fake_group(ctx.values[ctx.operation.metadata["inputs"][0]])

    return StaticRoleRunner("native_runtime", {
        OperationKind.GENERATOR_UPDATE.value: student,
        OperationKind.CRITIC_ROLLOUT.value: prepare,
        OperationKind.CRITIC_UPDATE.value: update,
    }, versions=VersionLedger(initial)).prepare(
        graph, order=order, retain_outputs=("g", "c0", "c1", "c2", "c3", "c4"))


def torch_rng_snapshot(torch, device):
    """Observed generators only; never seed/reset/replace a generator."""
    result = {name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in (
        ("torch_cpu", torch.get_rng_state()), ("torch_cuda", torch.cuda.get_rng_state(device)))}
    result["python_random"] = hashlib.sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest()
    if "numpy" in sys.modules:
        result["numpy_legacy"] = hashlib.sha256(pickle.dumps(sys.modules["numpy"].random.get_state(), protocol=4)).hexdigest()
    return result


def tensor_evidence(torch, value, *, value_mode="full"):
    """Full actual draw digest, correctness ONLY; host reads can synchronize.

    No replacement draws, sampled value sketch, or retained GPU copy. Canonical
    execution and all admitted timing must not enable this observer.
    """
    values = value if isinstance(value, (tuple, list)) else (value,)
    rows = []
    require(value_mode in ("full", "metadata"), "unknown H3 draw evidence mode")
    for tensor in values:
        require(torch.is_tensor(tensor), "H3 draw site returned a non-tensor")
        metadata = dict(shape=list(tensor.shape), dtype=str(tensor.dtype), device=str(tensor.device),
                        bytes=int(tensor.numel() * tensor.element_size()))
        if value_mode == "metadata":
            rows.append(metadata)
            continue
        local = tensor.detach().contiguous().cpu()
        finite = bool(torch.isfinite(local).all().item())
        require(finite, "H3 stochastic input contains nonfinite values")
        rows.append(dict(**metadata, finite=True,
            sha256=hashlib.sha256(local.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()))
    return rows


class H3ActionTrace:
    """Action-owned samples/draws and update neutrality; not a service-cost clock."""

    def __init__(self, *, rank, source_head, rng_snapshot, value_mode="full"):
        require(value_mode in ("full", "metadata"), "unknown H3 draw evidence mode")
        self.value_mode = value_mode
        self.rank, self.source_head, self.rng_snapshot = rank, source_head, rng_snapshot
        self.rows, self.active, self.cycle = [], None, None
        self._originals = {}

    def install_draw_hooks(self, trainer, torch):
        require(not self._originals, "H3 draw observer already installed")
        for method, site in (("sample_initial_latents", "initial_latents"),
                             ("_sample_renoise_sigmas", "renoise_sigmas"),
                             ("_randn_like_exact", "noise_like")):
            original = getattr(trainer, method)
            self._originals[method] = (method in trainer.__dict__, trainer.__dict__.get(method))

            def observed(*args, _original=original, _site=site, **kwargs):
                require(self.active is not None, f"H3 draw {_site} outside prepared action")
                result = _original(*args, **kwargs)
                self.active["draws"].append(dict(site=_site, tensors=tensor_evidence(torch, result, value_mode=self.value_mode)))
                return result

            setattr(trainer, method, observed)

    def close(self, trainer):
        for method, (had_local, original) in self._originals.items():
            if had_local:
                setattr(trainer, method, original)
            else:
                delattr(trainer, method)
        self._originals.clear()

    def begin_cycle(self, cycle):
        require(self.active is None and cycle in (1, 2)
                and cycle == (1 if self.cycle is None else self.cycle + 1), "H3 action cycle order changed")
        self.cycle, self.rows = cycle, []

    @contextmanager
    def scope(self, ctx, trainer):
        require(self.active is None, "nested H3 actions are not supported")
        key = ctx.operation.operation_id
        start_samples = len(trainer.matched_cycle_census.samples)
        row = dict(action=key, cycle=self.cycle, rank=self.rank, logical_fake_step=ctx.operation.logical_step,
                   inputs=list(ctx.operation.metadata["inputs"]),
                   draws=[], status="RUNNING", rng_before=self.rng_snapshot(),
                   actual_updates_before=dict(trainer.planner_cycle_observer.updates),
                   host_start_ns=perf_counter_ns())
        self.rows.append(row)
        self.active = row
        try:
            yield
            row["rng_after"] = self.rng_snapshot()
            if key.startswith("c") and row["rng_before"] != row["rng_after"]:
                raise RuntimeError(f"H3 RNG advanced inside {key}; reordering is not admitted")
            row["status"] = "PASS"
        except BaseException as error:
            row.update(status="FAIL", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            row["host_stop_ns"] = perf_counter_ns()
            row["samples"] = deepcopy(trainer.matched_cycle_census.samples[start_samples:])
            row["actual_updates_after"] = dict(trainer.planner_cycle_observer.updates)
            self.active = None

    def finish_cycle(self, dispatch, updates, census, *, require_draws=True):
        value = dict(schema_version="distillschedule.h3_cycle_actions.v1", source_head=self.source_head,
            rank=self.rank, cycle=self.cycle, dispatch=dispatch, actual_versions=versions(updates),
            rows=deepcopy(self.rows), status="CAPTURED", cost_admission=False,
            tensor_value_reads=("CORRECTNESS_ONLY_SYNCHRONIZING_FULL_DRAW_HASHES" if self.value_mode == "full"
                                else "DISABLED_METADATA_ONLY_PRIOR_CORRECTNESS_REQUIRED"),
            rng_scope="TORCH_DEFAULT_CPU_RANK_CUDA_PYTHON_RANDOM_NUMPY_LEGACY; CUSTOM_GENERATORS_UNOBSERVED",
            timer_eligible=False, census_binding=deepcopy(census))
        validate_cycle_actions(value, require_draws=require_draws)
        value["status"] = (("PASS_NATIVE_ACTION_BINDING" if self.value_mode == "full"
                            else "PASS_NATIVE_METADATA_ACTION_BINDING_VALUES_NOT_RECHECKED")
                           if require_draws else "PASS_CPU_FIXTURE_DISPATCH_ONLY")
        return value


def validate_cycle_actions(value, *, require_draws=True):
    full_values = value.get("tensor_value_reads") == "CORRECTNESS_ONLY_SYNCHRONIZING_FULL_DRAW_HASHES"
    require(full_values or value.get("tensor_value_reads") == "DISABLED_METADATA_ONLY_PRIOR_CORRECTNESS_REQUIRED",
            "H3 draw evidence mode missing")
    cycle, rows, dispatch = value["cycle"], value["rows"], value["dispatch"]
    require(cycle in (1, 2) and dispatch["status"] == "PASS", "H3 action execution incomplete")
    initial = dict(generator=cycle-1, critic=5*(cycle-1), ema=0)
    final = dict(generator=cycle, critic=5*cycle, ema=0)
    require(dispatch["initial_versions"] == initial
            and dispatch["final_versions"] == value["actual_versions"] == final, "H3 actual updates mismatch")
    window = dispatch["peak_live_inputs"]
    graph = build_cycle_graph(source_steps=tuple(5*(cycle-1)+i for i in range(5)),
                              initial_versions=initial, ema_after=())
    order = list(rollout_window_order(graph, max_live_rollouts=window))
    require([r["action"] for r in rows] == dispatch["executed_operations"] == order,
            "H3 action order/retained window mismatch")
    samples = []
    expected_updates = dict(student=cycle-1, fake=5*(cycle-1), ema=0)
    previous_rng = None
    operations = {op.operation_id: op for op in graph.operations}
    for row in rows:
        key = row["action"]
        require(row["status"] == "PASS" and row["cycle"] == cycle and row["rank"] == value["rank"],
                "H3 action row identity/status mismatch")
        require(row["logical_fake_step"] == operations[key].logical_step
                and row["inputs"] == list(operations[key].metadata["inputs"]),
                "H3 action input/step binding differs")
        require(row["rng_before"] and row["rng_after"]
                and (previous_rng is None or row["rng_before"] == previous_rng), "H3 action RNG chain differs")
        if require_draws:
            for state in (row["rng_before"], row["rng_after"]):
                require(isinstance(state, dict) and set(state) in (
                        {"torch_cpu", "torch_cuda", "python_random"},
                        {"torch_cpu", "torch_cuda", "python_random", "numpy_legacy"})
                        and all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in state.values()),
                        "H3 action generator identity/hash missing")
        previous_rng = row["rng_after"]
        require(row["actual_updates_before"] == expected_updates, "H3 native updates before action differ")
        if key == "g":
            expected_updates["student"] += 1
        elif key.startswith("c"):
            expected_updates["fake"] += 1
        require(row["actual_updates_after"] == expected_updates, "H3 native updates after action differ")
        if key.startswith("c"):
            require(not row["draws"] and not row["samples"] and row["rng_before"] == row["rng_after"],
                    "H3 Critic action consumed sample/RNG")
        else:
            expected_stage = "student" if key == "g" else f"fake_{int(key[1:])}"
            require(len(row["samples"]) == 1 and row["samples"][0]["stage"] == expected_stage,
                    "H3 action-owned sample differs")
            samples.extend(row["samples"])
            if require_draws:
                require([d["site"] for d in row["draws"]] ==
                        ["initial_latents", "renoise_sigmas", "noise_like", "noise_like"],
                        "H3 native draw inventory differs")
                for draw, count in zip(row["draws"], (2, 2, 1, 1)):
                    require(len(draw["tensors"]) == count, "H3 video/audio draw pair missing")
                    for tensor in draw["tensors"]:
                        require(tensor.get("bytes", 0) > 0
                                and isinstance(tensor.get("shape"), list)
                                and all(type(n) is int and n > 0 for n in tensor["shape"])
                                and tensor.get("dtype", "").startswith("torch.")
                                and tensor.get("device", "").startswith("cuda:"),
                                "H3 actual draw digest/finite evidence missing")
                        if full_values:
                            require(tensor.get("finite") is True and
                                    re.fullmatch(r"[0-9a-f]{64}", tensor.get("sha256", "")) is not None,
                                    "H3 full draw digest/finite evidence missing")
                        else:
                            require("finite" not in tensor and "sha256" not in tensor,
                                    "H3 metadata-only observation cannot claim tensor value proof")
    require(samples == value["census_binding"]["samples"], "H3 action/native sample census differs")


def compare_action_inputs(left, right):
    """Join by native action identity, not the reordered physical row position."""
    try:
        for value in (left, right):
            require(value.get("tensor_value_reads") == "CORRECTNESS_ONLY_SYNCHRONIZING_FULL_DRAW_HASHES",
                    "metadata-only action receipts cannot prove value parity")
            validate_cycle_actions(value)
        require(all(left[k] == right[k] for k in ("rank", "cycle", "source_head")),
                "H3 action comparison identity changed")
        a, b = ({row["action"]: row for row in value["rows"]} for value in (left, right))
        mismatches = []
        for key in a:
            # Updates can move relative to draws. Their pre/post RNG states
            # need only be neutral individually; r_i and g must match exactly.
            fields = ("samples", "draws", "inputs", "logical_fake_step")
            if not key.startswith("c"):
                fields += ("rng_before", "rng_after")
            mismatches.extend(f"{key}/{field}" for field in fields if a[key][field] != b[key][field])
        return dict(status="FAIL" if mismatches else "PASS", mismatches=mismatches,
                    scope="FULL_NATIVE_STOCHASTIC_DRAW_DIGESTS_AND_OPERATION_QUALIFIED_GENERATORS")
    except (ValueError, KeyError, TypeError) as error:
        return dict(status="FAIL", reason=str(error))
