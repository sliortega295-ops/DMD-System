"""Opt-in, two-cycle Helios adapter over the hardened canonical P0 helpers.

No model/loss/optimizer implementation lives here. Source cohorts are replayed
unchanged while actual G/C/EMA versions advance. Root application observations
are metadata only; tensor reads and correctness sketches are outside timers.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
import math
import platform
import struct
import sys
import time
from typing import Any

from distillschedule.core.enums import OperationKind as Kind
from distillschedule.core.graph import ExecutionGraph
from distillschedule.search.contracts import require
from distillschedule.search.helios_implementation import current_implementation
from distillschedule.search.helios_request import TRACE_KEY, TRACE_HASH_KEY
from distillschedule.workloads.best_known.registry import git_head, repository_root
from .cycle_plan import build_cycle_graph, rollout_window_order
from .role_runner import StaticRoleRunner
from .version_sync import VersionLedger

SCHEMA = "distillschedule.helios_native_two_cycle.v1"
COHORT_POLICY = "fixed_source_cohort_replay"
_COUNTS = ("generator_forward", "critic_forward", "teacher_forward", "generator_backward",
           "critic_backward", "generator_commit", "critic_commit", "generator_optimizer", "critic_optimizer")


def _bf16_timestep(value):
    """Native Torch FP64-to-BF16 projection, including its FP32 intermediate."""
    try:
        bits = struct.unpack(">I", struct.pack(">f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)
    if (bits & 0x7FFFFFFF) > 0x7F800000:
        return math.nan
    # Torch's BF16 constructor takes float and rounds to nearest, ties to even.
    # Dropping binary64 bits directly would disagree near FP32 rounding ties.
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def _versions(generator, critic, ema):
    return dict(generator=int(generator.global_steps), critic=int(critic.global_steps), ema=int(ema.optimization_step))


def _tensor_metadata(value, torch):
    if torch.is_tensor(value):
        return dict(shape=list(value.shape), dtype=str(value.dtype), stride=list(value.stride()),
                    bytes=int(value.numel()*value.element_size()), requires_grad=bool(value.requires_grad))
    if isinstance(value, (list, tuple)):
        return [_tensor_metadata(v, torch) for v in value]
    if isinstance(value, dict):
        return {k: _tensor_metadata(v, torch) for k,v in value.items()
                if torch.is_tensor(v) or isinstance(v, (list, tuple, dict))}
    return None


def _memory(torch, device, *, driver=False):
    cuda = torch.cuda
    value = dict(allocated_bytes=int(cuda.memory_allocated(device)), reserved_bytes=int(cuda.memory_reserved(device)),
                 max_allocated_bytes=int(cuda.max_memory_allocated(device)),
                 max_reserved_bytes=int(cuda.max_memory_reserved(device)))
    if driver:
        free, total = cuda.mem_get_info(device)
        value.update(driver_free_bytes=int(free), driver_total_bytes=int(total))
    return value


def _initial_adapter_state(model):
    """Read actual PEFT layer state outside timing (GAN's first fake call needs it)."""
    states = {not value for module in model.modules()
              if type(value := getattr(module, "disable_adapters", None)) is bool}
    explicit = getattr(model, "adapters_enabled", None)
    if type(explicit) is bool:
        states.add(explicit)
    require(len(states) == 1, "cannot establish a uniform actual PEFT adapter state")
    return states.pop()


def _include_helper_peaks(memory, value):
    """Legacy rollout helpers reset peaks; retain their observed subrange peaks."""
    if memory.get("status") == "NOT_APPLICABLE_CPU_REFERENCE":
        return
    if isinstance(value, dict):
        for key in ("max_allocated", "max_reserved"):
            for suffix, scale in (("_bytes", 1), ("_gib", 2**30)):
                sample = value.get(key + suffix)
                if type(sample) in (float, int):
                    memory[key + "_bytes"] = max(memory[key + "_bytes"], int(sample*scale))
        for child in value.values():
            _include_helper_peaks(memory, child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _include_helper_peaks(memory, child)


def _cycle_graph(impl, steps, versions):
    graph = build_cycle_graph(source_steps=steps, initial_versions=versions,
                              ema_after=(0,) if impl.variant == "gan" else (0,1,2,3,4))
    if impl.request is not None:
        require(tuple(steps) == impl.request.source_steps, "Helios request source cohort differs")
        graph = replace(graph, metadata={**graph.metadata, TRACE_KEY: impl.request.to_dict(),
                                       TRACE_HASH_KEY: impl.request.digest})
    order = rollout_window_order(graph, max_live_rollouts=impl.rollout_window)
    if not (impl.variant == "gan" and impl.rollout_window == 5):
        return graph, order
    # One native helper call owns acquire, ALL five B1 rollouts, and release.
    # Five singleton scopes would not be the same persistence implementation.
    # Its OFF control uses the same composite, never an invented decomposition.
    def name(key):
        return "r_all" if key in {f"r{i}" for i in range(5)} else key
    operations = []
    for op in graph.operations:
        if op.kind == Kind.CRITIC_ROLLOUT:
            if op.operation_id == "r0":
                operations.append(replace(op, operation_id="r_all", metadata={"inputs": (), "source_steps": steps}))
        else:
            operations.append(replace(op, metadata={**op.metadata,
                "inputs": tuple(name(key) for key in op.metadata["inputs"])}))
    edges, seen = [], set()
    for edge in graph.dependencies:
        source, target = name(edge.source), name(edge.target)
        if source != target and (source,target,edge.kind) not in seen:
            seen.add((source,target,edge.kind))
            edges.append(replace(edge, source=source, target=target))
    grouped_order = tuple(dict.fromkeys(name(key) for key in order))
    return ExecutionGraph(tuple(operations), tuple(edges), graph.metadata), grouped_order


class _ObservedTrace:
    """Observe returned entries without bypassing the hardened one-use cache."""
    def __init__(self, trace, observer):
        self._trace, self._observer = trace, observer

    def __getattr__(self, name):
        return getattr(self._trace, name)

    def load_generator_entry(self, **kwargs):
        value = self._trace.load_generator_entry(**kwargs)
        self._observer.entry(value, generator=True, rank=kwargs["rank"],
                             native_task=getattr(self._trace, "task", None))
        return value

    def load_critic_entry(self, **kwargs):
        value = self._trace.load_critic_entry(**kwargs)
        self._observer.entry(value, generator=False, rank=kwargs["rank"],
                             native_task=getattr(self._trace, "task", None))
        return value


class _Observer:
    def __init__(self, *, torch, device, rank, impl, generator, critic, ema, mode, version_offsets,
                 cpu_trace=None):
        self.torch, self.device, self.rank, self.impl = torch, device, rank, impl
        self.generator, self.critic, self.ema, self.mode = generator, critic, ema, mode
        self.version_offsets = dict(version_offsets)
        self.actions, self.losses = [], []
        self.current, self.current_entry = None, None
        self.last_entries = {}
        self.live, self.lifetimes = {}, []
        self.peak_live = 0
        self.in_backward = False
        self._timesteps = []
        self._cpu_trace = cpu_trace
        if mode == "cpu-diagnostic":
            from distillschedule.data.helios_nogan_reference_inputs import cpu_telemetry
            from distillschedule.search.helios_implementation import HeliosImplementation
            require(torch.device(device).type == "cpu" and rank == 0
                    and impl == HeliosImplementation("no-gan", "world16-fsdp", 512, 1, False)
                    and version_offsets == dict(generator=0, critic=0, ema=0)
                    and cpu_telemetry(device, cpu_trace), "explicit admitted W2 CPU diagnostic observer required")
        else:
            require(cpu_trace is None, "CPU trace requires explicit W2 CPU diagnostic mode")

    def memory(self, *, driver=False):
        if self.mode == "cpu-diagnostic":
            return dict(status="NOT_APPLICABLE_CPU_REFERENCE", allocated_bytes=None, reserved_bytes=None,
                        max_allocated_bytes=None, max_reserved_bytes=None,
                        timing="UNAVAILABLE_CPU_DIAGNOSTIC", claimable_speedup=False)
        return _memory(self.torch, self.device, driver=driver)

    def version_snapshot(self):
        physical = _versions(self.generator, self.critic, self.ema)
        logical = {key: value+self.version_offsets[key] for key, value in physical.items()}
        return logical, physical

    def versions(self):
        return self.version_snapshot()[0]

    def entry(self, value, *, generator, rank, native_task=None):
        require(self.current is not None, "trace load outside native action")
        member = value if generator else value.pool_entry
        actual_rank = member.source_id if generator else member.source_trace_rank
        require(int(actual_rank) == self.rank == rank, "rank-qualified trace changed")
        step = int(member.logical_step)
        require(step in self.current["source_steps"], "native action loaded a different source step")
        control = value.operation_trace.operation(member.rollout_operation)["control"]
        item = dict(source_rank=rank, source_step=step, trace_identity_sha256=member.identity_sha256,
                    operation_trace_hash=value.operation_trace.trace_hash,
                    rollout_operation=member.rollout_operation,
                    expected_call_count=int(control["call_count"]),
                    expected_stage_call_count={int(row["stage"]): int(row["call_count"])
                                               for row in control["pyramid_stage_call_sequence"]})
        if self.impl.request is not None:
            self.impl.request.validate_trace_metadata(value.operation_trace)
            item.update({TRACE_HASH_KEY: self.impl.request.digest,
                         "stage_schedule": control["pyramid_stage_call_sequence"],
                         "stage_schedule_policy": self.impl.request.stage_schedule_policy,
                         "all_history_zero": control.get("all_history_zero", False)})
        if getattr(native_task, "native_policy", None) is not None:
            if self.impl.variant == "gan":
                from .helios_gan_native_history import SEMANTICS, require_task
                require_task(native_task)
            else:
                from .helios_native_history import SEMANTICS
            require(self.impl.request is None and self.impl.variant == native_task.request.variant,
                    "native history cannot use a generated benchmark request")
            item.update(native_history_policy=native_task.native_policy, native_history_semantics=SEMANTICS,
                        native_task_sha256=native_task.digest, stage_schedule=control["pyramid_stage_call_sequence"],
                        all_history_zero=control["all_history_zero"])
        self.current["trace_entries"].append(item)
        self.current_entry = item
        self.last_entries[step] = item

    def _patch(self, stack, obj, name, wrapper):
        original = getattr(obj, name)
        # Restore an inherited method by removing the temporary instance member.
        own = name in getattr(obj, "__dict__", {})
        stack.callback(setattr if own else delattr, obj, name, *([original] if own else []))
        setattr(obj, name, wrapper(original))

    @contextmanager
    def installed(self):
        with ExitStack() as stack:
            for role, engine in (("generator", self.generator), ("critic", self.critic)):
                def backward(original, role=role):
                    def call(loss, *args, **kwargs):
                        require(self.current is not None, "backward outside native action")
                        self.current["counts"][f"{role}_backward"] += 1
                        scalar = loss.detach()
                        self.losses.append((self.current, role, scalar))
                        self.in_backward = True
                        try:
                            return original(loss, *args, **kwargs)
                        finally:
                            self.in_backward = False
                    return call
                self._patch(stack, engine, "backward", backward)
                for obj, method, counter in ((engine, "step", f"{role}_commit"),
                                             (engine.optimizer, "step", f"{role}_optimizer")):
                    def count(original, counter=counter):
                        def call(*args, **kwargs):
                            require(self.current is not None, "optimizer outside native action")
                            value = original(*args, **kwargs)
                            self.current["counts"][counter] += 1
                            return value
                        return call
                    self._patch(stack, obj, method, count)
                model = engine.module
                adapter = {"enabled": _initial_adapter_state(model)}
                for method, enabled in (("enable_adapters", True), ("disable_adapters", False)):
                    def track(original, enabled=enabled, adapter=adapter):
                        def call(*args, **kwargs):
                            result = original(*args, **kwargs)
                            adapter["enabled"] = enabled
                            return result
                        return call
                    self._patch(stack, model, method, track)
                pending = []
                def before(module, args, kwargs, role=role, adapter=adapter, pending=pending):
                    require(self.current is not None and self.current_entry is not None,
                            "root model forward has no action-qualified trace entry")
                    require(adapter["enabled"] is not None, "actual root adapter state was not observed")
                    physical_role = "teacher" if role == "critic" and not adapter["enabled"] else role
                    application = not self.in_backward
                    item = dict(role=physical_role, root_module=role, adapter_enabled=adapter["enabled"],
                                application_forward=application, grad_enabled=bool(self.torch.is_grad_enabled()),
                                gan_mode=bool(kwargs.get("gan_mode", False)),
                                versions=self.versions(), trace=dict(self.current_entry),
                                inputs=_tensor_metadata(kwargs if kwargs else args, self.torch))
                    if application and role == "generator" and (self.impl.request is not None
                            or self.current_entry.get("native_history_policy") is not None):
                        timestep = kwargs.get("timestep")
                        require(self.torch.is_tensor(timestep) and timestep.numel() == 1,
                                "Helios request requires an observed B1 model timestep")
                        # Retain tiny device tensors; inspect values only after
                        # the fenced native cycle, never synchronize per call.
                        self._timesteps.append((item, timestep.detach().reshape(-1)[0]))
                    if application:
                        self.current["counts"][f"{physical_role}_forward"] += 1
                    self.current["forwards"].append(item)
                    pending.append(item)
                def after(module, args, kwargs, output, pending=pending):
                    pending.pop()["output"] = _tensor_metadata(output, self.torch)
                stack.callback(model.register_forward_pre_hook(before, with_kwargs=True).remove)
                stack.callback(model.register_forward_hook(after, with_kwargs=True).remove)
            yield self

    @contextmanager
    def action(self, op, *, source_steps):
        require(self.current is None, "nested native actions are not supported")
        logical_versions, physical_versions = self.version_snapshot()
        row = dict(action_id=op.operation_id, kind=op.kind.value, source_steps=list(source_steps), source_rank=self.rank,
                   versions_before=logical_versions, physical_versions_before=physical_versions,
                   counts={key:0 for key in _COUNTS}, forwards=[], trace_entries=[],
                   produced=[], losses=[], calibration_seconds=None, calibration_start_unix_ns=None,
                   calibration_stop_unix_ns=None)
        self.actions.append(row)
        self.current, self.current_entry = row, None
        calibrated = self.mode == "calibration"
        if calibrated:
            self.torch.cuda.synchronize(self.device)
            self.torch.cuda.reset_peak_memory_stats(self.device)
        row["memory_before"] = self.memory(driver=calibrated)
        if calibrated:
            row["calibration_start_unix_ns"] = time.time_ns()
            started = time.perf_counter()
        try:
            yield row
        finally:
            if calibrated:
                self.torch.cuda.synchronize(self.device)
                row["calibration_seconds"] = time.perf_counter() - started
                row["calibration_stop_unix_ns"] = time.time_ns()
            row["versions_after"], row["physical_versions_after"] = self.version_snapshot()
            row["memory_after"] = self.memory(driver=calibrated)
            row["retained_after"] = dict(count=len(self.live), bytes=sum(x["bytes"] for x in self.live.values()))
            self.current, self.current_entry = None, None
        wanted = {**row["versions_before"], **op.produced_versions}
        require(row["versions_after"] == wanted, f"{op.operation_id}: actual native commit versions differ from graph")

    def produced(self, step, value):
        require(step not in self.live, "rollout tensor produced twice")
        metadata = _tensor_metadata(value, self.torch)
        shape = list(self.impl.request.latent_shape) if self.impl.request is not None else [1,16,9,48,80]
        require(metadata["shape"] == shape and metadata["dtype"] == "torch.bfloat16"
                and not metadata["requires_grad"], "retained Helios rollout must be physical B1 BF16 no-grad")
        physical_step = int(self.generator.global_steps)
        item = dict(source_step=step, producer=self.current["action_id"],
                    produced_after_action=len(self.actions)-1, released_after_action=None,
                    generator_version=physical_step+self.version_offsets["generator"],
                    generator_physical_step=physical_step, **metadata)
        self.current["produced"].append(item)
        self.lifetimes.append(item)
        self.live[step] = item
        self.peak_live = max(self.peak_live, len(self.live))
        require(self.peak_live <= self.impl.rollout_window, "native retained-rollout window exceeded")

    def consumed(self, step):
        require(step in self.live, "Critic consumed an unproduced rollout")
        self.live.pop(step)["released_after_action"] = len(self.actions)-1

    def finish(self):
        """Only called after the CUDA-fenced outer cycle wall has stopped."""
        require(not self.live, "native cycle retained unconsumed rollout tensors")
        require(bool(self.losses), "native cycle did not observe loss tensors")
        finite = self.torch.stack([self.torch.isfinite(scalar).all()
                                  for _, _, scalar in self.losses]).cpu().tolist()
        require(all(finite), "native cycle observed a non-finite backward loss")
        for action, role, scalar in self.losses:
            action["losses"].append(dict(role=role, value=float(scalar.float().cpu().item())))
        self.losses.clear()
        if self._timesteps:
            observed = self.torch.stack([value for _, value in self._timesteps]).cpu().tolist()
            for (call, _), value in zip(self._timesteps, observed, strict=True):
                call["timestep"] = float(value)
            self._timesteps.clear()
        for row in self.actions:
            _validate_action(row, self.impl)
        return dict(census="PASS", versions="PASS", finite="PASS", dtype="PASS", sample_rng="PASS",
                    sample_rng_scope="actual_rank_qualified_trace_entries_not_global_rng_state",
                    numerical_equivalence="NOT_RUN")


def _validate_action(row, impl):
    if impl.variant == "gan":
        from .helios_gan_native_history import validate_action as validate_native_history_action
    else:
        from .helios_native_history import validate_action as validate_native_history_action
    validate_native_history_action(row)
    name, counts = row["action_id"], row["counts"]
    expected = {key: 0 for key in _COUNTS}
    if name == "g":
        require(len(row["trace_entries"]) == 1, "Generator trace entry census changed")
        expected.update(generator_forward=row["trace_entries"][0]["expected_call_count"],
                        critic_forward=2 if impl.variant == "gan" else 1, teacher_forward=2,
                        generator_backward=1, generator_commit=1, generator_optimizer=1)
    elif name.startswith("r"):
        expected["generator_forward"] = sum(e["expected_call_count"] for e in row["trace_entries"])
        require(len(row["trace_entries"]) == len(row["source_steps"]), "rollout entry census changed")
    elif name.startswith("c"):
        expected.update(critic_forward=2 if impl.variant == "gan" else 1,
                        critic_backward=2 if impl.variant == "gan" else 1, critic_commit=1, critic_optimizer=1)
        require(len(row["trace_entries"]) == 1, "Critic trace entry census changed")
    require([entry["source_step"] for entry in row["trace_entries"]] == row["source_steps"],
            "actual action source step identity/order changed")
    require(counts == expected, f"{name}: actual root/backward/optimizer counts differ: {counts} != {expected}")
    if impl.request is not None:
        request=impl.request
        for entry in row['trace_entries']:
            require(entry.get(TRACE_HASH_KEY)==request.digest
                    and entry.get('stage_schedule_policy')==request.stage_schedule_policy
                    and type(entry.get('all_history_zero')) is bool,
                    'Helios action request/schedule binding differs')
            zero=entry['all_history_zero']
            require(not zero or request.stage_schedule_policy=='benchmark_v4_strict_global',
                    'normal Helios action cannot use amplified native history')
            schedule=entry.get('stage_schedule',[])
            require(len(schedule)==3 and [s.get('stage') for s in schedule]==[0,1,2],
                    'Helios action stage inventory differs')
            for index,item in enumerate(schedule):
                horizon=request.stage2_inference_steps[index];budget=horizon*(2 if zero else 1)
                count=item.get('call_count');grid=item.get('timestep_grid',[])
                require(item.get('latent_shape')==[*request.latent_shape[:3],*request.stage_spatial_shapes[index]]
                        and type(count) is int and 1<=count<=horizon
                        and item.get('denoising_exit')==count-1
                        and count==entry['expected_stage_call_count'].get(index,
                            entry['expected_stage_call_count'].get(str(index)))
                        and item.get('denoising_step_budget')==budget
                        and len(grid)==budget and all(type(v) in (int,float) and math.isfinite(v) for v in grid)
                        and item.get('timestep')==grid[:count],
                        'Helios action native geometry/horizon/grid differs from request')
            require(sum(s['call_count'] for s in schedule)==entry['expected_call_count'],
                    'Helios action stage/total call count differs')
    gan_calls = []
    for call in row["forwards"]:
        if not call["application_forward"]:
            continue
        hidden = call["inputs"].get("hidden_states")
        packed_gan_d = impl.variant == "gan" and name.startswith("c") and call["gan_mode"]
        if packed_gan_d:
            # Logical B1 GAN-D is an official physical fake/real/R1 B3 pack,
            # spatially cropped by 0.5. This is NOT a searched batch increase.
            shape = list(impl.request.gan_d_shape) if impl.request is not None else [3,16,9,24,40]
            require(call["role"] == "critic" and hidden["shape"] == shape,
                    "official GAN-D fake/real/R1 packed input shape changed")
        else:
            require(isinstance(hidden, dict) and hidden["shape"][0] == 1,
                    "actual root forward violated the logical B1 physical call contract")
        require(hidden["dtype"] == "torch.bfloat16", "actual root forward violated BF16")
        if impl.request is not None and call["root_module"] == "critic" and not packed_gan_d:
            shape = list(impl.request.latent_shape)
            if call["gan_mode"]:
                shape[-2:] = [s//2 for s in shape[-2:]]
            require(hidden["shape"] == shape, "actual Helios Score/Teacher geometry differs from request")
        if call["root_module"] == "critic":
            gan_calls.append(call["gan_mode"])
    if impl.variant == "gan":
        expected_modes = [False, False, False, True] if name == "g" else [False, True] if name.startswith("c") else []
        require(gan_calls == expected_modes, "actual GAN root forward mode/order changed")
    records = [row["helper_record"]["rollout"]] if name == "g" else row.get("rollout_records", [])
    if impl.candidate_bundle.enabled:
        for record in records:
            require(record.get("synchronize_after_rollout") is False,
                    "safe No-GAN actual rollout fence request differs")
            if name == "g":
                require(record.get("empty_cache_before_rollout") is False,
                        "safe No-GAN actual Generator empty-cache request differs")
    if "helper_record" in row:
        require(row["helper_record"].get("source_rank") == row["source_rank"]
                and row["helper_record"].get("logical_step") == row["source_steps"][0],
                "helper source rank/step identity changed")
    for entry, record in zip(row["trace_entries"], records, strict=bool(records)):
        if impl.request is not None:
            require(entry.get(TRACE_HASH_KEY) == impl.request.digest,
                    "actual Helios trace request binding differs")
            require(record.get("output_shape") == list(impl.request.latent_shape),
                    "actual Helios helper output geometry differs from request")
        identities = [record.get("trace_identity_sha256")] if name == "g" else record.get("member_trace_identities")
        require(identities == [entry["trace_identity_sha256"]], "helper member trace identity changed")
        if name != "g":
            require(record.get("logical_steps") == [entry["source_step"]] and record.get("batch_size") == 1,
                    "helper rollout source step or physical B1 changed")
        require(record["call_count"] == entry["expected_call_count"], "native helper vs trace rollout count differs")
        require({int(k): v for k,v in record["stage_call_count"].items()} ==
                {int(k): v for k,v in entry["expected_stage_call_count"].items()},
                "native helper vs trace stage count differs")
        calls = [x for x in row["forwards"] if x["application_forward"] and x["root_module"] == "generator"
                 and x["trace"]["rollout_operation"] == entry["rollout_operation"]]
        require(len(calls) == record["call_count"], "actual root forwards differ from helper rollout records")
        if impl.request is not None or entry.get("native_history_policy") is not None:
            schedule = entry.get("stage_schedule", [])
            expected_calls = [(item["latent_shape"], timestep)
                              for item in schedule for timestep in item["timestep"]]
            require(len(expected_calls) == len(calls), "Helios request native stage inventory differs")
            # The native FP64 schedule is checked before model construction by
            # validate_native_stage_grids. FSDP2's fixed BF16 input policy casts
            # that schedule before this observer's root hook runs. Project only
            # the audit expectation, outside timers, never the scientific grid.
            model_timesteps = [_bf16_timestep(t) for _, t in expected_calls]
            for call, (shape, _), timestep in zip(calls, expected_calls, model_timesteps, strict=True):
                require(call["inputs"]["hidden_states"]["shape"] == shape,
                        "actual Helios pyramid geometry differs from request trace")
                metadata = call["inputs"].get("timestep")
                require(isinstance(metadata, dict) and metadata.get("shape") == [1]
                        and metadata.get("dtype") == "torch.bfloat16",
                        "actual Helios model timestep must be native B1 BF16")
                require(type(call.get("timestep")) in (float, int)
                        and call["timestep"] == timestep,
                        "actual Helios model timestep differs from BF16-projected per-rank request grid")
    if impl.request is not None:
        require(all(entry.get(TRACE_HASH_KEY) == impl.request.digest for entry in row["trace_entries"]),
                "Helios action uses another request trace")
        require(all(value["shape"] == list(impl.request.latent_shape) for value in row["produced"]),
                "Helios produced metadata differs from request")


def _prepare_cycle(*, observer, p0, args, policy, steps, trace, negative, cfg, diagnostics,
                   generator_free_memory):
    if observer.mode == "cpu-diagnostic":
        from distillschedule.data.helios_nogan_reference_inputs import cpu_telemetry
        require(trace is observer._cpu_trace and tuple(steps) == trace.task.source_steps
                and cpu_telemetry(observer.device, trace), "W2 input-task source steps changed")
    graph, order = _cycle_graph(observer.impl, steps, observer.versions())
    observed_trace = _ObservedTrace(trace, observer)
    common = dict(rank=observer.rank, device=observer.device, trace=observed_trace,
                  collect_diagnostics=(not args.timing_mode) and args.profile_diagnostics, diagnostics=diagnostics)
    def generator_scope(*, with_grad, action_id):
        task = getattr(trace, "task", None)
        if getattr(task, "native_policy", None) is None:
            return nullcontext((observer.generator, None))
        if observer.impl.variant == "gan":
            from .helios_gan_native_history import generator_phase, require_task
            from distillschedule.training.helios_gan_training import implementation
            require_task(task)
            require(observer.impl == implementation() and policy.rollout_batch_size == 1
                    and policy.generator_nograd_persistent_unshard is True,
                    "native GAN history requires unchanged full world16/window5 bundle")
            if not with_grad:
                require(action_id == "r_all", "native GAN history requires original detached window5 owner")
                return nullcontext((observer.generator, None))
            return generator_phase(observer.generator, task=task, with_grad=with_grad, action_id=action_id)
        from .helios_native_history import generator_phase
        require(observer.impl.variant == "no-gan" and observer.impl.request is None
                and not observer.impl.candidate_bundle.enabled and observer.impl.mesh == "world16-fsdp"
                and observer.impl.rollout_window == policy.rollout_batch_size == 1
                and not policy.generator_nograd_persistent_unshard,
                "native history phase requires its original reference dispatch")
        return generator_phase(observer.generator, task=task, with_grad=with_grad, action_id=action_id)
    def gupdate(ctx):
        with observer.action(ctx.operation, source_steps=(steps[0],)) as row:
            with generator_scope(with_grad=True, action_id=ctx.operation.operation_id) as (engine, phase):
                if phase is not None:
                    row["native_history_phase"] = phase
                if observer.impl.variant == "gan":
                    result = p0.base._run_gan_generator_update(**common, args=args, generator_engine=engine,
                        critic_engine=observer.critic, cfg=cfg, negative=negative,
                        generator_free_memory_receipt=generator_free_memory,
                        elide_official_generator_free_memory=bool(policy.elide_official_generator_free_memory
                                                                  and (args.timing_mode or args.profile_mode)))
                else:
                    result = p0._run_nogan_generator_update_p0(**common, generator_engine=engine,
                        critic_engine=observer.critic, negative=negative, policy=policy,
                        score_pack_parity_check=(not args.timing_mode) and (not args.profile_mode))
            row["helper_record"] = result
        _include_helper_peaks(row["memory_after"], result)
        return result
    def rollout(ctx):
        selected = steps if ctx.operation.operation_id == "r_all" else (ctx.operation.logical_step,)
        with observer.action(ctx.operation, source_steps=selected) as row:
            with generator_scope(with_grad=False, action_id=ctx.operation.operation_id) as (engine, phase):
                if phase is not None:
                    row["native_history_phase"] = phase
                outputs, records, persistence = p0._precompute_critic_rollouts(**common, steps=selected,
                    generator_engine=engine, policy=policy,
                    rollout_parity_check=(not args.timing_mode) and (not args.profile_mode))
            require(set(outputs) == set(selected) and len(records) == len(selected), "native B1 rollout inventory changed")
            require(persistence.get("enabled") is observer.impl.persistent_unshard and persistence.get("restored") is True,
                    "native persistence scope did not restore its original policy")
            if observer.impl.persistent_unshard:
                require(persistence.get("rollout_batch_count") == 5 and persistence.get("set_no_reshard_count") == 41
                        and persistence.get("explicit_reshard_count") == 41
                        and persistence.get("unsharded_after_exit_count") == 0,
                        "persistent r_all did not observe full acquire/five-rollout/release scope")
            row.update(rollout_records=records, persistence=persistence,
                       persistence_timer_scope="acquire_all_selected_B1_rollouts_release")
            for step in selected:
                observer.produced(step, outputs[step])
        _include_helper_peaks(row["memory_after"], records)
        _include_helper_peaks(row["memory_after"], persistence)
        return outputs
    def critic(ctx):
        step = ctx.operation.logical_step
        # Pop promptly even for grouped r_all: remaining latents retain only
        # their own last-consumer lifetimes, not the whole scope's lifetime.
        x0 = ctx.values[ctx.operation.metadata["inputs"][0]].pop(step)
        with observer.action(ctx.operation, source_steps=(step,)) as row:
            kwargs = dict(**common, step=step, x0=x0, critic_engine=observer.critic)
            if observer.impl.variant == "gan":
                result = p0._run_gan_critic_precomputed(**kwargs, cfg=cfg,
                    collect_official_scalar_logs=not (policy.elide_critic_scalar_logs and (args.timing_mode or args.profile_mode)))
            else:
                result = p0._run_nogan_critic_precomputed(**kwargs)
            row["helper_record"] = result
            del x0, kwargs
            observer.consumed(step)
        return result
    def ema_update(ctx):
        with observer.action(ctx.operation, source_steps=(ctx.operation.logical_step,)) as row:
            # Bind EMA to the actual just-consumed C entry, not another load.
            row["trace_entries"].append(dict(observer.last_entries[ctx.operation.logical_step]))
            decay = observer.ema.step()
            diagnostics.add_bytes("ema_shadow", int(getattr(observer.ema, "last_transfer_bytes", 0)),
                direction="D2H" if args.ema_device == "cpu" else "GPU", source="fsdp2_ema_local_shard")
            row["ema"] = dict(decay=decay, optimization_step=int(observer.ema.optimization_step))
    runner = StaticRoleRunner("native_runtime", {
        Kind.GENERATOR_UPDATE.value: gupdate, Kind.CRITIC_ROLLOUT.value: rollout,
        Kind.CRITIC_UPDATE.value: critic, Kind.EMA_UPDATE.value: ema_update,
    }, versions=VersionLedger(observer.versions()))
    return runner.prepare(graph, order=order), graph, order


def _correctness_sketch(torch, generator, critic, ema):
    from distillschedule.evidence.trajectory_state_sketch import collect_numeric_state_sketch
    values = []
    for role, engine in (("generator", generator), ("critic", critic)):
        for name, parameter in engine.module.named_parameters():
            if not parameter.requires_grad:
                continue
            values.append((f"{role}/parameter/{name}", parameter, None))
            for key in ("exp_avg", "exp_avg_sq"):
                value = engine.optimizer.state.get(parameter, {}).get(key)
                if value is not None:
                    values.append((f"{role}/optimizer/{name}/{key}", value, parameter))
    for name, shadow, parameter in zip(ema._names, ema._shadows, ema._parameters, strict=True):
        values.append((f"ema/{name}", shadow, parameter))
    return collect_numeric_state_sketch(torch=torch, named_values=values, samples_per_tensor=16,
                                        max_tensors=96, verify_all_replicas=True)


def run_two_cycles(*, p0: Any, args: Any, policy: Any, result: dict[str, Any], output: Any,
                   rank: int, device: Any, logical: Any, trace: Any, negative: Any, cfg: Any,
                   generator_engine: Any, critic_engine: Any, ema: Any, diagnostics: Any,
                   generator_free_memory: dict[str, Any]) -> dict[str, Any]:
    """Mutate the caller's rank receipt; the hardened caller owns atomic write.

    All modes use the same timing-mode native helpers. Calibration adds explicit
    per-action host fences; correctness adds only outside-boundary state probes.
    Native admission and paired numerical comparison remain controller-owned.
    """
    impl = current_implementation()
    require(impl is not None, "native Helios bridge is default-off")
    binding = impl.receipt_binding(args=args, policy=policy)
    if impl.request is not None:
        require(getattr(trace, "request", None) == impl.request, "Helios runtime trace request differs")
        impl.request.validate_rank_controls(trace._control)
        if impl.variant == "gan": impl.request.validate_official_config(cfg)
    bundle_before = (impl.candidate_bundle.effect_snapshot(result.get("build", {}))
                     if impl.candidate_bundle.enabled else None)
    mode, torch = binding["mode"], p0.torch
    canonical = impl.lower_profile()["contract"]
    mapping = impl.version_mapping(canonical)
    physical_initial = _versions(generator_engine, critic_engine, ema)
    require(physical_initial == mapping["physical_initial"],
            "native physical G/C/EMA steps differ from canonical checkpoint")
    require(all(int(logical.component(key)["optimizer_step"]) == physical_initial[key]
                for key in ("generator", "critic")),
            "native engine physical steps differ from loaded logical checkpoint")
    if impl.variant == "gan":
        control = logical.manifest.get("systems_only_control", {})
        require(control.get("classification") == "systems_only_gan_bootstrap"
                and control.get("logical_schedule_versions") == mapping["logical_initial"]
                and control.get("logical_control_window") == canonical["logical_steps"]
                and control.get("official_checkpoint_1000") is False,
                "GAN systems-only logical version mapping differs from loaded checkpoint")
    initial = dict(mapping["logical_initial"])
    steps = tuple(trace.logical_steps)
    require(steps == tuple(canonical["logical_steps"]), "native source cohort must remain fixed")
    require(result.get("world_size") == 16 and result.get("rank") == rank, "Helios requires rank-qualified world16")
    require(generator_engine.module is not critic_engine.module, "Helios Generator and Score must remain separate roots")
    for engine in (generator_engine, critic_engine):
        if hasattr(engine, "train_batch_size"):
            require(engine.train_batch_size() == 16 and engine.train_micro_batch_size_per_gpu() == 1,
                    "actual engine batch contract differs from B1/B16")
    result.update(schema_version=SCHEMA, mode=mode, profiled=False, profiler_status="DISABLED",
                  source_head=git_head(repository_root()),
                  runtime_identity=dict(python_executable=sys.executable, python_version=platform.python_version(),
                      torch_version=getattr(torch, "__version__", None),
                      cuda_version=getattr(getattr(torch, "version", None), "cuda", None)),
                  candidate_id=impl.id, planner_binding=binding, source_cohort_policy=COHORT_POLICY,
                  source_steps=list(steps), same_process=True, initializations=1, measured_cycle=2,
                  timer_scope="NATIVE_CYCLE_OUTSIDE_PROBES", cycles=[], initial_versions=initial,
                  version_mapping=mapping, initial_physical_versions=physical_initial,
                  numerical_status="NOT_RUN", numerical_evidence_scope="diagnostic_sampled_state_not_strict_parity",
                  post_setup_memory=_memory(torch, device, driver=True), outside_timer_probes=[])
    if mode == "correctness":
        started = time.perf_counter()
        result["outside_timer_probes"].append(dict(boundary="initial", state=_correctness_sketch(
            torch, generator_engine, critic_engine, ema), seconds=time.perf_counter()-started))
    for cycle in (1, 2):
        cycle_bundle_before = (impl.candidate_bundle.effect_snapshot(result.get("build", {}))
                               if impl.candidate_bundle.enabled else None)
        observer = _Observer(torch=torch, device=device, rank=rank, impl=impl, generator=generator_engine,
                             critic=critic_engine, ema=ema, mode=mode,
                             version_offsets=mapping["logical_minus_physical"])
        prepared, graph, order = _prepare_cycle(observer=observer, p0=p0, args=args, policy=policy, steps=steps,
            trace=trace, negative=negative, cfg=cfg, diagnostics=diagnostics, generator_free_memory=generator_free_memory)
        logical_versions, physical_versions = observer.version_snapshot()
        row = dict(cycle=cycle, measured=cycle == 2, source_steps=list(steps), source_cohort_policy=COHORT_POLICY,
                   initial_versions=logical_versions, initial_physical_versions=physical_versions,
                   graph=graph.to_dict(), selected_order=list(order),
                   actions=observer.actions, status="RUNNING")
        result["cycles"].append(row)
        with observer.installed():
            p0.dist.barrier()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            row["start_unix_ns"] = time.time_ns()
            started = time.perf_counter()
            prepared.run()
            torch.cuda.synchronize(device)
            row["wall_seconds"] = time.perf_counter()-started
            row["stop_unix_ns"] = time.time_ns()
        logical_versions, physical_versions = observer.version_snapshot()
        row.update(final_versions=logical_versions, final_physical_versions=physical_versions,
                   dispatch=prepared.receipt(),
                   peak_live_rollouts=observer.peak_live, retained_lifetimes=observer.lifetimes,
                   memory=_memory(torch, device, driver=True))
        _include_helper_peaks(row["memory"], observer.actions)
        probe_started = time.perf_counter()
        row["semantic_validation"] = observer.finish()
        require(row["final_versions"] == graph.metadata["final_versions"], "native final versions differ from graph")
        if mode == "correctness":
            row["state_sketch"] = _correctness_sketch(torch, generator_engine, critic_engine, ema)
        row["outside_timer_probe_seconds"] = time.perf_counter()-probe_started
        if impl.candidate_bundle.enabled:
            row["candidate_bundle_effects"] = impl.candidate_bundle.executed_effects(
                result.get("build", {}), cycle_bundle_before)
        row["status"] = "PASS"
    final, physical_final = observer.version_snapshot()
    require(final == canonical["final_versions"], "native two-cycle final versions changed")
    if hasattr(trace, "receipt"):
        result["input_reuse"] = trace.receipt()
        require(result["input_reuse"].get("critic_entries_still_cached") == 0,
                "hardened trace cache retained an unconsumed input")
    if impl.candidate_bundle.enabled:
        # Counter reads occur outside both native timers, and exclude warmup.
        result["candidate_bundle_effects"] = impl.candidate_bundle.executed_effects(
            result.get("build", {}), bundle_before)
    result.update(status="PASS", semantics_status="PASS", final_versions=final,
                  final_physical_versions=physical_final,
                  cycle_wall_seconds={str(row["cycle"]): row["wall_seconds"] for row in result["cycles"]},
                  updates={key: final[key]-initial[key] for key in final},
                  final=dict(generator_physical_step=physical_final["generator"],
                             critic_physical_step=physical_final["critic"],
                             generator_logical_version=final["generator"], critic_logical_version=final["critic"],
                             ema_step=final["ema"], memory=_memory(torch, device, driver=True)))
    return result
