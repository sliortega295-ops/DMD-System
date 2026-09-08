"""Opt-in, two-cycle Helios adapter over the hardened canonical P0 helpers.

No model/loss/optimizer implementation lives here. Source cohorts are replayed
unchanged while actual G/C/EMA versions advance. Root application observations
are metadata only; tensor reads and correctness sketches are outside timers.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import replace
import platform
import sys
import time
from typing import Any

from distillschedule.core.enums import OperationKind as Kind
from distillschedule.core.graph import ExecutionGraph
from distillschedule.search.contracts import require
from distillschedule.search.helios_implementation import current_implementation
from distillschedule.workloads.best_known.registry import git_head, repository_root
from .cycle_plan import build_cycle_graph, rollout_window_order
from .role_runner import StaticRoleRunner
from .version_sync import VersionLedger

SCHEMA = "distillschedule.helios_native_two_cycle.v1"
COHORT_POLICY = "fixed_source_cohort_replay"
_COUNTS = ("generator_forward", "critic_forward", "teacher_forward", "generator_backward",
           "critic_backward", "generator_commit", "critic_commit", "generator_optimizer", "critic_optimizer")


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
        self._observer.entry(value, generator=True, rank=kwargs["rank"])
        return value

    def load_critic_entry(self, **kwargs):
        value = self._trace.load_critic_entry(**kwargs)
        self._observer.entry(value, generator=False, rank=kwargs["rank"])
        return value


class _Observer:
    def __init__(self, *, torch, device, rank, impl, generator, critic, ema, mode):
        self.torch, self.device, self.rank, self.impl = torch, device, rank, impl
        self.generator, self.critic, self.ema, self.mode = generator, critic, ema, mode
        self.actions, self.losses = [], []
        self.current, self.current_entry = None, None
        self.last_entries = {}
        self.live, self.lifetimes = {}, []
        self.peak_live = 0
        self.in_backward = False

    def versions(self):
        return _versions(self.generator, self.critic, self.ema)

    def entry(self, value, *, generator, rank):
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
        row = dict(action_id=op.operation_id, kind=op.kind.value, source_steps=list(source_steps), source_rank=self.rank,
                   versions_before=self.versions(), counts={key:0 for key in _COUNTS}, forwards=[], trace_entries=[],
                   produced=[], losses=[], calibration_seconds=None, calibration_start_unix_ns=None,
                   calibration_stop_unix_ns=None)
        self.actions.append(row)
        self.current, self.current_entry = row, None
        calibrated = self.mode == "calibration"
        if calibrated:
            self.torch.cuda.synchronize(self.device)
            self.torch.cuda.reset_peak_memory_stats(self.device)
        row["memory_before"] = _memory(self.torch, self.device, driver=calibrated)
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
            row["versions_after"] = self.versions()
            row["memory_after"] = _memory(self.torch, self.device, driver=calibrated)
            row["retained_after"] = dict(count=len(self.live), bytes=sum(x["bytes"] for x in self.live.values()))
            self.current, self.current_entry = None, None
        wanted = {**row["versions_before"], **op.produced_versions}
        require(row["versions_after"] == wanted, f"{op.operation_id}: actual native commit versions differ from graph")

    def produced(self, step, value):
        require(step not in self.live, "rollout tensor produced twice")
        metadata = _tensor_metadata(value, self.torch)
        require(metadata["shape"] == [1,16,9,48,80] and metadata["dtype"] == "torch.bfloat16"
                and not metadata["requires_grad"], "retained Helios rollout must be physical B1 BF16 no-grad")
        item = dict(source_step=step, producer=self.current["action_id"],
                    produced_after_action=len(self.actions)-1, released_after_action=None,
                    generator_version=int(self.generator.global_steps), **metadata)
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
        for row in self.actions:
            _validate_action(row, self.impl)
        return dict(census="PASS", versions="PASS", finite="PASS", dtype="PASS", sample_rng="PASS",
                    sample_rng_scope="actual_rank_qualified_trace_entries_not_global_rng_state",
                    numerical_equivalence="NOT_RUN")


def _validate_action(row, impl):
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
    gan_calls = []
    for call in row["forwards"]:
        if not call["application_forward"]:
            continue
        hidden = call["inputs"].get("hidden_states")
        packed_gan_d = impl.variant == "gan" and name.startswith("c") and call["gan_mode"]
        if packed_gan_d:
            # Logical B1 GAN-D is an official physical fake/real/R1 B3 pack,
            # spatially cropped by 0.5. This is NOT a searched batch increase.
            require(call["role"] == "critic" and hidden["shape"] == [3,16,9,24,40],
                    "official GAN-D fake/real/R1 packed input shape changed")
        else:
            require(isinstance(hidden, dict) and hidden["shape"][0] == 1,
                    "actual root forward violated the logical B1 physical call contract")
        require(hidden["dtype"] == "torch.bfloat16", "actual root forward violated BF16")
        if call["root_module"] == "critic":
            gan_calls.append(call["gan_mode"])
    if impl.variant == "gan":
        expected_modes = [False, False, False, True] if name == "g" else [False, True] if name.startswith("c") else []
        require(gan_calls == expected_modes, "actual GAN root forward mode/order changed")
    records = [row["helper_record"]["rollout"]] if name == "g" else row.get("rollout_records", [])
    if "helper_record" in row:
        require(row["helper_record"].get("source_rank") == row["source_rank"]
                and row["helper_record"].get("logical_step") == row["source_steps"][0],
                "helper source rank/step identity changed")
    for entry, record in zip(row["trace_entries"], records, strict=bool(records)):
        identities = [record.get("trace_identity_sha256")] if name == "g" else record.get("member_trace_identities")
        require(identities == [entry["trace_identity_sha256"]], "helper member trace identity changed")
        if name != "g":
            require(record.get("logical_steps") == [entry["source_step"]] and record.get("batch_size") == 1,
                    "helper rollout source step or physical B1 changed")
        require(record["call_count"] == entry["expected_call_count"], "native helper vs trace rollout count differs")
        require({int(k): v for k,v in record["stage_call_count"].items()} == entry["expected_stage_call_count"],
                "native helper vs trace stage count differs")
        calls = [x for x in row["forwards"] if x["application_forward"] and x["root_module"] == "generator"
                 and x["trace"]["rollout_operation"] == entry["rollout_operation"]]
        require(len(calls) == record["call_count"], "actual root forwards differ from helper rollout records")


def _prepare_cycle(*, observer, p0, args, policy, steps, trace, negative, cfg, diagnostics,
                   generator_free_memory):
    graph, order = _cycle_graph(observer.impl, steps, observer.versions())
    observed_trace = _ObservedTrace(trace, observer)
    common = dict(rank=observer.rank, device=observer.device, trace=observed_trace,
                  collect_diagnostics=(not args.timing_mode) and args.profile_diagnostics, diagnostics=diagnostics)
    def gupdate(ctx):
        with observer.action(ctx.operation, source_steps=(steps[0],)) as row:
            if observer.impl.variant == "gan":
                result = p0.base._run_gan_generator_update(**common, args=args, generator_engine=observer.generator,
                    critic_engine=observer.critic, cfg=cfg, negative=negative,
                    generator_free_memory_receipt=generator_free_memory,
                    elide_official_generator_free_memory=bool(policy.elide_official_generator_free_memory
                                                              and (args.timing_mode or args.profile_mode)))
            else:
                result = p0._run_nogan_generator_update_p0(**common, generator_engine=observer.generator,
                    critic_engine=observer.critic, negative=negative, policy=policy,
                    score_pack_parity_check=(not args.timing_mode) and (not args.profile_mode))
            row["helper_record"] = result
        _include_helper_peaks(row["memory_after"], result)
        return result
    def rollout(ctx):
        selected = steps if ctx.operation.operation_id == "r_all" else (ctx.operation.logical_step,)
        with observer.action(ctx.operation, source_steps=selected) as row:
            outputs, records, persistence = p0._precompute_critic_rollouts(**common, steps=selected,
                generator_engine=observer.generator, policy=policy,
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
    mode, torch = binding["mode"], p0.torch
    canonical = impl.lower_profile()["contract"]
    initial = _versions(generator_engine, critic_engine, ema)
    require(initial == canonical["initial_versions"], "native initial G/C/EMA versions differ from canonical checkpoint")
    require(all(int(logical.component(key)["optimizer_step"]) == initial[key] for key in ("generator", "critic")),
            "native engine versions differ from loaded logical checkpoint")
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
                  numerical_status="NOT_RUN", numerical_evidence_scope="diagnostic_sampled_state_not_strict_parity",
                  post_setup_memory=_memory(torch, device, driver=True), outside_timer_probes=[])
    if mode == "correctness":
        started = time.perf_counter()
        result["outside_timer_probes"].append(dict(boundary="initial", state=_correctness_sketch(
            torch, generator_engine, critic_engine, ema), seconds=time.perf_counter()-started))
    for cycle in (1, 2):
        observer = _Observer(torch=torch, device=device, rank=rank, impl=impl, generator=generator_engine,
                             critic=critic_engine, ema=ema, mode=mode)
        prepared, graph, order = _prepare_cycle(observer=observer, p0=p0, args=args, policy=policy, steps=steps,
            trace=trace, negative=negative, cfg=cfg, diagnostics=diagnostics, generator_free_memory=generator_free_memory)
        row = dict(cycle=cycle, measured=cycle == 2, source_steps=list(steps), source_cohort_policy=COHORT_POLICY,
                   initial_versions=observer.versions(), graph=graph.to_dict(), selected_order=list(order),
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
        row.update(final_versions=observer.versions(), dispatch=prepared.receipt(),
                   peak_live_rollouts=observer.peak_live, retained_lifetimes=observer.lifetimes,
                   memory=_memory(torch, device, driver=True))
        _include_helper_peaks(row["memory"], observer.actions)
        probe_started = time.perf_counter()
        row["semantic_validation"] = observer.finish()
        require(row["final_versions"] == graph.metadata["final_versions"], "native final versions differ from graph")
        if mode == "correctness":
            row["state_sketch"] = _correctness_sketch(torch, generator_engine, critic_engine, ema)
        row["outside_timer_probe_seconds"] = time.perf_counter()-probe_started
        row["status"] = "PASS"
    final = _versions(generator_engine, critic_engine, ema)
    require(final == canonical["final_versions"], "native two-cycle final versions changed")
    if hasattr(trace, "receipt"):
        result["input_reuse"] = trace.receipt()
        require(result["input_reuse"].get("critic_entries_still_cached") == 0,
                "hardened trace cache retained an unconsumed input")
    result.update(status="PASS", semantics_status="PASS", final_versions=final,
                  cycle_wall_seconds={str(row["cycle"]): row["wall_seconds"] for row in result["cycles"]},
                  updates={key: final[key]-initial[key] for key in final},
                  final=dict(generator_physical_step=final["generator"], critic_physical_step=final["critic"],
                             ema_step=final["ema"], memory=_memory(torch, device, driver=True)))
    return result
