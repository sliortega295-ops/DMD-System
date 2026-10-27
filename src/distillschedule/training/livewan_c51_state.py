"""C51 physical-pool data, bounded staging and serialization-only state.

No DMD math lives here. Host buffers are reused only after send completion on G
and H2D completion on Score. CUDA values returned to Score own separate storage.
Neither these buffers nor streamer/KV/shadow/SAC scratch enters a checkpoint.
"""
from concurrent.futures import ThreadPoolExecutor

from distillschedule.runtime.livewan_host_runahead import PinnedHostRunaheadTransport, runahead_tag
from distillschedule.workloads.best_known.registry import fingerprint
from .livewan_reference_assets import require

SCHEMA = "distillschedule.livewan_c51_pool_progress.v1"


def physical_versions(rank, cycle):
    return {"generator": cycle if rank < 8 else 0, "critic": 0 if rank < 8 else 5*cycle, "ema": 0}


class PoolInputs:
    """Observe actual G loads / Score contexts, not a physical-rank seek()."""
    def __init__(self, loader, *, rank, pairs):
        from distillschedule.runtime.livewan_crossscale_worker import InputLoader
        from distillschedule.data.livewan_inputs import C51_KIND
        require(type(loader) is InputLoader and loader.input_kind == C51_KIND,
                "full C51 requires actual original fresh InputLoader")
        require(type(rank) is int and 0 <= rank < 16, "explicit physical C51 rank required")
        self.loader, self.inputs, self.rank = loader, loader.inputs, rank
        self.pairs = tuple(tuple(p) for p in pairs)
        from distillschedule.runtime.livewan_crossscale_worker import validate_slot_pairs
        validate_slot_pairs(self.pairs)
        self.slots = self.pairs[rank % 8]
        self.cycle, self.pending, self.history = 0, [], []

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def descriptor(self, row):
        return dict(operation_key=row.operation_key, logical_step=row.logical_step, logical_slot=row.logical_slot,
                    record_id=row.record_id, sample_sha256=row.sample_sha256, prompt_idx=row.prompt_idx,
                    world_frames=row.world_frames, rope_gap=row.rope_gap, seeds=dict(row.seeds))

    def expected(self, cycle):
        require(type(cycle) is int and 0 <= cycle < self.inputs.cycles, "C51 finite lane cycle outside admitted horizon")
        # These are already admitted immutable operation rows, not record reads.
        # Actual load/context and checkpoint boundaries recheck source bytes.
        # Do not run ten Git/source probes for every metadata comparison.
        return [self.descriptor(self.inputs.rows[(5*cycle+step)*16+slot]) for step in range(5) for slot in self.slots]

    def _observe(self, row):
        wanted = self.expected(self.cycle)
        require(len(self.pending) < 10 and self.descriptor(row) == wanted[len(self.pending)],
                "C51 missing/duplicate/reordered physical lane sample or extra lookahead")

    def load(self, row):
        require(self.rank < 8, "Score must consume received real latents, not reload Teacher samples")
        self._observe(row)
        sample = self.loader.load(row)  # original TeacherLatents and every archive gate
        self.pending.append(self.descriptor(row))
        return sample

    def context_from_trace(self, row, *, device):
        require(self.rank >= 8, "G context consumption belongs to its actual load")
        self._observe(row)
        value = self.loader.context_from_trace(row, device=device)
        self.pending.append(self.descriptor(row))
        return value

    def complete(self, cycle, actual):
        require(type(cycle) is int and cycle == self.cycle+1 and self.pending == self.expected(self.cycle)
                and actual == physical_versions(self.rank, cycle), "C51 physical dual-wave/update completion differs")
        previous = self.history[-1] if self.history else "0"*64
        self.history.append(fingerprint(dict(previous=previous, cycle=cycle, rows=self.pending)))
        self.pending.clear()
        self.cycle = cycle

    def _history(self, cycle):
        result = []
        for index in range(cycle):
            result.append(fingerprint(dict(previous=result[-1] if result else "0"*64,
                                           cycle=index+1, rows=self.expected(index))))
        return result

    def state_dict(self):
        self.validate_checkpoint_boundary()
        return dict(schema_version=SCHEMA, input_sha256=self.inputs.input_sha256, rank=self.rank,
                    pairs=[list(p) for p in self.pairs], cycle=self.cycle, cursor=5*self.cycle,
                    history=list(self.history), next_keys=[r["operation_key"] for r in self.expected(self.cycle)[:2]]
                    if self.cycle < self.inputs.cycles else [])

    def validate_state_dict(self, value):
        require(type(value) is dict and type(value.get("cycle")) is int
                and 0 <= value["cycle"] <= self.inputs.cycles, "C51 saved finite cursor invalid")
        cycle = value["cycle"]
        expected = dict(schema_version=SCHEMA, input_sha256=self.inputs.input_sha256, rank=self.rank,
                        pairs=[list(p) for p in self.pairs], cycle=cycle, cursor=5*cycle,
                        history=self._history(cycle), next_keys=[r["operation_key"] for r in self.expected(cycle)[:2]]
                        if cycle < self.inputs.cycles else [])
        require(value == expected, "C51 saved source/input/rank/slot/wave/history/cursor changed")
        self.inputs._unchanged()

    def load_state_dict(self, value):
        self.validate_state_dict(value)
        require(not self.pending, "cannot restore over an in-flight produced sample")
        self.cycle, self.history = value["cycle"], list(value["history"])

    def validate_checkpoint_boundary(self):
        require(not self.pending and self.history == self._history(self.cycle),
                "C51 lookahead/half-wave cannot be checkpointed")
        self.inputs._unchanged()


class WindowedRunaheadTransport(PinnedHostRunaheadTransport):
    """Two reusable pinned slots per role, independent of finite horizon.

    Legacy transport and its immutable receipts are not changed. On-demand
    Gloo receives are safe because the sole sender runs off the CUDA thread;
    waiting for a reusable slot exerts genuine producer backpressure.
    """
    def __init__(self, *, start_cycle=0, **kwargs):
        self.start_cycle = start_cycle
        super().__init__(**kwargs)

    def __post_init__(self):
        require(self.waves_per_step == 2 and type(self.cycles) is int
                and 0 <= self.start_cycle < self.cycles <= 100 and self.cross_cycle_lookahead,
                "C51 transport requires finite B16 dual waves")
        self.shape = tuple(self.shape)
        import math
        self.payload_bytes = math.prod(self.shape)*self.torch.empty((), dtype=self.dtype).element_size()
        self.copy_stream = self.torch.cuda.Stream(device=self.device) if self.is_generator else None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"c51-rank{self.rank}") if self.is_generator else None
        self._slots = [self.torch.empty(self.shape, dtype=self.dtype, device="cpu", pin_memory=True) for _ in range(2)]
        self._owners, self._h2d = {}, {}
        self._closed, self._drained = False, self.start_cycle
        self.backpressure_waits = 0
        self._high_water = 0

    @property
    def expected_keys(self):
        return {(5*c+s, w) for c in range(self.start_cycle, self.cycles) for s in (1, 2, 3, 4) for w in (0, 1)} | {
            (5*c, 0) for c in range(max(1, self.start_cycle), self.cycles)}

    def _retire(self, wave):
        previous = self._owners.get(wave)
        if previous is None:
            return
        self.backpressure_waits += 1
        if self.is_generator:
            self.send_futures.pop(previous).result()
        else:
            self._h2d.pop(previous).synchronize()
        self.buffers.pop(previous)
        # wait_staged uses only the latest event for a streamer; its previous
        # event has completed before a later wave can own the same buffer.
        self.copy_events.pop(previous, None)
        self._owners.pop(wave)

    def enqueue_send(self, *, step, wave, tensor):
        key = (int(step), int(wave))
        require(not self._closed and key in self.expected_keys and key not in self.sent_keys,
                "C51 duplicate/out-of-range runahead send")
        self._retire(wave)
        self.buffers[key] = self._slots[wave]
        self._owners[wave] = key
        self._high_water = max(self._high_water, len(self.buffers))
        super().enqueue_send(step=step, wave=wave, tensor=tensor)

    def wait_staged(self, *, step, wave):
        key = (int(step), int(wave))
        require(key in self.sent_keys, "C51 missing staged payload")
        event = self.copy_events.get(key)
        if event is not None:
            event.synchronize()
        # A retired event is stronger than staged: send already completed.

    def receive_to_device(self, *, step, wave):
        key = (int(step), int(wave))
        require(not self.is_generator and not self._closed and key in self.expected_keys
                and key not in self.received_keys, "C51 duplicate/out-of-range receive")
        self._retire(wave)
        self.buffers[key] = self._slots[wave]
        self._owners[wave] = key
        self._high_water = max(self._high_water, len(self.buffers))
        self.dist.irecv(self.buffers[key], src=self.peer, group=self.group, tag=runahead_tag(*key)).wait()
        value = self.torch.empty(self.shape, dtype=self.dtype, device=self.device)
        value.copy_(self.buffers[key], non_blocking=True)
        event = self.torch.cuda.Event(enable_timing=False)
        event.record(self.torch.cuda.current_stream(self.device))
        self._h2d[key] = event
        self.received_keys.add(key)
        return value

    def drain_through(self, cycle):
        require(type(cycle) is int and self._drained <= cycle <= self.cycles,
                "C51 invalid nonterminal transport drain")
        expected = {key for key in self.expected_keys if key[0] < 5*cycle}
        observed = self.sent_keys if self.is_generator else self.received_keys
        require(observed == expected, "C51 incomplete waves or future lookahead at publication")
        for wave in tuple(self._owners):
            self._retire(wave)
        require(not self.buffers and not self.send_futures and not self._h2d,
                "C51 transport remains in flight")
        self._drained = cycle

    def finish(self):
        self.drain_through(self.cycles)
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        self._closed = True

    def receipt(self):
        return dict(enabled=True, name="c51_two_slot_pinned_gloo.v1", role="generator" if self.is_generator else "score",
                    execution_status="PASS" if self._drained == self.cycles else "INCOMPLETE",
                    start_cycle=self.start_cycle, cycles=self.cycles, buffer_count=len(self._slots),
                    pinned_buffer_count=sum(int(v.is_pinned()) for v in self._slots),
                    payload_bytes=self.payload_bytes, high_water_buffers=self._high_water, in_flight=len(self._owners),
                    send_event_count=len(self.sent_keys), receive_event_count=len(self.received_keys),
                    backpressure_waits=self.backpressure_waits, drained_through_cycle=self._drained)


class Progress:
    def __init__(self, cursor, actual, *, restore_cycle=None):
        self.cursor, self.actual, self.restore_cycle = cursor, actual, restore_cycle
        self.quiescent = True

    def state_dict(self):
        self.validate_checkpoint_boundary()
        return dict(data=self.cursor.state_dict(), versions=dict(self.actual), quiescent=True)

    def validate_state_dict(self, value):
        require(type(value) is dict and set(value) == {"data", "versions", "quiescent"}
                and value["quiescent"] is True, "C51 incomplete progress state")
        self.cursor.validate_state_dict(value["data"])
        cycle = value["data"]["cycle"]
        require(value["versions"] == physical_versions(self.cursor.rank, cycle)
                and all(type(v) is int for v in value["versions"].values())
                and (self.restore_cycle is None or cycle == self.restore_cycle), "C51 saved physical clocks changed")

    def validate_checkpoint_progress(self, value, progress):
        self.validate_state_dict(value)
        cycle = value["data"]["cycle"]
        require(type(progress) is dict and type(progress.get("cycle")) is int and type(progress.get("cursor")) is int
                and type(progress.get("versions")) is dict and all(type(v) is int for v in progress["versions"].values())
                and progress == dict(cycle=cycle, cursor=5*cycle, versions=dict(generator=cycle, critic=5*cycle, ema=0)),
                "C51 joint header does not match actual physical pool progress")

    def load_state_dict(self, value):
        self.validate_state_dict(value)
        self.cursor.load_state_dict(value["data"])
        self.actual.clear(); self.actual.update(value["versions"])
        self.quiescent = True

    def validate_checkpoint_boundary(self):
        self.cursor.validate_checkpoint_boundary()
        require(self.quiescent and self.actual == physical_versions(self.cursor.rank, self.cursor.cycle),
                "C51 pending transport/reducer/shadow or inconsistent physical clock")


def owner_state(bundle, *, source_root, progress):
    """Compose original resident named weights/Adam with strict step/finite gates."""
    from .livewan_state import model_bundle_state, _Optimizer, _ReferenceOptimizer, _Weights
    import torch

    class Weights(_Weights):
        def state_dict(self):
            value = super().state_dict()
            require(all(bool(torch.isfinite(t).all()) for t in (*value["tensors"].values(), *value["buffers"].values())),
                    "C51 live weights/buffers are nonfinite")
            return value

        def validate_state_dict(self, value):
            super().validate_state_dict(value)
            require(value["training"] == {n: m.training for n, m in self.engine.module.named_modules()},
                    "C51 saved original module modes changed")
            live_parameters = dict(self.engine.module.named_parameters())
            for name, tensor in value["tensors"].items():
                live = live_parameters[name]
                require(tensor.stride() == live.stride() and bool(torch.isfinite(tensor).all()),
                        "C51 saved trainable stride/nonfinite value")
            require(all(bool(torch.isfinite(t).all()) for t in value["buffers"].values()), "C51 nonfinite buffer")

    class Optimizer(_Optimizer):
        def state_dict(self):
            self._reading_live = True
            try:
                return super().state_dict()
            finally:
                self._reading_live = False

        def _admit(self, value):
            super()._admit(value)
            cycle = (progress.restore_cycle if progress.restore_cycle is not None
                     and not getattr(self, "_reading_live", False) else progress.cursor.cycle)
            _ReferenceOptimizer._steps(self, value, dict(generator=cycle, critic=5*cycle))
            require(all(bool(torch.isfinite(s["exp_avg"]).all())
                        and bool(torch.isfinite(s["exp_avg_sq"]).all()) and bool((s["exp_avg_sq"] >= 0).all())
                        for s in value["optimizer"]["state"].values()), "C51 named Adam step/moment differs from actual clock")

    base = model_bundle_state(bundle, source_root=source_root)
    return {key: (Optimizer(view.engine) if key.endswith(".optimizer") else Weights(view.engine))
            for key, view in base.items()} | {"pool.progress": progress}
