"""Pinned-host transport for Critic-only LiveWan run-ahead latents.

The production pipeline produces logical steps 1--4 ahead of the Score pool.
Keeping an unmatched NCCL ``isend`` alive while Score is still in the previous
Critic backward occupies GPU communication resources.  This helper moves only
those already-produced float32 latents through a dedicated Gloo group:

CUDA producer stream -> pinned CPU -> Gloo -> pinned CPU -> CUDA consumer stream.

The copy is ordered with CUDA events and a single bounded sender thread.  It
does not synchronize the producing stream, alter the tensor value, or touch
step 0's differentiable DMD exchange.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


RUNAHEAD_STEPS = (1, 2, 3, 4)
RUNAHEAD_WAVES = (0, 1)
RUNAHEAD_TAG_BASE = 27100


def runahead_tag(step: int, wave: int) -> int:
    if step <= 0:
        raise ValueError(f"invalid run-ahead step: {step}")
    if wave not in RUNAHEAD_WAVES:
        raise ValueError(f"invalid run-ahead wave: {wave}")
    return RUNAHEAD_TAG_BASE + step * len(RUNAHEAD_WAVES) + wave


@dataclass
class PinnedHostRunaheadTransport:
    torch: Any
    dist: Any
    group: Any
    rank: int
    peer: int
    device: Any
    is_generator: bool
    shape: Sequence[int]
    dtype: Any
    cycles: int = 1
    cross_cycle_lookahead: bool = False
    buffers: dict[tuple[int, int], Any] = field(default_factory=dict)
    receive_works: dict[tuple[int, int], Any] = field(default_factory=dict)
    copy_events: dict[tuple[int, int], Any] = field(default_factory=dict)
    send_futures: dict[tuple[int, int], Future[Any]] = field(default_factory=dict)
    sent_keys: set[tuple[int, int]] = field(default_factory=set)
    received_keys: set[tuple[int, int]] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.cycles < 1:
            raise RuntimeError("run-ahead transport cycle count is invalid")
        if self.cycles > 1 and not self.cross_cycle_lookahead:
            raise RuntimeError("continuous run-ahead requires the registered lookahead")
        self.shape = tuple(int(value) for value in self.shape)
        if not self.shape or any(value <= 0 for value in self.shape):
            raise RuntimeError("run-ahead transport shape is invalid")
        self.payload_bytes = 1
        for value in self.shape:
            self.payload_bytes *= value
        self.payload_bytes *= self.torch.empty((), dtype=self.dtype).element_size()
        self.copy_stream = (
            self.torch.cuda.Stream(device=self.device) if self.is_generator else None
        )
        self.executor = (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"livewan-gloo-runahead-rank{self.rank}",
            )
            if self.is_generator
            else None
        )
        for key in sorted(self.expected_keys):
            self.buffers[key] = self.torch.empty(
                self.shape,
                dtype=self.dtype,
                device="cpu",
                pin_memory=True,
            )
        if not self.is_generator:
            for key in sorted(self.expected_keys):
                self.receive_works[key] = self.dist.irecv(
                    self.buffers[key],
                    src=self.peer,
                    group=self.group,
                    tag=runahead_tag(*key),
                )

    @property
    def expected_keys(self) -> set[tuple[int, int]]:
        keys = {
            (cycle * 5 + step, wave)
            for cycle in range(self.cycles)
            for step in RUNAHEAD_STEPS
            for wave in RUNAHEAD_WAVES
        }
        if self.cross_cycle_lookahead:
            keys.update((cycle * 5, 0) for cycle in range(1, self.cycles))
        return keys

    def _send_after_copy(self, key: tuple[int, int], event: Any, source: Any) -> None:
        self.torch.cuda.set_device(self.device)
        event.synchronize()
        work = self.dist.isend(
            self.buffers[key],
            dst=self.peer,
            group=self.group,
            tag=runahead_tag(*key),
        )
        work.wait()
        # ``source`` deliberately remains captured until the D2H copy and Gloo
        # send complete.  This makes its CUDA/storage lifetime explicit.
        del source

    def enqueue_send(self, *, step: int, wave: int, tensor: Any) -> None:
        if not self.is_generator or self.executor is None or self.copy_stream is None:
            raise RuntimeError("Score transport cannot enqueue a run-ahead send")
        key = (int(step), int(wave))
        if key not in self.expected_keys or key in self.send_futures:
            raise RuntimeError(f"invalid or duplicate run-ahead send: {key}")
        if tuple(int(value) for value in tensor.shape) != tuple(self.shape):
            raise RuntimeError(f"run-ahead payload shape changed: {tuple(tensor.shape)}")
        if tensor.dtype != self.dtype or tensor.device.type != "cuda":
            raise RuntimeError(
                f"run-ahead payload contract changed: {tensor.dtype}/{tensor.device}"
            )
        source = tensor.detach().contiguous()
        producer_stream = self.torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(producer_stream)
        event = self.torch.cuda.Event(enable_timing=False)
        with self.torch.cuda.stream(self.copy_stream):
            self.buffers[key].copy_(source, non_blocking=True)
            event.record(self.copy_stream)
        self.copy_events[key] = event
        self.send_futures[key] = self.executor.submit(
            self._send_after_copy, key, event, source
        )
        self.sent_keys.add(key)

    def wait_staged(self, *, step: int, wave: int) -> None:
        """Wait only for the tiny D2H stage before reusing a rollout streamer."""

        key = (int(step), int(wave))
        event = self.copy_events.get(key)
        if event is None:
            raise RuntimeError(f"run-ahead staging event is missing: {key}")
        event.synchronize()

    def receive_to_device(self, *, step: int, wave: int) -> Any:
        if self.is_generator:
            raise RuntimeError("Generator transport cannot receive a run-ahead payload")
        key = (int(step), int(wave))
        if key not in self.expected_keys or key in self.received_keys:
            raise RuntimeError(f"invalid or duplicate run-ahead receive: {key}")
        work = self.receive_works.get(key)
        if work is None:
            raise RuntimeError(f"preposted run-ahead receive is missing: {key}")
        work.wait()
        value = self.torch.empty(self.shape, dtype=self.dtype, device=self.device)
        value.copy_(self.buffers[key], non_blocking=True)
        self.received_keys.add(key)
        return value

    def finish(self) -> None:
        if self.is_generator:
            if set(self.send_futures) != self.expected_keys:
                raise RuntimeError("Generator run-ahead send inventory is incomplete")
            for key in sorted(self.send_futures):
                self.send_futures[key].result()
            assert self.executor is not None
            self.executor.shutdown(wait=True)
            self.executor = None
        elif self.received_keys != self.expected_keys:
            raise RuntimeError("Score run-ahead receive inventory is incomplete")

    def receipt(self) -> Mapping[str, Any]:
        event_count = len(self.sent_keys if self.is_generator else self.received_keys)
        return {
            "enabled": True,
            "name": "pinned_cpu_gloo_latent_v1",
            "role": "generator" if self.is_generator else "score",
            "execution_status": (
                "PASS" if event_count == len(self.expected_keys) else "INCOMPLETE"
            ),
            "backend": "gloo",
            "scope": (
                "continuous_critic_only_steps_1_4_plus_next_cycle_step0_wave0"
                if self.cycles > 1
                else "critic_only_steps_1_4_latent_only"
            ),
            "cycles": int(self.cycles),
            "cross_cycle_lookahead": bool(self.cross_cycle_lookahead),
            "step0_transport": "cuda_nccl_p2p_unchanged",
            "context_transport": "trace_local_replay_unchanged",
            "payload_shape": list(self.shape),
            "payload_dtype": str(self.dtype),
            "payload_bytes": int(self.payload_bytes),
            "buffer_count": len(self.buffers),
            "pinned_buffer_count": sum(
                int(bool(buffer.is_pinned())) for buffer in self.buffers.values()
            ),
            "send_event_count": len(self.sent_keys),
            "receive_event_count": len(self.received_keys),
            "d2h_bytes": int(len(self.sent_keys) * self.payload_bytes),
            "h2d_bytes": int(len(self.received_keys) * self.payload_bytes),
            "tags": [runahead_tag(step, wave) for step, wave in sorted(self.expected_keys)],
            "sender_threads": 1 if self.is_generator else 0,
            "preposted_receives": not self.is_generator,
        }
