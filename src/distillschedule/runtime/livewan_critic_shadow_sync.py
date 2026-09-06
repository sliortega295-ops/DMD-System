"""Exact paired Critic-LoRA hot sync for continuous LiveWan pipelines.

The C20 split-DMD graph keeps a frozen Critic replica on each Generator rank.
That replica is C0 for one Full5.  A continuous sequence must update it to C5,
C10, ... before the next Generator DMD objective.  This helper owns one
preallocated FP32 flat buffer and one dedicated two-rank NCCL process group so
the sync cannot be confused with the latent P2P stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Any, Mapping, Sequence


def parameter_layout(parameters: Sequence[Any]) -> tuple[tuple[tuple[int, ...], str, int], ...]:
    return tuple(
        (
            tuple(int(value) for value in parameter.shape),
            str(parameter.dtype),
            int(parameter.numel()),
        )
        for parameter in parameters
    )


def layout_sha256(layout: Sequence[Sequence[Any]]) -> str:
    return hashlib.sha256(
        json.dumps(list(layout), separators=(",", ":"), sort_keys=False).encode("utf-8")
    ).hexdigest()


@dataclass
class PairedCriticShadowSync:
    torch: Any
    dist: Any
    group: Any
    rank: int
    peer: int
    score_rank: int
    device: Any
    parameters: Sequence[Any]
    is_generator: bool
    run_kind: str
    flat: Any = field(init=False)
    views: list[Any] = field(init=False, default_factory=list)
    pending_source_work: Any | None = field(init=False, default=None)
    pending_source_version: int | None = field(init=False, default=None)
    events: list[dict[str, Any]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.group is None:
            raise RuntimeError("Critic-shadow sync group is missing")
        if not self.parameters:
            raise RuntimeError("Critic-shadow sync parameter inventory is empty")
        dtypes = {parameter.dtype for parameter in self.parameters}
        devices = {parameter.device for parameter in self.parameters}
        if dtypes != {self.torch.float32}:
            raise RuntimeError(f"Critic-shadow sync requires FP32 LoRA: {dtypes}")
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise RuntimeError(f"Critic-shadow sync parameters are not one CUDA placement: {devices}")
        self.layout = parameter_layout(self.parameters)
        self.layout_sha256 = layout_sha256(self.layout)
        self.total_numel = sum(value[2] for value in self.layout)
        self.flat = self.torch.empty(
            self.total_numel, dtype=self.torch.float32, device=self.device
        )
        offset = 0
        for parameter in self.parameters:
            view = self.flat[offset : offset + parameter.numel()].view_as(parameter)
            self.views.append(view)
            offset += parameter.numel()
        if offset != self.total_numel:
            raise RuntimeError("Critic-shadow flat-buffer layout is incomplete")

    def validate_pair_contract(self) -> Mapping[str, Any]:
        local = self.torch.tensor(
            [len(self.parameters), self.total_numel],
            dtype=self.torch.int64,
            device=self.device,
        )
        gathered = [self.torch.empty_like(local) for _ in range(2)]
        self.dist.all_gather(gathered, local, group=self.group)
        values = [tuple(int(value) for value in row.cpu()) for row in gathered]
        if values[0] != values[1] or values[0] != (
            len(self.parameters),
            self.total_numel,
        ):
            raise RuntimeError(f"Critic-shadow pair layout differs: {values}")
        return {
            "status": "PASS",
            "parameter_count": len(self.parameters),
            "total_numel": self.total_numel,
            "payload_bytes": self.total_numel * 4,
            "dtype": "torch.float32",
            "layout_sha256": self.layout_sha256,
            "pair_ranks": [min(self.rank, self.peer), max(self.rank, self.peer)],
        }

    def _pack_source(self) -> None:
        for view, parameter in zip(self.views, self.parameters):
            view.copy_(parameter.detach())

    def _unpack_receiver(self) -> None:
        for parameter, view in zip(self.parameters, self.views):
            parameter.copy_(view)

    def _sha256(self) -> str | None:
        if self.run_kind != "correctness":
            return None
        value = self.flat.detach().cpu().contiguous()
        return hashlib.sha256(value.view(self.torch.uint8).numpy().tobytes()).hexdigest()

    def begin_source(self, *, version: int) -> None:
        if self.is_generator:
            raise RuntimeError("Generator cannot source the Critic-shadow sync")
        if self.pending_source_work is not None:
            raise RuntimeError("previous Critic-shadow source sync is still pending")
        started = time.perf_counter()
        self._pack_source()
        source_sha = self._sha256()
        work = self.dist.broadcast(
            self.flat,
            src=self.score_rank,
            group=self.group,
            async_op=True,
        )
        self.pending_source_work = work
        self.pending_source_version = int(version)
        self.events.append(
            {
                "role": "source",
                "version": int(version),
                "source_sha256": source_sha,
                "submit_wall_seconds": time.perf_counter() - started,
                "completion_wall_seconds": None,
                "status": "PENDING",
            }
        )

    def wait_source(self, *, expected_version: int) -> None:
        if self.is_generator:
            raise RuntimeError("Generator cannot wait as Critic-shadow source")
        if self.pending_source_work is None:
            raise RuntimeError("Critic-shadow source work is missing")
        if self.pending_source_version != int(expected_version):
            raise RuntimeError("Critic-shadow source version changed")
        started = time.perf_counter()
        self.pending_source_work.wait()
        self.events[-1]["completion_wall_seconds"] = time.perf_counter() - started
        self.events[-1]["status"] = "PASS"
        self.pending_source_work = None
        self.pending_source_version = None

    def receive_and_load(self, *, version: int) -> None:
        if not self.is_generator:
            raise RuntimeError("Score rank cannot load the Critic shadow")
        started = time.perf_counter()
        work = self.dist.broadcast(
            self.flat,
            src=self.score_rank,
            group=self.group,
            async_op=True,
        )
        work.wait()
        loaded_sha = self._sha256()
        with self.torch.no_grad():
            self._unpack_receiver()
        self.events.append(
            {
                "role": "receiver",
                "version": int(version),
                "loaded_sha256": loaded_sha,
                "wait_and_load_wall_seconds": time.perf_counter() - started,
                "status": "PASS",
            }
        )

    def finish(self) -> None:
        if self.pending_source_work is not None:
            assert self.pending_source_version is not None
            self.wait_source(expected_version=self.pending_source_version)

    def receipt(self) -> Mapping[str, Any]:
        return {
            "enabled": True,
            "name": "paired_fp32_critic_lora_hot_sync_v1",
            "role": "generator_receiver" if self.is_generator else "score_source",
            "pair_ranks": [min(self.rank, self.peer), max(self.rank, self.peer)],
            "parameter_count": len(self.parameters),
            "total_numel": self.total_numel,
            "payload_bytes": self.total_numel * 4,
            "dtype": "torch.float32",
            "layout_sha256": self.layout_sha256,
            "events": list(self.events),
            "execution_status": (
                "PASS"
                if self.events and all(row.get("status") == "PASS" for row in self.events)
                else "INCOMPLETE"
            ),
        }


__all__ = ["PairedCriticShadowSync", "layout_sha256", "parameter_layout"]
