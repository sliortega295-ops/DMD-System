"""Collective-only workload padding: discard extra no-grad forward results.

Never generate extra samples or append K/V. Each true rollout is left intact;
short ranks run the missing physical forwards against no mutable KV cache.
Padding is opt-in and separate from the application-level DiT census.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Iterable


_PADDING = ContextVar("livewan_collective_padding", default=False)


def padding_call_active() -> bool:
    return _PADDING.get()


class StudentRolloutPadding:
    def __init__(self, torch: Any, rows: Iterable[Any]) -> None:
        self.torch = torch
        self.max_blocks: dict[int, int] = {}
        for row in rows:
            step, count = int(row.logical_step), int(row.new_blocks)
            if not 4 <= count <= 6:
                raise ValueError("padding trace changed the original block range")
            self.max_blocks[step] = max(self.max_blocks.get(step, 0), count)
        self.records: list[dict[str, Any]] = []

    def pad(self, *, blockcausal: Any, streamer: Any, row: Any, recs: Any) -> None:
        maximum = self.max_blocks[int(row.logical_step)]
        if len(streamer.schedule) != 4 or streamer.noisy_context:
            raise ValueError("padding requires original four-step plus clean-refresh rollout")
        missing = maximum - int(row.new_blocks)
        if missing < 0 or not recs or len(recs) != row.new_blocks:
            raise ValueError("padding rollout/trace census mismatch")
        state_before = (streamer.n_frames, streamer.n_world,
                        streamer.cache.num_tokens, streamer.cache.evictions)
        count = 0
        if missing:
            z = recs[-1][0]["z"].detach()
            t_start = int(recs[-1][0]["block_start"])
            device = z.device.index if z.device.index is not None else 0
            devices = [device] if z.device.type == "cuda" else []
            token = _PADDING.set(True)
            try:
                with self.torch.no_grad(), self.torch.random.fork_rng(devices=devices):
                    for _ in range(missing):
                        for t in (*streamer.schedule, 0.0):
                            blockcausal.block_forward(
                                streamer.model, z, t, t_start, streamer.rope,
                                streamer.ctx, streamer.ctx_lens, kv=None,
                                dtype=streamer.dtype, time_scale=streamer.time_scale)
                            count += 1
            finally:
                _PADDING.reset(token)
        if state_before != (streamer.n_frames, streamer.n_world,
                            streamer.cache.num_tokens, streamer.cache.evictions):
            raise RuntimeError("padding mutated true rollout state")
        self.records.append({"operation_key": row.operation_key,
            "global_step": int(row.logical_step), "real_new_blocks": int(row.new_blocks),
            "maximum_new_blocks": maximum, "real_nograd_forward_calls": 1 + 5*int(row.new_blocks),
            "discarded_nograd_forward_calls": count,
            "aligned_nograd_forward_calls": 1 + 5*maximum})

    def receipt(self) -> dict[str, Any]:
        if any(r["discarded_nograd_forward_calls"] !=
               5*(r["maximum_new_blocks"] - r["real_new_blocks"]) for r in self.records):
            raise RuntimeError("padding physical census mismatch")
        return {"enabled": True, "name": "max_blocks_discarded_nograd_forward_v1",
                "sample_history_and_real_rollout_unchanged": True,
                "dummy_kv": "NONE_NO_CACHE_MUTATION", "rng": "FORK_AND_RESTORE",
                "dummy_outputs_used_in_loss": False, "dummy_gradients": False,
                "extra_compute_and_communication_inside_timer": True,
                "extra_physical_forward_calls": sum(r["discarded_nograd_forward_calls"] for r in self.records),
                "records": list(self.records)}
