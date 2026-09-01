"""Group-aware, gradient-ready reduction for replicated LiveWan LoRA state.

The training mathematics remain SUM over the role group followed by division by
the role-group size.  This module changes only when deterministic flat buckets
are issued: each bucket may enter NCCL as soon as all of its leaf gradients are
ready instead of waiting for the complete Critic backward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Sequence


def bucket_named_parameters(
    named_parameters: Sequence[tuple[str, Any]],
    *,
    bucket_bytes: int,
) -> list[list[tuple[str, Any]]]:
    """Pack whole tensors into deterministic reverse-order byte-limited buckets."""

    if bucket_bytes <= 0:
        raise ValueError("bucket_bytes must be positive")
    if not named_parameters:
        raise ValueError("gradient reducer requires at least one parameter")
    result: list[list[tuple[str, Any]]] = []
    current: list[tuple[str, Any]] = []
    current_bytes = 0
    for name, parameter in reversed(tuple(named_parameters)):
        size = int(parameter.numel() * parameter.element_size())
        if current and current_bytes + size > bucket_bytes:
            result.append(current)
            current = []
            current_bytes = 0
        current.append((name, parameter))
        current_bytes += size
    if current:
        result.append(current)
    return result


@dataclass
class _Bucket:
    index: int
    flat: Any
    names: tuple[str, ...]
    parameters: tuple[Any, ...]
    views: tuple[Any, ...]
    ready_parameter_ids: set[int] = field(default_factory=set)
    work: Any | None = None


class GradReadyAsyncGroupReducer:
    """Overlap exact replicated-gradient mean with backward on one role group.

    Parameters remain replicated and the optimizer, clipping, and update order
    remain unchanged.  Persistent FP32 gradient views eliminate temporary
    coalescing buffers.  Missing or duplicate autograd hooks fail before the
    optimizer is allowed to commit.
    """

    def __init__(
        self,
        torch: Any,
        named_parameters: Sequence[tuple[str, Any]],
        *,
        group: Any,
        group_size: int,
        bucket_bytes: int,
    ) -> None:
        if group is None:
            raise ValueError("gradient reducer group is required")
        if group_size <= 1:
            raise ValueError("gradient reducer group_size must exceed one")
        named = tuple(named_parameters)
        if len({name for name, _parameter in named}) != len(named):
            raise ValueError("gradient reducer parameter names are not unique")
        if len({id(parameter) for _name, parameter in named}) != len(named):
            raise ValueError("gradient reducer parameters are not unique")
        if {parameter.dtype for _name, parameter in named} != {torch.float32}:
            raise TypeError("gradient-ready reducer requires FP32 trainables")
        devices = {parameter.device for _name, parameter in named}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise TypeError("gradient-ready reducer requires one CUDA device")

        self._torch = torch
        self._group = group
        self._group_size = int(group_size)
        self._bucket_bytes = int(bucket_bytes)
        self._buckets: list[_Bucket] = []
        self._parameter_bucket: dict[int, _Bucket] = {}
        self._parameter_view: dict[int, Any] = {}
        self._hook_counts: dict[int, int] = {}
        self._hook_handles: list[Any] = []
        for index, rows in enumerate(
            bucket_named_parameters(named, bucket_bytes=self._bucket_bytes)
        ):
            parameters = tuple(parameter for _name, parameter in rows)
            flat = torch.zeros(
                sum(int(parameter.numel()) for parameter in parameters),
                dtype=torch.float32,
                device=parameters[0].device,
            )
            views: list[Any] = []
            offset = 0
            for parameter in parameters:
                count = int(parameter.numel())
                view = flat.narrow(0, offset, count).view_as(parameter)
                parameter.grad = view
                views.append(view)
                offset += count
            bucket = _Bucket(
                index=index,
                flat=flat,
                names=tuple(name for name, _parameter in rows),
                parameters=parameters,
                views=tuple(views),
            )
            self._buckets.append(bucket)
            for parameter, view in zip(parameters, views, strict=True):
                identity = id(parameter)
                self._parameter_bucket[identity] = bucket
                self._parameter_view[identity] = view
                self._hook_counts[identity] = 0
                self._hook_handles.append(
                    parameter.register_post_accumulate_grad_hook(
                        self._make_hook(identity)
                    )
                )

        self._comm_stream = torch.cuda.Stream(device=next(iter(devices)))
        self._active_backward = False
        self._pending_sync = False
        self._expected_backward_calls = 1
        self._next_bucket_to_issue = 0
        self._issue_order: list[int] = []
        self._issue_times_ns: list[int] = []

    def _make_hook(self, parameter_id: int):
        def hook(parameter: Any) -> None:
            if not self._active_backward:
                raise RuntimeError("gradient-ready hook fired outside reducer.backward")
            bucket = self._parameter_bucket[parameter_id]
            if id(parameter) != parameter_id:
                raise RuntimeError("gradient-ready hook parameter identity changed")
            expected_view = self._parameter_view[parameter_id]
            if parameter.grad is None or parameter.grad.data_ptr() != expected_view.data_ptr():
                raise RuntimeError("persistent gradient view was replaced")
            count = self._hook_counts[parameter_id] + 1
            self._hook_counts[parameter_id] = count
            if count > self._expected_backward_calls:
                raise RuntimeError("gradient-ready hook fired too many times")
            if count == self._expected_backward_calls:
                bucket.ready_parameter_ids.add(parameter_id)
                self._drain_ready_buckets()

        return hook

    def _drain_ready_buckets(self) -> None:
        torch = self._torch
        dist = torch.distributed
        while self._next_bucket_to_issue < len(self._buckets):
            bucket = self._buckets[self._next_bucket_to_issue]
            if len(bucket.ready_parameter_ids) != len(bucket.parameters):
                break
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(bucket.flat.device))
            with torch.cuda.stream(self._comm_stream):
                self._comm_stream.wait_event(ready)
                bucket.work = dist.all_reduce(
                    bucket.flat,
                    op=dist.ReduceOp.SUM,
                    group=self._group,
                    async_op=True,
                )
            self._issue_order.append(bucket.index)
            self._issue_times_ns.append(time.perf_counter_ns())
            self._next_bucket_to_issue += 1

    def _reset_backward_state(self) -> None:
        if self._pending_sync:
            raise RuntimeError("previous gradient synchronization was not finished")
        self._next_bucket_to_issue = 0
        self._expected_backward_calls = 1
        self._issue_order.clear()
        self._issue_times_ns.clear()
        for parameter_id in self._hook_counts:
            self._hook_counts[parameter_id] = 0
        for bucket in self._buckets:
            bucket.ready_parameter_ids.clear()
            bucket.work = None

    def zero_grad(self) -> None:
        if self._active_backward or self._pending_sync:
            raise RuntimeError("cannot zero gradients while reduction is active")
        for bucket in self._buckets:
            bucket.flat.zero_()
            for parameter, view in zip(bucket.parameters, bucket.views, strict=True):
                parameter.grad = view

    def backward(self, loss: Any) -> None:
        self.backward_accumulate(
            loss, accumulation_index=0, accumulation_count=1
        )

    def backward_accumulate(
        self,
        loss: Any,
        *,
        accumulation_index: int,
        accumulation_count: int,
    ) -> None:
        """Accumulate exact microbatch gradients and reduce only after the last.

        This retains one optimizer commit and one group mean.  Earlier
        microbatches remain local; the final backward issues each persistent
        flat bucket when its last contribution becomes ready.
        """

        if accumulation_count <= 0 or not 0 <= accumulation_index < accumulation_count:
            raise ValueError("invalid gradient accumulation position")
        if accumulation_index == 0:
            self._reset_backward_state()
            self._expected_backward_calls = int(accumulation_count)
        else:
            if self._active_backward or self._pending_sync:
                raise RuntimeError("gradient accumulation state is already active")
            if self._expected_backward_calls != accumulation_count:
                raise RuntimeError("gradient accumulation count changed")
            if set(self._hook_counts.values()) != {accumulation_index}:
                raise RuntimeError("previous gradient accumulation is incomplete")
            if self._issue_order:
                raise RuntimeError("gradient reduction started before final accumulation")
        self._active_backward = True
        try:
            loss.backward()
        finally:
            self._active_backward = False
        expected_hook_count = accumulation_index + 1
        if set(self._hook_counts.values()) != {expected_hook_count}:
            missing = [
                name
                for bucket in self._buckets
                for name, parameter in zip(bucket.names, bucket.parameters, strict=True)
                if self._hook_counts[id(parameter)] != expected_hook_count
            ]
            raise RuntimeError(
                f"gradient accumulation {expected_hook_count}/{accumulation_count} "
                f"is incomplete; missing={missing[:8]}"
            )
        if expected_hook_count < accumulation_count:
            if self._issue_order:
                raise RuntimeError("gradient reduction issued before final accumulation")
            return
        expected = list(range(len(self._buckets)))
        if self._issue_order != expected:
            missing = [
                name
                for bucket in self._buckets
                for name, parameter in zip(bucket.names, bucket.parameters, strict=True)
                if self._hook_counts[id(parameter)] != accumulation_count
            ]
            raise RuntimeError(
                f"gradient-ready collective order incomplete: {self._issue_order}; "
                f"missing={missing[:8]}"
            )
        self._pending_sync = True

    def finish(self) -> dict[str, Any]:
        if not self._pending_sync:
            raise RuntimeError("gradient synchronization is not pending")
        for bucket in self._buckets:
            if bucket.work is None:
                raise RuntimeError(f"gradient-ready bucket {bucket.index} was not issued")
            bucket.work.wait()
        current = self._torch.cuda.current_stream(self._buckets[0].flat.device)
        current.wait_stream(self._comm_stream)
        self._torch._foreach_mul_(
            [bucket.flat for bucket in self._buckets], 1.0 / self._group_size
        )
        self._pending_sync = False
        return {
            "backend": "GRAD_READY_ASYNC_FLAT_ALL_REDUCE",
            "reduction": "SUM_THEN_DIVIDE_GROUP_SIZE",
            "group_size": self._group_size,
            "gradient_tensors": len(self._parameter_bucket),
            "gradient_bytes": sum(
                int(bucket.flat.numel() * bucket.flat.element_size())
                for bucket in self._buckets
            ),
            "buckets": len(self._buckets),
            "bucket_bytes_limit": self._bucket_bytes,
            "bucket_bytes_observed": [
                int(bucket.flat.numel() * bucket.flat.element_size())
                for bucket in self._buckets
            ],
            "bucket_issue_order": list(self._issue_order),
            "hook_count_min": min(self._hook_counts.values()),
            "hook_count_max": max(self._hook_counts.values()),
            "backward_accumulation_count": self._expected_backward_calls,
            "overlap_with_backward": True,
            "persistent": True,
            "dedicated_comm_stream": True,
        }


__all__ = ["GradReadyAsyncGroupReducer", "bucket_named_parameters"]
