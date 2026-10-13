"""Run eight unmodified B1 host samplers through one GPipe8 model boundary.

History decisions, schedule budgets/timesteps, noise and trace reads remain
rank-local. Only matching physical DiT calls rendezvous. The real model proxy
executes eight B1 microbatches; this dispatcher never uses rank0's sampler for
another sample, modifies RNG, or calls a distributed collective in a worker.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
import queue
import threading
import time

import torch
from torch import nn


@dataclass
class _Call:
    sample: int
    ordinal: int
    kwargs: dict
    grad: bool
    ready: threading.Event = field(default_factory=threading.Event)
    result: object = None
    error: BaseException | None = None


class _SampleModel(nn.Module):
    def __init__(self, sample, submit, stopped, pending, guard, timeout):
        super().__init__()
        self.sample, self.submit, self.stopped = sample, submit, stopped
        self.pending, self.guard, self.timeout = pending, guard, timeout
        self.ordinal = 0

    def enable_adapters(self):
        # The owner configures the real Generator before launching these B1
        # samplers. Threads must never recursively mutate a shared model.
        pass

    def disable_adapters(self):
        raise ValueError('a native Generator sampler cannot select Teacher mode')

    def forward(self, *args, **kwargs):
        if args or self.stopped.is_set():
            raise RuntimeError('stopped or non-keyword native sampler call')
        call = _Call(self.sample, self.ordinal, kwargs, torch.is_grad_enabled())
        self.ordinal += 1
        with self.guard:
            if self.stopped.is_set():
                raise RuntimeError('native B1 cohort stopped before enqueue')
            self.pending.append(call)
        self.submit.put(('call', call))
        if not call.ready.wait(self.timeout):
            raise TimeoutError('native B1 DiT rendezvous timed out')
        if call.error is not None:
            raise call.error
        return call.result


def _pack_B1_kwargs(calls):
    first = calls[0]
    if any(c.grad != first.grad for c in calls):
        raise ValueError('native B1 gradient scope differs within a physical call')
    keys = set(first.kwargs)
    if any(set(c.kwargs) != keys for c in calls):
        raise ValueError('native B1 argument inventory differs')
    result = {}
    for key in first.kwargs:
        values = [c.kwargs[key] for c in calls]
        if isinstance(values[0], torch.Tensor):
            rows = []
            for value in values:
                if not isinstance(value, torch.Tensor):
                    raise ValueError('native tensor argument type differs: '+key)
                if value.ndim == 0:
                    value = value.reshape(1)
                if key.startswith('indices_') and value.ndim == 1:
                    value = value.unsqueeze(0)
                if value.shape[0] != 1:
                    raise ValueError('native host sampler must submit B1: '+key)
                if rows and (value.shape != rows[0].shape or value.dtype != rows[0].dtype
                             or value.device != rows[0].device):
                    raise ValueError('native physical call tensor signature differs: '+key)
                rows.append(value)
            result[key] = torch.cat(rows, dim=0)
        else:
            if any(type(value) is not type(values[0]) or value != values[0] for value in values):
                raise ValueError('native non-tensor model argument differs: '+key)
            result[key] = values[0]
    return result


def run_native_B1_cohort(model, sample_function, *, device, timeout_seconds=600):
    """Return eight native sampler results in their immutable sample order.

    ``sample_function(model_view, index)`` executes the original B1 sampler.
    Its per-sample autograd graphs remain attached through concatenation and
    slicing at the model boundary. Loss mean8 belongs to the later native loss,
    not this dispatcher. The caller owns adapter mode, optimizer and placement.
    """
    if timeout_seconds <= 0:
        raise ValueError('positive bounded native sampler timeout required')
    device = torch.device(device)
    stream = torch.cuda.current_stream(device) if device.type == 'cuda' else None
    initial_grad = torch.is_grad_enabled()
    messages = queue.Queue()
    stopped, guard = threading.Event(), threading.Lock()
    pending, threads = [], []
    deadline = time.monotonic() + timeout_seconds

    def worker(index):
        view = _SampleModel(index, messages, stopped, pending, guard, timeout_seconds)
        try:
            with ExitStack() as stack:
                stack.enter_context(torch.set_grad_enabled(initial_grad))
                if stream is not None:
                    stack.enter_context(torch.cuda.device(device))
                    stack.enter_context(torch.cuda.stream(stream))
                result = sample_function(view, index)
            messages.put(('done', (index, result, view.ordinal)))
        except BaseException as error:
            messages.put(('error', error))

    for index in range(8):
        thread = threading.Thread(target=worker, args=(index,),
                                  name=f'native-B1-sampler-{index}', daemon=True)
        threads.append(thread); thread.start()
    results, finished, ordinal = [None]*8, {}, 0
    try:
        while len(finished) != 8:
            cohort = {}
            while len(cohort) != 8:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError('bounded native B1 cohort timeout')
                kind, payload = messages.get(timeout=left)
                if kind == 'error':
                    raise payload
                if kind == 'done':
                    index, result, count = payload
                    if count != ordinal or index in finished or cohort:
                        raise ValueError('native B1 call count diverges across samples')
                    finished[index], results[index] = count, result
                    if len(finished) == 8:
                        break
                    continue
                call = payload
                if finished or call.ordinal != ordinal or call.sample in cohort:
                    raise ValueError('native B1 call count/order diverges across samples')
                cohort[call.sample] = call
            if len(finished) == 8:
                break
            calls = [cohort[i] for i in range(8)]
            packed = _pack_B1_kwargs(calls)
            with torch.set_grad_enabled(calls[0].grad):
                output = model(**packed)
                if (not isinstance(output, tuple) or len(output) != 1
                        or not isinstance(output[0], torch.Tensor) or output[0].shape[0] != 8):
                    raise ValueError('native GPipe8 proxy returned an invalid cohort output')
                parts = [(output[0].narrow(0, i, 1),) for i in range(8)]
            for call, part in zip(calls, parts, strict=True):
                call.result = part; call.ready.set()
            ordinal += 1
        if ordinal == 0:
            raise ValueError('native B1 samplers submitted no DiT calls')
    except BaseException as error:
        stopped.set()
        with guard:
            for call in pending:
                if not call.ready.is_set():
                    call.error = error; call.ready.set()
        raise
    finally:
        for thread in threads:
            thread.join(timeout=1)
    if any(thread.is_alive() for thread in threads):
        raise RuntimeError('native B1 sampler thread did not complete teardown')
    return tuple(results), dict(logical_samples=8, rendezvous_calls=ordinal,
        per_sample_call_counts=[finished[i] for i in range(8)],
        rank_local_controls_preserved=True, training_RNG_consumed_by_dispatcher=False,
        physical_model_batch=1, model_boundary='GPipe8_B1_NOT_BATCH8_DIT')
