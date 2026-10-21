"""Opt-in block-local autograd retention instead of checkpoint recomputation.

No result memoization, precision change, detach, new RNG draw, or offload. The
original module and optional adapter scope are executed once with autograd.
Unselected/auxiliary modules use the unchanged checkpoint dispatcher. Python
counts describe eager/capture construction, not CUDA Graph replay invocations.
"""
from __future__ import annotations


def install(model, *, indices, role, expected_block_count):
    if not isinstance(indices, tuple) or any(type(i) is not int for i in indices):
        raise TypeError('retained block indices must be an exact tuple of integers')
    blocks = tuple(getattr(model, 'blocks', ()))
    if len(blocks) != expected_block_count or len({id(b) for b in blocks}) != len(blocks):
        raise ValueError('block count or aliasing changed')
    if tuple(sorted(set(indices))) != indices or any(i < 0 or i >= len(blocks) for i in indices):
        raise ValueError('retained block indices must be sorted, unique and in range')
    receipt = dict(schema_version='distillschedule.checkpoint_block_retention.v1',
        enabled=bool(indices), role=role, block_indices=list(indices),
        scope='EXACT_ORIGINAL_AUTOGRAD_INTERMEDIATES_THROUGH_BLOCK_BACKWARD',
        retained_forward_calls=0, checkpoint_forward_calls=0,
        block_forward_counts={str(i): 0 for i in range(len(blocks))},
        graph_count_scope='PYTHON_EAGER_AND_CAPTURE_ONLY_NOT_GRAPH_REPLAY',
        parameter_dtype_changed=False, weights_merged=False, uses_offload=False,
        changes_application_calls=False, changes_optimizer=False,
        source_wrapper_preserved=True)
    if not indices:
        return receipt
    if hasattr(model, '_distillschedule_block_retention'):
        raise ValueError('block retention installed twice')
    if not getattr(model, 'gradient_checkpointing', False):
        raise ValueError('enabled original checkpoint policy required')
    original = getattr(model, '_gradient_checkpointing_func', None)
    if not callable(original):
        raise ValueError('missing original checkpoint dispatcher')
    known = {id(m) for m in model.modules()}
    selected = {id(blocks[i]): i for i in indices}

    def dispatch(module, *args, **kwargs):
        if id(module) not in known:
            raise ValueError('checkpoint received a module outside the original model')
        if id(module) not in selected:
            receipt['checkpoint_forward_calls'] += 1
            return original(module, *args, **kwargs)
        if any(k in kwargs for k in ('use_reentrant', 'context_fn', 'determinism_check')):
            raise ValueError('checkpoint-only options cannot be passed to retained module')
        import torch
        if not torch.is_grad_enabled():
            raise ValueError('retention dispatcher must only enter the original grad path')
        index = selected[id(module)]
        receipt['retained_forward_calls'] += 1
        receipt['block_forward_counts'][str(index)] += 1
        function = module.__call__
        wrapper = getattr(model, '_distillschedule_lora_checkpoint_function_wrapper', None)
        if wrapper is not None:
            function = wrapper(function)
        return function(*args, **kwargs)

    model._gradient_checkpointing_func = dispatch
    object.__setattr__(model, '_distillschedule_block_retention', receipt)
    return receipt
