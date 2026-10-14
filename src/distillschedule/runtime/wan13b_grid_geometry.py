"""Opt-in geometry binding for fresh Wan grid traces; default paths untouched.

Reuses the shape-only adapter validated by the prior resolution experiment.
Only tracer base dimensions and the final latent shape assertions change.
The original sampler, tensor operations, exits, RNG, dtype and loss remain.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import sys
import textwrap
from pathlib import Path

from .wan13b_parameter_grid import WanParameterGridSpec


def _adapt_rollout(function, spatial_shape):
    height, width = spatial_shape
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    counts = dict(base_height=0, base_width=0, output_shapes=0)

    class Geometry(ast.NodeTransformer):
        def visit_keyword(self, node):
            if node.arg in ('base_height', 'base_width'):
                expected = 48 if node.arg == 'base_height' else 80
                if not isinstance(node.value, ast.Constant) or node.value.value != expected:
                    raise ValueError('native rollout audit geometry source changed')
                node.value = ast.copy_location(ast.Constant(
                    height if node.arg == 'base_height' else width), node.value)
                counts[node.arg] += 1
            return self.generic_visit(node)

        def _shape(self, node):
            self.generic_visit(node)
            if (len(node.elts) == 5
                    and all(isinstance(x, ast.Constant) for x in node.elts[1:])
                    and [x.value for x in node.elts[1:]] == [16, 9, 48, 80]):
                first = node.elts[0]
                if not ((isinstance(first, ast.Constant) and first.value == 1)
                        or (isinstance(first, ast.Name) and first.id == 'batch_size')):
                    raise ValueError('unadmitted rollout batch assertion')
                node.elts[-2:] = [ast.Constant(height), ast.Constant(width)]
                counts['output_shapes'] += 1
            return node

        visit_List = _shape
        visit_Tuple = _shape

    tree = Geometry().visit(tree)
    ast.fix_missing_locations(tree)
    if counts['base_height'] != 2 or counts['base_width'] != 2 or not counts['output_shapes']:
        raise ValueError('incomplete native rollout audit geometry binding')
    namespace = dict(inspect.unwrap(function).__globals__)
    exec(compile(tree, '<explicit-Wan-grid-rollout-geometry>', 'exec'), namespace)
    adapted = namespace[function.__name__]
    adapted.grid_geometry_receipt = dict(
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        ast_sha256=hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
        substitutions=counts, latent_spatial_shape=[height, width],
        change_scope='TRACER_GEOMETRY_AND_OUTPUT_ASSERTIONS_ONLY')
    return adapted


def bind_grid_rollout_geometry(stack, root, workload):
    """Bind already loaded source aliases, with restoration owned by ExitStack."""
    if not isinstance(workload, WanParameterGridSpec):
        raise TypeError('explicit validated workload spec required')
    shape = workload.spatial_shapes[-1]
    if shape == (48, 80):
        return dict(status='CANONICAL_384_GEOMETRY_UNMODIFIED',
                    latent_spatial_shape=list(shape), endpoints=[])
    from runtime import gst_pool_helios, static_fsdp2_p0_helios, critic_pool_helios
    root = Path(root).resolve()
    endpoints = ((gst_pool_helios, 'run_generator_update_trace_rollout'),
                 (static_fsdp2_p0_helios, 'run_generator_trace_rollout_batch'),
                 (critic_pool_helios, 'run_generator_trace_rollout'))
    receipts = []
    for module, name in endpoints:
        original = getattr(module, name)
        adapted = _adapt_rollout(original, shape)
        bindings = []
        for loaded in tuple(sys.modules.values()):
            path = getattr(loaded, '__file__', None)
            if path is None or not Path(path).resolve().is_relative_to(root):
                continue
            for alias, value in tuple(vars(loaded).items()):
                if value is original:
                    bindings.append(f'{loaded.__name__}.{alias}')
                    setattr(loaded, alias, adapted)
                    stack.callback(setattr, loaded, alias, original)
        if f'{module.__name__}.{name}' not in bindings:
            raise RuntimeError('native rollout endpoint not bound in the admitted source root')
        receipts.append(dict(endpoint=f'{module.__name__}.{name}',
                             aliases=bindings, **adapted.grid_geometry_receipt))
    return dict(status='PASS_EXPLICIT_GRID_GEOMETRY_BINDING',
                latent_spatial_shape=list(shape), endpoints=receipts)
