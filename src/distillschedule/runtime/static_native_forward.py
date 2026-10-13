"""Source-guarded PP boundary extraction, not an alternate model forward.

This helper mechanically separates the admitted forward at its existing block
loop. It preserves prefix, checkpoint dispatch and postfix ASTs. No caching,
adapter merging, new kernels, precision change or native timing is implied.
The model-specific binder remains responsible for TP/PP parameter ownership.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from pathlib import Path
import textwrap


def _select_context(values, names):
    # Some fields exist only on the native NaViT branch. Retain their original
    # presence rather than inventing a value for an unexecuted source branch.
    return {name: values[name] for name in names if name in values}


def _context_return(names):
    return ast.Return(ast.Call(ast.Name('_static_select_context', ast.Load()),
        [ast.Call(ast.Name('locals', ast.Load()), [], []),
         ast.Tuple([ast.Constant(name) for name in names], ast.Load())], []))


def _function(name, args, body):
    return ast.FunctionDef(name=name, args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg=x) for x in args], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=body, decorator_list=[], returns=None)


class _SliceBlockLoop(ast.NodeTransformer):
    def visit_For(self, node):
        node = self.generic_visit(node)
        iterator = node.iter
        if (isinstance(iterator, ast.Call) and isinstance(iterator.func, ast.Name)
                and iterator.func.id == 'enumerate' and len(iterator.args) == 1
                and isinstance(iterator.args[0], ast.Attribute)
                and isinstance(iterator.args[0].value, ast.Name)
                and iterator.args[0].value.id == 'self' and iterator.args[0].attr == 'blocks'):
            iterator.args[0] = ast.Subscript(iterator.args[0],
                ast.Slice(ast.Name('_start', ast.Load()), ast.Name('_stop', ast.Load()), None), ast.Load())
            iterator.args.append(ast.Name('_start', ast.Load()))
        return node


class AdmittedForwardSplit:
    def __init__(self, *, prefix, blocks, suffix, receipt, block_count):
        self.prefix, self._blocks, self.suffix = prefix, blocks, suffix
        self.receipt, self.block_count = receipt, block_count

    def blocks(self, model, context, start, stop):
        if not (type(start) is int and type(stop) is int and 0 <= start < stop <= self.block_count):
            raise ValueError('invalid admitted native block range')
        return self._blocks(model, context, start, stop)


def create_admitted_forward_split(model, *, expected_source_sha256, expected_blocks):
    """Split checked source without manually recoding its projections/math.

    Decorator-based attention scaling is intentionally not synthesized. The
    native PP binder must reject non-default attention_kwargs/LoRA scaling.
    This component does not authorize an unsupported native model or GAN head.
    """
    function = inspect.unwrap(type(model).forward)
    source_path = Path(inspect.getsourcefile(function))
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_sha != expected_source_sha256:
        raise ValueError('native forward source SHA mismatch')
    if len(model.blocks) != expected_blocks or expected_blocks <= 0 or expected_blocks % 2:
        raise ValueError('admitted native block count mismatch')
    source = textwrap.dedent(inspect.getsource(function))
    node = ast.parse(source).body[0]
    boundaries = []
    for index, statement in enumerate(node.body):
        if not isinstance(statement, ast.If):
            continue
        loops = [x for x in ast.walk(statement) if isinstance(x, ast.For)
            and isinstance(x.iter, ast.Call) and isinstance(x.iter.func, ast.Name)
            and x.iter.func.id == 'enumerate' and len(x.iter.args) == 1
            and isinstance(x.iter.args[0], ast.Attribute)
            and isinstance(x.iter.args[0].value, ast.Name)
            and x.iter.args[0].value.id == 'self' and x.iter.args[0].attr == 'blocks']
        if loops:
            if len(loops) != 2 or 'gradient_checkpointing' not in ast.unparse(statement.test):
                raise ValueError('unsupported native checkpoint/block-loop contract')
            boundaries.append(index)
    if len(boundaries) != 1:
        raise ValueError('native block-loop boundary is not unique')
    boundary = boundaries[0]
    before, loop, after = node.body[:boundary], node.body[boundary], node.body[boundary+1:]
    defined = {a.arg for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs}
    defined |= {x.id for statement in before for x in ast.walk(statement)
        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)}
    used = {x.id for statement in [loop, *after] for x in ast.walk(statement)
        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
    names = tuple(sorted((defined & used) - {'self'}))
    # Conditional source fields must be read only where the original branch
    # reads them. Bind a missing field to None without changing branch guards.
    assignments = [ast.Assign([ast.Name(name, ast.Store())], ast.Call(
        ast.Attribute(ast.Name('_context', ast.Load()), 'get', ast.Load()),
        [ast.Constant(name)], [])) for name in names]
    prefix = copy.deepcopy(node)
    prefix.name, prefix.decorator_list = '_static_native_prefix', []
    prefix.body = copy.deepcopy(before) + [_context_return(names)]
    block_node = _SliceBlockLoop().visit(copy.deepcopy(loop))
    blocks = _function('_static_native_blocks', ['self', '_context', '_start', '_stop'],
        copy.deepcopy(assignments) + [block_node, _context_return(names)])
    suffix = _function('_static_native_suffix', ['self', '_context'],
        copy.deepcopy(assignments) + copy.deepcopy(after))
    generated = ast.fix_missing_locations(ast.Module(body=[prefix, blocks, suffix], type_ignores=[]))
    namespace = dict(function.__globals__)
    namespace['_static_select_context'] = _select_context
    exec(compile(generated, str(source_path) + '::<checked-PP-boundary>', 'exec'), namespace)
    receipt = dict(source_sha256=source_sha,
        forward_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        generated_ast_sha256=hashlib.sha256(ast.dump(generated).encode()).hexdigest(),
        context_names=list(names), original_checkpoint_dispatch_preserved=True,
        prefix_and_postfix='MECHANICALLY_EXTRACTED_ADMITTED_SOURCE_AST',
        optimizations=False, GPU='NOT_RUN_COMPONENT_ONLY', native_3D_timing='NOT_RUN')
    return AdmittedForwardSplit(prefix=namespace[prefix.name], blocks=namespace[blocks.name],
        suffix=namespace[suffix.name], receipt=receipt, block_count=expected_blocks)
