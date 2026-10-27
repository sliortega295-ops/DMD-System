"""Measured implementation helper, relocated from r1/scripts/sf_checkpoint_abi.py.

The public owner validates capabilities and lifecycle; this module does not
launch experiments or claim framework GPU acceptance. Source lock records all
original and relocated executable definitions.
"""
import ast
import contextlib
import hashlib
import inspect
import textwrap

@contextlib.contextmanager
def native_checkpoint_toggles(*classes):
    saved = []
    receipts = []
    try:
        for cls in classes:
            native = cls.__dict__.get('_set_gradient_checkpointing')
            if native is None or list(inspect.signature(native).parameters) != ['self', 'module', 'value']:
                raise RuntimeError('SF_CHECKPOINT_ABI_SOURCE_CHANGED')
            source = textwrap.dedent(inspect.getsource(native))
            tree = ast.parse(source).body[0]
            expected = ast.parse('self.gradient_checkpointing = value').body[0]
            if len(tree.body) != 1 or ast.dump(tree.body[0]) != ast.dump(expected):
                raise RuntimeError('SF_CHECKPOINT_ABI_SETTER_SEMANTICS_CHANGED')

            def enable(self, gradient_checkpointing_func=None, *, _native=native):
                if gradient_checkpointing_func is not None:
                    raise RuntimeError('SF_CUSTOM_CHECKPOINT_FUNCTION_NOT_ADMITTED')
                _native(self, self, value=True)

            def disable(self, *, _native=native):
                _native(self, self, value=False)
            for name, method in [('enable_gradient_checkpointing', enable), ('disable_gradient_checkpointing', disable)]:
                existed = name in cls.__dict__
                previous = cls.__dict__.get(name)
                setattr(cls, name, method)
                saved.append((cls, name, existed, previous, method))
            receipts.append(dict(model=cls.__module__ + '.' + cls.__name__, native_setter_sha256=hashlib.sha256(source.encode()).hexdigest(), action='invoke_unchanged_native_boolean_setter', training_forward_or_checkpoint_changed=False))
        yield receipts
    finally:
        for cls, name, existed, previous, method in reversed(saved):
            if cls.__dict__.get(name) is not method:
                raise RuntimeError('SF_CHECKPOINT_ABI_PATCH_OWNERSHIP_CHANGED')
            if existed:
                setattr(cls, name, previous)
            else:
                delattr(cls, name)
