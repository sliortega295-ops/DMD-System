"""Measured implementation helper, relocated from r2_fsdp2_baselines/scripts/sf_host_kv_indices.py.

The public owner validates capabilities and lifecycle; this module does not
launch experiments or claim framework GPU acceptance. Source lock records all
original and relocated executable definitions.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap
import types
NATIVE_PIPELINE_SHA256 = '0fbef7cae39a13338ae555acc2f114dc28a0e15f4dfcf4303640192e42ab7e96'
KEYS = ('global_end_index', 'local_end_index')

def require(value, reason):
    if not value:
        raise ValueError(reason)

def build_host_initializer(original):
    """Compile the source-pinned original with exactly two device literals."""
    function = original.__func__ if hasattr(original, '__func__') else original
    path = Path(inspect.getsourcefile(function)).resolve()
    require(path.name == 'self_forcing_training.py' and hashlib.sha256(path.read_bytes()).hexdigest() == NATIVE_PIPELINE_SHA256, 'NATIVE_KV_INITIALIZER_SOURCE_CHANGED')
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef) and (tree.body[0].name == '_initialize_kv_cache'), 'NATIVE_INITIALIZER_ABI')
    before = ast.dump(tree, include_attributes=False)
    changes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value if isinstance(k, ast.Constant) else None for k in node.keys]
        if not all((k in keys for k in KEYS)):
            continue
        require(set(keys) == {'k', 'v', *KEYS}, 'NATIVE_CACHE_KEYS_CHANGED')
        for key in KEYS:
            call = node.values[keys.index(key)]
            require(isinstance(call, ast.Call) and ast.unparse(call.func) == 'torch.tensor' and (len(call.args) == 1) and (ast.literal_eval(call.args[0]) == [0]) and ({k.arg for k in call.keywords} == {'dtype', 'device'}), 'NATIVE_POSITION_NOT_ZERO_INT')
            dtype = next((k for k in call.keywords if k.arg == 'dtype'))
            device = next((k for k in call.keywords if k.arg == 'device'))
            require(ast.unparse(dtype.value) == 'torch.long' and isinstance(device.value, ast.Name) and (device.value.id == 'device'), 'NATIVE_POSITION_LAYOUT_CHANGED')
            device.value = ast.Constant('cpu')
            changes.append(key)
    require(sorted(changes) == sorted(KEYS), 'HOST_INDEX_REWRITE_CENSUS')
    after = ast.dump(tree, include_attributes=False)
    scope = dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<sf-host-kv-position-metadata>', 'exec'), scope)
    return (scope['_initialize_kv_cache'], dict(source_sha256=NATIVE_PIPELINE_SHA256, before_ast_sha256=hashlib.sha256(before.encode()).hexdigest(), after_ast_sha256=hashlib.sha256(after.encode()).hexdigest(), changed_keys=changes, index_dtype='torch.int64', native_attention_unchanged=True, native_kv_tensors_unchanged=True, index_tensor_gpu_value_reads=0))

class HostKVIndices:

    def __init__(self, trainer):
        self.trainer = trainer
        require(trainer.model.inference_pipeline is None, 'HOST_INDICES_MUST_BIND_BEFORE_PIPELINE')
        self.original = trainer.model._initialize_inference_pipeline
        self.initializer_original = None
        self.handles = []
        self.record = dict(mechanism='HOST_ONLY_NATIVE_KV_POSITION_METADATA', pipeline_initializations=0, cache_initializations=0, index_tensors_constructed=0, native_attention_calls=0, index_tensor_gpu_value_reads=0)

        def initialize(_model):
            result = self.original()
            pipeline = trainer.model.inference_pipeline
            self.record['pipeline_initializations'] += 1
            require(self.record['pipeline_initializations'] == 1 and pipeline.kv_cache1 is None, 'PIPELINE_REINITIALIZED_OR_LIVE')
            original = pipeline._initialize_kv_cache
            function, proof = build_host_initializer(original)
            self.initializer_original = original
            self.record['source_proof'] = proof

            def initialize_cache(this, *args, **kwargs):
                value = function(this, *args, **kwargs)
                require(len(this.kv_cache1) == 30, 'NATIVE_CACHE_BLOCK_CENSUS')
                for cache in this.kv_cache1:
                    self.validate_indices(cache)
                self.record['cache_initializations'] += 1
                self.record['index_tensors_constructed'] += 2 * len(this.kv_cache1)
                return value
            pipeline._initialize_kv_cache = types.MethodType(initialize_cache, pipeline)
            return result
        trainer.model._initialize_inference_pipeline = types.MethodType(initialize, trainer.model)
        for block in trainer.model.generator.model.blocks:
            require(block.self_attn.local_attn_size == -1, 'LOCAL_EVICTION_NOT_QUALIFIED')

            def before(module, args, kwargs):
                cache = kwargs.get('kv_cache', args[5] if len(args) > 5 else None)
                require(cache is not None, 'CAUSAL_CACHE_REQUIRED')
                self.validate_indices(cache)
                self.record['native_attention_calls'] += 1
            self.handles.append(block.self_attn.register_forward_pre_hook(before, with_kwargs=True))

    @staticmethod
    def validate_indices(cache):
        import torch
        for key in KEYS:
            t = cache[key]
            require(isinstance(t, torch.Tensor) and t.device.type == 'cpu' and (t.dtype == torch.long) and (tuple(t.shape) == (1,)) and (not t.requires_grad), 'HOST_INDEX_EXECUTION_CONTRACT:' + key)

    def receipt(self):
        return dict(self.record, index_devices=['cpu'], k_v_placement='UNCHANGED_NATIVE_GPU', state_role='DERIVED_INTEGER_CONTROL_METADATA_NOT_MODEL_STATE')

    def close(self):
        self.trainer.model._initialize_inference_pipeline = self.original
        if self.initializer_original is not None:
            self.trainer.model.inference_pipeline._initialize_kv_cache = self.initializer_original
        for handle in self.handles:
            handle.remove()
