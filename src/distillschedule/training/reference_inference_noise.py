"""Opt-in reference-output inputs for the existing original rollout seam.

No training sampler or process environment is changed. The old reference
inference ABI continues to call its original sampler when this policy is absent.
"""
from pathlib import Path
import contextlib
import inspect
import sys
import types

from .helios_public_assets import require

POLICY = 'fresh_projection_v1'
SCHEMA = 'distillschedule.reference_cpu_inference_noise.v1'


def enabled(parameters):
    if 'inference_noise_policy' not in parameters:
        return False
    require(parameters['inference_noise_policy'] == POLICY, 'unsupported inference noise policy')
    return True


def _code_signature(code):
    # Same source-versus-live-code pattern as the existing backend attestation;
    # ignore location metadata but retain instructions, constants and ABI.
    return (code.co_code,tuple(_code_signature(v) if isinstance(v,types.CodeType) else v
        for v in code.co_consts),code.co_names,code.co_varnames,code.co_freevars,code.co_cellvars,
        code.co_argcount,code.co_posonlyargcount,code.co_kwonlyargcount,code.co_flags)


def _live_helpers():
    """Verify actual bound callables, including the contextmanager closure."""
    import torch
    from runtime import official_baseline
    from distillschedule.strategies import wan_capture
    functions = []
    for module,name in ((official_baseline,'stable_block_noise'),(wan_capture,'input_stream')):
        fn = getattr(module,name)
        require(type(fn) is types.FunctionType, 'live inference noise helper is not an original function: '+name)
        raw = inspect.unwrap(fn)
        source = Path(module.__file__).resolve()
        require(type(raw) is types.FunctionType and raw.__module__ == module.__name__
            and raw.__name__ == name and raw.__qualname__ == name
            and Path(raw.__code__.co_filename).resolve() == source and raw.__globals__ is vars(module),
            'live inference noise helper source/origin changed: '+name)
        compiled = compile(source.read_bytes(),str(source),'exec',dont_inherit=True,optimize=sys.flags.optimize)
        expected = [v for v in compiled.co_consts if isinstance(v,types.CodeType) and v.co_name == name]
        require(len(expected) == 1 and _code_signature(raw.__code__) == _code_signature(expected[0])
            and raw.__defaults__ is None and raw.__closure__ is None,
            'live inference noise helper code/defaults changed: '+name)
        if name == 'input_stream':
            wrapper = contextlib.contextmanager(raw)
            constructor = fn.__globals__.get('_GeneratorContextManager')
            require(fn is not raw and fn.__code__ is wrapper.__code__ and fn.__defaults__ is None
                and fn.__globals__ is wrapper.__globals__ is vars(contextlib)
                and constructor is contextlib._GeneratorContextManager and isinstance(constructor,type)
                and constructor.__module__ == contextlib.__name__
                and constructor.__qualname__ == '_GeneratorContextManager'
                and fn.__kwdefaults__ is None and raw.__kwdefaults__ is None
                and fn.__closure__ is not None and len(fn.__closure__) == 1
                and fn.__closure__[0].cell_contents is raw,
                'live inference noise helper contextmanager binding changed: '+name)
        else:
            require(fn is raw and raw.__kwdefaults__ == dict(device='cpu',dtype=torch.float32),
                'live inference noise helper default binding changed: '+name)
        functions.append(fn)
    return tuple(functions)


def protocol(dtype):
    from runtime import official_baseline
    from distillschedule.strategies import wan_capture
    from . import wan_native_inference
    from distillschedule.workloads.best_known.registry import file_sha256
    _live_helpers()
    require(dtype in ('torch.float32', 'torch.bfloat16'), 'unsupported inference noise dtype')
    return dict(schema_version=SCHEMA, policy=POLICY, gamma=1/3,
        operation_prefix=wan_native_inference.NOISE_PREFIX,
        tensor_draw_order=['entry_noise','rollout_noise','block_noise/1','block_noise/2'],
        iid_dtype=dtype, block_dtype='torch.float32',
        iid_distribution='CPU torch.randn BCTHW in separate labelled input_stream contexts',
        block_distribution='stable_block_noise: CPU FP32 iid4, subtract mean, multiply sqrt(4/3)',
        block_covariance='(4/3)*(I4-ones(4,4)/4), rank3 PSD; no jitter',
        seed_derivation='input_stream(torch.initial_seed(),schema_version,tensor_name); public runtime trainer.seed; caller CPU/Python RNG restored',
        live_binding='original source-compiled function code/origin/defaults and original contextmanager closure',
        exits='one section; configured last step in each of three original pyramid stages',
        compatibility='NEW_OPT_IN_OUTPUT_RNG_ABI_NOT_OLD_CHOLESKY_BITWISE_OR_TRAINING_CONTINUATION',
        source={name:file_sha256(path) for name,path in (
            ('policy',Path(__file__)), ('primitive',Path(official_baseline.__file__)),
            ('streams',Path(wan_capture.__file__)), ('consumer',Path(wan_native_inference.__file__)))})


def prepare(parameters, dtype):
    """Private draws, exact original consumer layout and complete input census."""
    import torch
    from .wan_native_inference import _FreshNoiseInputs, NOISE_PREFIX
    stable_block_noise,input_stream = _live_helpers()
    require(parameters.get('noise_protocol') == protocol(str(dtype)), 'inference noise protocol/source changed')
    # session._generate_values has just set the independent runtime seed. The
    # exported factory seed stays provenance and must not select output draws.
    seed = torch.initial_seed()
    require(type(seed) is int and 0 <= seed < 2**63, 'invalid inference noise seed')
    shape = parameters['geometry']['noise_shape']
    b,c,t,h,w = shape
    steps = parameters['native']['rollout']['stage2_num_inference_steps_list']
    values = {}
    for name in ('entry_noise','rollout_noise'):
        with input_stream(seed,SCHEMA,name):
            values[NOISE_PREFIX+'/'+name] = torch.randn(shape,dtype=dtype,device='cpu')
    for stage in (1,2):
        name = 'block_noise/'+str(stage)
        height,width = h//(2**(2-stage)),w//(2**(2-stage))
        with input_stream(seed,SCHEMA,name):
            noise = stable_block_noise(b*c*t*(height//2)*(width//2),device='cpu',dtype=torch.float32)
        values[NOISE_PREFIX+'/'+name] = noise.view(b,c,t,height//2,width//2,2,2).permute(
            0,1,2,3,5,4,6).reshape(b,c,t,height,width)
    sequence = [dict(stage=i,denoising_exit=count-1,call_count=count,
        denoising_step_budget=count,latent_shape=[b,c,t,h//(2**(2-i)),w//(2**(2-i))])
        for i,count in enumerate(steps)]
    return _FreshNoiseInputs(values,sequence)
