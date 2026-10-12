"""Execute original helper argument binding with a stub sampler; not GPU math."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
import sys
import pytest

torch = pytest.importorskip('torch')
from distillschedule.strategies.wan import OriginalWanTask as Task

class Captured(Exception):
    pass


def extract(path, name, scope):
    node = next(n for n in ast.parse(Path(path).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
                        names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, 'exec'), scope)
    return scope[name]

@pytest.mark.parametrize('grad', [False, True])
@pytest.mark.parametrize('task', [Task(),Task(resolution=(256,384),video_frames=17,
                                         fake_updates=1,stage2_steps=(3,3,2))])
def test_native_helper_uses_actual_parameters_without_new_sampler(monkeypatch, grad, task):
    observed={}
    def sampler(**kw):
        observed.update(kw, grad_enabled=torch.is_grad_enabled())
        raise Captured
    for name in ('helios','helios.scheduler','helios.scheduler.scheduling_helios',
                 'helios.utils','helios.utils.utils_helios_post'):
        monkeypatch.setitem(sys.modules,name,ModuleType(name))
    sys.modules['helios.scheduler.scheduling_helios'].HeliosScheduler=lambda **kw:NS(**kw)
    sys.modules['helios.utils.utils_helios_post'].run_generator=sampler
    control={'call_count':3,'pyramid_stage_call_sequence':[{'stage':i,'call_count':1} for i in range(3)]}
    op=NS(manifest={'contract':task.trace_contract()},operation=lambda name:{'control':control},
          validation={'call_count':3,'stage_call_count':{0:1,1:1,2:1}})
    names=('entry_noise','indices_hidden_states','indices_history_short','indices_history_mid',
           'indices_history_long','history_short','history_mid','history_long','continuous_history','text_embedding')
    tensors={name:torch.zeros(1) for name in names}
    loaded=NS(tensors=tensors,operation_trace=op,rollout_operation='step/10/generator/rollout',logical_step=10,
              pool_entry=NS(synthetic_trace_prefix='step/10/critic'))
    model=NS(enable_adapters=lambda:None,train=lambda:None)
    tracer=lambda model,**kw:NS(model=model,**kw)
    scope=dict(torch=torch,SimpleNamespace=NS,LightweightNestedCallAudit=tracer,
               make_batched_loaded_entry=lambda entries:loaded,ForceNoGradCalls=lambda x:x,
               generator_nograd_cross_kv_cache_scope=lambda *a,**k:nullcontext(),
               generator_nograd_static_context_cache_scope=lambda *a,**k:nullcontext(),
               generator_nograd_residual_fusion_scope=lambda:nullcontext(),_module=lambda x:x)
    path='runtime/gst_pool_helios.py' if grad else 'runtime/static_fsdp2_p0_helios.py'
    name='run_generator_update_trace_rollout' if grad else 'run_generator_trace_rollout_batch'
    fn=extract(path,name,scope)
    args=dict(device=torch.device('cpu'),clean_runtime=True,detailed_profile=False,
              collect_diagnostics=False,runtime_cache={})
    args.update(engine=model,entry=loaded) if grad else args.update(model=model,loaded_entries=(loaded,))
    with pytest.raises(Captured):fn(**args)
    assert observed['stage2_num_inference_steps_list']==list(task.stage2_steps)
    assert observed['num_critic_input_frames']==task.latent_shape[2]
    assert (observed['transformer'].base_height,observed['transformer'].base_width)==task.latent_shape[-2:]
    assert observed['grad_enabled'] is grad
    assert observed['operation_trace'] is op
    assert observed['last_step_only'] is False and observed['is_enable_stage2'] is True
    assert observed['history_sizes']==[16,2,1] and observed['num_rollout_sections']==1
    assert observed['noise'] is tensors['entry_noise']
