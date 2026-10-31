"""Byte-pinned real helper with explicitly tiny checkpointed CPU sentinel.

The production helper and scheduler are compiled unchanged. Only the model
is a small CPU checkpoint sentinel; this is not full-model GPU capacity evidence.
The original source is reconstructed in pytest-owned temporary storage by
reversing the two reviewed detach insertions. No historical payload is vendored.
No denoising loop, scheduler math, loss helper, or history math is synthesized.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager, nullcontext
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace as NS
from typing import Literal
import weakref

import numpy as np
import torch
import torch.nn.functional as F
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils.checkpoint import checkpoint


ROOT = Path(__file__).resolve().parents[1]  # Rebound to pytest-owned source fixtures.
POST = 'third_party/helios_compat/source/helios/utils/utils_helios_post.py'
BASE = 'third_party/helios_compat/source/helios/utils/utils_base.py'
HISTORY = 'third_party/helios_compat/source/helios/utils/utils_helios_base.py'
SCHEDULER = 'third_party/helios_compat/source/helios/scheduler/scheduling_helios.py'
EXPECTED_POST = 'e3ff9e5f8bf025ee740e436bc0c209c972f1e56eb0236b968fe6bd8284107004'

_provider = os.environ.get('DMD_TEST_REARRANGE_PROVIDER', 'einops')
assert _provider in ('functorch', 'einops'), 'unknown dependency provider: fail closed'
if _provider == 'einops':
    from einops import rearrange
    REARRANGE_PROVIDER = 'installed einops.rearrange'
else:
    from functorch.einops import rearrange
    REARRANGE_PROVIDER = 'Torch-bundled functorch.einops.rearrange (scoped substitution)'



@contextmanager
def isolated_cpu_test_state():
    """Restore CPU RNG, threads and the exact temporary import namespaces."""
    relevant = lambda name: (name == "runtime" or name.startswith("runtime.")
                             or name.startswith("helios_liveness_scheduler_"))
    modules = {name: module for name, module in sys.modules.copy().items() if relevant(name)}
    attributes = {name: vars(module).copy() for name, module in modules.items()}
    state = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state(), torch.get_num_threads())
    initialized = torch.cuda.is_initialized()
    try:
        yield
    finally:
        torch.set_rng_state(state[0])
        random.setstate(state[1])
        np.random.set_state(state[2])
        torch.set_num_threads(state[3])
        for name in tuple(sys.modules):
            if relevant(name) and name not in modules:
                del sys.modules[name]
        sys.modules.update(modules)
        for name, before in attributes.items():
            namespace = vars(modules[name])
            namespace.clear()
            namespace.update(before)
        assert torch.cuda.is_initialized() == initialized

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_functions(path, names, namespace):
    """Compile unchanged AST nodes, retaining original filename/line locations."""
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return {n.name: namespace[n.name] for n in nodes}


def alias_audit(path, *, candidate):
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == 'inference_with_trajectory_stage2')
    parents = {}
    for node in ast.walk(fn):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    refs = sorted((n for n in ast.walk(fn) if isinstance(n, ast.Name)
                   and n.id == 'start_point_list'), key=lambda n: n.lineno)
    assert [n.lineno for n in refs] == [1672, 1731, 1810], 'new alias consumer: audit fail closed'
    store, append, read = refs
    assert isinstance(parents[store], ast.Assign)
    assert isinstance(parents[append], ast.Attribute) and parents[append].attr == 'append'
    read_parent = parents[read]
    assert isinstance(read_parent, ast.Subscript)
    call = parents[read_parent]
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'add_noise'
    assert call.args[1] is read_parent
    ancestors = []
    n = call
    while n in parents:
        n = parents[n]
        ancestors.append(n)
    assert any(isinstance(n, ast.With) and any(
        isinstance(i.context_expr, ast.Call) and ast.unparse(i.context_expr.func) == 'torch.no_grad'
        for i in n.items) for n in ancestors), 'stored alias reaches grad consumer'
    expected = ['[noisy_model_input.detach()]', 'start_point_list.append(noisy_model_input.detach())'] if candidate else [
        '[noisy_model_input]', 'start_point_list.append(noisy_model_input)']
    assert ast.unparse(parents[store].value) == expected[0]
    assert ast.unparse(parents[parents[append]]) == expected[1]
    # The only consumer is exact out-of-place math, with no identity/autograd inspection.
    add = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'add_noise')
    assert not any(isinstance(n, ast.AugAssign) for n in ast.walk(add))
    assert not any(isinstance(n, ast.Attribute) and (n.attr.endswith('_') or n.attr in
        {'requires_grad', 'grad_fn', 'is_leaf', 'detach', 'data'}) for n in ast.walk(add))
    return dict(function=fn.name, references=[dict(line=n.lineno, context=type(n.ctx).__name__) for n in refs],
                sole_consumer='add_noise second argument under torch.no_grad',
                stores_detached=candidate)


def load(mode):
    source = ROOT / mode / POST
    original = ROOT / 'original' / POST
    assert digest(original) == EXPECTED_POST
    namespace = dict(torch=torch, F=F, math=math, random=random, os=os,
                     rearrange=rearrange, Literal=Literal, __name__='helios_liveness_exact_' + mode)
    # Single CPU process only: no collective or distributed claim.
    namespace['broadcast'] = lambda tensor, from_process: tensor
    exact_functions(ROOT / 'original' / BASE, ['calculate_shift', 'apply_schedule_shift'], namespace)
    exact_functions(ROOT / 'original' / HISTORY, ['downsample_corrupt', 'get_corrupt_noise_sigma',
        'corrupt_history_latents', 'add_saturation_to_history_latents'], namespace)
    exact_functions(source, ['generate_and_sync_flag', 'sample_block_noise', 'add_noise',
        'convert_flow_pred_to_x0', 'convert_xt_pred_to_x0', 'inference_with_trajectory_stage2'], namespace)
    # Real baseline stable PSD sampler is called through its unchanged source function.
    baseline_path = ROOT / 'original' / 'runtime/official_baseline.py'
    if 'runtime.official_baseline' not in sys.modules:
        import types
        if 'runtime' not in sys.modules:
            package = types.ModuleType('runtime')
            package.__path__ = [str(baseline_path.parent)]
            sys.modules['runtime'] = package
        spec = importlib.util.spec_from_file_location('runtime.official_baseline', baseline_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    scheduler_path = ROOT / 'original' / SCHEDULER
    spec = importlib.util.spec_from_file_location('helios_liveness_scheduler_' + mode, scheduler_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return NS(post=NS(**namespace), scheduler=module.HeliosScheduler, source=source)


def byte_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(
            a.detach().contiguous().reshape(-1).view(torch.uint8),
            b.detach().contiguous().reshape(-1).view(torch.uint8))
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(byte_equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(byte_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def detached(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, dict):
        return {k: detached(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return type(x)(detached(v) for v in x)
    return x


class SavedBox:
    def __init__(self, tensor):
        self.tensor = tensor


class StateTracker:
    """Weak tensor AND C++ storage witnesses; no tensor root retained by observer."""
    def __init__(self):
        self.records = []
        self.by_tensor = {}
        self.unpack_calls = []

    def boundary(self, tensor, label):
        record = dict(label=label, tensor=weakref.ref(tensor), storage=StorageWeakRef(tensor.untyped_storage()),
                      bytes=tensor.untyped_storage().nbytes(), box=None)
        self.records.append(record)
        self.by_tensor[id(tensor)] = record

    def pack(self, tensor):
        box = SavedBox(tensor)
        record = self.by_tensor.get(id(tensor))
        if record is not None and record['tensor']() is tensor:
            record['box'] = weakref.ref(box)
        return box

    def unpack(self, box):
        record = self.by_tensor.get(id(box.tensor))
        if record is not None and record['tensor']() is box.tensor:
            self.unpack_calls.append(record['label'])
        return box.tensor

    def snapshot(self):
        return [dict(label=r['label'], tensor_alive=r['tensor']() is not None,
                     storage_alive=not r['storage'].expired(),
                     checkpoint_box_alive=r['box'] is not None and r['box']() is not None,
                     bytes=r['bytes']) for r in self.records]


class Sentinel(torch.nn.Module):
    """Three independent tiny differentiable stage parameter groups, real checkpoint."""
    def __init__(self, tracker, *, observe_sections=False):
        super().__init__()
        self.scale = torch.nn.ParameterList(torch.nn.Parameter(torch.tensor(.7 + i * .1)) for i in range(3))
        self.a = torch.nn.ParameterList(torch.nn.Parameter(torch.tensor(.21 + i * .03)) for i in range(3))
        self.b = torch.nn.ParameterList(torch.nn.Parameter(torch.tensor(.13 + i * .02)) for i in range(3))
        self.bias = torch.nn.ParameterList(torch.nn.Parameter(torch.tensor(.07 + i * .01)) for i in range(3))
        self.frozen = torch.nn.Parameter(torch.tensor(.19), requires_grad=False)
        self.tracker = tracker
        self.calls, self.core_calls, self.clears = [], [], 0
        self.observe_sections=observe_sections
        self.section=-1
        self.section_by_label={}
        self.section_input_grads={}
        self.section_parameter_use_grads={}
        self.section_history_grads={}

    @staticmethod
    def _gradient_recorder(mapping,label):
        # Captures only a dictionary and text label, never the observed tensor.
        def record(gradient):
            mapping[label]=gradient.detach().clone()
        return record

    def _core(self, boundary, a, b, bias, frozen, label):
        self.core_calls.append(label)
        mask = (torch.rand_like(boundary) > .2).to(boundary.dtype)
        return torch.sin(boundary * a + bias) * b * mask + boundary * frozen

    def forward(self, *, hidden_states, timestep, encoder_hidden_states, latents_history_short,
                latents_history_mid, latents_history_long, is_first_denoising_step, **kwargs):
        stage = {2: 0, 4: 1, 8: 2}[hidden_states.shape[-1]]
        grad = torch.is_grad_enabled()
        label = f's{stage}-call{len(self.calls)}-grad{int(grad)}'
        scale=self.scale[stage]
        if self.observe_sections:
            if is_first_denoising_step:
                self.section+=1
            self.section_by_label[label]=self.section
            if grad:
                # One distinct parameter-use node per forward, with unchanged
                # value/storage. This distinguishes earlier-section parameter
                # contribution from the shared Parameter's aggregate gradient.
                scale=scale.view_as(scale)
                scale.register_hook(self._gradient_recorder(self.section_parameter_use_grads,label))
                if hidden_states.requires_grad:
                    hidden_states.register_hook(self._gradient_recorder(self.section_input_grads,label))
                if latents_history_short.requires_grad:
                    latents_history_short.register_hook(self._gradient_recorder(self.section_history_grads,label))
        self.calls.append(dict(label=label, stage=stage, grad=grad, timestep=detached(timestep),
            first=is_first_denoising_step, hidden=detached(hidden_states),
            indices=detached({k:v for k,v in kwargs.items() if k.startswith('indices_')}),
            history=detached((latents_history_short,latents_history_mid,latents_history_long)),
            saved_state_at_entry=self.tracker.snapshot()))
        conditioning = (encoder_hidden_states.mean() * .11 + latents_history_short.mean() * .07
                        + latents_history_mid.mean() * .05 + latents_history_long.mean() * .03
                        + timestep.to(hidden_states.dtype).mean() * .0001)
        boundary = hidden_states * scale + conditioning
        if grad:
            self.tracker.boundary(boundary, label)
        output = checkpoint(self._core, boundary, self.a[stage], self.b[stage], self.bias[stage],
                            self.frozen, label, use_reentrant=False, preserve_rng_state=True)
        return (output,)

    def clear_kv_cache(self):
        self.clears += 1


class Trace:
    def __init__(self, exits, leaves):
        self.exits, self.leaves, self.calls = exits, leaves, []

    def operation(self, key):
        self.calls.append(('operation', key))
        return dict(control=dict(pyramid_stage_call_sequence=[dict(denoising_exit=x) for x in self.exits]))

    def tensor(self, key):
        self.calls.append(('tensor', key))
        return self.leaves[key.removeprefix('unit/')]


def execute(native, exits=(0, 1, 1), *, consistency=False, amplify=False, gt=True, zero_history=False,
            dynamic=False, shift_type='linear', sections=1, last_section=False, skip_first=False,
            multi=False, pyramid=2, vae=False, kv=True, trace=True, seed=995,
            corrupt=False, saturation=False, keep=True, last_step=False, stages=3, hooks=True,
            critic_frames=2,observe_sections=False,objective_last_frames=None):
    """Actual helper invocation. Only sentinel model and one-rank broadcast are stand-ins."""
    assert len(exits) == stages
    torch.set_num_threads(1)
    torch.random.default_generator.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    tracker = StateTracker()
    model = Sentinel(tracker,observe_sections=observe_sections)
    gen = torch.Generator().manual_seed(812)
    def leaf(shape, zeros=False):
        return (torch.zeros(shape) if zeros else torch.randn(shape, generator=gen)).requires_grad_()
    leaves = dict(noise_argument=leaf((1,2,2,8,8)), prompt=leaf((1,3,4)),
                  history_short=leaf((1,2,2,8,8), zero_history),
                  history_mid=leaf((1,2,2,8,8), zero_history),
                  history_long=leaf((1,2,4,8,8), zero_history),
                  rollout_noise=leaf((1,2,2,8,8)),
                  **{'block_noise/1':leaf((1,2,2,4,4)), 'block_noise/2':leaf((1,2,2,8,8))})
    # For two stages, block stage 1 has final resolution 8 rather than 4.
    if stages == 2:
        leaves['block_noise/1'] = leaf((1,2,2,8,8))
    index = torch.arange(10)
    history = (None,index,index[:2],index[:2],index[:4],
               leaves['history_short'],leaves['history_mid'],leaves['history_long'],
               torch.cat([leaves['history_long'],leaves['history_mid'],leaves['history_short'][:,:,:1]],dim=2))
    trace_obj = Trace(exits, leaves) if trace else None
    config = NS(base_seq_len=256,max_seq_len=4096,base_shift=.5,max_shift=1.15,
        corrupt_mode_history='noise',corrupt_mode_prob_history=.9,is_frame_independent_corrupt_history=False,
        is_chunk_independent_corrupt_history=False,noise_corrupt_ratio_history_short=.1,
        noise_corrupt_ratio_history_mid=.2,noise_corrupt_ratio_history_long=.3,noise_corrupt_clean_prob_history=0.,
        downsample_min_corrupt_ratio_history=.9,downsample_max_corrupt_ratio_history=1.,
        saturation_ratio_min=.7,saturation_ratio_max=1.3,saturation_ratio_clean_prob=0.)
    scheduler = native.scheduler(gamma=1/3, stages=stages,
        stage_range=[i/stages for i in range(stages+1)])
    schedule_calls = []
    real_set = scheduler.set_timesteps
    def set_timesteps(*args, **kwargs):
        result = real_set(*args, **kwargs)
        schedule_calls.append(dict(args=args,kwargs={k:str(v) for k,v in kwargs.items()},
            timesteps=detached(scheduler.timesteps),sigmas=detached(scheduler.sigmas)))
        return result
    scheduler.set_timesteps = set_timesteps
    globals_ = native.post.inference_with_trajectory_stage2.__globals__
    real_flag = globals_['generate_and_sync_flag']
    flag_calls=[]
    def flag(*args,**kwargs):
        value = real_flag(*args,**kwargs)
        flag_calls.append(dict(budget=args[1],last_step=bool(args[2]),is_sync=kwargs.get('is_sync',True),value=value))
        return value
    globals_['generate_and_sync_flag'] = flag
    def rng_snapshot():
        numpy = np.random.get_state()
        return dict(torch=torch.get_rng_state().clone(),python=random.getstate(),
                    numpy=(numpy[0],torch.from_numpy(numpy[1].copy()),*numpy[2:]))
    rng_before = rng_snapshot()
    helper_call_order=[]
    codes={v.__code__:k for k,v in globals_.items() if callable(v) and hasattr(v,'__code__')}
    codes[real_flag.__code__]='generate_and_sync_flag'
    codes[real_set.__func__.__code__]='HeliosScheduler.set_timesteps'
    codes[sys.modules['runtime.official_baseline'].stable_block_noise.__code__]='stable_block_noise'
    old_profile=sys.getprofile()
    def profile(frame,event,arg):
        if event=='call' and frame.f_code in codes:
            helper_call_order.append(codes[frame.f_code])
    try:
        sys.setprofile(profile)
        with (torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack) if hooks else nullcontext()):
            result = native.post.inference_with_trajectory_stage2(
                NS(training_config=config),NS(device=torch.device('cpu'),is_main_process=True),model,scheduler,
                leaves['noise_argument'],leaves['prompt'],history_sizes=[4,2,1],is_keep_x0=keep,
                stage2_num_stages=stages,stage2_num_inference_steps_list=[2]*stages,
                num_critic_input_frames=critic_frames,num_rollout_sections=sections,last_section_grad_only=last_section,
                is_skip_first_section=skip_first,is_amplify_first_chunk=amplify,
                is_use_gt_history=gt,gt_all_data=history,is_consistency_align=consistency,
                use_dynamic_shifting=dynamic,time_shift_type=shift_type,
                is_multi_pyramid_stage_backward_simulated=multi,init_pyramid_stage_flag=pyramid,
                is_dmd_vae_decode=vae,use_kv_cache=kv,return_sim_step=True,
                operation_trace=trace_obj,trace_prefix='unit',last_step_only=last_step,
                is_corrupt_history_latents=corrupt,is_add_saturation=saturation,
                sigmas=torch.linspace(1,0,1000),timesteps=torch.linspace(1000,0,1000))
            helper_return_lifetime = tracker.snapshot()
            objective_output=result[0] if objective_last_frames is None else result[0][:,:,-objective_last_frames:]
            loss = objective_output.square().mean() + objective_output.mean()*.13 + result[3]
            before_backward = tracker.snapshot()
            loss.backward()
    finally:
        sys.setprofile(old_profile)
        globals_['generate_and_sync_flag'] = real_flag
    grads = dict(parameters={k:detached(v.grad) for k,v in model.named_parameters()},
                 inputs={k:detached(v.grad) for k,v in leaves.items()})
    calls_without_lifetime = [{k:v for k,v in call.items() if k!='saved_state_at_entry'} for call in model.calls]
    proof = dict(calls=model.calls,helper_return=helper_return_lifetime,before_backward=before_backward,
                 after_backward=tracker.snapshot(),unpacks=list(tracker.unpack_calls),
                 section_by_label=model.section_by_label)
    payload = dict(outputs=detached(result),output_requires_grad=result[0].requires_grad,
        loss=detached(loss),grads=grads,forward_calls=calls_without_lifetime,core_calls=model.core_calls,
        clears=model.clears,flag_calls=flag_calls,scheduler=schedule_calls,helper_call_order=helper_call_order,
        trace_calls=None if trace_obj is None else trace_obj.calls,rng_before=rng_before,
        rng_after=rng_snapshot(),
        parameter_values=detached(dict(model.named_parameters())))
    if observe_sections:
        payload['section_observer']=dict(section_by_label=model.section_by_label,
            parameter_use_grads=model.section_parameter_use_grads,input_grads=model.section_input_grads,
            history_short_grads=model.section_history_grads)
    assert not torch.cuda.is_initialized()
    return payload, proof


def paired(exits, **kwargs):
    original, candidate = load('original'), load('candidate')
    before, a = execute(original, exits, **kwargs)
    after, b = execute(candidate, exits, **kwargs)
    assert byte_equal(before,after), f'helper value/grad/call/RNG mismatch: {exits}, {kwargs}'
    return before,a,b


def saved_at_grad_entry(proof, stage):
    calls = [x for x in proof['calls'] if x['stage']==stage and x['grad']]
    assert len(calls)==1
    return calls[0]['saved_state_at_entry']
