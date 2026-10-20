"""Install the manual GAN mechanisms in the current native plan executor.

No trainer, timer, owner, model replacement or numerical acceptance lives here.
All mutations are process-scoped and restored on failure. The copied down40
policy still checks exact frozen Linear storage, graph/version and FIFO lifetime.
"""
from contextlib import ExitStack, contextmanager
import functools
import json
import os
from pathlib import Path

from distillschedule.search.contracts import require
from distillschedule.search.helios_gan_bundle import (
    HeliosGANBundle, ROW_SHA256, LORA_INVENTORY, LORA_LAYER_COUNT,
)
from distillschedule.workloads.best_known.registry import file_sha256
from . import backend_abi, backend_library
from .down_replay import NativeDownSAC, replace

TRANSFORMER_SHA = '5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886'


def _dense(q, k, v, mask=None):
    require(mask is None, 'manual GAN backend does not admit masks/varlen')
    require(all(x.ndim == 4 and str(x.dtype) == 'torch.bfloat16' and x.is_cuda for x in (q,k,v)),
            'manual GAN backend requires BF16 CUDA BSHD')
    require(q.device == k.device == v.device and q.shape[0] in (1,3)
            and q.shape[0] == k.shape[0] == v.shape[0], 'manual GAN backend physical batch/device differs')
    require(all(x.shape[-2:] == (40,128) and x.shape[1] > 0 and x.stride(-1) == 1 for x in (q,k,v))
            and k.shape[1] == v.shape[1], 'manual GAN backend head layout differs')


def _rows(rank):
    path = Path(__file__).with_name('down40_rows.json')
    require(file_sha256(path) == ROW_SHA256, 'manual GAN shape qualification drift')
    data = json.loads(path.read_text())
    require(len(data['ranks']) == 16 and {r['rank'] for r in data['ranks']} == set(range(16)),
            'manual GAN row qualification needs all16 ranks')
    row = next(r for r in data['ranks'] if r['rank'] == rank)
    for value in row['graphs']:
        require(file_sha256(Path(value['source_entry_path'])) == value['source_entry_sha256'],
                'manual GAN actual rank-qualified source entry changed')
    result = {(v['logical_step'],v['graph']):v['rows'] for v in row['graphs']}
    require(len(result) == 10, 'manual GAN exact five dual-graph row qualification missing')
    return result


def _peft_lora_types():
    from peft.tuners.lora.layer import Linear, Conv3d, LoraLayer
    return Linear, Conv3d, LoraLayer


def _lora_inventory(engine, role):
    linear_type, conv_type, lora_type = _peft_lora_types()
    modules, conv_names = [], []
    linears = 0
    for name, module in engine.module.named_modules():
        if not (isinstance(module, lora_type) or any(hasattr(module, key) for key in
                ('lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B'))):
            continue
        require(hasattr(module, 'lora_A') and hasattr(module, 'lora_B'),
                'manual GAN unknown or partial LoRA layer: ' + role + '/' + name)
        if isinstance(module, linear_type):
            linears += 1
        elif isinstance(module, conv_type):
            conv_names.append(name)
        else:
            raise ValueError('manual GAN unsupported LoRA layer type: ' + role + '/' + name)
        modules.append(module)
    inventory = dict(linear=linears, conv3d_names=sorted(conv_names))
    require(inventory == LORA_INVENTORY[role], 'manual GAN exact LoRA inventory differs: ' + role)
    return modules, inventory


def _audit_unmerged(state, engines, point):
    count = merged = 0
    inventories = {}
    for role, engine in zip(('generator','critic'), engines, strict=True):
        require(not hasattr(engine.module, '_distillschedule_merged_execution_state'),
                'manual GAN N70 execution state is forbidden')
        modules, inventory = _lora_inventory(engine, role)
        inventories[role] = inventory
        for module in modules:
            require(type(module.merged) is bool, 'manual GAN PEFT merge state unavailable')
            merged += int(module.merged)
            require(not module.merged and not getattr(module, 'merged_adapters', []),
                    'manual GAN merged adapter is forbidden')
        count += len(modules)
        state.setdefault('model_metadata_samples', []).append(dict(
            point=point, role=role, LoRA_count=len(modules),
            LoRA_linear_count=inventory['linear'], LoRA_conv3d_names=inventory['conv3d_names'],
            PEFT_merged_layers=0, N70_model_state_present=False))
    require(count == LORA_LAYER_COUNT and merged == 0, 'manual GAN full unmerged LoRA inventory differs')
    state.update(lora_layer_count=count, lora_inventory=inventories, merged_layer_count=merged, merge_count=0)


def _forbid_merges(stack, state, engines):
    state['merge_attempt_count'] = 0
    state['guarded_merge_methods'] = 0
    for role, engine in zip(('generator','critic'), engines, strict=True):
        modules, _ = _lora_inventory(engine, role)
        for module in modules:
            require(callable(getattr(module, 'merge', None)), 'manual GAN LoRA merge method unavailable')
            def forbidden(*args, **kwargs):
                state['merge_attempt_count'] += 1
                raise ValueError('manual GAN plan forbids every merged-LoRA execution')
            replace(stack,module,'merge',lambda original,fn=forbidden:fn)
            state['guarded_merge_methods'] += 1
    require(state['guarded_merge_methods'] == LORA_LAYER_COUNT, 'manual GAN merge guards incomplete')


@contextmanager
def manual_runtime_scope(p0):
    from distillschedule.search.helios_implementation import current_implementation
    impl = current_implementation()
    if impl is None or type(impl.candidate_bundle) is not HeliosGANBundle:
        yield
        return
    impl.candidate_bundle.source_identity()
    torch = p0.torch
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    require(os.environ.get('WORLD_SIZE') == '16' and 0 <= rank < 16 and local_rank == rank % 8,
            'manual GAN requires the exact A100 world16 rank map')
    require(torch.__version__ == '2.10.0+cu128' and torch.version.cuda == '12.8',
            'manual GAN pinned Torch/CUDA runtime differs')
    # This scope precedes the native setup. Select the owned local device
    # before a query/backend import can initialize CUDA on the default GPU.
    torch.cuda.set_device(local_rank)
    require(torch.cuda.get_device_capability(local_rank) == (8,0)
            and 'A100' in torch.cuda.get_device_name(local_rank),
            'manual GAN SM80 package is not H100 kernel admission')
    from helios.modules.helios_kernels import attention_dispatch as dispatch
    from helios.modules import transformer_helios as transformer
    from distillschedule.runtime import fsdp2_static as fsdp, helios_planner_cycle as native
    import torch.utils.checkpoint as checkpoint
    require(file_sha256(Path(transformer.__file__)) == TRANSFORMER_SHA,
            'manual GAN external transformer source differs')
    # Compare against the vendored dispatcher, not an eight-character prefix.
    from distillschedule.workloads.best_known.registry import repository_root
    expected_dispatch = repository_root() / 'third_party/helios_compat/source/helios/modules/helios_kernels/attention_dispatch.py'
    require(file_sha256(Path(dispatch.__file__)) == file_sha256(expected_dispatch)
            and transformer.attn_varlen_func is dispatch.attn_varlen_func,
            'manual GAN exact external dispatcher/alias differs')
    reference = p0._attention_backend_receipt('local-fa2')
    public, schema, identity = backend_library.load_actual(torch)
    state = dict(installed=True, effective_attention_backend='pinned-sm80-fa3',
                 actual_FA3_identity=identity, public_backend_call_count=0,
                 unsupported_request_count=0, merge_count=0, merged_layer_count=0,
                 lora_layer_count=0, down40={}, restored=False, numerical_status='REVIEW',
                 formal_speedup_eligible=False)
    with ExitStack() as stack:
        def checked(q,k,v):
            try:
                _dense(q,k,v)
            except BaseException:
                state['unsupported_request_count'] += 1
                raise
            state['public_backend_call_count'] += 1
            return public(q,k,v)
        stack.enter_context(backend_abi.actual_FA3_backend_scope(dispatch,fsdp,checked,
                                                               actual_schema_receipt=schema))
        def dense(original):
            @functools.wraps(original)
            def call(q,k,v,attention_mask=None):
                _dense(q,k,v,attention_mask)
                return original(q,k,v,attention_mask=attention_mask)
            return call
        replace(stack,transformer,'attn_varlen_func',dense)
        def no_varlen(*args,**kwargs):
            state['unsupported_request_count'] += 1
            raise ValueError('manual GAN varlen/FA2 fallback forbidden')
        replace(stack,dispatch,'flash_attn_varlen_func',lambda _:no_varlen)
        def receipt(original):
            def call(requested):
                require(requested == 'local-fa2' and dispatch.flash_attn_func is checked
                        and fsdp.SELECTIVE_FLASH_ATTN_OP == backend_abi.FA3_FW,
                        'manual GAN effective attention dispatcher differs')
                return dict(status='PASS_PINNED_GAN_BACKEND_IDENTITY', requested=requested,
                            runtime='pinned-sm80-fa3', effective_attention_backend='pinned-sm80-fa3',
                            reference_backend_receipt=reference, actual_FA3_identity=identity,
                            numerical_status='REVIEW', formal_speedup_eligible=False)
            return call
        replace(stack,p0,'_attention_backend_receipt',receipt)
        def configure(original):
            @functools.wraps(original)
            def call(*args, **kwargs):
                value = original(*args, **kwargs)
                require(value['selective_flash_op'] == backend_abi.FA3_FW,
                        'manual GAN checkpoint factory did not bind the actual FA3 op')
                value['reference_backend_scope'] = value['scope']
                if value['selective_flash_enabled']:
                    value['scope'] = 'SELECTIVE_PINNED_FA3_WHOLE_RESULT_REPLAY'
                value['effective_attention_backend'] = 'pinned-sm80-fa3'
                return value
            return call
        replace(stack,p0,'configure_fsdp2_checkpointing',configure)
        def run(original):
            @functools.wraps(original)
            def call(**kwargs):
                require(state.get('runner_calls',0) == 0, 'manual GAN executor invoked more than once')
                state['runner_calls'] = 1
                generator,critic,ema = (kwargs[k] for k in ('generator_engine','critic_engine','ema'))
                _audit_unmerged(state,(generator,critic),'AFTER_SETUP_BEFORE_BOTH_TIMERS')
                _forbid_merges(stack,state,(generator,critic))
                core = NativeDownSAC(torch=torch,rank=rank,enabled=True,rows=_rows(rank))
                state['down40'] = core.state
                kwargs['result']['build']['gan_manual_runtime'] = state
                core.bind(stack,p0=p0,native=native,fsdp=fsdp,checkpoint=checkpoint,
                          generator=generator,critic=critic,ema=ema)
                result = original(**kwargs)
                core.finish()
                _audit_unmerged(state,(generator,critic),'AFTER_BOTH_TIMERS')
                stack.close()
                core.restored()
                state['restored'] = True
                return result
            return call
        replace(stack,native,'run_two_cycles',run)
        try:
            yield state
        except BaseException:
            # Keep failed/unverified restoration distinct from success. The
            # ExitStack still unwinds all owned hooks on the exceptional path.
            state['failed'] = True
            raise
