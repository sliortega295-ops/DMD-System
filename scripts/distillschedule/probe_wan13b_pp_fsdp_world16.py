#!/usr/bin/env python3
"""Source-guarded placement-only PP/FSDP adapter of the native B1 benchmark.

The existing loss, samplers, optimizer order, finite/census gates and timer
are compiled unchanged. Only construction, layout, accumulation accounting,
and the report's physical sample multiplicity are adapted. TP is never used.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
import textwrap
import time
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

from scripts.distillschedule import probe_wan13b_static_3d_world16 as native
from distillschedule.runtime.pp_fsdp_layout import PPFSDPLayout
from distillschedule.runtime.pp_fsdp_native import (
    NativeFSDPSourcePPStage, NativePPFSDPProxy, PPFSDPTrainableState, shard_native_pp_stage,
)
from distillschedule.runtime.pp_fsdp_checkpoint import restore_stage_optimizer, LogicalStageShardEMA


class Engine(native.Native3DEngine):
    def __init__(self, module, state, *, lr, physical_step, package=None, component=None):
        torch = native.torch
        self.module, self.state = module, state
        self.optimizer = torch.optim.AdamW([x.parameter for x in state.placements],
            lr=lr, betas=(0., .999), eps=1e-8, weight_decay=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.)
        self.optimizer_restore = (native.materialize_adam_state(optimizer=self.optimizer,
            scheduler=self.lr_scheduler, physical_step=physical_step) if package is None else
            restore_stage_optimizer(module.native_model, self.optimizer, self.lr_scheduler, package, component))
        self.global_steps, self.step_calls = physical_step, 0
        self.grad_norms = []


def build_role(args, boot, role, device, pp, fsdp_mesh, pp_group, _dp_group, _model_group):
    torch = native.torch
    start = time.perf_counter()
    logical = args.workload == 'helios-nogan'
    if logical:
        root = args.model_root if role == 'generator' else args.score_model_root
        subfolder = 'transformer_ode' if role == 'generator' else 'transformer'
        precast = args.precast_generator_root if role == 'generator' else args.precast_score_root
        with native.suspend_transformers_zero3_init():
            model, metadata = native.build_helios_pool_model(role=role, model_root=root,
                subfolder=subfolder, precast_model_root=precast, lora_rank=256, enable_gan=False,
                chunked_residual_storage='cuda_out_of_place')
        restored = native.load_logical_master_into_trainables(model, boot, component=role,
            verify_payload_sha256=(native.dist.get_rank() == 0))
        frozen = native.mirror_fsdp_frozen_compute_dtype(model)
        blocks = 40
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(boot.component(role)['seed']))
            with native.suspend_transformers_zero3_init():
                model, metadata = native.build_helios_pool_model(role=role, model_root=args.model_root,
                    subfolder='transformer', lora_rank=256, enable_gan=False,
                    chunked_residual_storage='cuda_out_of_place',
                    small_mid_patch_seed=native.WAN13B_PYRAMID_PATCH_SEED)
        restored = boot.restore_trainables(model, component=role)
        frozen = dict(frozen_compute_dtype='BF16', converted_constant_tensors=0)
        blocks = 30
    if len(model.blocks) != blocks:
        raise ValueError('actual model architecture changed')
    model._distillschedule_split_modulation = False
    for block in model.blocks:
        block._distillschedule_split_modulation = False
    native.configure_fsdp2_checkpointing(model)
    split = native.create_admitted_forward_split(model,
        expected_source_sha256=native.TRANSFORMER_SHA, expected_blocks=blocks)
    with torch.no_grad():
        template = split.prefix(model, torch.zeros(1,16,1,4,4,dtype=torch.bfloat16),
            torch.tensor([500.],dtype=torch.float64), torch.zeros(1,3,4096,dtype=torch.bfloat16), return_dict=False)
    stage = NativeFSDPSourcePPStage(model, split=split, stage_index=pp, context_template=template,
        remove_root_modules=('norm_out','proj_out') if pp == 0 else
            ('condition_embedder','patch_embedding','patch_short','patch_mid','patch_long'))
    del template
    policy = shard_native_pp_stage(stage, mesh=fsdp_mesh, layout=PPFSDPLayout(shard=args.fsdp_shard),
                                  param_dtype=torch.bfloat16 if logical else None)
    proxy = NativePPFSDPProxy(stage, pp_group=pp_group, device=device, role=role)
    state = PPFSDPTrainableState(model, mesh=fsdp_mesh, pp_group=pp_group)
    engine = Engine(proxy, state, lr=2e-6 if role == 'generator' else 4e-7,
        physical_step=int(boot.component(role)['optimizer_step']), package=boot if logical else None, component=role)
    native.gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    return engine, dict(role=role, cold_construction_seconds=time.perf_counter()-start,
        source_metadata=metadata, bootstrap_restore=restored, TP=1, FSDP=policy, PP=stage.receipt(),
        optimizer=engine.optimizer_restore, after_setup=native.memory(device),
        compute_precision=dict(master_dtype='FP32', forward_compute_dtype='BF16' if logical else 'FP32', merged_count=0),
        frozen_precision=frozen, static_baseline_split_modulation=False)


def compiled_main():
    """Placement substitutions are explicit and reject source drift."""
    source = textwrap.dedent(inspect.getsource(native._main))
    replacements = {
        "layout=Static3DLayout()": "layout=PPFSDPLayout(shard=args.fsdp_shard)",
        "source_ranks=tuple(layout.logical_sample_rank(dp,i) for i in range(8))":
            "source_ranks=tuple(layout.source_rank(rank,i) for i in range(2))",
        "group_size = {'gpipe8':8, 'group2-accumulate8':2, 'serial1-accumulate8':1}[args.pipeline_memory_policy]":
            "group_size = 2",
        "range(0, 8, group_size)": "range(0, 2, group_size)",
        "coordinate=dict(pp=pp,dp=dp,tp=tp)": "coordinate=dict(pp=pp,dp=dp,fsdp=tp,tp=0)",
        "gradient_accumulation_groups_per_update=8//group_size": "gradient_accumulation_groups_per_update=1",
        "static_layout='TP4_DP2_PP2'": "static_layout=layout.receipt()['topology']",
        "mesh=init_device_mesh('cuda',(2,2,4),mesh_dim_names=('pp','dp','tp'))":
            "mesh=init_device_mesh('cuda',(2,layout.replicate,layout.shard),mesh_dim_names=('pp','replica','fsdp'))",
        "tp_group,pp_group,dp_group=(mesh[name].get_group() for name in ('tp','pp','dp'))":
            "tp_group=mesh['replica','fsdp']; pp_group=mesh['pp'].get_group(); dp_group=None",
        "model_groups=[dist.new_group([layout.physical_rank(p,d,t) for p in range(2) for t in range(4)]) for d in range(2)]":
            "model_groups=[pp_group for _ in range(layout.replicate)]",
        "ema=StaticLogicalEMA(placements=generator.state.placements, package=boot,\n                                tp_rank=tp, device=device)":
            "ema=LogicalStageShardEMA(generator.module.native_model, boot, device=device)",
        "sample_count=8*(1+len(logical_steps))": "sample_count=2*(1+len(logical_steps))",
        "reshape(1+len(logical_steps),8//group_size)": "reshape(1+len(logical_steps),1)",
        "actual_B1_stage_G_calls=g_calls*8": "actual_B1_stage_G_calls=g_calls*2",
        "actual_B1_stage_C_calls=c_calls*8": "actual_B1_stage_C_calls=c_calls*2",
        "schema='distillschedule.native_static_3d_wan.v1'": "schema='distillschedule.native_pp_fsdp_world16.v1'",
        "'PASS_CPU_NATIVE_STATIC_3D_SOURCE_ENV_NO_GPU'": "'PASS_CPU_PP_FSDP_SOURCE_ENV_NO_GPU'",
        "'FAIL_NATIVE_STATIC_3D'": "'FAIL_NATIVE_PP_FSDP'",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError('native PP/FSDP placement source drift: ' + old)
        source = source.replace(old, new)
    anchor = "ap.add_argument('--pipeline-memory-policy',choices=('gpipe8','group2-accumulate8','serial1-accumulate8'),default='gpipe8')"
    if source.count(anchor) != 1:
        raise ValueError('native CLI contract drift')
    source = source.replace(anchor, "ap.add_argument('--pipeline-memory-policy',choices=('gpipe2',),default='gpipe2')\n    ap.add_argument('--fsdp-shard',type=int,choices=(4,8),default=4)")
    namespace = dict(vars(native), PPFSDPLayout=PPFSDPLayout, build_role=build_role,
                     LogicalStageShardEMA=LogicalStageShardEMA)
    tree = ast.parse(source)
    exec(compile(tree, str(Path(__file__).resolve()), 'exec'), namespace)
    return namespace['_main']


def main():
    with native.ExitStack() as stack:
        return compiled_main()(stack)


if __name__ == '__main__':
    raise SystemExit(main())
