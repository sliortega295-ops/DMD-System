#!/usr/bin/env python3
"""Real native AR8 G1F1, world8 physical/global-B16, Resident/FSDP controls.

The private original DMD2Trainer owns every mathematical expression. Its two
optimizer callback sites are staged by the source-pinned logical B1 adapter.
Tiny mode qualifies that integration only; no tiny timing is claimable.
"""
from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.h3_diffsynth_logical_microbatches import run_native_microbatches

SOURCE_SHA = '9ca4ad8ca13b0a9e7397f6313e4eca225f10f52802e90423b06f7e786a58b989'
ROOT = Path('/home/jiangzhikun/yongyan_liu/Helios-H100')
ASSET = ROOT / 'assets/five_workload_h100_20260930/h3_ar_frozen_r1'
SOURCE = ASSET / 'assets/exact_reference_source_r2'
sys.path[:0] = [str(SOURCE / 'examples/minimax_h3/model_training'), str(SOURCE)]
from h3_dmd2 import DMD2Config, DMD2Trainer, H3Adapter, VelocityDiscriminator, prepare_batch
from train_dmd2 import load_teacher, make_optimizers, seed_all, rng_state, restore_rng
from diffsynth.models.minimax_h3_dit import MiniMaxH3DiT


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, data):
    with Path(path).open('x') as handle:
        json.dump(data, handle, indent=2); handle.write('\n')


def move(value, device):
    if isinstance(value, torch.Tensor): return value.to(device)
    if isinstance(value, dict): return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, tuple): return tuple(move(v, device) for v in value)
    if isinstance(value, list): return [move(v, device) for v in value]
    return value


def memory():
    free, total = torch.cuda.mem_get_info()
    status = Path('/proc/self/status').read_text().splitlines()
    rss = next(int(line.split()[1])*1024 for line in status if line.startswith('VmRSS:'))
    return {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved(),
            'peak_allocated': torch.cuda.max_memory_allocated(), 'peak_reserved': torch.cuda.max_memory_reserved(),
            'driver_free': free, 'driver_total': total, 'host_rss': rss,
            'cgroup_current': int(Path('/sys/fs/cgroup/memory.current').read_text()),
            'cgroup_events': {k: int(v) for k, v in (line.split() for line in Path('/sys/fs/cgroup/memory.events').read_text().splitlines())}}


def tensor_state_sha(named_parameters):
    # Called on CPU before placement, never in the performance window.
    digest = hashlib.sha256()
    for name, parameter in named_parameters:
        require(parameter.device.type == 'cpu', 'INITIAL_SHA_CPU_ONLY')
        digest.update(name.encode() + b'\0'); digest.update(str(parameter.dtype).encode())
        digest.update(parameter.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class SlotRNG:
    def __init__(self, slots):
        caller = rng_state(); self.states = {}
        for slot in slots:
            seed_all(42 + slot); self.states[slot] = rng_state()
        restore_rng(caller); self.active = None

    def scope(self, slot):
        from contextlib import contextmanager
        @contextmanager
        def scoped():
            require(self.active is None and slot in self.states, 'INDEPENDENT_NONNESTED_LOGICAL_RNG')
            caller = rng_state(); restore_rng(self.states[slot]); self.active = slot
            try:
                yield
            finally:
                self.states[slot] = rng_state(); restore_rng(caller); self.active = None
        return scoped()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--placement', choices=('resident', 'fsdp'), required=True)
    parser.add_argument('--mode', choices=('correctness', 'timing'), required=True)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--cache-manifest', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--receipt-dir', required=True, type=Path)
    parser.add_argument('--tiny', action='store_true')
    args = parser.parse_args()
    rank, local, world = [int(os.environ[name]) for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')]
    require(world == 8 and local == rank and int(os.environ['LOCAL_WORLD_SIZE']) == 8, 'PHYSICAL_WORLD8_NODE_LOCAL')
    require(os.environ.get('H3_W6_RUN_GPU_LOCK_HELD') == '1', 'EXPLICIT_OWNED_CONTROLLER_REQUIRED')
    require(sha(SOURCE / 'source_receipt.json') == SOURCE_SHA, 'PINNED_PRIVATE_SOURCE')
    slots = [rank, rank+8]
    device = torch.device('cuda', local); torch.cuda.set_device(device)
    dist.init_process_group('nccl', timeout=timedelta(seconds=600))
    args.receipt_dir.mkdir(exist_ok=True, parents=True)
    rec = {'rank': rank, 'local_rank': local, 'physical_world': world, 'logical_global_batch': 16,
           'logical_slots': slots, 'hostname': socket.gethostname(), 'placement': args.placement,
           'worker_sha256': sha(__file__), 'source_receipt_sha256': SOURCE_SHA,
           'comparison_boundary': 'H100_INTERNAL_CPU_FP32_LORA_COMMON_START_NOT_A100_ZER03_REPLAY',
           'mode': 'TINY_NATIVE_INTEGRATION_ONLY' if args.tiny else args.mode,
           'status': 'NOT_RUN', 'memory': [], 'cycles': [], 'census': {}, 'phase': 'setup',
           'parameter_offload': False, 'activation_offload': False, 'model_swap': False,
           'sigma_distribution': 'UNCHANGED_CONTINUOUS_SHIFTED_LOGIT_NORMAL', 'EMA_updates': 0,
           'strict_cross_arm_numerical_parity': 'NOT_RUN'}
    phase = 'setup'; last_block = {'index': None}; census = collections.Counter(); commits = collections.Counter()
    try:
        config = DMD2Config(**json.loads(args.config.read_text()))
        require(config.num_inference_steps == 8 and config.critic_updates == 1 and config.gan_start_step == 500,
                'ORIGINAL_AR8_G1F1_INACTIVE_GAN_PREDICATE')
        seed_all(42); rec['memory'].append({'phase': 'before_CPU_load', **memory()}); started=time.monotonic()
        if args.tiny:
            dit = MiniMaxH3DiT(num_layers=2, token_refiner_num_layers=1, hidden_size=24,
                              num_attention_heads=2, attention_head_dim=12, ffn_hidden_size=48,
                              timestep_input_dim=8, time_embed_hidden_size=24, time_embed_dim=12,
                              adaln_out_features=432, final_adaln_out_features=48, rope_inv_freq_len=2).to(torch.bfloat16)
            config.lora_rank = config.lora_alpha = 2
            raws = []
            for slot in slots:
                # Diagnostic-only shapes; never used for a model-capacity claim.
                gen = torch.Generator().manual_seed(42+slot)
                raws.append({'input_latents': torch.randn(1,24,2,4,4,generator=gen,dtype=torch.bfloat16),
                             'audio_input_latents': torch.randn(2,32,3,generator=gen,dtype=torch.bfloat16),
                             'ref_video_latents': torch.randn(1,24,2,4,4,generator=gen,dtype=torch.bfloat16),
                             'ref_audio_latents': torch.randn(2,32,3,generator=gen,dtype=torch.bfloat16),
                             'prompt_embeds': torch.randn(3,5120,generator=gen,dtype=torch.bfloat16)})
            rec['samples'] = [{'logical_rank': slot, 'identity': f'TINY_DIAGNOSTIC_{slot}'} for slot in slots]
        else:
            data = json.loads(args.cache_manifest.read_text())
            require(data['status'] == 'PASS_CACHE_SOURCE_AND_SCHEMA' and len(data['rows']) == 16, 'EXACT_16_REAL_ROWS')
            raws, identities = [], []
            for slot in slots:
                row = data['rows'][slot]; path = Path(row['path'])
                require(sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'REAL_ROW_PAYLOAD_SHA')
                raws.append(torch.load(path, map_location='cpu', weights_only=True))
                identities.append({'logical_rank': slot, 'identity': row['sha256'], 'path': str(path)})
            rec['samples'] = identities
            dit, architecture, checkpoint = load_teacher(args.checkpoint, 'cpu')
            rec.update(architecture=architecture, checkpoint_metadata=checkpoint)
            require(len(dit.blocks) == 50 and config.lora_rank == 64, 'EXACT_H3_50BLOCK_LORA64')
        rec['model_load_seconds_excluded'] = time.monotonic()-started
        rec['memory'].append({'phase': 'after_CPU_load', **memory()})
        adapter = H3Adapter(dit, config); discriminator = VelocityDiscriminator(config)
        rec['initial_LoRA_sha256'] = {role: tensor_state_sha((n,p) for n,p in adapter.named_parameters()
                                                     if f'.adapters.{role}.' in n) for role in ('student','fake_score')}
        for role in ('student','fake_score'):
            for p in adapter.role_parameters(role): p.requires_grad_(True)
        started=time.monotonic()
        if args.placement == 'fsdp':
            mesh = init_device_mesh('cuda',(world,),mesh_dim_names=('shard',))
            mp = MixedPrecisionPolicy(param_dtype=None,reduce_dtype=torch.float32,output_dtype=None,cast_forward_inputs=False)
            for module in list(dit.token_refiner.blocks)+list(dit.blocks)+[dit,discriminator]:
                fully_shard(module,mesh=mesh,mp_policy=mp,reshard_after_forward=True)
                module.set_force_sum_reduction_for_comms(True);module.set_gradient_divide_factor(float(world))
        else:
            adapter.to(device);discriminator.to(device)
        rec['construction_seconds_excluded'] = time.monotonic()-started
        adapter.select('teacher');discriminator.requires_grad_(False)
        gc.collect();torch.cuda.empty_cache()
        for index, block in enumerate(dit.blocks):
            def marker(module, inputs, index=index):last_block['index']=index
            block.register_forward_pre_hook(marker,prepend=True)
        require(not set(map(id,adapter.role_parameters('student')))&set(map(id,adapter.role_parameters('fake_score'))),
                'DISJOINT_NATIVE_ADAPTERS')
        optimizers, parameters = make_optimizers(adapter,discriminator,config)
        batches=[]
        for raw in raws:
            positive,negative=prepare_batch(raw,config.teacher_cfg_scale)
            positive,negative=move(positive,device),move(negative,device)
            real=(positive.pop('input_latents'),positive.pop('audio_input_latents'))
            require(positive.get('ref_visual_anchor') is not None and positive.get('ref_audio_anchor') is not None,
                    'PREVIOUS_AV_CHUNK_REQUIRED')
            batches.append((real,positive,negative))
        native_predict=adapter.predict
        def predict(*a,**kw):
            nonlocal phase
            roles={layer.role for _,layer in adapter.layers};require(len(roles)==1,'UNIFORM_NATIVE_ROLE')
            role=next(iter(roles));census[role]+=1;phase=role+'_forward'
            if torch.is_grad_enabled():census[role+'_grad']+=1
            result=native_predict(*a,**kw)
            if args.mode=='correctness':rec['memory'].append({'phase':'after_'+phase,**memory()})
            return result
        adapter.predict=predict
        def backward(role,loss):
            nonlocal phase
            phase=role+'_backward';finite=torch.isfinite(loss.detach()).int();dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            require(bool(finite.item()),'FINITE_NATIVE_LOSS');loss.backward();census[role+'_backward']+=1
            if args.mode=='correctness':rec['memory'].append({'phase':'after_'+phase,**memory()})
        def commit(role):
            nonlocal phase
            phase=role+'_optimizer'
            if args.placement=='resident':
                for p in parameters[role]:
                    if not p.requires_grad:continue
                    if p.grad is None:p.grad=torch.zeros_like(p)
                    dist.all_reduce(p.grad,op=dist.ReduceOp.SUM);p.grad.div_(world)
            norm=torch.nn.utils.clip_grad_norm_(parameters[role],config.max_grad_norm if role=='student' else config.critic_max_grad_norm)
            local_norm=norm.full_tensor() if hasattr(norm,'full_tensor') else norm
            require(bool(torch.isfinite(local_norm).item()),'FINITE_NATIVE_GRAD_NORM')
            optimizers[role].step();optimizers[role].zero_grad(set_to_none=True);commits[role]+=1
        trainers=[DMD2Trainer(adapter,discriminator,config,lambda *a: (_ for _ in ()).throw(RuntimeError('UNSTAGED_COMMIT'))) for _ in slots]
        logical_rng=SlotRNG(slots)
        rec['precision']={'base_storage':'BF16','LoRA_storage':'FP32','autocast':'BF16_cache_enabled_false','reduction':'FP32_SUM_then_div8'}
        rec['memory'].append({'phase':'after_setup',**memory()});write(args.receipt_dir/f'rank_{rank:02d}_setup.json',rec)
        expected={'student':18,'fake_score':4,'teacher':2,'student_grad':2,'fake_score_grad':2,'student_backward':2,'critic_backward':2}
        for cycle in range(2):
            census.clear();before_commits=dict(commits);torch.cuda.reset_peak_memory_stats(device)
            dist.barrier();torch.cuda.synchronize(device);start=time.perf_counter()
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,cache_enabled=False):
                metrics=run_native_microbatches(trainers,batches,backward=backward,commit=commit,
                                               rng_scope=logical_rng.scope,logical_slots=slots)
            torch.cuda.synchronize(device);dist.barrier();stop=time.perf_counter()
            require(dict(census)==expected,'FAIL_CENSUS:'+str(dict(census)))
            require(all(v['generator_step']==v['critic_step']==cycle+1 and not v['gan_active'] for v in metrics),
                    'FINAL_NATIVE_ROLE_VERSION_OR_GAN_PREDICATE')
            require({r:commits[r]-before_commits.get(r,0) for r in ('student','critic')}=={'student':1,'critic':1},'ONE_COMMIT_PER_LOGICAL_ROLE')
            rec['cycles'].append({'cycle':cycle+1,'timer':{'start':start,'stop':stop,'seconds':stop-start},
                                  'performance_denominator':cycle==1 and not args.tiny,
                                  'actual_census':dict(census),'metrics':metrics,'memory_after_timer':memory()})
        require(not any(p.grad is not None for n,p in adapter.named_parameters() if '.adapters.' not in n),'FROZEN_BASE_GRADIENT')
        rec.update(status='PASS_REAL_NATIVE_AR8_G1F1_LOGICAL_B16' if not args.tiny else 'PASS_TINY_NATIVE_INTEGRATION_ONLY',
                   actual_optimizer_counts=dict(commits),final_versions={'student':2,'fake':2,'ema':0},
                   formal_seconds=rec['cycles'][1]['timer']['seconds'] if not args.tiny else None)
        write(args.receipt_dir/f'rank_{rank:02d}.json',rec)
    except BaseException as exc:
        rec.update(status='FAIL',failure_type='CUDA_OOM' if isinstance(exc,torch.OutOfMemoryError) else 'OTHER_RUNTIME_ERROR',
                   failure_phase=phase,last_block=last_block['index'],error_type=type(exc).__name__,error=str(exc),
                   traceback=traceback.format_exc(),census_at_failure=dict(census),commits_at_failure=dict(commits),memory_at_failure=memory())
        write(args.receipt_dir/f'rank_{rank:02d}_failure.json',rec);raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
