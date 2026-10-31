"""Explicit single-device ports of hash-locked official Self-Forcing.

Numerical model, loss, scheduler, rollout and cache-write bodies remain upstream
definitions. Only constructors, attention provider, cache allocations and six
training-boundary ``.cuda()`` calls need an explicit single-device port.
No aliases are installed in the user's ``wan``, ``model`` or ``pipeline`` modules.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
from pathlib import Path
from types import SimpleNamespace

from .upstream import PinnedSource

REVISION = '33593df3e81fa3ec10239271dd2c100facac6de1'
REPOSITORY = 'https://github.com/guandeh17/Self-Forcing'
SOURCE_FILES = {
    'LICENSE': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4',
    'README.md': '68e3596953c9f9dc2f42863a123fd5325f6007c62d8a7fd92e00cf7b215ef19d',
    'requirements.txt': '0997abbb523ce7781195be3d556b70a16ec0d9d680733ed7f10afaa5a8c7fe94',
    'model/base.py': '3e39cd61b25a753cf064101e8ffca653d6cce07a7c60a50fd8e3475425362d4f',
    'model/dmd.py': '7b8dd78a48220e6868956a7cd639215756747b7b8e58c07f27e50090c4291e19',
    'pipeline/self_forcing_training.py': '0fbef7cae39a13338ae555acc2f114dc28a0e15f4dfcf4303640192e42ab7e96',
    'pipeline/causal_inference.py': 'a06f5653f266dce154116f85d33cee3f2923e8531fbb15b4702b3751abdaab69',
    'utils/wan_wrapper.py': 'da919629bac4472cd9072223f1ab03200de276e202ab0d9b91e6789d500ff8fa',
    'utils/scheduler.py': '4194ef1324bb06527621749c7f9a82f431ccf011823aa65099cb7bdf116b642e',
    'utils/loss.py': '5a08ba589ef54d5deb50d1ff1984f2969b1fdc2f03c53751167eb23a7082ecfe',
    'utils/distributed.py': '8560b35c3e5fa77ff7f98c54344775393b1cd3c1ed37740f6284b27aad4f0ba7',
    'utils/misc.py': '610bead16d50d99fabdccb2e9ce3c20fa1fa1cd3bdf735d7ff7ffda978bfe195',
    'wan/modules/model.py': 'f15214485a5297e6f9e79684bbad80acbcdc88c3272596b256a739a734529449',
    'wan/modules/causal_model.py': 'ae2644311ad212977260549e4b6034618647b82e2ee47a244bdaac50d8ff0778',
    'wan/modules/attention.py': '02ad719a76e6bdec79d18fc62b094ef2ad7c28b2d593c9f395ae1226041b7c9d',
    'wan/modules/t5.py': '8b0cebf3192c542f92a344255a06c203df3ba24160715899a055cc8de0cd930f',
    'wan/modules/tokenizers.py': 'cf5b189c1ae017326efadab2fc6380ba223fa7a59d0be6464bd264db53c76934',
    'wan/modules/vae.py': 'a4c1d9362f25a1bef463ac51caed4097317c007d4d87887b02a3a733fdbc48d5',
    'trainer/distillation.py': 'ec7b49b31c06411e913c46c7ac5d86c63fee0033e4b8cb6e12ed073e0a5f1fde',
    'demo_utils/memory.py': 'ca0b31c93019f802e1312d512591ae6bea35b450064cdde19b6202779f50bb7f',
    'configs/self_forcing_dmd.yaml': 'f248ae59463c1b557868e88384be69e5d6d6c41a740726424dfa24f0f02db4d4',
    'configs/default_config.yaml': '18303f1bacbfbebaa02893cc8e106941753f21a7d1a7109519a90762375ee772',
}


def require(condition, message):
    if not condition:
        raise ValueError('official Self-Forcing: ' + message)


class SingletonBroadcast:
    """Local rank0 only; a real group always retains its real broadcast."""
    @staticmethod
    def is_initialized():
        import torch.distributed as dist
        return dist.is_initialized()

    @staticmethod
    def get_rank():
        import torch.distributed as dist
        return dist.get_rank() if dist.is_initialized() else 0

    @staticmethod
    def broadcast(tensor, src):
        import os
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.broadcast(tensor, src=src)
        require(src == 0 and tensor.device.type in ('cpu', 'cuda')
                and all(int(os.environ.get(key, default)) == int(default)
                        for key, default in (('WORLD_SIZE', '1'), ('LOCAL_WORLD_SIZE', '1'),
                                             ('RANK', '0'), ('LOCAL_RANK', '0'))),
                'uninitialized broadcast is only valid for a true single-device singleton')
        return None


def fp32_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None,
                  q_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
                  dtype=None, version=None, fa_version=None):
    """FP32 SDPA reference to native varlen attention, including real masks.

    Clone the packed K/V reads, as FlashAttention's native varlen packing does:
    later causal cache writes must not invalidate a saved backward view.
    Native RoPE, projections, cache writes and end-index updates are unchanged.
    """
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    require(q.device.type in ('cpu', 'cuda') and all(t.device == q.device and t.dtype == torch.float32
                and t.layout == torch.strided for t in (q, k, v)) and dropout_p == 0.,
            'single-device attention requires FP32/no-dropout')
    require(q.ndim == k.ndim == v.ndim == 4 and q.shape[0] == k.shape[0] == v.shape[0]
            and k.shape[:3] == v.shape[:3] and q.shape[2] == k.shape[2], 'unsupported attention layout')
    outputs = []
    for index in range(q.shape[0]):
        nq = q.shape[1] if q_lens is None else int(q_lens[index])
        nk = k.shape[1] if k_lens is None else int(k_lens[index])
        require(0 < nq <= q.shape[1] and 0 < nk <= k.shape[1], 'invalid variable attention length')
        query = q[index:index + 1, :nq].transpose(1, 2).clone()
        key = k[index:index + 1, :nk].transpose(1, 2).clone()
        value = v[index:index + 1, :nk].transpose(1, 2).clone()
        if q_scale is not None:
            query = query * q_scale
        mask = None
        if causal or tuple(window_size) != (-1, -1):
            qp = torch.arange(nq, device=q.device)[:, None] + nk - nq  # FA bottom-right alignment
            kp = torch.arange(nk, device=q.device)[None, :]
            mask = torch.ones(nq, nk, dtype=torch.bool, device=q.device)
            if causal:
                mask &= kp <= qp
            if window_size[0] >= 0:
                mask &= kp >= qp - window_size[0]
            if window_size[1] >= 0:
                mask &= kp <= qp + window_size[1]
        with sdpa_kernel(SDPBackend.MATH):
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, scale=softmax_scale, dropout_p=0.)
        output = output.transpose(1, 2)
        outputs.append(torch.cat([output, output.new_zeros(1, q.shape[1] - nq, *output.shape[2:])], dim=1))
    return torch.cat(outputs)


cpu_attention = fp32_attention  # Retain the original CPU reference import.


def _uncached_not_admitted(*args, **kwargs):
    raise ValueError('official Self-Forcing: uncached causal/flex path is not admitted by the cached owner')


def _definitions(path, *, exclude=()):
    return tuple(node.name for node in ast.parse(Path(path).read_bytes()).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name not in exclude)


def _portable_boundary(source, module):
    """Audited per-function port, never a global Tensor.cuda monkeypatch."""
    path = source.root / 'pipeline/self_forcing_training.py'
    tree = ast.parse(path.read_bytes(), filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'SelfForcingTrainingPipeline')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'inference_with_trajectory')

    class BoundaryDevice(ast.NodeTransformer):
        count = 0

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
                require(not node.args and not node.keywords, 'unexpected upstream CUDA boundary')
                self.count += 1
                node.func.attr = 'to'
                node.keywords = [ast.keyword(arg='device', value=ast.Attribute(
                    value=ast.Name(id='noise', ctx=ast.Load()), attr='device', ctx=ast.Load()))]
            return node

    port = BoundaryDevice(); method = port.visit(method)
    require(port.count == 6, 'expected exactly six explicit CUDA boundary lookups')
    tree = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    scope = dict(vars(module))
    exec(compile(tree, str(path), 'exec'), scope)
    return scope[method.name], dict(cuda_to_noise_device=port.count,
        ast_sha256=hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest())


class GeometryCache:
    """Only allocation geometry changes; native cache read/write bodies do not."""
    def configure_geometry(self, *, shape):
        model = self.generator.model
        _, frames, _, height, width = shape
        self.num_transformer_blocks = len(model.blocks)
        self.frame_seq_length = (height // model.patch_size[1]) * (width // model.patch_size[2])
        self.cache_heads = model.num_heads
        self.cache_width = model.dim // model.num_heads
        self.context_length = model.text_len
        local = model.local_attn_size
        self.kv_cache_size = (frames if local == -1 else local) * self.frame_seq_length
        for block in model.blocks:
            block.self_attn.max_attention_size = (32760 if local == -1 else local * self.frame_seq_length)

    def _initialize_kv_cache(self, batch_size, dtype, device):
        import torch
        self.kv_cache1 = [dict(k=torch.zeros(batch_size, self.kv_cache_size, self.cache_heads,
                                self.cache_width, dtype=dtype, device=device),
                              v=torch.zeros(batch_size, self.kv_cache_size, self.cache_heads,
                                self.cache_width, dtype=dtype, device=device),
                              global_end_index=torch.zeros(1, dtype=torch.long, device=device),
                              local_end_index=torch.zeros(1, dtype=torch.long, device=device))
                          for _ in range(self.num_transformer_blocks)]

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        import torch
        self.crossattn_cache = [dict(k=torch.zeros(batch_size, self.context_length, self.cache_heads,
                                      self.cache_width, dtype=dtype, device=device),
                                    v=torch.zeros(batch_size, self.context_length, self.cache_heads,
                                      self.cache_width, dtype=dtype, device=device), is_init=False)
                                for _ in range(self.num_transformer_blocks)]


@lru_cache(maxsize=8)
def _original(root):
    import math
    import types
    import gc
    import logging
    import time
    from typing import List, Optional, Tuple
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from einops import repeat, rearrange
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin

    definitions = {
        name: _definitions(Path(root) / (name.replace('.', '/') + '.py'), exclude=exclude)
        for name, exclude in [('wan.modules.model', ()), ('wan.modules.causal_model', ()),
            ('wan.modules.t5', ('T5EncoderModel', '_t5', 'umt5_xxl')),
            ('utils.wan_wrapper', ()), ('pipeline.self_forcing_training', ()),
            ('pipeline.causal_inference', ()), ('model.base', ()), ('model.dmd', ()),
            ('trainer.distillation', ()), ('utils.distributed', ('fsdp_state_dict', 'fsdp_wrap',
                'barrier', 'launch_distributed_job'))]}
    # Globally selecting function/class definitions is safe only after all whole
    # files are verified by PinnedSource. No unverified snippet is executable.
    common = dict(torch=torch, nn=nn, F=F, math=math, types=types,
                  List=List, Optional=Optional, Tuple=Tuple)
    scopes = {name: dict(common) for name in definitions}
    scopes['wan.modules.model'].update(ConfigMixin=ConfigMixin, register_to_config=register_to_config,
        ModelMixin=ModelMixin, repeat=repeat, flash_attention=fp32_attention)
    scopes['wan.modules.causal_model'].update(ConfigMixin=ConfigMixin, register_to_config=register_to_config,
        ModelMixin=ModelMixin, attention=fp32_attention, create_block_mask=_uncached_not_admitted,
        flex_attention=_uncached_not_admitted, BlockMask=object, dist=SingletonBroadcast)
    scopes['model.base'].update(rearrange=rearrange, dist=SingletonBroadcast)
    scopes['pipeline.self_forcing_training']['dist'] = SingletonBroadcast
    source = PinnedSource(root, SOURCE_FILES, import_roots=('wan', 'model', 'utils', 'pipeline', 'demo_utils'),
        definitions=definitions, definition_globals=scopes)
    api = SimpleNamespace(source=source)
    api.wan = source.import_module('wan.modules.model')
    api.wan.WAN_CROSSATTENTION_CLASSES = dict(t2v_cross_attn=api.wan.WanT2VCrossAttention,
                                            i2v_cross_attn=api.wan.WanI2VCrossAttention)
    scopes['wan.modules.causal_model'].update({name: getattr(api.wan, name) for name in
        ('WanRMSNorm', 'rope_apply', 'WanLayerNorm', 'WAN_CROSSATTENTION_CLASSES',
         'rope_params', 'MLPProj', 'sinusoidal_embedding_1d')})
    api.causal = source.import_module('wan.modules.causal_model')
    api.vae = source.import_module('wan.modules.vae')
    api.t5 = source.import_module('wan.modules.t5')
    api.scheduler = source.import_module('utils.scheduler')
    api.loss = source.import_module('utils.loss')
    scopes['utils.wan_wrapper'].update(SchedulerInterface=api.scheduler.SchedulerInterface,
        FlowMatchScheduler=api.scheduler.FlowMatchScheduler, WanModel=api.wan.WanModel,
        CausalWanModel=api.causal.CausalWanModel, _video_vae=api.vae._video_vae)
    api.wrapper = source.import_module('utils.wan_wrapper')
    for name in ('pipeline.self_forcing_training', 'pipeline.causal_inference', 'model.base'):
        scopes[name].update({key: getattr(api.wrapper, key) for key in
            ('WanDiffusionWrapper', 'WanTextEncoder', 'WanVAEWrapper')})
        scopes[name]['SchedulerInterface'] = api.scheduler.SchedulerInterface
    api.training = source.import_module('pipeline.self_forcing_training')
    boundary, api.boundary_identity = _portable_boundary(source, api.training)
    api.TrainingPipeline = type('PortableTrainingPipeline', (GeometryCache, api.training.SelfForcingTrainingPipeline),
                                {'inference_with_trajectory': boundary})
    scopes['model.base'].update(SelfForcingTrainingPipeline=api.TrainingPipeline,
                               get_denoising_loss=api.loss.get_denoising_loss)
    api.base = source.import_module('model.base')
    scopes['model.dmd'].update(SelfForcingModel=api.base.SelfForcingModel,
                              SelfForcingTrainingPipeline=api.TrainingPipeline)
    api.dmd = source.import_module('model.dmd')
    api.inference = source.import_module('pipeline.causal_inference')
    api.InferencePipeline = type('PortableInferencePipeline', (GeometryCache, api.inference.CausalInferencePipeline), {})
    api.ema = source.import_module('utils.distributed')
    misc = source.import_module('utils.misc')
    scopes['trainer.distillation'].update(EMA_FSDP=api.ema.EMA_FSDP,
        merge_dict_list=misc.merge_dict_list, gc=gc, logging=logging, time=time)
    api.trainer = source.import_module('trainer.distillation')
    return api


def original(root):
    api = _original(str(Path(root).expanduser().resolve(strict=True)))
    api.source.validate()
    return api
