"""Manual-only, SHA-bound large self-attention backend experiment.

No canonical strategy, checkpoint, dtype, sampler or planner changes. Loader
runs only AFTER the original worker selected LOCAL_RANK, before any capture.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from distillschedule.core.contracts import require

SCHEMA = 'distillschedule.wan_fa3_manual_backend_experiment.v1'
LENGTHS = (2160, 8640, 11040)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LargeSelfAttention:
    def __init__(self, native, backend, receipt):
        self.native, self.backend, self.receipt = native, backend, receipt

    @staticmethod
    def eligible(q, k, v, mask, dropout, causal, scale, gqa):
        import torch
        return (mask is None and dropout == 0 and causal is False and scale is None and gqa is False
                and q.ndim == k.ndim == v.ndim == 4 and q.shape == k.shape == v.shape
                and q.shape[:2] == (1, 12) and q.shape[2] in LENGTHS and q.shape[3] == 128
                and all(x.is_cuda and x.dtype == torch.bfloat16 and x.stride(-1) == 1 for x in (q,k,v)))

    def __call__(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                 *, scale=None, enable_gqa=False):
        import torch
        r = self.receipt
        if not self.eligible(query,key,value,attn_mask,dropout_p,is_causal,scale,enable_gqa):
            r['native_fallback_calls'] += 1
            return self.native(query,key,value,attn_mask,dropout_p,is_causal,
                               scale=scale,enable_gqa=enable_gqa)
        r['fa3_calls'] += 1
        r['fa3_grad_calls' if torch.is_grad_enabled() else 'fa3_nograd_calls'] += 1
        name = str(query.shape[2]); r['fa3_lengths'][name] = r['fa3_lengths'].get(name,0)+1
        result = self.backend.flash_attn_func(query.transpose(1,2),key.transpose(1,2),value.transpose(1,2),
            softmax_scale=None,causal=False,deterministic=False,return_attn_probs=False).transpose(1,2)
        require(result.shape == query.shape and result.dtype == query.dtype and result.device == query.device,
                'FA3 output metadata changed',path='wan.fa3')
        return result


@contextmanager
def dispatch_scope(backend, receipt):
    import torch.nn.functional as F
    native = F.scaled_dot_product_attention
    require(getattr(native,'__module__','') == 'torch._C._nn',
            'pre-existing SDPA override is not admitted',path='wan.fa3')
    wrapper = LargeSelfAttention(native,backend,receipt)
    F.scaled_dot_product_attention = wrapper
    try:
        yield
    finally:
        changed = F.scaled_dot_product_attention is not wrapper
        F.scaled_dot_product_attention = native
        require(not changed,'concurrent SDPA override',path='wan.fa3')


def validate_packet(packet, *, head, task, assets, implementation):
    require(set(packet)=={'schema_version','source_head','arm','run_id','cycles','task_sha256',
        'assets_sha256','strategy_sha256','kernel_path','kernel_files','kernel_import_root'},'unknown packet fields',path='wan.fa3')
    require(packet['schema_version']==SCHEMA and packet['source_head']==head
            and packet['arm']in('native','fa3-large-v1') and packet['cycles']==2,
            'wrong source/arm/scope',path='wan.fa3')
    require(isinstance(packet['run_id'],str) and len(packet['run_id'])>8
            and Path(packet['kernel_import_root']).is_dir(),'explicit run/import identity required',path='wan.fa3')
    require(packet['task_sha256']==sha(task) and packet['assets_sha256']==sha(assets)
            and packet['strategy_sha256']==sha(implementation),'input packet drift',path='wan.fa3')
    from distillschedule.strategies.wan_strategy import WanStrategy
    from distillschedule.strategies.wan_task import OriginalWanTask
    t=OriginalWanTask.from_dict(json.loads(Path(task).read_text()))
    s=WanStrategy.from_dict(json.loads(Path(implementation).read_text()));s.validate_for(t)
    require(t==OriginalWanTask() and s.placement=='resident' and s.candidate_bundle.enabled
            and s.retained_critic_blocks==(26,27,28,29) and s.rollout_window==5
            and s.rollout_graph_pool_policy=='isolated.v1' and s.retention_execution=='manual-only.v1',
            'only exact current manual retain4 admitted',path='wan.fa3')
    require(len(packet['kernel_files'])>=3 and all(Path(p).is_relative_to(Path(packet['kernel_path']))
            and sha(p)==h for p,h in packet['kernel_files'].items()),'backend source/binary drift',path='wan.fa3')


def load_backend(packet):
    # Borrow only the pre-existing loader. Do NOT leave its whole package
    # overlay visible to later model/dependency imports.
    before=list(sys.path)
    try:
        sys.path.append(packet['kernel_import_root'])
        from kernels.utils import get_local_kernel
        return get_local_kernel(Path(packet['kernel_path']),'flash_attn3')
    finally:
        sys.path[:]=before


@contextmanager
def explicit_worker(worker, packet, packet_sha256):
    original = worker.OriginalWanSession; scopes=[]
    class Session(original):
        def __init__(self, **kwargs):
            receipt=dict(schema_version=SCHEMA,source_head=packet['source_head'],arm=packet['arm'],
                packet_sha256=packet_sha256,enabled=packet['arm']=='fa3-large-v1',planner=False,
                weights_merged=False,dtype_unchanged=True,fa3_calls=0,fa3_grad_calls=0,
                fa3_nograd_calls=0,native_fallback_calls=0,fa3_lengths={},
                count_scope='PYTHON_EAGER_AND_GRAPH_CAPTURE_ONLY_NOT_GRAPH_REPLAYS',
                kernel_files=packet['kernel_files'])
            if receipt['enabled']:
                import torch
                require(torch.cuda.current_device()==kwargs['device'].index,
                        'original rank device must precede backend import',path='wan.fa3')
                module=load_backend(packet)
                receipt['loaded_backend_file']=str(Path(module.__file__).resolve())
                scope=dispatch_scope(module,receipt);scope.__enter__();scopes.append(scope)
            super().__init__(**kwargs)
            for role in ('generator','critic'):
                self.build[role]['metadata']['self_attention_backend_experiment']=receipt
    worker.OriginalWanSession=Session
    try:
        yield
    finally:
        worker.OriginalWanSession=original
        for scope in reversed(scopes):scope.__exit__(None,None,None)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in('task','assets','implementation','output','packet'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--packet-sha256',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    require(a.execute and sha(a.packet)==a.packet_sha256,'explicit execution and packet hash required',path='wan.fa3')
    from distillschedule.strategies import wan_worker as worker
    from distillschedule.strategies.wan_strategy import WanStrategy
    from distillschedule.strategies.wan_task import OriginalWanTask
    packet=json.loads(a.packet.read_text())
    head=subprocess.check_output(['git','-C',str(worker.repository_root()),'rev-parse','HEAD'],text=True).strip()
    validate_packet(packet,head=head,task=a.task,assets=a.assets,implementation=a.implementation)
    with explicit_worker(worker,packet,a.packet_sha256):
        r=worker.run_worker(OriginalWanTask.from_dict(json.loads(a.task.read_text())),
            worker.OriginalWanAssets.from_dict(json.loads(a.assets.read_text())),output=a.output,
            mode='manual',strategy=WanStrategy.from_dict(json.loads(a.implementation.read_text())),
            cycles=2,record_memory_highwater=True)
    print(json.dumps(dict(status=r['status'],rank=r['rank'],arm=packet['arm'])),flush=True)


if __name__=='__main__':main()
