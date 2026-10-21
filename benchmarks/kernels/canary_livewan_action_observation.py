#!/usr/bin/env python3
"""No-model, two-node canary for the native LiveWan observation seam.

Use one owned GPU per node with the ordinary node lock held by the launcher.
This exercises the native copy-stream/Gloo sender and paired Critic-shadow
primitives. It is not a training correctness gate or a service-speed benchmark.
"""
import argparse
from collections import Counter
from datetime import timedelta
import json
from pathlib import Path
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node-rank', type=int, choices=(0,1), required=True)
    parser.add_argument('--master-addr', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('immutable canary receipt already exists')
    root = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip()
    if head != args.expected_head:
        raise RuntimeError('canary source differs')
    subprocess.run(['git','diff','--exit-code','HEAD','--','src','scripts/distillschedule'], cwd=root, check=True)
    import torch
    import torch.distributed as dist
    from distillschedule.runtime.livewan_host_runahead import PinnedHostRunaheadTransport
    from distillschedule.runtime.livewan_critic_shadow_sync import PairedCriticShadowSync
    from distillschedule.search.livewan_action_observation import TransportObserver
    if torch.cuda.device_count() != 1:
        raise RuntimeError('canary requires exactly one visible owned GPU per node')
    rank, device = args.node_rank, torch.device('cuda:0')
    torch.cuda.set_device(device)
    started = time.perf_counter()
    cgroup_path = Path('/sys/fs/cgroup/memory.events')
    def cgroup():
        return {k:int(v) for k,v in (line.split() for line in cgroup_path.read_text().splitlines())}
    before = cgroup()
    dist.init_process_group('gloo',init_method=f'tcp://{args.master_addr}:{args.port}',
                            world_size=2,rank=rank,timeout=timedelta(seconds=90))
    shadow_group = dist.new_group([0,1],backend='nccl',timeout=timedelta(seconds=90))
    params = [torch.nn.Parameter(torch.full((8,),float(3+i if rank else 0),device=device)) for i in range(2)]
    observer = TransportObserver(torch=torch,device=device,rank=rank)
    transport = PinnedHostRunaheadTransport(torch=torch,dist=dist,group=dist.group.WORLD,
        rank=rank,peer=1-rank,device=device,is_generator=rank == 0,shape=(4,4),dtype=torch.float32,
        cycles=2,cross_cycle_lookahead=True,observer=observer)
    shadow = PairedCriticShadowSync(torch=torch,dist=dist,group=shadow_group,rank=rank,peer=1-rank,
        score_rank=1,device=device,parameters=params,is_generator=rank == 0,run_kind='timing',observer=observer)
    pair_contract = shadow.validate_pair_contract()  # Setup-only native handshake.
    dist.barrier()
    received = {}
    reuse_keys = {(s,w) for s in (1,2,3,6,7,8) for w in (0,1)} | {(4,0)}
    if rank == 0:
        for step,wave in sorted(transport.expected_keys):
            value = torch.full((4,4),float(10*step+wave),device=device)
            transport.enqueue_send(step=step,wave=wave,tensor=value)
            if (step,wave) in reuse_keys:
                transport.wait_staged(step=step,wave=wave)
        transport.finish()
        shadow.receive_and_load(version=5)
    else:
        for step,wave in sorted(transport.expected_keys):
            received[step,wave] = transport.receive_to_device(step=step,wave=wave)
        transport.finish()
        shadow.begin_source(version=5)
    shadow.finish()
    torch.cuda.synchronize(device)
    dist.barrier()
    observed = observer.finish()
    # Tensor values are checked only after all observed native work is complete.
    if rank == 1:
        for (step,wave),value in received.items():
            torch.testing.assert_close(value,torch.full_like(value,float(10*step+wave)),rtol=0,atol=0)
    for i,value in enumerate(params):
        torch.testing.assert_close(value,torch.full_like(value,float(3+i)),rtol=0,atol=0)
    counts = Counter(r['kind'] for r in observed['rows'])
    expected = ({'latent_d2h':17,'sender_copy_ready_wait':17,'gloo_send_completion':17,
                 'streamer_reuse_copy_wait':13,'shadow_receiver_transfer_wait':1,'shadow_target_load':1}
                if rank == 0 else {'latent_h2d':17,'gloo_receive_wait':17,'shadow_source_pack':1,'shadow_source_completion_wait':1})
    if counts != expected or observed['status'] != 'PASS':
        raise RuntimeError(f'observer inventory failed: {counts}')
    after = cgroup()
    if any(after.get(k,0) != before.get(k,0) for k in ('oom','oom_kill')):
        raise RuntimeError('new cgroup OOM')
    free,total = torch.cuda.mem_get_info()
    result = dict(status='PASS_TRANSPORT_OBSERVER_CANARY',rank=rank,host=socket.gethostname(),
        source_head=head,torch_version=torch.__version__,cuda=torch.version.cuda,nccl=dist.get_backend(shadow_group),
        training='NOT_RUN',model_load='NOT_RUN',performance_claim=False,pair_contract=pair_contract,
        transport=transport.receipt(),shadow=shadow.receipt(),observation=observed,
        max_allocated=torch.cuda.max_memory_allocated(),max_reserved=torch.cuda.max_memory_reserved(),
        driver_free=free,driver_total=total,cgroup_before=before,cgroup_after=after,
        diagnostic_process_wall_seconds=time.perf_counter()-started)
    dist.barrier()
    dist.destroy_process_group(shadow_group)
    dist.destroy_process_group()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps({'status':result['status'],'rank':rank,'receipt':str(args.output)}),flush=True)


if __name__ == '__main__':
    main()
