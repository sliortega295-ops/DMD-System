#!/usr/bin/env python3
"""No-weight eight-rank native Triton cache-construction gate; not DMD timing."""
from __future__ import annotations
import argparse,json,os,runpy,sys,pathlib,hashlib,traceback
ROOT=pathlib.Path(__file__).resolve().parents[2]
def load_native_rmsnorm():
    """Use the production offline import policy, without constructing a model."""
    from runtime.critic_pool_helios import configure_offline_helios
    configure_offline_helios()
    from helios.modules.helios_kernels.triton_norm import flash_rms_layernorm
    return flash_rms_layernorm

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=pathlib.Path,required=True);args=p.parse_args()
    rank=int(os.environ["RANK"]);assert os.environ["WORLD_SIZE"]=="8"
    result=dict(status="FAIL",rank=rank,model="NOT_LOADED",formal_timing="NOT_RUN")
    try:
        worker=runpy.run_path(str(ROOT/"scripts/distillschedule/probe_wan13b_two_cycle_timing_world16.py"),run_name="rank_cache_import_only")
        cache=worker["_RANK_PRIVATE_CACHE"];assert cache["enabled"] and cache["fail_fast_on_error"] and cache["local_rank"]==rank
        import torch,torch.distributed as dist
        torch.cuda.set_device(rank);device=torch.device("cuda",rank)
        dist.init_process_group("gloo")
        flash_rms_layernorm = load_native_rmsnorm()
        import inspect
        source=hashlib.sha256(inspect.getsource(flash_rms_layernorm).encode()).hexdigest()
        norm=torch.nn.RMSNorm(128,eps=1e-6).to(device=device,dtype=torch.bfloat16)
        dist.barrier()
        for rows in (540,2160,8640):
            x=torch.ones((1,rows,128),device=device,dtype=torch.bfloat16,requires_grad=True)
            y=flash_rms_layernorm(norm,x);y.float().square().mean().backward()
            assert torch.isfinite(y).all().item() and torch.isfinite(x.grad).all().item()
            norm.zero_grad(set_to_none=True)
        torch.cuda.synchronize();dist.barrier()
        path=pathlib.Path(cache["path"]);metadata=list(path.rglob("*.json"));assert metadata,"NO_REAL_COMPILER_METADATA"
        hashes={}
        for f in metadata:
            json.loads(f.read_text());hashes[str(f.relative_to(path))]=hashlib.sha256(f.read_bytes()).hexdigest()
        result.update(status="PASS_ACTUAL_NATIVE_RMSNORM_FW_BW_PRIVATE_COMPILE_CACHE",cache=cache,
            metadata_sha256=hashes,native_function_source_sha256=source,
            real_forward_backward_layouts=[540,2160,8640],physical_GPUs=8,
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            minimum_free="NOT_CONTINUOUSLY_MEASURED",driver_free_after=torch.cuda.mem_get_info()[0],
            tensor_finite=True,training_capacity_or_parity="NOT_RUN")
        with (args.output/f"rank_{rank:02d}.json").open("x") as f:json.dump(result,f,indent=2)
        dist.destroy_process_group()
    except BaseException as e:
        result.update(error=repr(e),traceback=traceback.format_exc())
        with (args.output/f"rank_{rank:02d}.json").open("x") as f:json.dump(result,f,indent=2)
        os._exit(1)
if __name__=="__main__":main()
