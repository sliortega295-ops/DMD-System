"""Ownership of an explicitly requested torchrun group, not a cluster launcher.

Native recipes retain their original process-group/role-pool owners. This
module serves the existing generic Program backend and never acquires GPUs or
changes network/environment settings.
"""
import hashlib
import os
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from pathlib import Path


def rank_world():
    import torch.distributed as dist
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def checked(fn, label):
    if rank_world()[1] == 1:
        return fn()
    from distillschedule.open_workload.replicated import checked as collective_checked
    return collective_checked(fn, label)


def gather(value):
    if rank_world()[1] == 1:
        return [value]
    from distillschedule.open_workload.replicated import gather as collective_gather
    return collective_gather(value)


def primary(fn, label):
    return checked(lambda: fn() if rank_world()[0] == 0 else None, label)


def admit_run_controls(config):
    """Agree on optional collective branches before construction or restart.

    Restart paths may be explicit mount aliases: compare completed manifest
    bytes, not their local spelling. Exchange local read failures before any
    rank can enter a model factory or a checkpoint collective.
    """
    def local():
        trainer = config.trainer
        manifest_sha = None
        if trainer.get("resume_from"):
            root = Path(trainer["resume_from"])
            manifest_sha = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
            if (root / "_SUCCESS").read_text().strip() != manifest_sha:
                raise ValueError("resume checkpoint is not complete or its manifest changed")
        return {"cycles": trainer["cycles"], "checkpoint_every": trainer.get("checkpoint_every", 1),
                "output_root": str(Path(trainer["output_dir"]).resolve()),
                "resume_manifest_sha256": manifest_sha}

    controls = checked(local, "training run/restart control admission")
    if any(row != controls for row in gather(controls)):
        raise ValueError("ranks disagree on training run/restart controls: "
                         "cycles, checkpoint cadence, output root or resume manifest")


def topology():
    import torch.distributed as dist
    if rank_world()[1] == 1:
        return {"world_size": 1}
    from distillschedule.open_workload.topology import discover
    allocated = discover()
    return {"world_size": rank_world()[1], "backend": str(dist.get_backend()),
            "node_groups": allocated["node_groups"]}


@contextmanager
def context(config):
    import torch
    import torch.distributed as dist
    requested = config.trainer.get("distributed")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    owned = False
    metadata_group = None
    if not dist.is_initialized() and world > 1:
        if not requested:
            raise ValueError("torchrun training requires explicit trainer.distributed configuration")
        backend = requested["backend"]
        if backend == "nccl":
            if not torch.cuda.is_available() or not dist.is_gloo_available():
                raise ValueError("NCCL training requires CUDA and a Gloo metadata backend")
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend, timeout=timedelta(seconds=requested.get("timeout_seconds", 180)))
        owned = True
    try:
        rank, actual_world = rank_world()
        if actual_world > 1 and str(dist.get_backend()) == "nccl":
            metadata_group = dist.new_group(backend="gloo", timeout=timedelta(
                seconds=requested.get("timeout_seconds", 180) if requested else 180))
        with ExitStack() as stack:
            if metadata_group is not None:
                from distillschedule.open_workload.replicated import control_plane
                stack.enter_context(control_plane(metadata_group))
            def validate():
                if actual_world > 1:
                    if world != actual_world:
                        raise ValueError("launcher and initialized process-group world sizes differ")
                    if requested and str(dist.get_backend()) != requested["backend"]:
                        raise ValueError("initialized process group differs from trainer.distributed")
                    if config.data.get("sharding") != "rank_strided":
                        raise ValueError("distributed training requires explicit data.sharding=rank_strided")
                elif requested:
                    raise ValueError("trainer.distributed requires at least two explicitly launched ranks")
            checked(validate, "training resource contract")
            yield rank, actual_world
    finally:
        # Do not destroy a group owned by an embedding application/native adapter.
        try:
            if metadata_group is not None:
                dist.destroy_process_group(metadata_group)
        finally:
            if owned:
                dist.destroy_process_group()
