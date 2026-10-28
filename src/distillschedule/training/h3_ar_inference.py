"""Backend-native W6 current-Student snapshot at a restored cycle boundary.

The installed ZeRO-3 implementation owns parameter gathering. This helper only
admits capacity and selects the unchanged base plus unmerged Student from its
result. It is not a trainer, sampler or public inference-artifact publisher.
"""
import importlib.metadata
import inspect
import math
from pathlib import Path

from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from . import distributed
from .checkpoint import rng_state, restore_rng
from .host_memory import admit_host_memory

SCHEMA = "distillschedule.h3_ar_native_student_snapshot.v1"
ENGINE_SHA256 = "ef3b4797809453f4941eaa769c9cd4e397e3b52085131e8916990ea78d4f0f6c"


def require(ok, message):
    if not ok:
        raise ValueError("native H3 Student snapshot: " + message)


def owner_state(session):
    """Host-only ownership witness, never a model-sized tensor/value scan."""
    return dict(versions=session.versions(), native_global_steps=session.engine.global_steps,
        parameters={n: (id(p), p.requires_grad) for n, p in session.engine.module.named_parameters()},
        training={n: m.training for n, m in session.engine.module.named_modules()},
        roles=[layer.role for _, layer in session.adapter.layers],
        optimizer=dict(active_role=session.optimizer.active_role, train_heads=session.optimizer.train_heads),
        cursor=session.input_provider.completed, pending=session.input_provider.pending)


def _layout(module):
    import torch
    result = {}
    # keep_vars retains real ds_shape for partitioned Parameter placeholders.
    for name, value in module.state_dict(keep_vars=True).items():
        require(isinstance(value, torch.Tensor), "native model state contains unsupported extra state")
        shape = list(getattr(value, "ds_shape", value.shape))
        require(all(type(n) is int and n > 0 for n in shape), "invalid logical native parameter shape")
        result[name] = dict(shape=shape, dtype=str(value.dtype),
                            bytes=math.prod(shape)*value.element_size())
    return result


def _admit(session, native, budget):
    import torch
    from deepspeed.runtime.engine import DeepSpeedEngine
    from .h3_ar_state import NativeCheckpoint
    from distillschedule.strategies.h3_ar_session import H3ARSession
    rank, world = distributed.rank_world()
    require(type(session) is H3ARSession and type(native) is NativeCheckpoint and native.session is session
        and type(session.engine) is DeepSpeedEngine and world == 16 and session.rank == rank,
        "actual same-owner native world16 H3ARSession/DeepSpeed checkpoint required")
    require(importlib.metadata.version("deepspeed") == "0.18.4"
        and file_sha256(Path(inspect.getsourcefile(DeepSpeedEngine))) == ENGINE_SHA256
        and getattr(session.engine._zero3_consolidated_16bit_state_dict, "__func__", None)
            is DeepSpeedEngine._zero3_consolidated_16bit_state_dict,
        "exact installed unmodified native ZeRO-3 consolidation API required")
    require(session.local_completed == 0 and native.progress is not None
        and native.progress.cursor is session.input_provider and session.input_provider.pending is None
        and session.input_provider.completed == session.trainer.generator_step
        and session.engine.module.adapter is session.adapter
        and type(session.adapter) is session.native.H3Adapter,
        "snapshot requires a freshly restored complete original owner, not an active training segment")
    native.validate_checkpoint_boundary()
    limits = dict(max_parameters=40_000_000_000, max_consolidated_bytes=256*2**30,
                  host_reserve_bytes=512*2**30, device_reserve_bytes=128*2**30)
    require(isinstance(budget, dict) and set(budget) == set(limits)
        and all(type(budget[k]) is int and 0 < budget[k] <= limit for k, limit in limits.items()),
        "explicit bounded consolidation/host/device capacity required")
    layout = _layout(session.engine.module)
    selected = {"adapter."+name: name for name in _layout(session.adapter)
                if ".adapters.fake_score." not in name}
    require(selected and set(selected) <= set(layout)
        and all(set(layer.adapters) == {"student", "fake_score"} for _, layer in session.adapter.layers)
        and any(".adapters.student." in name for name in selected)
        and not any("discriminator" in name for name in selected),
        "native current Student/base inventory is incomplete or contains another role")
    count = sum(math.prod(row["shape"]) for row in layout.values())
    full_bytes = sum(row["bytes"] for row in layout.values())
    # Match the actual installed method: it gathers direct layer parameters,
    # releases them, then recurses. Do not budget a full 33B GPU materialization.
    layer_bytes = max(sum(math.prod(getattr(p, "ds_shape", p.shape))*p.element_size()
                          for p in layer.parameters(recurse=False))
                      for layer in session.engine.module.modules())
    require(count <= budget["max_parameters"] and full_bytes <= budget["max_consolidated_bytes"],
        "native consolidated model capacity exceeded")
    host = admit_host_memory(2*full_bytes+layer_bytes if rank == 0 else 0, budget["host_reserve_bytes"])
    free, _ = torch.cuda.mem_get_info(session.device)
    require(free >= layer_bytes+budget["device_reserve_bytes"], "native layer gather/device reserve capacity exceeded")
    return layout, selected, dict(schema_version=SCHEMA, cycle=session.input_provider.completed,
        versions=session.versions(), native_global_steps=session.engine.global_steps,
        engine_source_sha256=ENGINE_SHA256, model_layout_sha256=fingerprint(layout),
        student_layout_sha256=fingerprint({selected[k]: layout[k] for k in selected}),
        parameters=count, full_consolidated_bytes=full_bytes, max_layer_gather_bytes=layer_bytes,
        retained_on_primary_only=True, merged_lora=False, host_capacity=host,
        selection="FROZEN_BASE_PLUS_CURRENT_UNMERGED_STUDENT_NO_FAKE_OR_DISCRIMINATOR")


def consolidate_current_student(session, native_checkpoint, *, budget):
    """All sixteen owners call; only primary receives an independent CPU state.

    No forward/update/cache selection or model mutation is performed here.
    Native gathering remains a real collective; its CUDA behavior requires
    separate native qualification. A backend failure is propagated, never
    converted into a partial usable snapshot or retried in this helper.
    """
    import torch
    layout, selected, receipt = distributed.checked(lambda: _admit(session, native_checkpoint, budget),
        "native H3 current Student/source/boundary/capacity admission")
    binding = {k: v for k, v in receipt.items() if k != "host_capacity"}
    require(all(row == binding for row in distributed.gather(binding)), "ranks disagree on native Student snapshot")
    before = owner_state(session)
    random = rng_state(device=session.device)
    try:
        full = session.engine._zero3_consolidated_16bit_state_dict(exclude_frozen_parameters=False)
        def retain():
            if session.rank != 0:
                require(full is None, "nonprimary received unexpected consolidated CPU weights")
                return None
            require(isinstance(full, dict) and set(full) == set(layout), "native consolidated inventory changed")
            result = {}
            for name, target in selected.items():
                value, spec = full[name], layout[name]
                require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                    and list(value.shape) == spec["shape"] and str(value.dtype) == spec["dtype"]
                    and bool(torch.isfinite(value).all()), "native Student payload shape/dtype/finite mismatch")
                result[target] = value.detach().clone()
            return result
        values = distributed.checked(retain, "native H3 independent current Student CPU snapshot")
        require(owner_state(session) == before, "native consolidation changed owner flags/modes/versions")
        return values, receipt
    finally:
        restore_rng(random)
