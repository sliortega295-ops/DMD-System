from __future__ import annotations

import inspect
import json
from pathlib import Path

from distillschedule.runtime import wan13b_fair as wan


ROOT = Path(__file__).resolve().parents[1]


def test_fair_contract_has_reference_ddp_fast_and_fsdp_endpoints() -> None:
    value = json.loads(
        (
            ROOT
            / "configs/distillschedule/baselines/wan13b_nogan_world16_fair.json"
        ).read_text()
    )
    assert tuple(value["placements"]) == wan.FAIR_PLACEMENTS
    assert wan.FAIR_PLACEMENTS == (
        "resident",
        "resident-ddp",
        "resident-fast",
        "world16-fsdp",
    )
    assert value["model"]["frozen_base_compute_dtype"] == "torch.bfloat16"
    assert value["model"]["trainable_lora_compute_dtype"] == "torch.float32"
    assert value["common"]["gradient_mean"] == "SUM_THEN_DIVIDE_WORLD_SIZE"
    assert value["common"]["parameter_offload"] is False
    assert value["common"]["activation_offload"] is False
    assert value["common"]["optimizer_offload"] is False


def test_fair_fsdp_preserves_original_dtypes_and_matches_mean_convention() -> None:
    source = inspect.getsource(wan.apply_wan13b_fair_fsdp2_policy)
    assert "param_dtype=None" in source
    assert "reduce_dtype=torch.float32" in source
    assert "cast_forward_inputs=False" in source
    assert "set_force_sum_reduction_for_comms(True)" in source
    assert "set_gradient_divide_factor(float(WORLD_SIZE))" in source
    assert '"PRESERVE_ORIGINAL_PER_PARAMETER"' in source


def test_resident_ddp_is_full_residency_with_overlapped_trainable_sync() -> None:
    source = inspect.getsource(wan.ResidentDDPEngine)
    assert "DistributedDataParallel" in source
    assert "init_sync=True" in source
    assert "broadcast_buffers=False" in source
    assert "gradient_as_bucket_view=True" in source
    assert "static_graph=True" in source
    assert "register_comm_hook" in source
    hook = inspect.getsource(wan.ddp_sum_then_divide_hook)
    assert "ReduceOp.SUM" in hook
    assert "async_op=True" in hook
    assert "1.0 / state.world_size" in hook
    signature = inspect.signature(wan.ddp_sum_then_divide_hook)
    assert signature.parameters["bucket"].annotation is inspect.Parameter.empty
    assert signature.return_annotation is inspect.Signature.empty


def test_resident_fast_changes_only_ema_implementation_above_resident_ddp() -> None:
    source = inspect.getsource(wan.FairTrainableEMA.step)
    assert "torch._foreach_lerp_" in source
    builder = inspect.getsource(wan.build_fair_engine)
    assert "placement in {DDP_RESIDENT, FAST_RESIDENT}" in builder
    assert '"PYTORCH_FUSED" if fused else "PYTORCH_DEFAULT"' in builder
    assert "ReferenceResidentEngine" in builder


def test_fair_worker_reuses_frozen_workload_and_audits_runtime_dtypes() -> None:
    source = (
        ROOT / "scripts/distillschedule/probe_wan13b_fair_world16.py"
    ).read_text()
    assert "probe_wan13b_upper_bound_world16.py" in source
    assert "apply_wan13b_fair_fsdp2_policy" in source
    assert "LoRAComputeDtypeAudit" in source
    assert "configure_fsdp2_checkpointing(model)" in source
    assert 'chunked_residual_storage="cuda_out_of_place"' in source


def test_fair_launcher_is_fail_closed_and_never_profiles_as_timing() -> None:
    source = (
        ROOT / "scripts/distillschedule/launch_wan13b_fair_world16_node.sh"
    ).read_text()
    assert "resident-ddp" in source
    assert "resident-fast" in source
    assert "world16-fsdp" in source
    assert "DISTILLSCHEDULE_WAN13B_FAIR_CORRECTNESS_RECEIPT" in source
    assert "claimable_for_paired_timing" in source
    assert "frozen_base_compute_dtype=torch.bfloat16" in source
    assert "lora_compute_dtype=torch.float32" in source


def test_fair_comparator_keeps_default_p0_thresholds() -> None:
    source = (
        ROOT / "scripts/distillschedule/compare_wan13b_fair_correctness.py"
    ).read_text()
    assert "DEFAULT_P0_PARITY" in source
    assert "runtime LoRA compute dtype evidence is missing" in source
    assert '"claimable_for_paired_timing": not errors' in source


def test_resident_fast_uses_fused_adam_with_device_step_state() -> None:
    source = inspect.getsource(wan.build_fair_engine)
    state_source = inspect.getsource(wan._materialize_resident_adam_state)
    assert 'optimizer_kwargs["fused"] = True' in source
    assert '"PYTORCH_FUSED" if fused else "PYTORCH_DEFAULT"' in source
    assert 'device=parameter.device' in state_source
    assert '"step_tensor_device": "parameter_device" if fused else "cpu"' in state_source


def test_fair_controller_and_pair_driver_are_receipt_bound() -> None:
    controller = (
        ROOT / "scripts/distillschedule/run_wan13b_fair_world16.py"
    ).read_text()
    paired = (ROOT / "scripts/distillschedule/run_wan13b_fair_paired.py").read_text()
    assert "launch_wan13b_fair_world16_node.sh" in controller
    assert "summarize_wan13b_fair_world16.py" in controller
    assert "DISTILLSCHEDULE_WAN13B_FAIR_CORRECTNESS_RECEIPT" in controller
    assert "correctness receipt covers" in paired
    assert "warmup" in paired
    assert "geometric_mean_candidate_speedup" in paired
