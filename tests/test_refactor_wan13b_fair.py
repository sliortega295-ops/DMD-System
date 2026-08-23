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
        "resident-clean",
        "resident-async128",
        "resident-local-fast",
        "resident-flat256",
        "resident-flat512",
        "resident-lora-zero2",
        "resident-rollout-fused",
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


def test_resident_local_fast_keeps_manual_sync_and_uses_local_fast_kernels() -> None:
    builder = inspect.getsource(wan.build_fair_engine)
    engine = inspect.getsource(wan.LocalFastResidentEngine)
    assert "LOCAL_FAST_RESIDENT" in builder
    assert "LocalFastResidentEngine" in builder
    assert "_synchronize_trainable_gradients" in engine
    assert "foreach=True" in engine
    assert "DistributedDataParallel" not in engine
    assert '"PYTORCH_FOREACH_TRUE"' in builder


def test_persistent_flat_reducer_uses_stable_grad_views_and_one_scale_per_bucket() -> None:
    source = inspect.getsource(wan.PersistentFlatResidentEngine)
    assert "parameter.grad = view" in source
    assert "gradient.data_ptr() != view.data_ptr()" in source
    assert "dist.all_reduce(flat" in source
    assert "flat.mul_(1.0 / WORLD_SIZE)" in source
    assert '"persistent": True' in source


def test_grad_ready_async_reducer_is_ordered_overlapped_and_reference_local_math() -> None:
    source = inspect.getsource(wan.GradReadyAsyncResidentEngine)
    builder = inspect.getsource(wan.build_fair_engine)
    assert "register_post_accumulate_grad_hook" in source
    assert "reversed(named)" in source
    assert "torch.cuda.Stream" in source
    assert "async_op=True" in source
    assert "bucket.work.wait()" in source
    assert "wait_stream(self._comm_stream)" in source
    assert '"overlap_with_backward": True' in source
    assert '"hook_count_min"' in source
    assert "GradReadyAsyncResidentEngine" in builder
    assert "ASYNC_128_RESIDENT: 128" in builder
    assert "ASYNC_128_RESIDENT" not in inspect.getsource(wan.LocalFastResidentEngine)


def test_lora_zero2_keeps_model_resident_and_shards_only_trainable_state() -> None:
    source = inspect.getsource(wan.LoRAZero2ResidentEngine)
    assert "dist.reduce_scatter_tensor" in source
    assert "dist.all_gather_into_tensor" in source
    assert "optimizer_state_sharded" in source
    assert "parameter.copy_" in source
    assert "FULL_SHARD" not in source


def test_rollout_fused_routes_only_post_commit_no_grad_calls_to_shadow() -> None:
    source = inspect.getsource(wan.RolloutFusedLoRAZero2Engine)
    assert "not torch.is_grad_enabled()" in source
    assert "super().step()" in source
    assert "fuse_lora" in source
    assert "shadow_refresh_count" in source


def test_resident_clean_keeps_reference_engine_and_makes_state_requests_idempotent() -> None:
    builder = inspect.getsource(wan.build_fair_engine)
    assert "REFERENCE_RESIDENT" in builder
    assert "CLEAN_RESIDENT" in builder
    assert "LOCAL_FAST_RESIDENT" in builder
    assert "ReferenceResidentEngine" in builder

    class Model:
        def __init__(self) -> None:
            self.training = True
            self.adapter_enabled = True
            self.train_calls = 0
            self.enable_calls = 0
            self.disable_calls = 0

        def train(self, mode: bool = True):
            self.train_calls += 1
            self.training = mode
            return self

        def enable_adapters(self) -> None:
            self.enable_calls += 1
            self.adapter_enabled = True

        def disable_adapters(self) -> None:
            self.disable_calls += 1
            self.adapter_enabled = False

    model = Model()
    state = wan.install_resident_clean_model_state(model)
    initial_train_calls = model.train_calls
    initial_enable_calls = model.enable_calls
    model.train()
    model.train()
    model.enable_adapters()
    model.enable_adapters()
    assert model.train_calls == initial_train_calls
    assert model.enable_calls == initial_enable_calls
    model.disable_adapters()
    model.disable_adapters()
    model.enable_adapters()
    assert model.disable_calls == 1
    assert model.enable_calls == initial_enable_calls + 1
    assert state.adapter_disable_transitions == 1
    assert state.adapter_enable_transitions == 1


def test_fair_worker_reuses_frozen_workload_and_audits_runtime_dtypes() -> None:
    source = (
        ROOT / "scripts/distillschedule/probe_wan13b_fair_world16.py"
    ).read_text()
    assert "probe_wan13b_upper_bound_world16.py" in source
    assert "apply_wan13b_fair_fsdp2_policy" in source
    assert "LoRAComputeDtypeAudit" in source
    assert "configure_fsdp2_checkpointing(model)" in source
    assert 'chunked_residual_storage="cuda_out_of_place"' in source
    assert "install_resident_clean_model_state" in source

    base_worker = (
        ROOT / "scripts/distillschedule/probe_wan13b_upper_bound_world16.py"
    ).read_text()
    assert '"resident-flat256"' in base_worker
    assert '"resident-async128"' in base_worker
    assert '"resident-flat512"' in base_worker
    assert '"resident-lora-zero2"' in base_worker
    assert '"resident-rollout-fused"' in base_worker
    assert "entry_outputs=critic_entries" in base_worker
    assert "deferred_finite_checks=deferred_finite_checks" in base_worker
    assert "resident-clean deferred cycle finite gate failed" in base_worker


def test_fair_launcher_is_fail_closed_and_never_profiles_as_timing() -> None:
    source = (
        ROOT / "scripts/distillschedule/launch_wan13b_fair_world16_node.sh"
    ).read_text()
    assert "resident-ddp" in source
    assert "resident-clean" in source
    assert "resident-async128" in source
    assert "resident-local-fast" in source
    assert "resident-flat256" in source
    assert "resident-flat512" in source
    assert "resident-lora-zero2" in source
    assert "resident-rollout-fused" in source
    assert "resident-fast" in source
    assert "world16-fsdp" in source
    assert "DISTILLSCHEDULE_WAN13B_FAIR_CORRECTNESS_RECEIPT" in source
    assert "claimable_for_paired_timing" in source
    assert "frozen_base_compute_dtype=torch.bfloat16" in source
    assert "lora_compute_dtype=torch.float32" in source
    assert "diagnostic-timing" in source
    assert "diagnostic timing requires a PASS semantic correctness summary" in source
    assert '"${MODE}" == profile' in source
    assert "DISTILLSCHEDULE_STATIC_FSDP2_NSYS_EXACT_RANGE=1" in source
    assert "capture_mode=process_wide_clipped_to_exact_nvtx" in source
    assert "nsys_profile_validation_failed" in source
    assert "claim_boundary=$(" in source
    assert "DIAGNOSTIC_ONLY" in source
    assert "correctness receipt uses a different Helios residual runtime" in source
    assert "helios_transformer_sha256=${EXPECTED_HELIOS_TRANSFORMER_SHA256}" in source


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
    assert '"profile"' in controller
    assert "DISTILLSCHEDULE_NSYS_BIN" in controller
    assert "correctness receipt covers" in paired
    assert "warmup" in paired
    assert "geometric_mean_candidate_speedup" in paired
