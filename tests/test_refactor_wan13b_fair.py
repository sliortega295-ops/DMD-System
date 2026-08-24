from __future__ import annotations

import inspect
import json
from pathlib import Path

from distillschedule.runtime import wan13b_fair as wan
from distillschedule.runtime import wan13b_lora_cast as lora_cast


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
        "resident-async128-tree",
        "resident-async128-tree-chunk2048",
        "resident-async128-tree-chunk2048-compile",
        "resident-async256",
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
    assert "ASYNC_128_TREE_RESIDENT: 128" in builder
    assert "ASYNC_128_TREE_CHUNK2048_RESIDENT: 128" in builder
    assert "ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT: 128" in builder
    assert "ASYNC_256_RESIDENT: 256" in builder
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
    assert "install_generator_nograd_lora_cast_cache" in source
    assert "install_lora_operator_attribution" in source

    base_worker = (
        ROOT / "scripts/distillschedule/probe_wan13b_upper_bound_world16.py"
    ).read_text()
    assert '"resident-flat256"' in base_worker
    assert '"resident-async128"' in base_worker
    assert '"resident-async128-tree"' in base_worker
    assert '"resident-async128-tree-chunk2048"' in base_worker
    assert '"resident-async128-tree-chunk2048-compile"' in base_worker
    assert '"resident-async256"' in base_worker
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
    assert "resident-async128-tree" in source
    assert "resident-async128-tree-chunk2048" in source
    assert "resident-async128-tree-chunk2048-compile" in source
    assert "GENERATOR_NOGRAD_TRANSFORMER_BLOCKS_ONLY" in source
    assert source.index("torch_compile_scope=$( [[") < source.index(
        'cat >"${NODE_LOG}/launch_manifest.txt"'
    )
    assert "export NCCL_ALGO=Tree" in source
    assert "unset NCCL_ALGO" in source
    assert "resident-async256" in source
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
    assert "lora_operator_attribution_requires_profile_mode" in source
    assert "lora_operator_attribution=${LORA_OPERATOR_ATTRIBUTION}" in source
    assert "generator_nograd_lora_cast_cache=${GENERATOR_NOGRAD_LORA_CAST_CACHE}" in source


def test_lora_operator_attribution_is_default_off_and_profile_only(monkeypatch) -> None:
    monkeypatch.delenv(lora_cast.ATTRIBUTION_ENV, raising=False)
    receipt = lora_cast.install_lora_operator_attribution(
        object(), role="generator", profile_mode=False, global_rank=0
    )
    assert receipt["requested"] is False
    assert receipt["enabled"] is False
    assert receipt["disabled_reason"] == "NOT_REQUESTED"

    monkeypatch.setenv(lora_cast.ATTRIBUTION_ENV, "1")
    try:
        lora_cast.install_lora_operator_attribution(
            object(), role="generator", profile_mode=False, global_rank=0
        )
    except RuntimeError as error:
        assert "only in profile mode" in str(error)
    else:
        raise AssertionError("attribution unexpectedly enabled outside profile mode")


def test_lora_operator_attribution_pins_peft_and_preserves_operator_order() -> None:
    source = inspect.getsource(lora_cast.install_lora_operator_attribution)
    assert lora_cast.PEFT_0181_LINEAR_FORWARD_SHA256 == (
        "ec0a80f5c5ce05f5dd90027952bb5d29aaf910e9d2ab40486a90e8a03c7e1cd9"
    )
    assert source.index('op=BASE') < source.index('op=INPUT_CAST')
    assert source.index('op=INPUT_CAST') < source.index('op=LORA_A')
    assert source.index('op=LORA_A') < source.index('op=LORA_B')
    assert source.index('op=LORA_B') < source.index('op=SCALE_ADD')
    assert source.index('op=SCALE_ADD') < source.index('op=OUTPUT_CAST')
    assert "result = result + update * scaling" in source
    assert "result = result.to(torch_result_dtype)" in source
    assert "synchronize" not in source


def test_generator_nograd_lora_cast_cache_reuses_only_exact_input(monkeypatch) -> None:
    import torch

    class FakeLinear(torch.nn.Module):
        cast_input_dtype_enabled = True

        def _cast_input_dtype(self, value, dtype):
            return value.to(dtype=dtype)

        def forward(self, value):
            return self._cast_input_dtype(value, torch.float64)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = FakeLinear()
            self.second = FakeLinear()

        def forward(self, value):
            return self.first(value) + self.second(value)

    model = Model()
    value = torch.tensor([1.0, 2.0], dtype=torch.float32)
    with torch.no_grad():
        reference = model(value)
    monkeypatch.setenv(lora_cast.CAST_CACHE_ENV, "1")
    receipt = lora_cast.install_generator_nograd_lora_cast_cache(
        model,
        role="generator",
        linear_type=FakeLinear,
    )
    with torch.no_grad():
        candidate = model(value)
    assert torch.equal(reference, candidate)
    assert receipt["enabled"] is True
    assert receipt["model_call_count"] == 1
    assert receipt["eligible_cast_requests"] == 2
    assert receipt["cache_miss_count"] == 1
    assert receipt["cache_hit_count"] == 1
    assert receipt["avoided_source_read_bytes"] == value.numel() * value.element_size()
    assert receipt["avoided_destination_write_bytes"] == value.numel() * 8

    before = dict(receipt)
    model(value)
    assert receipt["model_call_count"] == before["model_call_count"]
    assert receipt["cache_hit_count"] == before["cache_hit_count"]


def test_generator_nograd_lora_fused_epilogue_is_no_grad_generator_only(
    monkeypatch,
) -> None:
    import torch

    class FakeLinear(torch.nn.Module):
        disable_adapters = False
        merged = False
        active_adapters = ["default"]
        lora_variant = {}

        def __init__(self) -> None:
            super().__init__()
            self.base_layer = torch.nn.Linear(2, 2, bias=False).to(torch.bfloat16)
            self.lora_A = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 1, bias=False).to(torch.float32)}
            )
            self.lora_B = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(1, 2, bias=False).to(torch.float32)}
            )
            self.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Identity()})
            self.scaling = {"default": 0.5}

        def _check_forward_args(self, value, *args, **kwargs):
            del value, args, kwargs

        def _cast_input_dtype(self, value, dtype):
            return value.to(dtype=dtype)

        def forward(self, value, *args, **kwargs):
            del args, kwargs
            result = self.base_layer(value)
            update = self.lora_B["default"](
                self.lora_A["default"](value.to(torch.float32))
            )
            return (result + update * self.scaling["default"]).to(result.dtype)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = FakeLinear()

        def forward(self, value):
            return self.linear(value)

    calls = []

    def fused(base, update, scaling):
        calls.append((base.dtype, update.dtype, scaling))
        return (base + update * scaling).to(base.dtype)

    model = Model()
    value = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    with torch.no_grad():
        reference = model(value)
    monkeypatch.setenv(lora_cast.FUSED_EPILOGUE_ENV, "1")
    receipt = lora_cast.install_generator_nograd_lora_fused_epilogue(
        model,
        role="generator",
        linear_type=FakeLinear,
        fused_epilogue=fused,
    )
    with torch.no_grad():
        candidate = model(value)
    assert torch.equal(reference, candidate)
    assert receipt["enabled"] is True
    assert receipt["model_call_count"] == 1
    assert receipt["fused_invocation_count"] == 1
    assert calls == [(torch.bfloat16, torch.float32, 0.5)]

    before = dict(receipt)
    model(value)
    assert receipt["model_call_count"] == before["model_call_count"]
    assert receipt["fused_invocation_count"] == before["fused_invocation_count"]


def test_controller_forwards_bounded_lora_attribution_only_to_profile() -> None:
    source = (
        ROOT / "scripts/distillschedule/run_wan13b_fair_world16.py"
    ).read_text()
    assert '"--lora-operator-attribution"' in source
    assert '"--lora-attribution-target-rank"' in source
    assert '"--lora-attribution-target-nograd-calls"' in source
    assert "LoRA operator attribution requires --mode profile" in source
    assert "DISTILLSCHEDULE_WAN13B_LORA_OPERATOR_ATTRIBUTION=1" in source
    assert '"--generator-nograd-lora-cast-cache"' in source
    assert "DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_CAST_CACHE=1" in source
    assert '"--generator-nograd-lora-fused-epilogue"' in source
    assert "DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE=1" in source


def test_tree_chunk2048_changes_only_pinned_modulation_chunk_granularity() -> None:
    class Block:
        pass

    class Model:
        def __init__(self) -> None:
            self.blocks = [Block() for _ in range(wan.WAN13B_BLOCK_COUNT)]

    reference = Model()
    reference_receipt = wan.configure_modulation_chunk_tokens(
        reference, placement=wan.ASYNC_128_TREE_RESIDENT
    )
    assert reference_receipt["effective_tokens"] == [512]
    assert reference_receipt["block_count"] == 30
    assert reference_receipt["all_blocks_match"] is True

    candidate = Model()
    candidate_receipt = wan.configure_modulation_chunk_tokens(
        candidate, placement=wan.ASYNC_128_TREE_CHUNK2048_RESIDENT
    )
    assert candidate_receipt["effective_tokens"] == [2048]
    assert candidate_receipt["block_count"] == 30
    assert candidate_receipt["all_blocks_match"] is True
    assert all(
        block._distillschedule_modulation_chunk_tokens == 2048
        for block in candidate.blocks
    )


def test_compile_arm_routes_only_generator_nograd_blocks(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeTorch:
        grad_enabled = False
        compile_calls = 0

        @classmethod
        def compile(cls, function, **kwargs):
            cls.compile_calls += 1
            assert kwargs == {
                "backend": "inductor",
                "mode": "default",
                "fullgraph": False,
                "dynamic": False,
            }

            def compiled(*args, **call_kwargs):
                return function(*args, **call_kwargs)

            return compiled

        @classmethod
        def is_grad_enabled(cls):
            return cls.grad_enabled

    class Block:
        def forward(self, value):
            return value + 1

    class Model:
        def __init__(self):
            self.blocks = [Block() for _ in range(wan.WAN13B_BLOCK_COUNT)]

    monkeypatch.setattr(wan, "_torch", lambda: FakeTorch)
    monkeypatch.setenv("DISTILLSCHEDULE_WAN13B_COMPILE_CACHE_ROOT", str(tmp_path))
    generator = Model()
    receipt = wan.install_generator_nograd_block_compile(
        generator,
        placement=wan.ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT,
        role="generator",
    )
    assert receipt["enabled"] is True
    assert receipt["all_blocks_wrapped"] is True
    assert FakeTorch.compile_calls == wan.WAN13B_BLOCK_COUNT
    assert [block.forward(1) for block in generator.blocks] == [2] * 30
    assert receipt["compiled_dispatch_count"] == 30
    assert receipt["eager_dispatch_count"] == 0
    assert all(
        value is not None for value in receipt["first_dispatch_wall_seconds_by_block"]
    )

    FakeTorch.grad_enabled = True
    assert [block.forward(2) for block in generator.blocks] == [3] * 30
    assert receipt["compiled_dispatch_count"] == 30
    assert receipt["eager_dispatch_count"] == 30

    critic = Model()
    critic_receipt = wan.install_generator_nograd_block_compile(
        critic,
        placement=wan.ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT,
        role="critic",
    )
    assert critic_receipt["enabled"] is False
    assert critic_receipt["disabled_reason"] == "GENERATOR_ONLY"


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
