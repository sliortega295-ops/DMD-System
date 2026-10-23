"""Versioned opt-in bindings to existing native implementations, not plans.

Presets contain no saved run, costs, asset paths or historical initial state.
Their typed owners still enforce source, task, immutable assets, capacity and
numerical gates. All six bindings remain bounded research integrations.
"""
from __future__ import annotations

from copy import deepcopy


SCHEMA = "distillschedule.native_training_preset.v1"
_PRESETS = {
    "W1": dict(id="wan_ar.incumbent21a.v1", numerical_status="REVIEW",
        source_origin="21a0993db64137489e120dd6614c48a1bf1a6002",
        implementation_paths=["strategies/wan_strategy.py", "search/wan_candidate_bundle.py"],
        mechanisms=["resident", "modulation_chunk2048", "lora_nograd_epilogue",
                    "grad_lora_fused_epilogue", "modulated_norm_fusion", "generator_rollout_cuda_graph",
                    "generator_nograd_lora_cast_cache", "critic_grad_cuda_graph"],
        runtime="distillschedule.strategies.wan_worker:run_worker",
        execution_evidence="native bundle installation, version checks and actual graph/cast counters"),
    "W2": dict(id="helios_nogan.safe_ed8.v1", numerical_status="REVIEW",
        source_origin="ed8e1ccbee44e039c73b303d33179964795ea71b",
        implementation_paths=["search/helios_implementation.py", "search/helios_safe_bundle.py"],
        mechanisms=["node_hsdp", "modulation_chunk512", "rollout_window5", "exact_modulated_norm",
                    "elide_explicit_rollout_sync", "lora_b_gemm_epilogue", "lora_checkpoint_replay_scope"],
        runtime="distillschedule.search.execution:run_plan",
        execution_evidence="HeliosSafeBundle.bind_effective and per-cycle executed_effects call deltas"),
    "W3": dict(id="helios_gan.fa3_down40_nomerge.v1", numerical_status="REVIEW",
        source_origin="ba52a2446b7f090cd31af9504679e68770d32603",
        implementation_paths=["search/helios_implementation.py", "search/helios_gan_bundle.py"],
        mechanisms=["world16_fsdp", "modulation_chunk2048", "persistent_unshard", "rollout_window5",
                    "pinned_sm80_fa3", "frozen_down40_replay", "separate_gan_head_fsdp",
                    "lora_b_gemm_epilogue", "lora_checkpoint_replay_scope", "merged_lora_off"],
        runtime="distillschedule.search.execution:run_plan",
        execution_evidence="HeliosGANBundle.bind_effective/effect_snapshot, FA3/down40/head counters and merge_count=0"),
    "W4": dict(id="self_forcing.c51.v1", numerical_status="REVIEW",
        source_origin="aa30527db5d048e2ab23222eef587cb7cfe909f6",
        implementation_paths=["search/livewan_implementation.py", "runtime/livewan_c51_bundle.py"],
        mechanisms=["pipeline_8g8score", "generator_modulation", "generator_gate", "row_major_residual",
                    "c51_full_v1"],
        runtime="distillschedule.search.execution:run_plan",
        execution_evidence="C51 exact effective node plans, full source-template contract and native bundle counters"),
    "W5": dict(id="h3_fullseq.cpu_staged_trim.v1", numerical_status="FAIL_UNRESOLVED",
        source_origin="cfecbb4188e1f3820451657f744aa2c823e24da4",
        implementation_paths=["search/h3_implementation.py", "backends/training/minimax_h3/trainer.py"],
        mechanisms=["world16_fsdp2", "grid1000", "fa3_cpu_staged_trim", "prefetch_depth2", "lora_epilogue_n128"],
        runtime="distillschedule.search.execution:run_plan",
        execution_evidence="H3Implementation.receipt_binding and original sampled numerical/capacity gates; FAIL retained"),
    "W6": dict(id="h3_ar.zero3_ap.v1", numerical_status="REVIEW",
        source_origin="e88d21c168ce54be99134cecd195c961319cb45a",
        implementation_paths=["strategies/h3_ar_task.py", "strategies/h3_ar_assets.py", "strategies/h3_ar_worker.py"],
        mechanisms=["zero3", "selected_cache_prefetch", "cpu_staged_attention_replay", "unit_prefetch",
                    "pointwise_fusion", "lora_inplace", "swiglu_fusion", "rms_fusion", "cutlass_lora", "overlap_comm"],
        runtime="distillschedule.strategies.h3_ar_cli:main",
        execution_evidence="native implementation/helper hashes, input/source/checkpoint admission and live capacity gate"),
}


def describe_preset(workload: str, preset_id: str) -> dict:
    """Metadata only: no Torch, model, asset access or numerical promotion."""
    value = _PRESETS.get(workload)
    if value is None or value["id"] != preset_id:
        raise ValueError(f"native training: unknown or mismatched preset {preset_id!r} for {workload}")
    return dict(schema_version=SCHEMA, workload=workload, **deepcopy(value),
        opt_in="execution.mode=manual; not a stable/recommended default",
        source_archived=True, runtime_binding="EXISTING_NATIVE_IMPLEMENTATION",
        continuous_training=False, cycles=2, gpu_execution="NOT_RUN",
        mechanism_execution="NOT_RUN; requires actual native worker counters",
        source_binding="current source HEAD and owner hashes; origin is provenance, not a reusable plan",
        reference_alternative="select an explicit supported reference recipe; never substituted automatically")


def preset_implementation(workload: str, preset_id: str):
    """Construct the exact registered type; do not duplicate any lowerer."""
    describe_preset(workload, preset_id)
    if workload == "W1":
        from distillschedule.strategies.wan_strategy import WanStrategy
        from distillschedule.search.wan_candidate_bundle import INCUMBENT
        return WanStrategy("resident", 2048, True, 5, INCUMBENT)
    if workload in ("W2", "W3"):
        from distillschedule.search.helios_implementation import HeliosImplementation
        if workload == "W2":
            from distillschedule.search.helios_safe_bundle import HeliosSafeBundle
            return HeliosImplementation("no-gan", "node-hsdp", 512, 5, False, HeliosSafeBundle.safe())
        from distillschedule.search.helios_gan_bundle import HeliosGANBundle
        return HeliosImplementation("gan", "world16-fsdp", 2048, 5, True, HeliosGANBundle.manual())
    if workload == "W4":
        from distillschedule.search.livewan_implementation import LiveWanImplementation
        return LiveWanImplementation("pipeline", True, True, True, bundle="c51_full_v1")
    if workload == "W5":
        from distillschedule.search.h3_implementation import H3Implementation
        return H3Implementation(cache_policy="cpu_staged_trim")
    from distillschedule.strategies.h3_ar_task import H3ARImplementation
    return H3ARImplementation("zero3-ap")
