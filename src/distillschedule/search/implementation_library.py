"""Inventory existing kernels without importing Torch, Triton or model code.

An inventoried source is not a new correctness receipt. Shape/backend gates in
the implementation remain authoritative, and candidate parameters need fresh
validation. This library intentionally does not generate GPU code.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from distillschedule.workloads.best_known.registry import file_sha256, repository_root


# Exact existing entry modules, not copied kernels or inferred CUDA support.
# Only a public, already-supported tuning surface is listed as tunable.
ENTRIES = (
    ("modulation_chunks", "src/distillschedule/runtime/helios_modulation_chunks.py", ("wan", "helios"),
     {"chunk_tokens": [512, 1024, 2048]}, "MATCH_NATIVE_CHUNK_ROUNDING_AND_GRADIENT_CONTRACT"),
    ("wan_lora_epilogue", "src/distillschedule/runtime/wan13b_lora_epilogue.py", ("wan",),
     {}, "BF16_BASE_FP32_LORA_CONTIGUOUS_NOGRAD_RN_MUL_ADD"),
    ("helios_modulated_norm", "src/distillschedule/runtime/helios_modulated_norm_kernel.py", ("helios",),
     {}, "SOURCE_GATED_DTYPE_SHAPE_AND_BACKWARD"),
    ("helios_residual", "src/distillschedule/runtime/helios_residual_kernel.py", ("helios",),
     {}, "SOURCE_GATED_RESIDUAL_STORAGE_AND_DTYPE"),
    ("livewan_score_rmsnorm", "src/distillschedule/runtime/livewan_score_rmsnorm.py", ("livewan",),
     {}, "SCORE_POOL_ONLY_BF16_SOURCE_HASH_GATED"),
    ("livewan_generator_modulation", "src/distillschedule/runtime/livewan_generator_modulation.py", ("livewan",),
     {}, "GENERATOR_NOGRAD_ONLY_B1_WIDTH1536_FP32_SCALE_SHIFT"),
    ("livewan_generator_gate", "src/distillschedule/runtime/livewan_generator_gate.py", ("livewan",),
     {}, "SOURCE_GATED_GENERATOR_GATE"),
    ("livewan_c51_fused_rms", "src/distillschedule/runtime/livewan_generator_fused_rms.py", ("livewan",),
     {}, "EXACT_C51_BUNDLE_GENERATOR_NOGRAD_BF16_ROUND_FP32_REDUCTION"),
    ("livewan_c51_readonly_splitkv", "src/distillschedule/runtime/livewan_generator_splitkv.py", ("livewan",),
     {}, "EXACT_C51_BUNDLE_SOURCE_PINNED_FA2_READONLY_CURRENT_KV"),
    ("livewan_c51_score_fa3", "src/distillschedule/runtime/livewan_score_fa3.py", ("livewan",),
     {}, "EXACT_C51_BUNDLE_FIXED_FA3_SOURCE_OPTIONS_AND_SAC_REPLAY"),
    ("h3_block_pointwise", "src/distillschedule/backends/training/minimax_h3/fused_block_pointwise.py", ("h3",),
     {}, "GRID_CONTRACT_FIXED_REPLAY_SCOPE_REQUIRED"),
    ("h3_swiglu", "src/distillschedule/backends/training/minimax_h3/fused_swiglu.py", ("h3",),
     {}, "SOURCE_GATED_JOINT_VIDEO_AUDIO_GRADIENTS"),
    ("h3_qk_rmsnorm_rotary", "src/distillschedule/backends/training/minimax_h3/fused_qk_rmsnorm_rotary.py", ("h3",),
     {}, "SOURCE_GATED_LAYOUT_DTYPE_ROTARY_CONVENTION"),
    ("h3_lora_epilogue", "src/distillschedule/backends/training/minimax_h3/triton_lora_epilogue.py", ("h3",),
     {}, "SOURCE_GATED_LORA_PRECISION_AND_ROUNDING"),
)


def implementation_inventory(root: Path | None = None) -> dict[str, Any]:
    root = root or repository_root()
    rows = []
    for name, source, families, knobs, contract in ENTRIES:
        path = root / source
        if not path.is_file():
            rows.append({"id": name, "source": source, "status": "MISSING_SOURCE",
                         "gpu_validation": "NOT_RUN", "search_admission": "REJECTED"})
            continue
        tree = ast.parse(path.read_text())
        rows.append({"id": name, "source": source, "source_sha256": file_sha256(path),
                     "families": list(families), "supported_parameters": knobs,
                     "contract": contract, "status": "SOURCE_INVENTORIED",
                     "functions": [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))],
                     "gpu_validation": "NOT_RUN_ON_PLANNER_HEAD",
                     "cost_calibration": "REQUIRED", "composition_validation": "REQUIRED",
                     "search_admission": "REQUIRES_WORKLOAD_ADAPTER_AND_CALIBRATION"})
    return {"schema_version": "distillschedule.implementation_inventory.v1", "entries": rows,
            "gpu_execution": "NOT_RUN", "new_kernel_generation": False}
