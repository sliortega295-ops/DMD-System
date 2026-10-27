"""Source-bound official optimization bundles; discovery is metadata only.

Historical measurements identify the imported implementation, not the speed or
GPU qualification of the public continuous-training integration. Resolving a
profile neither installs it nor bypasses its owner's runtime admission.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
from distillschedule.workloads.best_known.registry import file_sha256


SCHEMA = "distillschedule.official_optimization_profile.v1"
_SOURCE_SCHEMA = "distillschedule.official_optimization_sources.v1"
_TRAINING_ROOT = Path(__file__).resolve().parent
_SOURCE_LOCK = _TRAINING_ROOT / "official_optimizations" / "source_lock.json"
_RUNTIME_FILES = {
    'dmd2-sd15-resident-tree-v1': ('official_optimizations/dmd2.py',),
    'self-forcing-wan-fsdp2-r6-v1': ('self_forcing_fsdp.py',
        'official_optimizations/bound_source.py', 'official_optimizations/self_forcing.py'),
}

_PROFILES = {
    "dmd2-sd15-resident-tree-v1": {
        "model": "sd15",
        "algorithm": "official-dmd2-gan",
        "backend": "native-role-ddp-bf16-v1",
        "contract": {
            "world_size": 16,
            "per_rank_batch": 1,
            "global_batch": 16,
            "image_height": 512,
            "image_width": 512,
            "updates": {"generator": 1, "critic": 10, "ema": 0},
            "precision": {
                "trainable_parameters": "float32",
                "gradient_compute": "bfloat16",
                "generator_nograd": "float32",
                "teacher": "bfloat16",
                "vae": "float16",
                "text_encoder": "float32",
                "classifier_heads": "float32",
            },
        },
        "mechanisms": [
            "native_role_ddp_ready_buckets",
            "fused_adamw",
            "fake_epsilon_activation_retention",
            "packed_real_fake_classifier_trunk",
            "critic_update_scoped_cast_cache",
        ],
        "environment": {"NCCL_ALGO": "allreduce:Tree", "NCCL_MIN_CTAS": "16"},
        "origin": {
            "run_id": "official_dmd2_min16_tree_r10",
            "upstream_commit": "8d8fa55633d47cfb81bbc7a892e7248f9518763f",
            "launch_manifest_sha256": "8b88a2b518448534219e46ca567044a92697e879ff2319051b8b046734342acc",
            "source_manifest_sha256": "772fa2ebff6a18cf4db3cd1ff8bc36e0c9856adb03434cc862c8931fcd47ac80",
            "numerical_status": "REVIEW",
            "native_cycle2_seconds": 3.4750529816374183,
            "historical_reference": {
                "kind": "fixed_historical_ddp",
                "native_cycle2_seconds": 6.113556639291346,
            },
            "input_scope": "FULL_SIZE_SYNTHETIC_INITIALIZATION_AND_INPUTS; NOT_GENERATION_QUALITY",
        },
    },
    "self-forcing-wan-fsdp2-r6-v1": {
        "model": "wan",
        "algorithm": "official-self-forcing",
        "backend": "native-self-forcing-fsdp2-bf16-v1",
        "contract": {
            "world_size": 16,
            "per_rank_batch": 1,
            "global_batch": 16,
            "height": 368,
            "width": 640,
            "output_frames": 81,
            "latent_frames": 21,
            "updates": {"generator": 1, "critic": 5, "ema": 0},
            "precision": {"master_parameters": "float32", "compute": "bfloat16"},
            "layout": "full16",
            "phase_retention_max_bytes": 3221225472,
        },
        "mechanisms": [
            "text_kv_retention",
            "host_kv_indices",
            "dense_fa2_b1",
            "nograd_rms",
            "phase_weight_retention",
            "nograd_modulation",
            "native_layer_norm",
            "nograd_fp64_rope",
            "rollout_context_hoist",
            "critic_fa2_checkpoint_reuse",
            "grad_fp64_rope",
            "critic_ffn_down_reuse",
            "grad_rms_fusion",
            "grad_pointwise_fusion",
            "critic_output_projection_reuse",
            "teacher_cfg_block_pair",
        ],
        "environment": {"NCCL_CROSS_NIC": "1"},
        "origin": {
            "run_id": "official_sf_teacher_pair_r6",
            "upstream_commit": "33593df3e81fa3ec10239271dd2c100facac6de1",
            "launch_manifest_sha256": "11d46be1942299bb903ee57a874087d94779c61c3d324217a1c299893fcf9c0c",
            "source_manifest_sha256": "baf7d9331b64bcf31cd14b0f6e7d2921d471cbbff9dd529e1fb6fce5138061bc",
            "numerical_status": "REVIEW",
            "native_cycle2_seconds": 30.959485488943756,
            "historical_reference": {
                "kind": "frozen_fsdp2",
                "native_cycle2_seconds": 50.739418774843216,
            },
            "input_scope": "WAN_BASE_WEIGHTS; NO_OFFICIAL_ODE_INIT_OR_GENERATION_QUALITY_VALIDATION",
        },
    },
}


def resolve(profile_id: str, *, model: str | None = None) -> dict:
    """Return an independent metadata copy; no model or optional imports."""
    value = _PROFILES.get(profile_id)
    if value is None:
        raise ValueError(f"unknown official optimization profile: {profile_id!r}")
    if model is not None and model != value["model"]:
        raise ValueError(f"official optimization profile {profile_id!r} rejects model {model!r}")
    return dict(
        schema_version=SCHEMA,
        id=profile_id,
        **deepcopy(value),
        runtime_sources={name: file_sha256(_TRAINING_ROOT / name)
                         for name in _RUNTIME_FILES[profile_id]},
        source_integration="PUBLIC_NATIVE_OWNER_PROGRAM_ENTRY; EXPLICIT_SOURCE_AND_DEVICE_ADMISSION",
        integration_gpu_validation="NOT_RUN",
        performance_guarantee=False,
        continuous_lifecycle_validation="NOT_RUN",
        automatic_enable=False,
    )


def describe() -> dict:
    """List the two measured-source bundles, separately from W1-W6."""
    return {name: resolve(name) for name in _PROFILES}


class _NormalizeHelperImports(ast.NodeTransformer):
    def __init__(self, helper_names):
        self.helper_names = helper_names

    def visit_ImportFrom(self, node):
        # Source relocation changes only sibling-helper imports to relative.
        if node.module in self.helper_names:
            node.level = 0
        return node


def _definition_sha256(node, helper_names):
    normalized = _NormalizeHelperImports(helper_names).visit(node)
    # The immutable import lock was produced with Python 3.12's AST schema.
    # 3.10/3.11 omit the empty PEP-695 field, even for identical source bytes.
    # Restore only that absent empty field; never normalize executable syntax
    # or change the locked file hash to make a different definition match.
    for item in ast.walk(normalized):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and "type_params" not in item._fields:
            item._fields = (*item._fields, "type_params")
            item.type_params = []
    return hashlib.sha256(ast.dump(normalized, include_attributes=False).encode()).hexdigest()


def verify_sources(profile_id: str) -> list[dict]:
    """Check packaged bytes and original normalized AST identities, without imports.

    Only the reviewed lock and packaged helper sources are needed. Original
    experiment directories are deliberately not a runtime dependency. This
    verifies provenance; it does not establish numerical or GPU equivalence.
    """
    if profile_id not in _PROFILES:
        raise ValueError(f"unknown official optimization profile: {profile_id!r}")
    lock = json.loads(_SOURCE_LOCK.read_text(encoding="utf-8"))
    if lock.get("schema_version") != _SOURCE_SCHEMA or not isinstance(lock.get("files"), list):
        raise ValueError("official optimization source lock schema mismatch")
    files = lock["files"]
    helper_names = {PurePosixPath(row["target"]).stem for row in files}
    selected = [row for row in files if row.get("profile") == profile_id]
    if not selected:
        raise ValueError(f"official optimization source lock missing profile: {profile_id!r}")
    result, seen = [], set()
    for row in selected:
        relative = PurePosixPath(row["target"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("official_optimizations", "_native"):
            raise ValueError("official optimization source lock target outside packaged helper directory")
        if row["target"] in seen:
            raise ValueError("official optimization source lock duplicate target")
        seen.add(row["target"])
        path = (_TRAINING_ROOT / relative).resolve()
        if not path.is_relative_to((_TRAINING_ROOT / "official_optimizations" / "_native").resolve()):
            raise ValueError("official optimization source target escapes packaged helper directory")
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != row["integrated_sha256"]:
            raise ValueError(f"official optimization source SHA256 mismatch: {row['target']}")
        definitions = {
            node.name: _definition_sha256(node, helper_names)
            for node in ast.parse(source, filename=str(relative)).body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in row["definitions"]
        }
        if not row["definitions"] or definitions != row["definitions"]:
            raise ValueError(f"official optimization source definition mismatch: {row['target']}")
        result.append(dict(
            target=row["target"],
            origin=row["origin"],
            origin_file_sha256=row["origin_file_sha256"],
            integrated_sha256=row["integrated_sha256"],
            definitions=deepcopy(definitions),
            definitions_match_origin=True,
        ))
    return result
