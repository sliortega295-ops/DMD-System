"""Fail-closed contracts for the No-GAN Generator/Student/Teacher pool.

N2 is a capacity search, not a license to change the workload.  This module is
kept free of Helios imports so launchers can reject a malformed candidate or a
wrong negative-prompt asset before constructing a 14B model.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    import torch


TARGET_EXECUTION_GRAPH = "target_no_gan_gst_pool"
CANDIDATE_SCHEMA = "distillschedule.g_offload_candidate.v1"
NEGATIVE_PROMPT_SCHEMA = "distillschedule.helios_negative_prompt_asset.v1"
BLOCK_COUNT = 40
GENERATOR_WORLD_SIZE = 8
GENERATOR_MICROBATCHES = 2
GLOBAL_BATCH = GENERATOR_WORLD_SIZE * GENERATOR_MICROBATCHES
LOGICAL_STEP = 15

OFFICIAL_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn "
    "hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused "
    "fingers, still picture, messy background, three legs, many people in the "
    "background, walking backwards"
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_hash(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return hashlib.sha256(canonical_json(unsigned)).hexdigest()


@dataclass(frozen=True)
class GeneratorOffloadCandidate:
    path: Path
    run_id: str
    contract_digest: str
    streamed_blocks: tuple[int, ...]
    manifest_sha256: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_contract_digest: str | None = None,
    ) -> "GeneratorOffloadCandidate":
        path = Path(path).resolve()
        value = json.loads(path.read_text())
        if value.get("schema_version") != CANDIDATE_SCHEMA:
            raise ValueError("unsupported Generator offload candidate schema")
        if value.get("execution_graph") != TARGET_EXECUTION_GRAPH:
            raise ValueError("candidate is not the target G/S/T execution graph")
        if value.get("student_residency") != "resident":
            raise ValueError("Student residency changed")
        if value.get("teacher_residency") != "resident":
            raise ValueError("Teacher residency changed")
        if value.get("activation_policy_mutable") is not False:
            raise ValueError("N2 may not mutate the activation policy")
        if value.get("loss_optimizer_batch_trace_version_ema_mutable") is not False:
            raise ValueError("N2 may not mutate semantic training state")
        if value.get("historical_candidate_substitution_allowed") is not False:
            raise ValueError("historical Route-A bounds cannot substitute for N2")

        run_id = str(value.get("run_id", "")).strip()
        if not run_id:
            raise ValueError("candidate run_id is missing")
        contract_digest = str(value.get("contract_digest", ""))
        if len(contract_digest) != 64:
            raise ValueError("candidate contract digest is not SHA-256")
        if (
            expected_contract_digest is not None
            and contract_digest != expected_contract_digest
        ):
            raise ValueError("candidate evidence contract drift")

        raw_blocks = value.get("generator_offload_blocks")
        if not isinstance(raw_blocks, list):
            raise ValueError("candidate Generator block set is missing")
        blocks = tuple(int(block) for block in raw_blocks)
        if len(blocks) != int(value.get("generator_offload_block_count", -1)):
            raise ValueError("candidate Generator block count disagrees")
        if len(set(blocks)) != len(blocks):
            raise ValueError("candidate Generator block set contains duplicates")
        if any(block < 0 or block >= BLOCK_COUNT for block in blocks):
            raise ValueError("candidate Generator block index is out of range")
        return cls(
            path=path,
            run_id=run_id,
            contract_digest=contract_digest,
            streamed_blocks=blocks,
            manifest_sha256=sha256_file(path),
        )


@dataclass(frozen=True)
class NegativePromptAsset:
    root: Path
    manifest: dict[str, Any]
    embedding: "torch.Tensor"

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        verify_payload_sha256: bool = True,
    ) -> "NegativePromptAsset":
        import torch

        from .operation_trace import tensor_sha256

        root = Path(root).resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("schema_version") != NEGATIVE_PROMPT_SCHEMA:
            raise ValueError("unsupported negative-prompt asset schema")
        if manifest.get("manifest_hash") != _manifest_hash(
            manifest, "manifest_hash"
        ):
            raise ValueError("negative-prompt manifest hash mismatch")
        if manifest.get("prompt") != OFFICIAL_NEGATIVE_PROMPT:
            raise ValueError("negative-prompt text changed")
        if manifest.get("prompt_sha256") != hashlib.sha256(
            OFFICIAL_NEGATIVE_PROMPT.encode("utf-8")
        ).hexdigest():
            raise ValueError("negative-prompt text digest mismatch")

        payload = manifest.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("negative-prompt payload descriptor is missing")
        payload_path = root / str(payload.get("path"))
        if payload_path.stat().st_size != int(payload.get("file_bytes", -1)):
            raise ValueError("negative-prompt payload file size changed")
        if verify_payload_sha256 and sha256_file(payload_path) != payload.get(
            "file_sha256"
        ):
            raise ValueError("negative-prompt payload file SHA-256 changed")
        embedding = torch.load(payload_path, map_location="cpu", weights_only=True)
        if not isinstance(embedding, torch.Tensor):
            raise TypeError("negative-prompt payload is not a Tensor")
        if list(embedding.shape) != [1, 512, 4096]:
            raise ValueError("negative-prompt embedding shape changed")
        if embedding.dtype != torch.bfloat16:
            raise ValueError("negative-prompt embedding must be BF16")
        if tensor_sha256(embedding) != payload.get("tensor_sha256"):
            raise ValueError("negative-prompt logical tensor identity changed")
        return cls(root=root, manifest=manifest, embedding=embedding)


def write_negative_prompt_asset(
    root: str | Path,
    *,
    embedding: "torch.Tensor",
    model_root: str | Path,
    tokenizer_subfolder: str = "tokenizer",
    text_encoder_subfolder: str = "text_encoder",
) -> NegativePromptAsset:
    """Write an immutable official negative-prompt tensor plus provenance."""

    import torch

    from .operation_trace import tensor_sha256

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    payload_path = root / "negative_prompt_embedding.pt"
    if manifest_path.exists() or payload_path.exists():
        raise FileExistsError(f"refusing to overwrite negative-prompt asset {root}")
    value = embedding.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    if list(value.shape) != [1, 512, 4096]:
        raise ValueError("encoded negative prompt violates [1,512,4096]")
    temporary = payload_path.with_suffix(f".pt.{__import__('os').getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(payload_path)
    manifest: dict[str, Any] = {
        "schema_version": NEGATIVE_PROMPT_SCHEMA,
        "prompt": OFFICIAL_NEGATIVE_PROMPT,
        "prompt_sha256": hashlib.sha256(
            OFFICIAL_NEGATIVE_PROMPT.encode("utf-8")
        ).hexdigest(),
        "encoding_contract": {
            "function": "helios.utils.utils_base.encode_prompt",
            "max_sequence_length": 512,
            "num_videos_per_prompt": 1,
            "caption_dropout_p": 0.0,
            "dtype": "torch.bfloat16",
        },
        "source": {
            "model_root": str(Path(model_root).resolve()),
            "tokenizer_subfolder": tokenizer_subfolder,
            "text_encoder_subfolder": text_encoder_subfolder,
        },
        "payload": {
            "path": payload_path.name,
            "file_bytes": payload_path.stat().st_size,
            "file_sha256": sha256_file(payload_path),
            "tensor_sha256": tensor_sha256(value),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "stride": list(value.stride()),
        },
    }
    manifest["manifest_hash"] = _manifest_hash(manifest, "manifest_hash")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return NegativePromptAsset.load(root)


def source_identity(source_id: int) -> dict[str, int]:
    """Map global sample slot to its rank-qualified G wave."""

    if not 0 <= source_id < GLOBAL_BATCH:
        raise ValueError(f"source_id must be in [0,{GLOBAL_BATCH})")
    return {
        "source_id": int(source_id),
        "generator_rank": int(source_id % GENERATOR_WORLD_SIZE),
        "generator_wave": int(source_id // GENERATOR_WORLD_SIZE),
        "generator_world_size": GENERATOR_WORLD_SIZE,
        "generator_microbatches": GENERATOR_MICROBATCHES,
        "global_batch": GLOBAL_BATCH,
    }
