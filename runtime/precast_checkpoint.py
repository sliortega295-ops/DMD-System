"""Build and load validated, pre-cast sharded safetensors checkpoints.

The conversion is intentionally offline. Runtime loading can then instantiate a
model on the meta device and bind the already-cast CPU tensors without first
allocating an FP32 model or copying an FP32 state dict into it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from safetensors.torch import load_file, save_file

PRECAST_SCHEMA = "distillschedule.precast_safetensors.v1"
PRECAST_MANIFEST_NAME = "distillschedule_precast_manifest.json"
DEFAULT_INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
DEFAULT_CONFIG_NAME = "config.json"
HELIOS_KEEP_IN_FP32_MODULES = (
    "time_embedder",
    "scale_shift_table",
    "norm1",
    "norm2",
    "norm3",
    "history_key_scale",
)

_DTYPE_BY_NAME = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _sha256(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read checkpoint JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"checkpoint JSON must contain an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _safe_relative_file(root: Path, filename: str) -> Path:
    relative = Path(filename)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name != filename:
        raise RuntimeError(f"checkpoint shard must be a plain filename: {filename!r}")
    return root / relative


def _load_index(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    index = _read_json(path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"checkpoint index has no weight_map: {path}")
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in weight_map.items()
    ):
        raise RuntimeError(f"checkpoint index weight_map is malformed: {path}")
    return index, weight_map


def _target_tensor_dtype(
    name: str,
    tensor: torch.Tensor,
    *,
    target_dtype: torch.dtype,
    keep_in_fp32_modules: tuple[str, ...],
) -> torch.dtype:
    if not tensor.is_floating_point():
        return tensor.dtype
    if any(pattern in name for pattern in keep_in_fp32_modules):
        return torch.float32
    return target_dtype


def _dtype_counter_add(counter: Counter[str], tensor: torch.Tensor) -> None:
    counter[dtype_name(tensor.dtype)] += 1


def convert_pretrained_safetensors(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    target_dtype: torch.dtype = torch.bfloat16,
    keep_in_fp32_modules: Iterable[str] = HELIOS_KEEP_IN_FP32_MODULES,
    hash_output_shards: bool = True,
) -> dict[str, Any]:
    """Convert one indexed checkpoint shard at a time and publish atomically."""

    if target_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported target dtype: {target_dtype}")
    keep_patterns = tuple(keep_in_fp32_modules)
    if not keep_patterns or any(not pattern for pattern in keep_patterns):
        raise ValueError("keep_in_fp32_modules must contain non-empty patterns")
    if len(set(keep_patterns)) != len(keep_patterns):
        raise ValueError("keep_in_fp32_modules must not contain duplicates")

    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    config_path = source / DEFAULT_CONFIG_NAME
    index_path = source / DEFAULT_INDEX_NAME
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    index, weight_map = _load_index(index_path)

    staging = output.parent / f".{output.name}.incomplete-{os.getpid()}-{time.time_ns()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)

    source_dtype_counts: Counter[str] = Counter()
    output_dtype_counts: Counter[str] = Counter()
    source_tensor_bytes = 0
    output_tensor_bytes = 0
    shard_records: list[dict[str, Any]] = []
    keys_by_shard: dict[str, set[str]] = {}
    for key, filename in weight_map.items():
        keys_by_shard.setdefault(filename, set()).add(key)

    try:
        shutil.copy2(config_path, staging / DEFAULT_CONFIG_NAME)
        for filename in sorted(keys_by_shard):
            source_shard = _safe_relative_file(source, filename)
            if not source_shard.is_file():
                raise FileNotFoundError(source_shard)
            source_state = load_file(str(source_shard), device="cpu")
            expected_keys = keys_by_shard[filename]
            observed_keys = set(source_state)
            if observed_keys != expected_keys:
                missing = sorted(expected_keys - observed_keys)
                extra = sorted(observed_keys - expected_keys)
                raise RuntimeError(
                    f"source shard/index mismatch for {filename}: missing={missing}, extra={extra}"
                )

            output_state: dict[str, torch.Tensor] = {}
            for name, tensor in source_state.items():
                _dtype_counter_add(source_dtype_counts, tensor)
                source_tensor_bytes += int(tensor.numel() * tensor.element_size())
                destination_dtype = _target_tensor_dtype(
                    name,
                    tensor,
                    target_dtype=target_dtype,
                    keep_in_fp32_modules=keep_patterns,
                )
                converted = tensor.to(dtype=destination_dtype).contiguous()
                output_state[name] = converted
                _dtype_counter_add(output_dtype_counts, converted)
                output_tensor_bytes += int(converted.numel() * converted.element_size())

            output_shard = staging / filename
            save_file(
                output_state,
                str(output_shard),
                metadata={
                    "format": "pt",
                    "distillschedule_schema": PRECAST_SCHEMA,
                    "target_dtype": dtype_name(target_dtype),
                },
            )
            shard_records.append(
                {
                    "file": filename,
                    "size_bytes": output_shard.stat().st_size,
                    "sha256": _sha256(output_shard) if hash_output_shards else None,
                    "tensor_count": len(output_state),
                }
            )
            del output_state
            del source_state

        output_index = dict(index)
        output_index["metadata"] = dict(index.get("metadata") or {})
        output_index["metadata"]["total_size"] = output_tensor_bytes
        output_index_path = staging / DEFAULT_INDEX_NAME
        _write_json(output_index_path, output_index)

        manifest = {
            "schema_version": PRECAST_SCHEMA,
            "source": {
                "path": str(source),
                "config_sha256": _sha256(config_path),
                "index_sha256": _sha256(index_path),
                "tensor_bytes": source_tensor_bytes,
                "dtype_tensor_counts": dict(sorted(source_dtype_counts.items())),
                "shards": [
                    {
                        "file": filename,
                        "size_bytes": _safe_relative_file(source, filename).stat().st_size,
                    }
                    for filename in sorted(keys_by_shard)
                ],
            },
            "artifact": {
                "target_dtype": dtype_name(target_dtype),
                "keep_in_fp32_modules": list(keep_patterns),
                "config_file": DEFAULT_CONFIG_NAME,
                "config_sha256": _sha256(staging / DEFAULT_CONFIG_NAME),
                "index_file": DEFAULT_INDEX_NAME,
                "index_sha256": _sha256(output_index_path),
                "tensor_count": len(weight_map),
                "tensor_bytes": output_tensor_bytes,
                "dtype_tensor_counts": dict(sorted(output_dtype_counts.items())),
                "shards": shard_records,
            },
        }
        _write_json(staging / PRECAST_MANIFEST_NAME, manifest)
        os.replace(staging, output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_precast_checkpoint(
    checkpoint_dir: str | Path,
    *,
    expected_target_dtype: torch.dtype,
    expected_keep_in_fp32_modules: Iterable[str],
    expected_source_dir: str | Path | None = None,
    verify_shard_hashes: bool = False,
) -> dict[str, Any]:
    """Validate the cheap runtime contract, optionally including full hashes."""

    root = Path(checkpoint_dir).resolve()
    manifest_path = root / PRECAST_MANIFEST_NAME
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != PRECAST_SCHEMA:
        raise RuntimeError(f"unsupported pre-cast checkpoint schema: {manifest_path}")
    source = manifest.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("path"), str):
        raise RuntimeError(f"pre-cast checkpoint has no source identity: {manifest_path}")
    if expected_source_dir is not None and Path(source["path"]).resolve() != Path(expected_source_dir).resolve():
        raise RuntimeError(
            "pre-cast source identity mismatch: "
            f"expected {Path(expected_source_dir).resolve()}, observed {Path(source['path']).resolve()}"
        )
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise RuntimeError(f"pre-cast checkpoint has no artifact record: {manifest_path}")
    if artifact.get("target_dtype") != dtype_name(expected_target_dtype):
        raise RuntimeError(
            "pre-cast target dtype mismatch: "
            f"expected {dtype_name(expected_target_dtype)}, observed {artifact.get('target_dtype')}"
        )
    expected_patterns = list(expected_keep_in_fp32_modules)
    if artifact.get("keep_in_fp32_modules") != expected_patterns:
        raise RuntimeError(
            "pre-cast FP32 retention policy mismatch: "
            f"expected {expected_patterns}, observed {artifact.get('keep_in_fp32_modules')}"
        )

    config_filename = artifact.get("config_file")
    index_filename = artifact.get("index_file")
    if not isinstance(config_filename, str) or not isinstance(index_filename, str):
        raise RuntimeError("pre-cast manifest is missing config/index filenames")
    config_path = _safe_relative_file(root, config_filename)
    index_path = _safe_relative_file(root, index_filename)
    if _sha256(config_path) != artifact.get("config_sha256"):
        raise RuntimeError("pre-cast config hash mismatch")
    if _sha256(index_path) != artifact.get("index_sha256"):
        raise RuntimeError("pre-cast index hash mismatch")
    index, weight_map = _load_index(index_path)
    if len(weight_map) != artifact.get("tensor_count"):
        raise RuntimeError("pre-cast tensor count mismatch")
    metadata = index.get("metadata") or {}
    if metadata.get("total_size") != artifact.get("tensor_bytes"):
        raise RuntimeError("pre-cast tensor byte count mismatch")

    shard_records = artifact.get("shards")
    if not isinstance(shard_records, list):
        raise RuntimeError("pre-cast shard inventory is missing")
    record_by_file: dict[str, dict[str, Any]] = {}
    for record in shard_records:
        if not isinstance(record, dict) or not isinstance(record.get("file"), str):
            raise RuntimeError("pre-cast shard record is malformed")
        filename = record["file"]
        if filename in record_by_file:
            raise RuntimeError(f"duplicate pre-cast shard record: {filename}")
        record_by_file[filename] = record
    indexed_files = set(weight_map.values())
    if set(record_by_file) != indexed_files:
        raise RuntimeError("pre-cast shard inventory does not match the index")
    for filename, record in record_by_file.items():
        shard_path = _safe_relative_file(root, filename)
        if shard_path.stat().st_size != record.get("size_bytes"):
            raise RuntimeError(f"pre-cast shard size mismatch: {filename}")
        if verify_shard_hashes:
            expected_hash = record.get("sha256")
            if not expected_hash:
                raise RuntimeError(f"pre-cast shard has no recorded hash: {filename}")
            if _sha256(shard_path) != expected_hash:
                raise RuntimeError(f"pre-cast shard hash mismatch: {filename}")
    return manifest


def _load_shards(root: Path, filenames: list[str], *, max_workers: int) -> dict[str, torch.Tensor]:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")

    def load(filename: str) -> tuple[str, dict[str, torch.Tensor]]:
        path = _safe_relative_file(root, filename)
        return filename, load_file(str(path), device="cpu")

    shards: dict[str, dict[str, torch.Tensor]] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(filenames))) as executor:
        futures = {executor.submit(load, filename): filename for filename in filenames}
        for future in as_completed(futures):
            filename, state = future.result()
            shards[filename] = state

    combined: dict[str, torch.Tensor] = {}
    for filename in filenames:
        for name, tensor in shards[filename].items():
            if name in combined:
                raise RuntimeError(f"duplicate tensor across pre-cast shards: {name}")
            combined[name] = tensor
    return combined


def load_precast_checkpoint_into_meta_model(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
    *,
    expected_target_dtype: torch.dtype,
    expected_keep_in_fp32_modules: Iterable[str],
    expected_source_dir: str | Path | None = None,
    max_workers: int = 8,
    key_remap: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Strictly bind a validated pre-cast checkpoint to a meta model."""

    keep_patterns = tuple(expected_keep_in_fp32_modules)
    manifest = validate_precast_checkpoint(
        checkpoint_dir,
        expected_target_dtype=expected_target_dtype,
        expected_keep_in_fp32_modules=keep_patterns,
        expected_source_dir=expected_source_dir,
    )
    root = Path(checkpoint_dir).resolve()
    artifact = manifest["artifact"]
    _, weight_map = _load_index(root / artifact["index_file"])
    filenames = sorted(set(weight_map.values()))
    source_state = _load_shards(root, filenames, max_workers=max_workers)
    if set(source_state) != set(weight_map):
        missing = sorted(set(weight_map) - set(source_state))
        extra = sorted(set(source_state) - set(weight_map))
        raise RuntimeError(f"pre-cast shard/index key mismatch: missing={missing}, extra={extra}")

    state: dict[str, torch.Tensor] = {}
    for source_name, tensor in source_state.items():
        destination_name = key_remap(source_name) if key_remap is not None else source_name
        if destination_name in state:
            raise RuntimeError(f"pre-cast key remap collision: {destination_name}")
        state[destination_name] = tensor

    expected_state = model.state_dict()
    if set(state) != set(expected_state):
        missing = sorted(set(expected_state) - set(state))
        extra = sorted(set(state) - set(expected_state))
        raise RuntimeError(f"pre-cast model key mismatch: missing={missing}, extra={extra}")
    dtype_counts: Counter[str] = Counter()
    for name, tensor in state.items():
        expected_tensor = expected_state[name]
        if tuple(tensor.shape) != tuple(expected_tensor.shape):
            raise RuntimeError(
                f"pre-cast shape mismatch for {name}: expected {tuple(expected_tensor.shape)}, observed {tuple(tensor.shape)}"
            )
        expected_dtype = _target_tensor_dtype(
            name,
            expected_tensor,
            target_dtype=expected_target_dtype,
            keep_in_fp32_modules=keep_patterns,
        )
        if tensor.dtype != expected_dtype:
            raise RuntimeError(
                f"pre-cast dtype mismatch for {name}: expected {dtype_name(expected_dtype)}, observed {dtype_name(tensor.dtype)}"
            )
        _dtype_counter_add(dtype_counts, tensor)

    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict pre-cast load unexpectedly returned {incompatible}")
    remaining_meta = [
        name
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
        if tensor.is_meta
    ]
    if remaining_meta:
        raise RuntimeError(f"pre-cast load left meta tensors: {remaining_meta}")
    return {
        "tensor_count": len(state),
        "tensor_bytes": artifact["tensor_bytes"],
        "dtype_tensor_counts": dict(sorted(dtype_counts.items())),
        "shard_count": len(filenames),
    }


def verify_source_equivalence(
    source_dir: str | Path,
    checkpoint_dir: str | Path,
    *,
    chunk_elements: int = 16 * 1024 * 1024,
) -> dict[str, int]:
    """Verify bitwise equality to the declared source after the dtype policy."""

    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    root = Path(checkpoint_dir).resolve()
    manifest = _read_json(root / PRECAST_MANIFEST_NAME)
    artifact = manifest.get("artifact") or {}
    target_dtype_name = artifact.get("target_dtype")
    if target_dtype_name not in _DTYPE_BY_NAME:
        raise RuntimeError(f"unsupported manifest target dtype: {target_dtype_name}")
    target_dtype = _DTYPE_BY_NAME[target_dtype_name]
    keep_patterns = tuple(artifact.get("keep_in_fp32_modules") or ())
    validate_precast_checkpoint(
        root,
        expected_target_dtype=target_dtype,
        expected_keep_in_fp32_modules=keep_patterns,
        verify_shard_hashes=True,
    )

    source = Path(source_dir).resolve()
    source_record = manifest.get("source") or {}
    source_config = source / DEFAULT_CONFIG_NAME
    source_index_path = source / DEFAULT_INDEX_NAME
    if _sha256(source_config) != source_record.get("config_sha256"):
        raise RuntimeError("source config no longer matches the conversion manifest")
    if _sha256(source_index_path) != source_record.get("index_sha256"):
        raise RuntimeError("source index no longer matches the conversion manifest")
    _, source_weight_map = _load_index(source_index_path)
    _, output_weight_map = _load_index(root / artifact["index_file"])
    if source_weight_map != output_weight_map:
        raise RuntimeError("source and pre-cast weight maps differ")

    tensor_count = 0
    element_count = 0
    for filename in sorted(set(source_weight_map.values())):
        source_state = load_file(str(_safe_relative_file(source, filename)), device="cpu")
        output_state = load_file(str(_safe_relative_file(root, filename)), device="cpu")
        if set(source_state) != set(output_state):
            raise RuntimeError(f"source/output shard key mismatch: {filename}")
        for name, source_tensor in source_state.items():
            output_tensor = output_state[name]
            expected_dtype = _target_tensor_dtype(
                name,
                source_tensor,
                target_dtype=target_dtype,
                keep_in_fp32_modules=keep_patterns,
            )
            if output_tensor.dtype != expected_dtype or output_tensor.shape != source_tensor.shape:
                raise RuntimeError(f"source/output metadata mismatch for {name}")
            source_flat = source_tensor.reshape(-1)
            output_flat = output_tensor.reshape(-1)
            for start in range(0, source_flat.numel(), chunk_elements):
                end = min(start + chunk_elements, source_flat.numel())
                expected = source_flat[start:end].to(dtype=expected_dtype)
                if not torch.equal(output_flat[start:end], expected):
                    raise RuntimeError(f"pre-cast value mismatch for {name}")
            tensor_count += 1
            element_count += source_tensor.numel()
        del output_state
        del source_state
    return {"tensor_count": tensor_count, "element_count": element_count}
