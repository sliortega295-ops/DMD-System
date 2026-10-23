#!/usr/bin/env python3
"""CPU-only admission of the exact remotely transferred H3 asset inventory.

Default is index validation, never a fabricated complete-asset receipt. Full
hashing is remote-only, holds the existing transfer/timing exclusion lock, and
does not import torch or initialize CUDA. All outputs use a new directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import time
from h100_campaign_lock_adapter import open_campaign_lock

EXPECTED_COUNTS = {"model": 16, "cache": 97, "table": 2, "fa3": 7}
EXPECTED_TOTAL_BYTES = 86_533_781_312
GROUP_DIR = {"model": "model/transformer", "cache": "prompt_cache", "table": "grid_table", "fa3": "fa3"}
REMOTE_ROOT = Path("/home/jiangzhikun/yongyan_liu/Helios-H100")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_index(path: Path, expected_sha: str, root: Path) -> dict:
    if sha256_file(path) != expected_sha:
        raise ValueError("frozen transfer index SHA mismatch")
    plan = json.loads(path.read_text())
    if plan.get("status") != "SOURCE_METADATA_AND_HASH_INDEX_PASS_TRANSFER_NOT_RUN":
        raise ValueError("source transfer index is not admitted")
    if plan.get("diffusers_head") != "9284607295a09f759aadd65ed08f48b35feea6d9":
        raise ValueError("Diffusers source identity mismatch")
    if plan.get("lightx2v_head") != "d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be":
        raise ValueError("LightX2V source identity mismatch")
    rows = plan["files"]
    if Counter(row["group"] for row in rows) != Counter(EXPECTED_COUNTS):
        raise ValueError("incomplete or extra asset group inventory")
    if len({row["destination"] for row in rows}) != len(rows):
        raise ValueError("duplicate destination")
    total = sum(row["bytes"] for row in rows)
    if total != EXPECTED_TOTAL_BYTES or total != plan["total_bytes"]:
        raise ValueError("total byte inventory mismatch")
    for row in rows:
        destination = Path(row["destination"])
        group_root = root / GROUP_DIR[row["group"]]
        if not destination.is_absolute() or not destination.is_relative_to(group_root):
            raise ValueError("asset path escapes its assigned group")
        if ".." in destination.parts or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("unsafe path or malformed content identity")
        if type(row["bytes"]) is not int or row["bytes"] <= 0 or row.get("aliases"):
            raise ValueError("invalid bytes or unaudited aliases")
    return plan


def verify_file(row: dict, root: Path) -> dict:
    path = Path(row["destination"])
    # A symlink must not redirect a verified asset into another project.
    if path.resolve() != path or not path.resolve().is_relative_to(root):
        raise ValueError(f"symlink or escaping asset: {path}")
    before = path.stat()
    if before.st_size != row["bytes"]:
        raise ValueError(f"asset length mismatch: {path}")
    observed = sha256_file(path)
    after = path.stat()
    signature = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if signature(before) != signature(after):
        raise ValueError(f"asset changed while hashing: {path}")
    if observed != row["sha256"]:
        raise ValueError(f"asset SHA mismatch: {path}")
    return {"group": row["group"], "path": str(path.relative_to(root)),
            "bytes": after.st_size, "sha256": observed, "status": "PASS_ACTUAL_FULL_HASH"}


def validate_metadata(root: Path, rows: list[dict]) -> dict:
    config = json.loads((root / "model/transformer/config.json").read_text())
    if config.get("_class_name") != "MiniMaxH3Transformer3DModel" or "num_refiner_layers" not in config:
        raise ValueError("not a converted MiniMax-H3 transformer config")
    index = json.loads((root / "model/transformer/diffusion_pytorch_model.safetensors.index.json").read_text())
    actual_shards = {Path(row["path"]).name for row in rows if row["group"] == "model" and row["path"].endswith(".safetensors")}
    if len(actual_shards) != 14 or set(index["weight_map"].values()) != actual_shards:
        raise ValueError("model shard index does not cover the verified 14 shards")
    cache_lines = (root / "prompt_cache/metadata.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in cache_lines if line.strip()]
    if len(records) != 96 or any(not isinstance(row, dict) for row in records):
        raise ValueError("immutable cache must have exactly 96 metadata records")
    conditions = {Path(row["path"]).name for row in rows if row["group"] == "cache" and row["path"].endswith(".pt")}
    if conditions != {f"condition_{i:08d}.pt" for i in range(96)}:
        raise ValueError("cache condition inventory is incomplete")
    grid = json.loads((root / "grid_table/adaln_grid1000.json").read_text())
    expected = {"schema_version": "h3_a100.adaln_grid.v1", "binary_file": "adaln_grid1000.bf16.bin",
                "grid_size": 1000, "num_rollout_entries": 4, "num_entries": 1004,
                "num_blocks": 50, "hidden_size": 5376, "rows_per_entry": 6,
                "modulation_chunks": 6, "dtype": "torch.bfloat16",
                "renoise_sigma_min": 0.02, "renoise_sigma_max": 0.98,
                "video_shift": 6.0, "audio_shift": 3.0, "num_inference_steps": 4,
                "binary_bytes": 19_431_014_400}
    for key, value in expected.items():
        if grid.get(key) != value:
            raise ValueError(f"Grid table contract mismatch: {key}")
    bits = grid.get("timestep_bits", [])
    if len(bits) != 1004 or any(not isinstance(row, list) or len(row) != 2 for row in bits):
        raise ValueError("Grid timestep identity inventory mismatch")
    if len({tuple(row) for row in bits}) != 1004 or grid.get("dropped_adaln_parameter_numel", 0) <= 12_000_000_000:
        raise ValueError("Grid timetable or original AdaLN parameter inventory mismatch")
    return {"model_layout": "PASS", "model_shards": 14, "cache_samples": 96,
            "grid_layout": "PASS", "Grid_sigma_semantics": "approximate_discrete_grid1000",
            "FA3_runtime_import": "NOT_RUN", "GPU_model_load": "NOT_RUN"}


def group_identity(rows: list[dict], group: str) -> str:
    records = [{key: row[key] for key in ("path", "bytes", "sha256")}
               for row in rows if row["group"] == group]
    # Stable content-inventory identity, not a mutable absolute path identity.
    encoded = json.dumps(sorted(records, key=lambda row: row["path"]), sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-hash", action="store_true")
    args = parser.parse_args()
    root = args.assets_root.resolve()
    plan = verify_index(args.index, args.index_sha256, root)
    output = args.output_dir.resolve()
    if args.full_hash and (not root.is_relative_to(REMOTE_ROOT / "assets") or
                           not output.is_relative_to(REMOTE_ROOT / "research_artifacts")):
        raise ValueError("actual full asset gate is REMOTE persistent storage only")
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "PASS_INDEX_ONLY_FULL_ASSETS_NOT_RUN", "index_sha256": args.index_sha256,
              "file_count": len(plan["files"]), "total_bytes": plan["total_bytes"],
              "GPU": "NOT_RUN", "training": "NOT_RUN", "comparison_boundary":
              "H100_ONLY_COMMON_START_NOT_A100_INITIAL_STATE_REPLAY",
              "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    started = time.monotonic()
    try:
        if args.full_hash:
            with open_campaign_lock(REMOTE_ROOT / "locks/helios_asset_transfer.lock") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                missing = [row["destination"] for row in plan["files"]
                           if not Path(row["destination"]).is_file()]
                if missing:
                    result.update(status="NOT_READY_INCOMPLETE_ASSETS", missing_files=missing)
                else:
                    rows = []
                    with (output / "full_hash_rows.jsonl").open("x") as journal:
                        for source in plan["files"]:
                            row = verify_file(source, root); rows.append(row)
                            journal.write(json.dumps(row, sort_keys=True) + "\n"); journal.flush()
                    checks = validate_metadata(root, rows)
                    result.update(status="PASS_FULL_HASH_H3_ASSETS", checks=checks,
                                  model_root=str(root / "model"), prompt_cache_root=str(root / "prompt_cache"),
                                  grid_manifest=str(root / "grid_table/adaln_grid1000.json"),
                                  FA3_root=str(root / "fa3"),
                                  model_identity_sha256=group_identity(rows, "model"),
                                  prompt_cache_identity_sha256=group_identity(rows, "cache"),
                                  grid_identity_sha256=group_identity(rows, "table"),
                                  FA3_identity_sha256=group_identity(rows, "fa3"),
                                  identity_definition="SHA256 compact sorted content inventory of relative path/bytes/SHA256",
                                  source_inventory_sha256=plan["source_model_inventory_sha256"])
    except BlockingIOError as error:
        result.update(status="NOT_RUN_ACTIVE_TRANSFER_OR_TIMING", error_type=type(error).__name__, error=str(error))
    except Exception as error:
        result.update(status="FAIL_ASSET_GATE", error_type=type(error).__name__, error=str(error))
    result["elapsed_seconds"] = time.monotonic() - started
    (output / "asset_receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
