#!/usr/bin/env python3
"""Execute one native or DistillSchedule MiniMax-H3 matched DMD cycle."""
from __future__ import annotations

import argparse
import hashlib
import os
import random
from pathlib import Path

import numpy as np
import torch


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--mode", choices=("native", "distillschedule"), required=True)
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--receipt-dir", type=Path, required=True)
    value.add_argument("--seed", type=int, default=20260817)
    value.add_argument("--dmd-system-head", required=True)
    value.add_argument("--upstream-commit", required=True)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    rank = int(os.environ.get("RANK", "0"))
    value = int(seed) + rank
    random.seed(value)
    np.random.seed(value % (2**32 - 1))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def main() -> int:
    args = parser().parse_args()
    from lightx2v_train.data import build_data, prepare_data
    from lightx2v_train.model_zoo import build_model
    from lightx2v_train.runtime import cleanup_distributed, init_distributed, load_config, setup_logger
    from runtime.lightx2v_minimax_matched import create_minimax_matched_trainer

    config = load_config(args.config)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    init_distributed(config)
    setup_logger(config)
    seed_everything(args.seed)

    prompt_cache = Path(os.environ["LIGHTX2V_MINIMAX_PROMPT_CACHE"]).resolve()
    metadata_path = prompt_cache / "metadata.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"MiniMax prompt cache lacks metadata.jsonl: {metadata_path}")
    metadata = {
        "benchmark_seed": int(args.seed),
        "dmd_system_head": args.dmd_system_head,
        "upstream_commit": args.upstream_commit,
        "config_sha256": sha256_file(args.config),
        "prompt_metadata_sha256": sha256_file(metadata_path),
        "model_path": str(Path(os.environ["LIGHTX2V_MINIMAX_MODEL_PATH"]).resolve()),
        "prompt_cache": str(prompt_cache),
        "attention_backend": os.environ.get("LIGHTX2V_MINIMAX_ATTENTION_BACKEND", "_flash_3_hub"),
        "student_activation_offload": os.environ.get("LIGHTX2V_MINIMAX_ACTIVATION_OFFLOAD", "0"),
        "checkpoint_segment": os.environ.get("LIGHTX2V_MINIMAX_CHECKPOINT_SEGMENT", "0"),
        "activation_offload_disabled_for_segmented_canary": os.environ.get(
            "LIGHTX2V_MINIMAX_ACTIVATION_OFFLOAD", "0"
        ) == "0",
    }

    trainer = None
    try:
        prepare_data(config)
        model = build_model(config)
        model.load_components()
        dataloader_train = build_data(config, train_or_val="train")
        trainer_class = create_minimax_matched_trainer(
            args.mode,
            receipt_dir=args.receipt_dir,
            metadata=metadata,
        )
        trainer = trainer_class(config)
        trainer.set_model(model)
        trainer.set_data(dataloader_train, None)
        trainer.train()
        trainer.write_benchmark_receipt()
        return 0
    finally:
        close_attribution = getattr(trainer, "close_memory_attribution", None)
        if close_attribution is not None:
            close_attribution()
        cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
