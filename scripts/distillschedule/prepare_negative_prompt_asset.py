#!/usr/bin/env python3
"""Encode and freeze the official Helios negative prompt once."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from runtime.gst_pool_contract import (
    OFFICIAL_NEGATIVE_PROMPT,
    write_negative_prompt_asset,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--model-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--device", default="cuda:0")
    return result


def main() -> int:
    args = parser().parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer, UMT5EncoderModel
    from helios.utils.utils_base import encode_prompt

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_root), subfolder="tokenizer", local_files_only=True
    )
    encoder = UMT5EncoderModel.from_pretrained(
        str(args.model_root),
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True,
    )
    encoder.eval()
    with torch.no_grad():
        embedding, _ = encode_prompt(
            tokenizer=tokenizer,
            text_encoder=encoder,
            prompt=OFFICIAL_NEGATIVE_PROMPT,
            device=device,
            dtype=torch.bfloat16,
        )
    asset = write_negative_prompt_asset(
        args.output, embedding=embedding, model_root=args.model_root
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "root": str(asset.root),
                "manifest_hash": asset.manifest["manifest_hash"],
                "tensor_sha256": asset.manifest["payload"]["tensor_sha256"],
                "shape": list(asset.embedding.shape),
                "dtype": str(asset.embedding.dtype),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
