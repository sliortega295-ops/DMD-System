#!/usr/bin/env python3
"""Exec one torchrun child under a rank-qualified Nsight Systems capture."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        raise ValueError("rank Nsight wrapper requires a command")
    rank = int(os.environ["RANK"])
    args.profile_dir.mkdir(parents=True, exist_ok=True)
    output = args.profile_dir / f"rank_{rank:02d}"
    command = [
        "nsys",
        "profile",
        "--force-overwrite=true",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--cpuctxsw=none",
        "--output",
        str(output),
        *args.command,
    ]
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
