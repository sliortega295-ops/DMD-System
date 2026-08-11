#!/usr/bin/env python3
"""Build strict GAN fairness and compute receipts from a real run."""
from __future__ import annotations

import argparse
import json

from distillschedule.runtime.gan_execution_receipts import (
    write_gan_execution_receipts,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("baseline", "pipeline"), required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--trace-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-streamed-block-count", type=int, default=40)
    args = parser.parse_args()
    try:
        paths = write_gan_execution_receipts(
            family=args.family,
            run_root=args.run_root,
            trace_bundle=args.trace_bundle,
            output_dir=args.output_dir,
            expected_streamed_block_count=args.expected_streamed_block_count,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "PASS", **paths}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
