#!/usr/bin/env python3
"""Render one node-qualified Accelerate config from the checked-in template."""

from __future__ import annotations

import argparse
import json

from distillschedule.runtime.launch_config import render_accelerate_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--deepspeed-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-machines", type=int, required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--machine-rank", type=int, required=True)
    args = parser.parse_args()
    contract = render_accelerate_config(
        template_path=args.template,
        deepspeed_config_path=args.deepspeed_config,
        output_path=args.output,
        num_machines=args.num_machines,
        num_processes=args.num_processes,
        machine_rank=args.machine_rank,
    )
    print(json.dumps(contract.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
