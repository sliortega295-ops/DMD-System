#!/usr/bin/env python3
"""Fail CI when a required real CPU qualification silently skips or disappears."""
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def check(path, required=()):
    root = ET.parse(path).getroot()
    cases = list(root.iter('testcase'))
    if not cases:
        raise ValueError('CPU receipt contains no test cases')
    for case in cases:
        if any(case.find(kind) is not None for kind in ('failure', 'error', 'skipped')):
            raise ValueError('CPU qualification failed/skipped: ' + str(case.get('name')))
    names = {case.get('name') for case in cases}
    missing = set(required) - names
    if missing:
        raise ValueError('required CPU proofs missing: ' + ', '.join(sorted(missing)))
    return {'status': 'PASS_REAL_CPU_RECEIPT', 'tests': len(cases), 'skips': 0,
            'required': sorted(set(required)), 'gpu_execution': 'NOT_RUN'}


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--require', action='append', default=[])
    args = parser.parse_args(argv)
    print(json.dumps(check(args.receipt, args.require), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
