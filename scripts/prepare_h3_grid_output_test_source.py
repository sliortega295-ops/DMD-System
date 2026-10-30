#!/usr/bin/env python3
"""Reproduce the existing Grid inference source identity in a test checkout.

This is the R44 source snapshot, not a new training patch. The public upstream
checkout remains the provenance; the one historical benchmark change is named
and its exact before/after bytes are checked. No model payload is fetched.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source):
    source = Path(source).resolve()
    fixture = ROOT / 'tests/fixtures/h3_grid_output_source.json'
    review = json.loads(fixture.read_text())
    patch = fixture.with_suffix('.patch')
    assert _sha(patch) == review['patch_sha256'], 'reviewed patch changed'
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == review['upstream_commit'], 'wrong original LightX2V revision'
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', 'HEAD'], check=True)
    trainer = source / review['trainer']
    assert _sha(trainer) == review['upstream_trainer_sha256'], 'original trainer bytes changed'
    original = ast.parse(trainer.read_text())
    subprocess.run(['git', '-C', str(source), 'apply', '--check', str(patch)], check=True)
    subprocess.run(['git', '-C', str(source), 'apply', str(patch)], check=True)
    assert _sha(trainer) == review['reviewed_trainer_sha256'], 'R44 trainer bytes differ'

    # Apart from the named historical forward_loss opt-in and its os import,
    # every definition (including all inference methods) must remain original.
    current = ast.parse(trainer.read_text())
    os_import = ast.dump(ast.parse('import os').body[0], include_attributes=False)
    current.body = [node for node in current.body
                    if ast.dump(node, include_attributes=False) != os_import]
    old_class = next(node for node in original.body if isinstance(node, ast.ClassDef))
    new_class = next(node for node in current.body if isinstance(node, ast.ClassDef))
    old_loss = next(node for node in old_class.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'forward_loss')
    new_class.body = [deepcopy(old_loss) if isinstance(node, ast.FunctionDef)
                      and node.name == 'forward_loss' else node for node in new_class.body]
    assert ast.dump(current, include_attributes=False) == ast.dump(original, include_attributes=False)

    from distillschedule.training.h3_grid_inference import SOURCE_FILES
    assert SOURCE_FILES[review['trainer']] == review['reviewed_trainer_sha256']
    assert all(_sha(source / path) == expected for path, expected in SOURCE_FILES.items())
    return {'status': 'PASS_EXISTING_GRID_SOURCE_IDENTITY', 'source': str(source),
            'upstream_commit': head, 'files': SOURCE_FILES, 'review': str(fixture),
            'gpu_execution': 'NOT_RUN'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='isolated clean checkout, modified in place')
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.source), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
