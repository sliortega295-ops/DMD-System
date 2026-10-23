"""Reuse checked campaign FDs; otherwise keep the original independent lock path."""
import hashlib
import importlib.util
import os
from pathlib import Path

ROOT = Path('/home/jiangzhikun/yongyan_liu/Helios-H100')
CHECKER = ROOT / 'repos/DMD-System-h100-resident-baselines-20261014/scripts/distillschedule/verify_inherited_gpu_locks.py'
CHECKER_SHA = 'b7233f29f577dbd8c53198da905778f4b9f670722b5a428cd1e19faab706ff7b'


def open_campaign_lock(path):
    mode = os.environ.get('DISTILLSCHEDULE_INHERITED_GPU_LOCKS', '0')
    if mode == '0':
        return Path(path).open('a')
    if mode != '1':
        raise ValueError('invalid campaign lock inheritance mode')
    paths = {ROOT / 'locks/helios_asset_transfer.lock': 7,
             ROOT / 'locks/helios_world8_gpu.lock': 9,
             Path('/tmp/distillschedule_gpu_node.lock'): 8}
    if Path(path) not in paths:
        raise ValueError('unregistered campaign lock path')
    if hashlib.sha256(CHECKER.read_bytes()).hexdigest() != CHECKER_SHA:
        raise ValueError('committed lock verifier source SHA mismatch')
    spec = importlib.util.spec_from_file_location('lease_check', CHECKER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.verify_locks(ROOT, 9, 8, 7)
    # Duplicate the same open-file description; closing this duplicate cannot
    # unlock or close the campaign parent's remaining descriptors.
    return os.fdopen(os.dup(paths[Path(path)]), 'a')
