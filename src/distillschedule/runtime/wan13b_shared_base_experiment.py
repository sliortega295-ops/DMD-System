"""Explicit manual-only frozen-storage experiment; never a planner/default.

Reuse the ORIGINAL two-cycle worker, timers, loss, engines and state exporters.
Storage aliasing is allowed only before any Critic CUDA Graph capture. A cold
controller's address snapshot is rebound once; all subsequent checks remain.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess

from distillschedule.core.contracts import require

SCHEMA = 'distillschedule.wan_shared_frozen_manual_experiment.v1'
ARMS = {'private-retain4': (False, (26, 27, 28, 29)),
        'shared-retain6': (True, (24, 25, 26, 27, 28, 29))}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_packet(packet, *, head, task, assets, strategy, model_root):
    require(set(packet) == {'schema_version', 'source_head', 'arm', 'run_id',
                           'share_frozen_base', 'cycles', 'task_sha256',
                           'assets_sha256', 'strategy_sha256', 'checkpoint_identity'},
            'unexpected experiment fields', path='wan.shared_base')
    require(packet['schema_version'] == SCHEMA and packet['source_head'] == head
            and packet['arm'] in ARMS and packet['cycles'] == 2,
            'experiment schema/source/arm/cycles mismatch', path='wan.shared_base')
    enabled, blocks = ARMS[packet['arm']]
    require(type(packet['share_frozen_base']) is bool and packet['share_frozen_base'] == enabled,
            'arm must bind actual storage behavior', path='wan.shared_base')
    require(isinstance(packet['run_id'], str) and len(packet['run_id']) > 8,
            'explicit run identity required', path='wan.shared_base')
    require(packet['task_sha256'] == sha(task) and packet['assets_sha256'] == sha(assets)
            and packet['strategy_sha256'] == sha(strategy)
            and packet['checkpoint_identity'] == str(Path(model_root).resolve()),
            'experiment assets/strategy/checkpoint mismatch', path='wan.shared_base')
    document = json.loads(Path(strategy).read_text())
    require(document['placement'] == 'resident' and tuple(document['retained_critic_blocks']) == blocks
            and document.get('rollout_graph_pool_policy', 'isolated.v1') == 'isolated.v1',
            'unreviewed sharing/retention/pool combination', path='wan.shared_base')


def cold_controller(model):
    controller = getattr(model, '_distillschedule_critic_grad_cuda_graph', None)
    require(controller is not None and not controller.cache and all(
        controller.receipt.get(k) == 0 for k in
        ('capture_entries', 'replay_calls', 'backward_replay_calls', 'reference_calls')),
        'frozen alias requires a completely cold Critic graph controller', path='wan.shared_base')
    require(controller.parameter_identity() == controller.pointer_contract,
            'pre-alias parameter identity already changed', path='wan.shared_base')
    return controller


def session_class(original, packet, packet_sha256):
    class ExplicitSharedSession(original):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._frozen_alias = None
            import torch
            receipt = dict(schema_version=SCHEMA, packet_sha256=packet_sha256,
                           arm=packet['arm'], source_head=packet['source_head'],
                           run_id=packet['run_id'], enabled=packet['share_frozen_base'],
                           planner_used=False, merged_weights=False,
                           alias_before_any_capture=True, audit_checks=0)
            if packet['share_frozen_base']:
                from .shared_frozen_base import share_frozen_parameters
                controller = cold_controller(self.c.module)
                before = torch.cuda.memory_allocated(self.device)
                self._frozen_alias = share_frozen_parameters(
                    self.g.module, self.c.module,
                    reference_identity=packet['checkpoint_identity'],
                    candidate_identity=str(Path(kwargs['model_root']).resolve()))
                require(not self._frozen_alias.receipt['unmatched_frozen_candidate_parameters'],
                        'incomplete equal-checkpoint frozen coverage', path='wan.shared_base')
                require(not controller.cache and controller.receipt['capture_entries'] == 0,
                        'capture entered during setup alias', path='wan.shared_base')
                # No graph/autograd saved tensors exist. This is the only
                # permitted address change, BEFORE the ORIGINAL first capture.
                controller.pointer_contract = controller.parameter_identity()
                receipt.update(storage=self._frozen_alias.receipt,
                    setup_allocated_before_bytes=before,
                    setup_allocated_after_bytes=torch.cuda.memory_allocated(self.device),
                    cold_critic_pointer_snapshot_rebound_once=True)
            self._alias_receipt = receipt
            self.build['critic']['metadata']['shared_frozen_base_experiment'] = receipt
            self.build['generator']['metadata']['shared_frozen_base_experiment'] = receipt

        def _audit_alias(self):
            if self._frozen_alias is not None:
                self._alias_receipt['last_audit'] = self._frozen_alias.audit()
            self._alias_receipt['audit_checks'] += 1

        def versions(self):
            # The original worker calls this only outside its native timer.
            if hasattr(self, '_alias_receipt'):
                self._audit_alias()
            return super().versions()

        def prepare(self, **kwargs):
            self._audit_alias()  # before original barrier / START, not hot path
            return super().prepare(**kwargs)

    return ExplicitSharedSession


@contextmanager
def install_explicit_session(worker, packet, packet_sha256):
    original = worker.OriginalWanSession
    worker.OriginalWanSession = session_class(original, packet, packet_sha256)
    try:
        yield
    finally:
        worker.OriginalWanSession = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('task', 'assets', 'implementation', 'output', 'packet'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--packet-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    require(args.execute and sha(args.packet) == args.packet_sha256,
            'explicit execution and immutable packet required', path='wan.shared_base')
    from distillschedule.strategies import wan_worker as worker
    from distillschedule.strategies.wan_strategy import WanStrategy
    from distillschedule.strategies.wan_task import OriginalWanTask
    assets = worker.OriginalWanAssets.from_dict(json.loads(args.assets.read_text()))
    packet = json.loads(args.packet.read_text())
    head = subprocess.check_output(['git', '-C', str(worker.repository_root()),
                                    'rev-parse', 'HEAD'], text=True).strip()
    validate_packet(packet, head=head, task=args.task, assets=args.assets,
                    strategy=args.implementation, model_root=assets.model_root)
    task = OriginalWanTask.from_dict(json.loads(args.task.read_text()))
    strategy = WanStrategy.from_dict(json.loads(args.implementation.read_text()))
    with install_explicit_session(worker, packet, args.packet_sha256):
        result = worker.run_worker(task, assets, output=args.output, mode='manual',
                                   strategy=strategy, cycles=2, record_memory_highwater=True)
    print(json.dumps(dict(status=result['status'], rank=result['rank'],
                          arm=packet['arm'], source_head=head)), flush=True)


if __name__ == '__main__':
    main()
