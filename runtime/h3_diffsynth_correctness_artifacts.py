"""Opt-in real-output evidence for native W6 correctness runs only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


class CorrectnessArtifacts:
    def __init__(self, directory, *, enabled=False, mode, max_prediction_bytes=8 << 30):
        if enabled and mode != 'correctness':
            raise ValueError('full parity exports are correctness-only')
        self.enabled = bool(enabled)
        self.active = self.enabled
        self.directory = Path(directory)
        self.maximum = int(max_prediction_bytes)
        self.prediction_bytes = 0
        self.rows = []
        if self.enabled:
            self.directory.mkdir(exist_ok=False)

    def _metadata(self, value):
        import torch
        if isinstance(value, torch.Tensor):
            cpu = value.detach().cpu().contiguous()
            digest = hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
            return {'tensor': True, 'shape': list(value.shape), 'stride': list(value.stride()),
                    'dtype': str(value.dtype), 'device_type': value.device.type, 'sha256': digest}
        if isinstance(value, dict):
            return {key: self._metadata(item) for key, item in sorted(value.items())}
        if isinstance(value, (tuple, list)):
            return [self._metadata(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f'unsupported native correctness input type: {type(value).__name__}')

    def before_prediction(self, inputs, *, identity):
        if not self.active:
            return None
        return {'identity': identity, 'input_consumption': self._metadata(inputs)}

    def after_prediction(self, record, outputs):
        if not self.active:
            return
        import torch
        from torch.utils._pytree import tree_flatten, tree_unflatten
        leaves, spec = tree_flatten(outputs)
        if not all(isinstance(value, torch.Tensor) for value in leaves):
            raise TypeError('native prediction must contain only tensors')
        count = sum(value.numel() * value.element_size() for value in leaves)
        if self.prediction_bytes + count > self.maximum:
            raise RuntimeError('bounded correctness prediction export exceeded')
        cpu = tree_unflatten([value.detach().cpu().contiguous() for value in leaves], spec)
        path = self.directory / f'prediction_{len(self.rows):04d}.pt'
        with path.open('xb') as handle:
            torch.save(cpu, handle)
        self.prediction_bytes += count
        record.update(path=str(path), sha256=file_sha(path), bytes=path.stat().st_size,
                      tensor_logical_bytes=count)
        self.rows.append(record)

    def export_trainable_state(self, adapter, optimizers, *, name='final_trainable_state.pt'):
        if not self.enabled:
            return None
        import torch
        def local_cpu(value):
            if hasattr(value, 'to_local'):
                value = value.to_local()
            return value.detach().cpu().contiguous()
        state = {'parameters': {name: local_cpu(value) for name, value in adapter.named_parameters() if '.adapters.' in name},
                 'Adam': {role: [{key: local_cpu(value) for key, value in row.items() if isinstance(value, torch.Tensor)}
                                 for row in optimizer.state.values()] for role, optimizer in optimizers.items()}}
        if Path(name).name != name:
            raise ValueError('state export name must be local to its evidence directory')
        path = self.directory / name
        with path.open('xb') as handle:
            torch.save(state, handle)
        return {'path': str(path), 'sha256': file_sha(path), 'bytes': path.stat().st_size,
                'scope': 'after both cycles and outside timer; local FSDP shards only'}

    def receipt(self):
        return {'enabled': self.enabled, 'active': self.active, 'mode': 'CORRECTNESS_ONLY_EXCLUDED_FROM_PERFORMANCE',
                'predictions': self.rows, 'prediction_logical_bytes': self.prediction_bytes}


def compare_artifacts(reference, candidate):
    """CPU comparison of actual predictions and corresponding local shards."""
    import torch
    from torch.utils._pytree import tree_flatten
    from distillschedule.runtime.p0_parity import tensor_parity_metrics
    results, mismatches = [], []
    if reference['identities'] != candidate['identities']:
        mismatches.append('native_input_or_initial_state_identity')
    rrows, crows = reference['predictions'], candidate['predictions']
    if len(rrows) != len(crows):
        mismatches.append('prediction_count')
    def checked_load(row):
        if file_sha(row['path']) != row['sha256']:
            raise RuntimeError('parity payload SHA mismatch')
        return torch.load(row['path'], map_location='cpu', weights_only=True)
    def metric(a, b, **identity):
        if a.numel() == b.numel() == 0 and a.shape == b.shape and a.dtype == b.dtype:
            # Empty local shards contain no values; global reconstruction is
            # unnecessary here because all ranks must independently pass.
            return {**identity, 'status': 'PASS', 'empty_local_shard': True,
                    'bitwise_equal': True}
        return {**identity, 'bitwise_equal': torch.equal(a, b), **tensor_parity_metrics(a, b)}
    for index, (a, b) in enumerate(zip(rrows, crows)):
        if a['identity'] != b['identity'] or a['input_consumption'] != b['input_consumption']:
            mismatches.append(f'actual_input_consumption_{index}')
        x, spec_x = tree_flatten(checked_load(a)); y, spec_y = tree_flatten(checked_load(b))
        if spec_x != spec_y:
            mismatches.append(f'prediction_tree_{index}')
            continue
        results.extend(metric(v, w, kind='prediction', index=index, leaf=leaf)
                       for leaf, (v, w) in enumerate(zip(x, y)))
    x, y = checked_load(reference['trainable_state']), checked_load(candidate['trainable_state'])
    if set(x['parameters']) != set(y['parameters']) or set(x['Adam']) != set(y['Adam']):
        raise RuntimeError('native trainable/Adam structure mismatch')
    results.extend(metric(value, y['parameters'][name], kind='parameter', name=name)
                   for name, value in x['parameters'].items())
    for role, ordered in x['Adam'].items():
        observed = y['Adam'][role]
        if len(ordered) != len(observed):
            raise RuntimeError('native Adam ordered parameter list mismatch')
        for index, (a, b) in enumerate(zip(ordered, observed)):
            if set(a) != set(b):
                raise RuntimeError('native Adam state field mismatch')
            results.extend(metric(value, b[field], kind='Adam', role=role, ordered_state_index=index, field=field)
                           for field, value in a.items())
    return {'status': 'PASS' if not mismatches and all(row['status'] == 'PASS' for row in results) else 'FAIL',
            'identity_mismatches': mismatches, 'comparisons': len(results),
            'failed_comparisons': sum(row['status'] != 'PASS' for row in results),
            'bitwise_equal_comparisons': sum(row['bitwise_equal'] for row in results),
            'thresholds': 'unchanged DEFAULT_P0_PARITY', 'rows': results}
