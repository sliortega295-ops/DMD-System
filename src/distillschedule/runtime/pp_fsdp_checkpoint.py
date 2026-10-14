"""Restore only parameters owned by a PP stage, in genuine FSDP DTensors."""
from __future__ import annotations

import json
import torch
from torch.distributed.tensor import distribute_tensor

from .fsdp2_static import restore_fsdp2_optimizer_from_logical, ema_update_reference


class _StagePackageView:
    def __init__(self, package, component, names):
        self.package, self.root, self.component_name = package, package.root, component
        self.names = tuple(names)
        full = tuple(package.component(component)['parameter_names'])
        if not set(names).issubset(full) or len(set(names)) != len(names):
            raise ValueError('stage names must be a unique subset of the original checkpoint')

    def component(self, component):
        if component != self.component_name:
            raise ValueError('wrong checkpoint role')
        result = dict(self.package.component(component))
        result['parameter_names'] = list(self.names)
        return result


def restore_stage_optimizer(native, optimizer, scheduler, package, component):
    names = tuple(n for n,p in native.named_parameters() if p.requires_grad)
    view = _StagePackageView(package, component, names)
    return restore_fsdp2_optimizer_from_logical(module=native, optimizer=optimizer,
        scheduler=scheduler, package=view, component=component)


class LogicalStageShardEMA:
    def __init__(self, native, package, *, device):
        root = package.root / 'ema'
        metadata = json.loads((root/'sparse_ema_manifest.json').read_text())
        if metadata.get('schema_version') != 'distillschedule.sparse_ema.v1':
            raise ValueError('unadmitted EMA format')
        parameters = {n:p for n,p in sorted(native.named_parameters()) if p.requires_grad}
        normalize = lambda n: n[7:] if n.startswith('module.') else n
        if not set(parameters).issubset({normalize(n) for n in metadata['tracked_names']}):
            raise ValueError('PP stage EMA parameters are not present in original checkpoint')
        loaded = {}
        for shard in metadata['shards']:
            values = torch.load(root/shard['file'], map_location='cpu', weights_only=True, mmap=True)
            for name, full in values.items():
                key = normalize(name)
                if key not in parameters:
                    continue
                if key in loaded:
                    raise ValueError('duplicate original EMA parameter')
                p = parameters[key]
                if tuple(full.shape) != tuple(p.shape):
                    raise ValueError('EMA original global shape changed')
                distributed = distribute_tensor(full.to(device), device_mesh=p.device_mesh, placements=p.placements)
                loaded[key] = distributed.to_local().detach().clone()
            del values
        if set(loaded) != set(parameters):
            raise ValueError('missing stage EMA state')
        self.parameters = tuple(parameters.values())
        self.shadows = tuple(loaded[n] for n in parameters)
        self.decay = float(metadata['decay'])
        self.update_after_step = int(metadata['update_after_step'])
        self.optimization_step = int(metadata['optimization_step'])
        self.initial_step = self.optimization_step
        if self.optimization_step != int(package.manifest['ema']['optimization_step']):
            raise ValueError('logical EMA version mismatch')

    @torch.no_grad()
    def step(self):
        self.optimization_step += 1
        step = max(0, self.optimization_step-self.update_after_step-1)
        decay = 0.0 if step <= 0 else min(self.decay, (1+step)/(10+step))
        for shadow, parameter in zip(self.shadows, self.parameters, strict=True):
            ema_update_reference(shadow, parameter.to_local().detach().to(shadow.dtype), decay)

    def receipt(self):
        return dict(scope='UNIQUELY_OWNED_PP_AND_FSDP_LOCAL_SHARDS', device='cuda',
                    optimization_step=self.optimization_step, updates=self.optimization_step-self.initial_step,
                    merge_count=0)
