"""Validated source features for task-specific original-Wan input capture.

Only existing pre-encoded features are read. This module never resizes data,
changes a task, samples a replacement clip, loads a model, or launches a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from .wan_task import OriginalWanTask

SCHEMA = 'distillschedule.original_wan_sources.v1'


@dataclass(frozen=True)
class WanInputSources:
    features: tuple[str, ...]
    seed: int = 42
    cohort: int = 0
    schema_version: str = SCHEMA

    def __post_init__(self):
        require(self.schema_version == SCHEMA, 'unknown source schema', path='wan.inputs')
        require(len(self.features) == 16 and all(isinstance(p, str) and p for p in self.features),
                'one explicit pre-encoded feature file per rank is required', path='wan.inputs')
        paths = [Path(p).expanduser().resolve() for p in self.features]
        require(len(set(paths)) == 16, 'rank feature paths must be distinct', path='wan.inputs')
        require(type(self.seed) is int and 0 <= self.seed < 2**63,
                'seed must be a non-negative 63-bit integer', path='wan.inputs')
        require(type(self.cohort) is int and 0 <= self.cohort < 2**31,
                'invalid cohort index', path='wan.inputs')

    def to_dict(self):
        return {'schema_version': self.schema_version,
                'features': [str(Path(p).expanduser().resolve()) for p in self.features],
                'seed': self.seed, 'cohort': self.cohort}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        require(isinstance(value, Mapping) and set(value) <= set(cls.__dataclass_fields__),
                'unknown source fields are not ignored', path='wan.inputs')
        require(isinstance(value.get('features'), (list, tuple)), 'feature sequence required', path='wan.inputs')
        return cls(**{**value, 'features': tuple(value['features'])})

    def inventory(self):
        rows = []
        for rank, name in enumerate(self.to_dict()['features']):
            path = Path(name)
            require(path.is_file() and path.suffix in {'.pt', '.safetensors'},
                    'source must be a local .pt or .safetensors feature file', path='wan.inputs')
            rows.append({'rank': rank, 'path': str(path), 'sha256': file_sha256(path)})
        return {'sources': self.to_dict(), 'files': rows, 'digest': fingerprint(rows)}


def load_feature(task: OriginalWanTask, row):
    """Read the native dataset's vae_latent/prompt_embed keys, without its retry loop."""
    import torch
    path = Path(row['path'])
    require(file_sha256(path) == row['sha256'], 'source feature changed', path='wan.inputs')
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file
        value = load_file(str(path), device='cpu')
    else:
        value = torch.load(path, map_location='cpu', weights_only=True)
    require(isinstance(value, dict) and {'vae_latent', 'prompt_embed'} <= set(value),
            'native vae_latent and prompt_embed features required', path='wan.inputs')
    latent, text = value['vae_latent'], value['prompt_embed']
    require(isinstance(latent, torch.Tensor) and latent.ndim == 5
            and latent.shape[0] >= 1 and tuple(latent.shape[1:]) == task.latent_shape[1:],
            'encoded sections do not match requested frames/resolution; re-encode, never resize', path='wan.inputs')
    if isinstance(text, torch.Tensor) and text.ndim == 3 and text.shape[0] == 1:
        text = text.squeeze(0)
    require(isinstance(text, torch.Tensor) and text.ndim == 2 and text.shape[0] > 0
            and text.shape[1] == 4096, 'original Wan expects unbatched T5 embeddings of width 4096', path='wan.inputs')
    require(latent.is_floating_point() and text.is_floating_point()
            and bool(torch.isfinite(latent).all()) and bool(torch.isfinite(text).all()),
            'non-finite or non-floating source features', path='wan.inputs')
    mask = value.get('attention_mask')
    if mask is None:
        raise ValueError('explicit native attention_mask required; padding semantics are not guessed')
    if isinstance(mask, torch.Tensor) and mask.ndim == 2 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    require(isinstance(mask, torch.Tensor) and tuple(mask.shape) == (text.shape[0],)
            and bool(((mask == 0) | (mask == 1)).all()), 'invalid native attention mask', path='wan.inputs')
    require(file_sha256(path) == row['sha256'], 'feature changed while loading', path='wan.inputs')
    return latent, text, mask


def preflight_sources(task, sources):
    inventory = sources.inventory()
    shapes = []
    for row in inventory['files']:
        latent, text, mask = load_feature(task, row)
        shapes.append({'rank': row['rank'], 'sections': latent.shape[0],
                       'text_shape': list(text.shape), 'latent_shape': list(latent.shape)})
    # All ranks execute the same native physical batch/sequence geometry.
    require(len({tuple(row['text_shape']) for row in shapes}) == 1,
            'rank text shapes must match', path='wan.inputs')
    return {**inventory, 'task_sha256': task.digest, 'shapes': shapes,
            'status': 'PASS_SOURCE_FEATURES', 'gpu_execution': 'NOT_RUN'}
