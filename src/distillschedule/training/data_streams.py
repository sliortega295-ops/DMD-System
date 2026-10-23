"""Independent prepared input streams with resumable, RNG-free sampling."""
from __future__ import annotations

import hashlib

from distillschedule.data import PreparedDataset
from distillschedule.workloads.best_known.registry import fingerprint
from .sampling import ProgramSampling, SamplePosition


class InputStreams:
    def __init__(self, streams, *, seed, world_size=1):
        self.world_size = world_size
        self.datasets, self.samplers, self.counts = {}, {}, {}
        for name, spec in sorted(streams.items()):
            data = PreparedDataset(spec['prepared'])
            stream_seed = int(hashlib.sha256(f'{seed}:{name}'.encode()).hexdigest()[:15], 16)
            self.datasets[name] = data
            self.counts[name] = spec.get('samples_per_cycle', 1)
            self.samplers[name] = ProgramSampling(spec.get('sampling'), size=len(data),
                                                seed=stream_seed, world_size=world_size)
        self.manifest = {'schema_version': 'distillschedule.input_streams.v1',
                         'streams': {name: data.manifest for name, data in self.datasets.items()}}
        self.identity = fingerprint({'streams': {name: data.identity for name, data in self.datasets.items()},
                                     'sampling': self.describe()})

    def __len__(self):
        return min(len(data) // self.counts[name] for name, data in self.datasets.items())

    def _positions(self, cursor, name):
        cycle, rank = divmod(cursor, self.world_size)
        count = self.counts[name]
        return [self.samplers[name].position((cycle * count + draw) * self.world_size + rank)
                for draw in range(count)]

    def cycle_sample(self, cursor):
        return {name: [data.sample(p.sample_index) for p in self._positions(cursor, name)]
                for name, data in self.datasets.items()}

    def sample(self, cursor):
        return self.cycle_sample(cursor)

    def sample_id(self, cursor):
        return {name: [data.sample_id(p.sample_index) for p in self._positions(cursor, name)]
                for name, data in self.datasets.items()}

    def position(self, cursor):
        if type(cursor) is not int or cursor < 0:
            raise ValueError('stream cycle cursor must be nonnegative')
        return SamplePosition(cursor, 0, cursor)

    def admit_cycles(self, cycles):
        for name, sampler in self.samplers.items():
            sampler.admit_cycles(cycles * self.counts[name])

    def validate_progress(self, cycle, cursor):
        if type(cycle) is not int or type(cursor) is not int or cursor != cycle * self.world_size:
            raise ValueError('stream progress must describe a complete cycle')
        for name, sampler in self.samplers.items():
            sampler.validate_progress(cycle * self.counts[name], cursor * self.counts[name])

    def progress(self, cursor):
        return {name: {'cursor': cursor * self.counts[name], 'samples_per_cycle': self.counts[name],
                       'sampling': sampler.describe()}
                for name, sampler in self.samplers.items()}

    def validate_data_progress(self, value, cursor):
        if value != self.progress(cursor):
            raise ValueError('checkpoint stream data progress changed')

    def describe(self):
        return {name: {'samples_per_cycle': self.counts[name], **sampler.describe()}
                for name, sampler in self.samplers.items()}


def training_data(config, *, world_size):
    if 'streams' in config.data:
        return InputStreams(config.data['streams'], seed=config.trainer.get('seed', 0), world_size=world_size)
    return PreparedDataset(config.data['prepared'])


def training_sampling(config, data, *, world_size):
    return (data if isinstance(data, InputStreams) else
            ProgramSampling(config.data.get('sampling'), size=len(data),
                            seed=config.trainer.get('seed', 0), world_size=world_size))


def data_progress(data, cursor):
    return data.progress(cursor) if isinstance(data, InputStreams) else None


def validate_data_progress(data, payload):
    if isinstance(data, InputStreams):
        data.validate_data_progress(payload.get('data_progress'), payload['cursor'])
