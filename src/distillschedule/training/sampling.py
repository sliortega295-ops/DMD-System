"""Resolve Program training positions to immutable manifest record indexes.

No training RNG, tensor imports, dataset mutation or checkpoint sampler state:
the policy, seed, dataset length and total consumed-record cursor determine it.
"""
from dataclasses import dataclass
import hashlib


def sampling_policy(value):
    if (not isinstance(value, dict) or set(value) != {"kind", "shuffle"}
            or value["kind"] not in ("finite", "epochs") or type(value["shuffle"]) is not bool):
        raise ValueError("data.sampling requires exactly kind=finite|epochs and an explicit boolean shuffle")
    if value["kind"] == "finite" and value["shuffle"]:
        raise ValueError("finite sampling requires shuffle=false and preserves manifest order")
    return dict(value)


@dataclass(frozen=True)
class SamplePosition:
    sample_index: int
    epoch: int
    epoch_offset: int


class ProgramSampling:
    """One full permutation per epoch, traversed by fixed rank-strided cohorts."""

    def __init__(self, policy=None, *, size, seed, world_size=1):
        self.policy = sampling_policy({"kind": "finite", "shuffle": False} if policy is None else policy)
        if type(size) is not int or size < 1:
            raise ValueError("sampling dataset size must be a positive integer")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("sampling seed must be an integer in [0, 2**63)")
        if type(world_size) is not int or world_size < 1:
            raise ValueError("sampling world_size must be a positive integer")
        if self.policy["kind"] == "epochs" and size % world_size:
            raise ValueError("epoch sampling dataset length must be divisible by actual world_size; no padding or dropping")
        self.size, self.seed, self.world_size = size, seed, world_size
        self._epoch, self._order = None, ()

    def admit_cycles(self, cycles):
        if type(cycles) is not int or cycles < 1:
            raise ValueError("sampling cycles must be a positive integer")
        if self.policy["kind"] == "finite" and cycles * self.world_size > self.size:
            raise ValueError("trainer.cycles exceeds prepared samples; explicit epoch sampling is required")

    def validate_progress(self, cycle, cursor):
        if (type(cycle) is not int or cycle < 0 or type(cursor) is not int or cursor < 0
                or cursor != cycle * self.world_size):
            raise ValueError("checkpoint data cursor must match complete rank-strided cycles")
        if self.policy["kind"] == "finite" and cursor > self.size:
            raise ValueError("finite checkpoint data cursor exceeds prepared samples")

    def position(self, cursor):
        if type(cursor) is not int or cursor < 0:
            raise ValueError("sampling cursor must be a nonnegative integer")
        if self.policy["kind"] == "finite" and cursor >= self.size:
            raise ValueError("prepared input exhausted; add samples or explicitly opt in to epoch sampling")
        epoch, offset = divmod(cursor, self.size)
        index = offset
        if self.policy["shuffle"]:
            if self._epoch != epoch:
                # Versioned ASCII encoding + digest/index total ordering is
                # stable across Python versions and independent of every RNG.
                prefix = f"distillschedule.program_sampling.sha256-sort-v1:{self.seed}:{epoch}:"
                self._order = tuple(sorted(range(self.size), key=lambda index: (
                    hashlib.sha256((prefix + str(index)).encode("ascii")).digest(), index)))
                self._epoch = epoch
            index = self._order[offset]
        return SamplePosition(index, epoch, offset)

    def describe(self):
        return {**self.policy, "algorithm": "sha256-sort-v1" if self.policy["shuffle"] else "manifest-order-v1",
                "seed": self.seed, "samples_per_epoch": self.size}
