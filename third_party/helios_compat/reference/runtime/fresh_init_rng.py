from __future__ import annotations

import hashlib
import random
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch


FRESH_INIT_RNG_CONTRACT = "distillschedule.precast_fresh_init_rng.v1"


def derive_component_seed(base_seed: int, component: str) -> int:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")
    if not isinstance(component, str) or not component:
        raise ValueError("component must be a non-empty string")
    payload = f"{FRESH_INIT_RNG_CONTRACT}\0{base_seed}\0{component}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


@contextmanager
def isolated_component_rng(base_seed: int, component: str) -> Iterator[int]:
    component_seed = derive_component_seed(base_seed, component)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=[], enabled=True):
        try:
            random.seed(component_seed)
            np.random.seed(component_seed % (1 << 32))
            torch.manual_seed(component_seed)
            yield component_seed
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
