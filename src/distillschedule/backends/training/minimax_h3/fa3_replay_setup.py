"""Shared FA3 replay-cache configuration; for continuous full-sequence training."""
from __future__ import annotations
import os
from .fa3_replay_cache import (install_fa3_replay_cache, parse_block_indices,
    parse_max_d2h_inflight, parse_storage)


def configure_fa3_replay(self):
    replay_cache = self.training_config.get("a100", {}).get("fa3_replay_cache", {})
    env_blocks = os.environ.get("H3_FA3_REPLAY_CACHE_BLOCKS")
    configured_blocks = replay_cache.get("blocks", ())
    if env_blocks is not None:
        self.fa3_replay_cache_blocks = parse_block_indices(env_blocks)
    elif bool(replay_cache.get("enabled", False)):
        self.fa3_replay_cache_blocks = parse_block_indices(configured_blocks)
    else:
        self.fa3_replay_cache_blocks = ()
    self.fa3_replay_cache_storage = parse_storage(
        os.environ.get(
            "H3_FA3_REPLAY_CACHE_STORAGE", replay_cache.get("storage", "cuda")
        )
    )
    if any(key in replay_cache for key in ("fixed_registered_direct", "release_gpu_references")) or any(
        key in os.environ for key in ("H3_FA3_REPLAY_CACHE_FIXED_REGISTERED_DIRECT",
                                     "H3_FA3_REPLAY_CACHE_RELEASE_GPU_REFERENCES")
    ):
        raise ValueError("continuous v1 does not support retired FA3 transport options")
    self.fa3_replay_cache_max_d2h_inflight = parse_max_d2h_inflight(
        os.environ.get(
            "H3_FA3_REPLAY_CACHE_MAX_D2H_INFLIGHT",
            replay_cache.get("max_d2h_inflight", 2),
        )
    )
    trim_value = str(
        os.environ.get(
            "H3_FA3_REPLAY_CACHE_TRIM_BEFORE_BACKWARD",
            int(bool(replay_cache.get("trim_before_backward", False))),
        )
    ).strip().lower()
    if trim_value not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise ValueError(
            "H3_FA3_REPLAY_CACHE_TRIM_BEFORE_BACKWARD must be boolean, "
            f"got {trim_value!r}"
        )
    self.fa3_replay_cache_trim_before_backward = trim_value in {
        "1", "true", "yes", "on"
    }
    self.fa3_replay_cache_registration = None


def install_configured_fa3_replay(self):
    self.fa3_replay_cache_registration = install_fa3_replay_cache(
        self.shared_model.denoiser_module(),
        block_indices=self.fa3_replay_cache_blocks,
        parent_split_registration=self.fa3_nograd_split_registration,
        storage=self.fa3_replay_cache_storage,
        max_d2h_inflight=self.fa3_replay_cache_max_d2h_inflight,
        trim_before_backward=self.fa3_replay_cache_trim_before_backward,
        tuple_capture=getattr(self, "attention_tuple_capture", None),
    )
    return self.fa3_replay_cache_registration
