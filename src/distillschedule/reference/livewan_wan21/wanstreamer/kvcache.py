"""Preallocated K/V cache for block causal streaming.

Writes remain pending during denoising and are committed only when a block is
finished. Context views contain cached history and the pending block.
"""
import torch


class StreamingKVCache:
    """Preallocated K/V store per layer with no ring and an uncommitted scratch tail."""

    def __init__(self, num_layers, max_tokens, num_heads, head_dim,
                 device, dtype=torch.bfloat16, scratch_tokens=None):
        self.num_layers = num_layers
        self.max_tokens = max_tokens
        self.scratch = scratch_tokens if scratch_tokens is not None else max_tokens
        cap = max_tokens + self.scratch
        self.k = torch.zeros(num_layers, cap, num_heads, head_dim,
                             device=device, dtype=dtype)
        self.v = torch.zeros(num_layers, cap, num_heads, head_dim,
                             device=device, dtype=dtype)
        self.length = 0
        self._pending = 0
        # recorded prefixes become invalid after an eviction
        self.evictions = 0

    def reset(self):
        self.length = 0
        self._pending = 0
        self.evictions = 0

    def write(self, layer, k, v):
        """Write the current (uncommitted) chunk's K/V at [length, length+S)."""
        s = k.shape[1]
        if self.length + s > self.k.shape[1]:
            raise RuntimeError(
                f'K/V cache overflow: {self.length} committed + {s} pending > '
                f'capacity {self.k.shape[1]}. Raise max_tokens or evict.')
        self.k[layer, self.length:self.length + s].copy_(k[0])
        self.v[layer, self.length:self.length + s].copy_(v[0])
        self._pending = s

    def context(self, layer, s):
        """View of past + current: ([1, length+s, n, d], same for v). No copy."""
        end = self.length + s
        return (self.k[layer, :end].unsqueeze(0),
                self.v[layer, :end].unsqueeze(0))

    def commit(self, s):
        """Promote the pending chunk to committed history."""
        self.length += s
        self._pending = 0

    def evict_front(self, s, protect=0):
        """Drop `s` tokens from the front of the evictable region.

        `protect` pins the world prefix so eviction removes only old events.
        """
        if s <= 0:
            return
        evictable = self.length - protect
        s = min(s, max(0, evictable))
        if s == 0:
            return
        keep = self.length - protect - s          # events surviving the eviction
        if keep > 0:
            src = protect + s
            self.k[:, protect:protect + keep].copy_(self.k[:, src:src + keep])
            self.v[:, protect:protect + keep].copy_(self.v[:, src:src + keep])
        self.length = protect + keep
        self.evictions += 1

    @property
    def num_tokens(self):
        return self.length

    def memory_bytes(self):
        return self.k.numel() * self.k.element_size() * 2
