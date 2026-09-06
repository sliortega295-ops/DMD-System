"""Compatibility for H3 rollout step 0 in the fixed-shape Grid table.

At base sigma=1, video_shift(1)==audio_shift(1)==1, so build_row_timesteps
returns one unique timestep.  The mmap table stores a fixed six-row layout by
computing that identical timestep twice; the first three rows are exactly the
runtime rows and the duplicated tail is never indexed.  Map runtime [t] to the
manifest key [t, t].
"""

from __future__ import annotations

from . import grid_adaln as grid


def _entry_index(self, timesteps):
    bits = grid._f32_bits(timesteps)
    if len(bits) == 1:
        bits = (bits[0], bits[0])
    elif len(bits) != 2:
        raise RuntimeError(
            f"Grid-1000 H3 expects one or two unique AV timesteps, got {len(bits)}"
        )
    try:
        return self._entry_by_timestep_bits[bits]
    except KeyError as exc:
        raise RuntimeError(
            "Runtime sigma/timestep is not represented in the frozen AdaLN grid. "
            f"bits={bits}; this would violate the Grid-1000 contract."
        ) from exc


grid.GridAdaLNController._entry_index = _entry_index
