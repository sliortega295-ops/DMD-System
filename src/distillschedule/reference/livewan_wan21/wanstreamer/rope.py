"""Precomputed 3D RoPE with absolute temporal offsets."""
import torch


class RopeTable:
    """Per temporal index RoPE frequency tables for a fixed spatial grid.

    freqs: the model's [1024, c] complex buffer (WanModel.freqs), c = head_dim/2.
    Table t holds the flattened (1, h, w) grid at temporal position t, shaped
    [h*w, 1, c] so it broadcasts over batch and heads.
    """

    def __init__(self, freqs, h, w, max_frames, device, dtype=torch.complex64):
        c = freqs.shape[1]
        # temporal uses the dimensions left after the spatial split
        f_t, f_h, f_w = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        assert max_frames <= f_t.shape[0], (
            f'max_frames={max_frames} exceeds rope_params table ({f_t.shape[0]})')
        assert h <= f_h.shape[0] and w <= f_w.shape[0], (
            f'grid {h}x{w} exceeds rope_params table '
            f'({f_h.shape[0]}x{f_w.shape[0]}) -- raise rope_params(1024, ...)')

        spatial = torch.cat([
            f_h[:h].view(h, 1, -1).expand(h, w, -1),
            f_w[:w].view(1, w, -1).expand(h, w, -1),
        ], dim=-1).reshape(h * w, -1)

        tables = []
        for t in range(max_frames):
            temporal = f_t[t].view(1, -1).expand(h * w, -1)
            tables.append(torch.cat([temporal, spatial], dim=-1))
        self.table = torch.stack(tables).unsqueeze(2).to(device=device, dtype=dtype)
        self.h, self.w, self.seq = h, w, h * w
        self.max_frames = max_frames

    def frame(self, t_index):
        """[S, 1, c] complex table for a single latent frame at temporal index t."""
        if t_index >= self.max_frames:
            raise IndexError(f'temporal index {t_index} >= max_frames {self.max_frames}')
        return self.table[t_index]

    def span(self, t_start, num_frames):
        """[num_frames*S, 1, c] for a contiguous run of latent frames."""
        end = t_start + num_frames
        if end > self.max_frames:
            raise IndexError(f'span [{t_start},{end}) exceeds max_frames {self.max_frames}')
        return self.table[t_start:end].reshape(num_frames * self.seq, 1, -1)


def apply_rope(x, table):
    """x: [B, L, n, d] real -> [B, L, n, d] real, rotated by `table` [L, 1, c].

    Done in float32 complex rather than upstream's float64. Verified equivalent
    within bf16 tolerance by tests/test_streaming_core.py.
    """
    b, l, n, d = x.shape
    xc = torch.view_as_complex(x.float().reshape(b, l, n, d // 2, 2))
    out = torch.view_as_real(xc * table.unsqueeze(0))
    return out.flatten(3).to(x.dtype) if x.dtype != torch.float32 else out.flatten(3)
