"""Block causal streaming forward for Wan2.1."""
import torch

from ..wan.modules.attention import flash_attention

from .rope import RopeTable, apply_rope
from .kvcache import StreamingKVCache


def timestep_to_train_scale(t_val, num_train_timesteps=1000):
    """Convert a scheduler timestep to the checkpoints training scale."""
    return float(t_val) / float(num_train_timesteps)


class ModulationCache:
    """Cache per block AdaLN modulation for one timestep."""

    def __init__(self, blocks, e0):
        self.chunks = []
        with torch.amp.autocast('cuda', dtype=torch.float32):
            for blk in blocks:
                self.chunks.append((blk.modulation + e0).chunk(6, dim=1))

    def __getitem__(self, i):
        return self.chunks[i]


def block_forward(blk, x, mod, rope_tbl, t_index, cache, layer, ctx, ctx_lens):
    """One WanAttentionBlock in block causal streaming mode.

    x: [B, S, dim] tokens of the CURRENT latent frame only.
    Returns the updated x. K/V for this frame is written (uncommitted) to `cache`.
    """
    ec = mod
    sa_in = blk.norm1(x).float() * (1 + ec[1]) + ec[0]

    b, s = sa_in.shape[0], sa_in.shape[1]
    n = blk.num_heads
    d = blk.dim // n
    sa = blk.self_attn

    q = sa.norm_q(sa.q(sa_in)).view(b, s, n, d)
    k = sa.norm_k(sa.k(sa_in)).view(b, s, n, d)
    v = sa.v(sa_in).view(b, s, n, d)

    tbl = rope_tbl.frame(t_index)
    q = apply_rope(q, tbl)
    k = apply_rope(k, tbl)

    # attend over cached history and the current chunk
    cache.write(layer, k.to(cache.k.dtype), v.to(cache.v.dtype))
    ctx_k, ctx_v = cache.context(layer, s)

    y = flash_attention(q=q.to(ctx_k.dtype), k=ctx_k, v=ctx_v,
                        window_size=(-1, -1), causal=False)
    y = sa.o(y.flatten(2))

    with torch.amp.autocast('cuda', dtype=torch.float32):
        x = x + y * ec[2]
    x = x + blk.cross_attn(blk.norm3(x), ctx, ctx_lens)
    y = blk.ffn(blk.norm2(x).float() * (1 + ec[4]) + ec[3])
    with torch.amp.autocast('cuda', dtype=torch.float32):
        x = x + y * ec[5]
    return x


@torch.no_grad()
def frame_forward(model, latent_frame, e, mod, rope_tbl, t_index, cache,
                  ctx, ctx_lens):
    """Denoise ONE latent frame against the cached clean past.

    latent_frame: [B, C, 1, H, W]. Returns predicted velocity [C, 1, H, W].
    Does not commit the frame's K/V -- the caller commits once the chunk is final.
    """
    x = model.patch_embedding(latent_frame)
    grid = torch.stack([torch.tensor(x.shape[2:], dtype=torch.long, device=x.device)
                        for _ in range(x.shape[0])])
    x = x.flatten(2).transpose(1, 2)

    for i, blk in enumerate(model.blocks):
        x = block_forward(blk, x, mod[i], rope_tbl, t_index, cache, i, ctx, ctx_lens)

    return model.unpatchify(model.head(x, e), grid)[0]


@torch.no_grad()
def sequence_forward(model, latents, e, mod, rope_tbl, cache, ctx, ctx_lens,
                     start_index=0, commit=True):
    """Run a run of latent frames causally, committing each as it completes.

    Used to (a) prime the cache from clean context frames and (b) evaluate the
    model over a whole sequence for diagnostics. latents: [B, C, F, H, W].
    """
    outs = []
    for f in range(latents.shape[2]):
        out = frame_forward(model, latents[:, :, f:f + 1], e, mod, rope_tbl,
                            start_index + f, cache, ctx, ctx_lens)
        outs.append(out)
        if commit:
            cache.commit(rope_tbl.seq)
    return torch.cat(outs, dim=1)


def make_rope_table(model, h_patches, w_patches, max_frames, device):
    return RopeTable(model.freqs.to(device), h_patches, w_patches, max_frames, device)


def make_cache(model, tokens_per_frame, max_frames, device, dtype=torch.bfloat16):
    n = model.num_heads
    d = model.dim // n
    return StreamingKVCache(
        num_layers=len(model.blocks),
        max_tokens=tokens_per_frame * max_frames,
        num_heads=n, head_dim=d, device=device, dtype=dtype,
        scratch_tokens=tokens_per_frame)


def latent_geometry(width, height, vae_stride=(4, 8, 8), patch_size=(1, 2, 2)):
    """Pixel (W,H) -> latent (H_lat, W_lat) and patch grid (H_p, W_p).

    Mirrors WanModel/run_demo: target_shape[2] = size[1]//vae_stride[1] (height),
    target_shape[3] = size[0]//vae_stride[2] (width).
    """
    h_lat = height // vae_stride[1]
    w_lat = width // vae_stride[2]
    if h_lat % patch_size[1] or w_lat % patch_size[2]:
        raise ValueError(f'{width}x{height} -> latent {h_lat}x{w_lat} not '
                         f'divisible by patch size {patch_size[1:]}')
    return h_lat, w_lat, h_lat // patch_size[1], w_lat // patch_size[2]
