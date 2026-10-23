"""Shared block causal Wan forward for training and streaming inference.

Both paths attend over a clean K/V prefix and the current noisy block. The key
set contains no future blocks, so no causal mask is needed.
"""
import torch
import torch.utils.checkpoint as ckpt

from ..wan.modules.attention import flash_attention
from ..wan.modules.model import sinusoidal_embedding_1d


def time_embed(model, t, device, time_scale=1000.0):
    """Embed one or more flow fractions using the checkpoints time scale."""
    if not torch.is_tensor(t):
        t = torch.as_tensor([float(t)], device=device)
    t = t.to(device=device, dtype=torch.float32).reshape(-1)
    # match the module dtype because the frozen teacher can use bf16
    wd = next(model.time_embedding.parameters()).dtype
    with torch.amp.autocast('cuda', enabled=False):
        e = model.time_embedding(
            sinusoidal_embedding_1d(model.freq_dim, t * time_scale).to(wd))
        e0 = model.time_projection(e).unflatten(1, (6, model.dim))
    return e, e0


class Modulation:
    """Cache per layer AdaLN chunks for one set of timesteps."""

    def __init__(self, blocks, e0):
        with torch.amp.autocast('cuda', enabled=False):
            self.chunks = [(blk.modulation + e0).chunk(6, dim=1) for blk in blocks]

    def __getitem__(self, i):
        return self.chunks[i]


def _bcast(g, x, per_frame, n_frames):
    """Broadcast a modulation chunk [N,1,dim] against tokens x [B,L,dim].

    per_frame=False: N is 1 or B, which already broadcasts.
    per_frame=True:  N is the number of latent frames; tokens are frame major,
                     so view x as [B, F, S, dim] and give g a leading axis.
    """
    if not per_frame:
        return g
    return g.unsqueeze(0)              # [1, F, 1, dim] against [B, F, S, dim]


def _as_frames(x, per_frame, n_frames):
    if not per_frame:
        return x
    b, l, d = x.shape
    return x.view(b, n_frames, l // n_frames, d)


def _modulate(x, norm, shift, scale, per_frame, n_frames):
    y = _as_frames(norm(x).float(), per_frame, n_frames)
    y = y * (1 + _bcast(scale, x, per_frame, n_frames)) \
        + _bcast(shift, x, per_frame, n_frames)
    return y.reshape(x.shape[0], x.shape[1], x.shape[2]) if per_frame else y


def _gate(x, y, g, per_frame, n_frames):
    """x + y * g, in float32, with broadcasting per frame."""
    with torch.amp.autocast('cuda', dtype=torch.float32):
        if not per_frame:
            return x + y * g
        b, l, d = x.shape
        out = _as_frames(x, True, n_frames) + \
            _as_frames(y, True, n_frames) * g.unsqueeze(0)
        return out.reshape(b, l, d)


def _head_from(head, x, e, per_frame, n_frames):
    """Split out from `_head` so the FSDP shard unit can enter it with the head
    module it owns rather than reaching through the model (see wanstreamer.fsdp)."""
    with torch.amp.autocast('cuda', enabled=False):
        m = (head.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        y = _modulate(x, head.norm, m[0], m[1], per_frame, n_frames)
        return head.head(y.to(head.head.weight.dtype))


def _head(model, x, e, per_frame, n_frames):
    return _head_from(model.head, x, e, per_frame, n_frames)


def apply_rope(x, tbl):
    """x [B, L, n, d] real -> rotated, by tbl [L, 1, c] (shared) or [B, L, 1, c].

    float32 complex rather than upstream's float64; verified equivalent within
    bf16 tolerance by tests/test_streaming_core.py.
    """
    b, l, n, d = x.shape
    xc = torch.view_as_complex(x.float().reshape(b, l, n, d // 2, 2))
    if tbl.dim() == 3:
        tbl = tbl.unsqueeze(0)
    return torch.view_as_real(xc * tbl).flatten(3)


# K/V stores
class BufferKV:
    """Adapter over the preallocated StreamingKVCache used at inference."""

    def __init__(self, cache):
        self.cache = cache

    def context(self, layer, k, v, **_):
        self.cache.write(layer, k, v)
        ck, cv = self.cache.context(layer, k.shape[1])
        return ck, cv, None


class BufferPrefixKV:
    """Read only slice of a StreamingKVCache, in TrainKV's calling convention.

    Training uses this view to repeat a recorded step against its original prefix.
    """

    def __init__(self, cache):
        self.cache = cache

    def context(self, layer, k, v, upto=None, rows=1, k_lens=None):
        if not upto:
            return k, v, None
        pk = self.cache.k[layer, :upto].unsqueeze(0)
        pv = self.cache.v[layer, :upto].unsqueeze(0)
        if rows > 1:
            pk, pv = pk.expand(rows, -1, -1, -1), pv.expand(rows, -1, -1, -1)
        return torch.cat([k, pk], 1), torch.cat([v, pv], 1), k_lens


class TrainKV:
    """Clean K/V prefix per layer for a whole clip: plain tensors safe for autograd.

    Built once per clip by `build_clean_kv` under no_grad; every noisy block then
    reads the slice its start index implies, so each block trains against exactly
    the prefix it would see at deployment.
    """

    def __init__(self, num_layers):
        self.k = [None] * num_layers
        self.v = [None] * num_layers

    def append(self, layer, k, v):
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=1)
            self.v[layer] = torch.cat([self.v[layer], v], dim=1)

    @property
    def tokens(self):
        return 0 if self.k[0] is None else self.k[0].shape[1]

    def context(self, layer, k, v, upto=None, rows=1, k_lens=None):
        """Return keys as `[current block; clean prefix]`.

        This makes each visible row a contiguous slice for the key length mask.
        """
        pk, pv = self.k[layer], self.v[layer]
        if pk is None or upto == 0:
            return k, v, None
        pk, pv = pk[:, :upto], pv[:, :upto]
        if rows > 1:
            pk = pk.expand(rows, -1, -1, -1)
            pv = pv.expand(rows, -1, -1, -1)
        return torch.cat([k, pk], 1), torch.cat([v, pv], 1), k_lens


# layer forward
def _layer(blk, x, ec, tbl, kv_ctx, ctx, ctx_lens, dtype, per_frame, n_frames):
    """One WanAttentionBlock in block causal mode. kv_ctx(k,v) -> (K, V, k_lens)."""
    sa_in = _modulate(x, blk.norm1, ec[0], ec[1], per_frame, n_frames)
    b, s = sa_in.shape[0], sa_in.shape[1]
    n, d = blk.num_heads, blk.dim // blk.num_heads
    sa = blk.self_attn

    q = apply_rope(sa.norm_q(sa.q(sa_in)).view(b, s, n, d), tbl).to(dtype)
    k = apply_rope(sa.norm_k(sa.k(sa_in)).view(b, s, n, d), tbl).to(dtype)
    v = sa.v(sa_in).view(b, s, n, d).to(dtype)

    ck, cv, k_lens = kv_ctx(k, v)
    y = flash_attention(q=q, k=ck, v=cv, k_lens=k_lens,
                        window_size=(-1, -1), causal=False)
    x = _gate(x, sa.o(y.flatten(2)), ec[2], per_frame, n_frames)
    x = x + blk.cross_attn(blk.norm3(x), ctx, ctx_lens)
    yf = blk.ffn(_modulate(x, blk.norm2, ec[3], ec[4], per_frame, n_frames))
    return _gate(x, yf, ec[5], per_frame, n_frames)


def _run(model, z, t, tbl, ctx, ctx_lens, kv_ctx_factory, dtype, time_scale,
         per_frame, grad_checkpoint, emb=None):
    """Run patch embedding, block causal layers, head and unpatchify."""
    B, _, F = z.shape[0], z.shape[1], z.shape[2]
    e, e0 = emb if emb is not None else time_embed(model, t, z.device, time_scale)
    nf = F if per_frame else 1
    # enter attached shard modules so FSDP gathers their parameters
    layers = getattr(model, 'causal_layers', None)
    head_mod = getattr(model, 'causal_head', None)
    mod = Modulation(model.blocks, e0) if layers is None else None
    with torch.amp.autocast('cuda', dtype=dtype):
        x = model.patch_embedding(z.to(dtype))
        gf, gh, gw = (int(s) for s in x.shape[2:])
        x = x.flatten(2).transpose(1, 2)
        for li in range(len(model.blocks)):
            kv_ctx = kv_ctx_factory(li)
            if layers is None:
                fn = _layer
                args = (model.blocks[li], x, mod[li], tbl, kv_ctx, ctx,
                        ctx_lens, dtype, per_frame, nf)
            else:
                # build modulation after FSDP gathers the block parameters
                fn = layers[li]
                args = (x, e0, tbl, kv_ctx, ctx, ctx_lens, dtype, per_frame, nf)
            if grad_checkpoint and torch.is_grad_enabled():
                x = ckpt.checkpoint(fn, *args, use_reentrant=False)
            else:
                x = fn(*args)
        h = head_mod(x, e, per_frame, nf) if head_mod is not None \
            else _head(model, x, e, per_frame, nf)
        return _unpatchify(model, h, gf, gh, gw)


def _unpatchify(model, x, gf, gh, gw):
    """Convert patched tokens `[B, L, D]` to latents `[B, C, F, H, W]`.

    Python grid dimensions avoid a device sync during CUDA graph capture.
    """
    p0, p1, p2 = model.patch_size
    c = model.out_dim
    b = x.shape[0]
    u = x[:, :gf * gh * gw].view(b, gf, gh, gw, p0, p1, p2, c)
    u = u.permute(0, 7, 1, 4, 2, 5, 3, 6)          # b c f p0 h p1 w p2
    return u.reshape(b, c, gf * p0, gh * p1, gw * p2)


# entry points
def block_forward(model, z, t, t_start, rope, ctx, ctx_lens, kv=None,
                  collect=None, dtype=torch.bfloat16, time_scale=1000.0,
                  prefix_upto=None, per_frame=False, grad_checkpoint=False,
                  tbl=None, emb=None):
    """Velocity for ONE block of latent frames against the clean prefix.

    z        [B, C, F, H, W] noisy latents of the current block
    t        float, or [F] flow fractions in [0,1] when per_frame
    t_start  absolute latent frame index of z[:, :, 0] (drives temporal RoPE)
    kv       BufferKV (inference) | TrainKV (training) | None (self attention only)
    collect  list receiving this block's (k, v) per layer, or None
    tbl      precomputed RoPE table; overrides t_start. The CUDA graph path
             supplies it from a static buffer, since it is the one input that
             changes with the absolute frame index.
    """
    if tbl is None:
        tbl = rope.span(t_start, z.shape[2])

    def factory(li):
        def kv_ctx(k, v):
            if collect is not None:
                collect.append((k, v))
            if kv is None:
                return k, v, None
            if prefix_upto is None:
                return kv.context(li, k, v)
            return kv.context(li, k, v, upto=prefix_upto)
        return kv_ctx

    return _run(model, z, t, tbl, ctx, ctx_lens, factory, dtype, time_scale,
                per_frame, grad_checkpoint, emb=emb)


@torch.no_grad()
def build_clean_kv(model, z, rope, ctx, ctx_lens, world_frames, block_frames,
                   dtype=torch.bfloat16, time_scale=1000.0, rope_gap=0):
    """Build clean K/V from a world followed by event blocks.

    `rope_gap` shifts event positions away from the world positions.
    """
    kv = TrainKV(len(model.blocks))
    F, S = z.shape[2], rope.seq
    spans, f = ([(0, world_frames)] if world_frames else []), world_frames
    while f < F:
        n = min(block_frames, F - f)
        spans.append((f, n))
        f += n
    for t0, n in spans:
        collect = []
        idx = t0 if t0 < world_frames or t0 == 0 else t0 + rope_gap
        block_forward(model, z[:, :, t0:t0 + n], 0.0, idx, rope, ctx, ctx_lens,
                      kv=kv, collect=collect, dtype=dtype,
                      time_scale=time_scale, prefix_upto=t0 * S)
        for li, (k, v) in enumerate(collect):
            kv.append(li, k, v)
    return kv


def event_rope_index(frame, world_frames, rope_gap):
    """Return an absolute event position with an optional world gap.

    The gap simulates the relative distance produced by evicted events while the
    world remains pinned.
    """
    return frame if frame < world_frames else frame + rope_gap


def parallel_blocks_forward(model, z_noisy, t_rows, starts, rope, ctx, ctx_lens,
                            kv, block_frames, dtype=torch.bfloat16,
                            time_scale=1000.0, grad_checkpoint=True):
    """Every event block of a clip in ONE forward, as batch rows.

    Row i carries its own noise level and RoPE offset, and attends over
    [clean prefix[:starts[i]*S] ; its own noisy tokens]. Because the visible key
    set depends only on the row, this costs a
    [nb, 1, 1, Lk] key padding mask instead of a dense [Lq, Lk] one (see
    sdpa_attention's fast path for key padding).

    z_noisy [1, C, F, H, W] noisy latents for the full clip; returns [nb, C, b, H, W].
    """
    S, nb = rope.seq, len(starts)
    blocks = torch.cat([z_noisy[:, :, s:s + block_frames] for s in starts], 0)
    tbl = torch.stack([rope.span(s, block_frames).squeeze(1) for s in starts]
                      ).unsqueeze(2)                       # [nb, b*S, 1, c]
    max_prefix = max(starts) * S
    # each row keeps its own block followed by its clean prefix
    k_lens = torch.tensor([block_frames * S + s * S for s in starts],
                          device=z_noisy.device, dtype=torch.long)
    ctx_b = ctx.expand(nb, -1, -1) if ctx.shape[0] == 1 else ctx
    cl = ctx_lens.expand(nb) if ctx_lens.numel() == 1 else ctx_lens

    def factory(li):
        return lambda k, v: kv.context(li, k, v, upto=max_prefix, rows=nb,
                                       k_lens=k_lens)

    return _run(model, blocks, t_rows, tbl, ctx_b, cl, factory, dtype,
                time_scale, False, grad_checkpoint)


def block_starts(num_frames, world_frames, block_frames):
    return [s for s in range(world_frames, num_frames, block_frames)
            if s + block_frames <= num_frames]
