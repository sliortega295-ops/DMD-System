# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""STATIC SELECTED-DEFINITION DERIVATIVE, NOT the entire upstream attention file.

Original LiveWan _needs_mask and sdpa_attention are retained verbatim below.
Changes: explicit dependency imports; CUDA/optional backend selection removed;
CPU MATH context delegates to the maintained private bridge. Original default
BF16 is NOT the CPU public contract: the bridge always explicitly passes FP32.
See source_manifest.json and retained licenses/origin notice for exact evidence.
"""
import torch
import torch.nn.functional as F


def sdpa_backend_ctx():
    from .attention import math_sdpa_context
    return math_sdpa_context()


def _needs_mask(lens, length):
    """True if `lens` describes anything shorter than the padded `length`."""
    if lens is None:
        return False
    return bool(torch.as_tensor(lens).min().item() < length)


def sdpa_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
):
    """SDPA replacement for `flash_attn_varlen_func`.

    Shapes follow the flash attention convention:
        q: [B, Lq, Nq, C1]   k: [B, Lk, Nk, C1]   v: [B, Lk, Nk, C2]
    Returns: [B, Lq, Nq, C2]

    Lengths mask padded keys and queries. Nonsquare causal attention uses bottom
    right alignment where query i sees keys `0 .. i + Lk - Lq`.
    """
    assert tuple(window_size) == (-1, -1), (
        f'sliding-window attention (window_size={window_size}) has no SDPA '
        'fallback; it would be silently ignored.')

    b, lq, nq, _ = q.shape
    lk, nk = k.shape[1], k.shape[2]
    out_dtype = q.dtype

    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    if q_scale is not None:
        q = q * q_scale

    # transpose sequence and head dimensions
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # replicate K/V heads for grouped query attention
    if nq != nk:
        assert nq % nk == 0, f'Nq ({nq}) must be divisible by Nk ({nk})'
        rep = nq // nk
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    mask_k = _needs_mask(k_lens, lk)
    mask_q = _needs_mask(q_lens, lq)

    if not mask_k and (not causal or lq == lk):
        # square causal attention has the same alignment in either convention
        with sdpa_backend_ctx():
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=causal,
                dropout_p=dropout_p, scale=softmax_scale)
    elif (mask_k and not causal
          and int(torch.as_tensor(k_lens).min()) > 0):
        # broadcast key padding without allocating a dense query mask
        kl = torch.as_tensor(k_lens, device=q.device).reshape(b, 1, 1, 1)
        keep = torch.arange(lk, device=q.device).reshape(1, 1, 1, lk) < kl
        with sdpa_backend_ctx():
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=keep, is_causal=False,
                dropout_p=dropout_p, scale=softmax_scale)
    else:
        # boolean keep mask shaped B 1 Lq Lk
        keep = torch.ones(b, 1, lq, lk, dtype=torch.bool, device=q.device)

        if mask_k:
            kl = torch.as_tensor(k_lens, device=q.device).reshape(b, 1, 1, 1)
            key_idx = torch.arange(lk, device=q.device).reshape(1, 1, 1, lk)
            keep &= key_idx < kl

        if causal:
            # bottom right alignment uses j <= i + lk - lq
            qi = torch.arange(lq, device=q.device).reshape(1, 1, lq, 1)
            kj = torch.arange(lk, device=q.device).reshape(1, 1, 1, lk)
            keep &= kj <= qi + (lk - lq)

        # avoid NaN for empty rows and zero them after attention
        dead = ~keep.any(dim=-1, keepdim=True)
        if dead.any():
            keep = keep | (dead & (torch.arange(lk, device=q.device) == 0).reshape(1, 1, 1, lk))

        with sdpa_backend_ctx():
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=keep, is_causal=False,
                dropout_p=dropout_p, scale=softmax_scale)
        if dead.any():
            out = out.masked_fill(dead, 0.0)

    out = out.transpose(1, 2).contiguous()  # -> [B, Lq, Nq, C2]

    if mask_q:
        ql = torch.as_tensor(q_lens, device=out.device).reshape(b, 1, 1, 1)
        qi = torch.arange(lq, device=out.device).reshape(1, lq, 1, 1)
        out = out * (qi < ql)

    return out.to(out_dtype)
