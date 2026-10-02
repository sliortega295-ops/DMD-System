# C51: opt-in Generator no-grad fused RMS row

Default C38/C49 plans are unchanged. C51 inherits C49 and changes only its
no-grad Generator RMS kernel. Score, gradient forward, checkpoint replay,
optimizer, EMA, source/trace, B16/two-wave scheduling and application calls
are unchanged.

Before:

```python
variance = cast_square_bf16_to_fp32(x).mean(-1, keepdim=True)
normalized = (x.float() * rsqrt(variance + 1e-6)).bfloat16()
output = normalized.float() * weight_fp32
```

After, one row kernel:

```python
variance = sum_fp32(x.float() * x.float()) / 1536
normalized = (x.float() * rsqrt(variance + 1e-6)).bfloat16()
output = normalized.float() * weight_fp32
```

The FP32 reduction tree changes. This is not bitwise-equivalent, nor an
approximate RMS formula; unchanged DEFAULT_P0_PARITY applies. BF16 intermediate
rounding and FP32 affine/output are explicitly retained. C50's BF16-output
experiment is not included.

The source-bound temporary no-grad scope and exception restoration are reused
from C31. Any grad-enabled input to the new kernel fails closed. Receipt must
prove actual scope entries/exits, call/shape counts, dtype, row size/warps and
zero failures. C51 requires the original C49 split-KV/G-FA2/Score-FA3 parent
without C39/C40/C44 or step1 schedule changes.

Component evidence, 2026-10-02: 32 unchanged-threshold real-shape/edge cases
passed; worst normalized L2 2.06e-5. Weighted kernel ratio 3.195x, projected
component saving about 1.503s/cycle using the actual cumulative two-cycle
histogram divided by two. **Neither number is a native speedup or proved
exposed critical-path saving.**

Production scope/numerical qualification and matched native C2 are required.
Cold C1, profiler time and component estimates never enter the native result.
