# H3 actual-attention tuple capture v2

This opt-in diagnostic observes the original first Fake update at cycle 1,
rank 0, block 10. It calls the original backward once and saves its inputs and
gradient outputs. Ordinary training and performance measurement do not enable it.

## Bind the sample, not the cache capacity

H3 packs `[text | stereo audio | video]` without sequence padding. The text
length depends on the selected condition. A maximum-sized replay cache is not
the shape of every attention call.

Version 2 uses:

- schema: `distillschedule.h3_native_attention_capture_opt_in.v2`;
- selector: `h3.w5.c1.fake0.block10.actualtuple.v2`;
- a sealed, exact Q/K/V/output shape and corresponding LSE shape;
- the existing sample, native-config, source, provider, owner and capacity
  bindings.

After the original Fake input has been prepared, the observer reads its text,
audio and video **shape metadata**, and the loaded model's head count and head
dimension. Their packed shape must equal the sealed geometry before entering
the selected Fake forward/backward. No tensor values are read and no extra
layout, device copy or random sample is produced.

For example, 37,296 video rows, 414 stereo-audio rows and 17 text rows yield
37,727 attention rows, not the 37,760-row maximum. The original input is neither
padded nor cropped to make the diagnostic fit.

The reviewed maximum shape and byte ceilings remain unchanged. Version 1
retains its original fixed geometry. Schema/selector combinations cannot be
mixed. At capture, every tensor must still match its exact admitted dtype,
shape, device, layout and storage constraints.

## Evidence consumers

`native_capture_geometry(packet)` provides the same versioned geometry check to
the worker and tuple consumers. A v2 tuple includes `prepared_geometry` in its
binding; readers must compare it with the sealed packet and validate all nine
tensor files against that geometry rather than a maximum-size constant.

Capture receipts remain correctness diagnostics, excluded from timing/cost
claims. An intact tuple is not a numerical-equivalence result: the original
native completion checks and subsequent same-tuple analysis remain separate.
