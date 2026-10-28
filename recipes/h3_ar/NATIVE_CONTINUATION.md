# Explicit native W6 continuation

Status: source/CPU protocol integration only; native world16 continuation is
**NOT_RUN**. This is not a pretrained quality or decoded-AV tutorial. See
[`original_h3_ar_continuation_v1.md`](../../docs/contracts/original_h3_ar_continuation_v1.md).

Use existing, licensed original native source/model/helper/library assets and
the exact supported environment. No asset/environment download, GPU reservation
or launcher is provided. The original source receipt, helper/library pins,
50-block model audit, Torch2.10.0+cu128/CUDA12.8 and pinned DeepSpeed0.18.4 API are
required. Preserve the original two-node/world16 startup device identities and
explicit per-node capacity/reserve policy. Start only within an independently
authorized resource allocation.

1. Supply a JSON list of `16 * cycles` fresh source rows, ordered by cycle then
   rank. Each row has this shape (all paths and hashes must be real):

   ```json
   {"cycle": 1, "rank": 0, "record_id": "unique-native-record-id",
    "path": "/project/data/record.pth", "sha256": "SHA256_OF_CACHED_RECORD",
    "sources": {"original_features": {"path": "/project/data/source.file",
                                      "sha256": "SHA256_OF_SOURCE"}}}
   ```

   Cached records must already carry original previous/current AV and native
   text/packing conditions. Preparation refuses wrong rank geometry, masks,
   missing conditions or dtype/scalar differences rather than transforming them.

2. On CPU, invoke:

   ```sh
   distillctl data prepare-h3-ar-stream --sources /project/data/sources.json \
     --assets /project/native-assets.json --cycles 100 --seed 42 \
     --output /project/data/prepared-w6.json
   ```

   Preparation uses only the source/parser part of the assets document. For
   training, set its `cache_manifest` to this *actual* prepared path and returned
   SHA256. Do not retain an unused private historical cache dependency.

3. Copy `training.native-stream.json` to your project, resolve its task/assets/
   input/capacity paths and choose a new output directory outside the checkout.
   Public describe/validate/dry-run are metadata/admission only and start no GPU.
   Actual execution must enter `distillctl train CONFIG` on all already-launched
   ranks with the original explicit attention/allocator environment:

   ```sh
   DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,pinned_use_cuda_host_register:True,pinned_num_register_threads:4
   ```

   This recipe is opt-in manual ZeRO3-AP. To explicitly select the same native
   reference owner, use `mode: reference`, remove `preset`, set
   `implementation: zero3-native`, and use allocator `expandable_segments:True`.
   It is never substituted automatically. Saved historical plans, partial
   bundles, generic auto mode and the separate CPU reference are not this entry.

4. Keep `cycles: 100`, `seed: 42`, `checkpoint_every: 50` fixed. The first segment
   stops at `stop_after_cycle: 50`. Require the complete checkpoint's common
   manifest, every native model/optimizer shard, and `_SUCCESS`; a rank receipt
   alone is not a restart checkpoint. Restart in a new process/output directory
   with `resume_from` pointing to `checkpoints/cycle-000050` and
   `stop_after_cycle: 100`. Do **not** change the horizon, data/assets, source,
   topology, precision, capacity or optimizer layout. Actual native steps go
   100 -> 102 on the first resumed cycle, not 0 -> 2. Caches restart cold.

Only complete-cycle, same-topology restart is supported. There is no mid-cycle,
elastic or failed-update replay. The frozen 33B backbone is reconstructed from
assets, not duplicated in checkpoints. Never restore untrusted native pickles.

## Current Student output

Use the same training identity with an explicit output policy to export a
complete intermediate or terminal checkpoint. The native owner restores all
state but executes zero additional cycles. See the
[native export and independent inference guide](../../docs/guides/h3_ar_native_inference.md)
and `export.native.template.json`. Independent generation accepts new original
AV conditions; [decode](../../docs/guides/h3_audio_video_decode.md) uses the actual
video/audio codecs as a separate step. Native GPU output qualification and
pretrained quality remain separate from source/CPU acceptance.
