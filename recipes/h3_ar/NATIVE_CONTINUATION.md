# MiniMax H3 DMD2 native continuation

Use [`training.native-stream.json`](training.native-stream.json) for complete-cycle
checkpoints and restart with the native ZeRO-3 owner. See the
[continuation contract](../../docs/contracts/h3_native.md#original_h3_ar_continuation_v1)
for the state and input formats.

Use an allocated two-node job with eight ranks per node and the configured
per-node capacity and reserve policy. Supply the licensed native source and
model assets, helper/library pins and 50-block model audit. The recipe requires
PyTorch 2.10.0+cu128, CUDA 12.8 and the pinned DeepSpeed 0.18.4 API.

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

   Preparation reads the source/parser settings. For training, set
   `cache_manifest` to the prepared path and SHA256 returned by this command.

3. Copy `training.native-stream.json` to your project, resolve its task/assets/
   input/capacity paths and choose a new output directory outside the checkout.
   Use `--dry-run` to check the launch configuration. On each rank of your
   existing torchrun job, run `distillctl train CONFIG` with these attention and
   allocator settings:

   ```sh
   DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,pinned_use_cuda_host_register:True,pinned_num_register_threads:4
   ```

   This recipe is opt-in manual ZeRO3-AP. To explicitly select the same native
   reference owner, use `mode: reference`, remove `preset`, set
   `implementation: zero3-native`, and use allocator `expandable_segments:True`.
   Select this mode explicitly to run the reference owner.

4. Keep `cycles: 100`, `seed: 42`, `checkpoint_every: 50` fixed. The first segment
   stops at `stop_after_cycle: 50`. Require the complete checkpoint's common
   manifest, every native model/optimizer shard, and `_SUCCESS`; a rank receipt
   alone is not a restart checkpoint. Restart in a new process/output directory
   with `resume_from` pointing to `checkpoints/cycle-000050` and
   `stop_after_cycle: 100`. Do **not** change the horizon, data/assets, source,
   topology, precision, capacity or optimizer layout. Actual native steps go
   100 -> 102 on the first resumed cycle, not 0 -> 2. Caches restart cold.

Restart from a complete-cycle checkpoint on the same topology. The worker
reconstructs the frozen 33B backbone from assets. Load checkpoints only from
trusted sources.

## Explicit epochs over a finite cached dataset

To train for more cycles than the cached dataset has distinct records, select
the [versioned record-epoch policy](../../docs/contracts/h3_native.md#original_h3_ar_record_epochs_v1)
instead of duplicating files or inventing fresh record IDs. Supply a source list
with `rank`, `record_id`, `path`, `sha256` and `sources`, but no `cycle` or
caller-authored `metadata`. Each original rank needs at least one genuine
compatible record. The original parser and rank-specific geometry still apply.

```sh
distillctl data prepare-h3-ar-stream --sources /project/data/rank-pool.json \
  --assets /project/native-assets.json --cycles 50 --seed 42 \
  --input-policy distillschedule.h3_ar_record_epochs.v1 \
  --output /project/data/prepared-w6-epochs.json
```

Bind the returned path/hash in `assets.cache_manifest`, and set training data to
`{"input_kind":"original_h3_ar_epoch_stream.v1","prepared":"/project/data/prepared-w6-epochs.json"}`.
Keep the immutable horizon at 50, checkpoint cadence at 25, and first stop at
25; resume the same document in a new process through cycle 50. With one record
per rank this uses 16 real sources and 800 explicit logical draws. Scheduling
uses a private RNG; the original training RNG and update loop are unchanged.
The same native export entry accepts either declared input kind.

## Current Student output

Use the same training identity with an explicit output policy to export a
complete intermediate or terminal checkpoint. The native owner restores all
state but executes zero additional cycles. See the
[native export and independent inference guide](../../docs/guides/training.md#h3_ar_native_inference)
and `export.native.template.json`. Independent generation accepts new original
AV conditions; [decode](../../docs/guides/training.md#h3_audio_video_decode) uses
the video/audio codecs as a separate step. Integration checks cover source
binding and the CPU continuation protocol.
