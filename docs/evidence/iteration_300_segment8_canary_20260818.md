# Segment-8 MiniMax-H3 world16 canary evidence

Run ID: iteration_300_segment8_canary_20260818.

The complete raw evidence bundle (forward and update canaries, 16-rank
segment-debug streams, memory attribution, manifests, tables, and reports) is
in the adjacent .tar.gz file. It is a compressed copy of the persistent
evidence directory:

/home/lyy/Helios/research_artifacts/DMD-System/lightx2v_minimax_fsdp_first_init_20260817/iteration_300_segment8_canary_20260818/

Uncompressed size: approximately 20 MiB. Archive size: approximately 357 KiB.

Result:
- Student grad-enabled forward: PASS (16/16; all 50 blocks).
- Student update: FAIL_CUDA_OOM during backward/recompute at segment 6,
  block 49, on all 16 ranks.
- Full native Student1+Fake5, timing, and Nsight: NOT_RUN.

The archived manifest records the exact code/source/asset identities and
runtime flags. The archive does not contain model weights or prompt tensors.
