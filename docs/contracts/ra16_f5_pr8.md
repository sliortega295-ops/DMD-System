# Frozen contract: RA16-F5-PR8

`RA16-F5-PR8` is one bounded successor to the failed `RA16-F5-PR4` admission
test. It is not a rename or repair of PR4.

## Frozen workload

- two nodes, 16 ranks, one complete Route-A flow per rank;
- official 384x640, latent `[1,16,9,48,80]`, BF16;
- rank16 operation trace v3 and checkpoint-10;
- continuous steps 10--14;
- ZeRO-1 with CPU optimizer offload, MB1/GAS1/global batch 16;
- one Generator update, five Critic updates and five EMA updates;
- `G2/C10/EMA10 -> G3/C15/EMA15`;
- default no-GAN/no-Reward/no-VAE Stage-3 branch.

## Only lifecycle change from PR4

Step10-A Generator uses window-1 frozen-block streaming for exactly:

```text
[0,24,1,5,8,25,10,32]
```

Score has no block streamer. During each complete no-grad Critic Generator
rollout, those eight externally managed blocks are phase-resident: one balanced
phase enter/exit, not one reload per physical Transformer call. The existing
Generator-selected/Critic-train saved-tensor policy remains enabled with its
32-MiB threshold. No allocator-cache recovery is enabled.

## Functional and memory gates

The single preregistered functional attempt passes only if all 16 ranks:

1. replay the exact trace and reach checkpoint-15;
2. complete the required update/version sequence;
3. emit no Score block-stream events;
4. stream only the eight frozen Generator block indices;
5. balance all phase-resident handles and finish with zero live handles;
6. add no CUDA or cgroup OOM.

The same run may be analyzed offline using already-emitted phase, block-stream
and saved-tensor events. That analysis must be per-rank and storage-aware. It
may identify candidate low-reuse memory, but it must not label unclassified
allocator bytes as redundant or turn this one functional run into a speedup
claim. Full allocator snapshots require a separate diagnostic run.

## Refutation

Any Step10 backward OOM, trace/version mismatch, missing update/EMA, lifecycle
leak, unexpected streamed block, or invalid/missing rank evidence refutes this
candidate. Timing remains `Not Run` regardless of functional success.
