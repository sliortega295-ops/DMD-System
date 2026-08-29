# Iteration 355: Critic selective Flash checkpoint replay

## Candidate

- Code HEAD: `e75c80e95198690ee070a8d7f77c6db9bfb85ce9`
- Parent substrate: C4 (`143.206713110 s`)
- Single changed mechanism: on Critic blocks `0,2,...,38`, save the exact
  pinned FA2 self/cross-attention result during the original non-reentrant
  checkpoint forward and reuse it during replay.
- Unchanged: world16 FSDP2, GAN-on steps 1000--1004, B1/global-B16, physical
  Score/DiT calls, trace/RNG, loss, precision, optimizer/EMA ordering, residual
  residency, local FA2 identity and timer boundaries.

## Gates

- Focused tests: `99 passed`.
- Validate-only:
  `iter355_gan_cuda_selectiveflash20_validate_r1_20260829`, 16/16 PASS,
  summary SHA-256
  `54182ef75715792f9373bb9c5a2cc49df5a37b9fd0c8b678e480370f7da41399`.
- Correctness/capacity:
  `iter355_gan_cuda_selectiveflash20_correctness_r1_20260829`, 16/16 PASS,
  G1/C5/EMA1 and final G201/C1005/EMA801, summary SHA-256
  `6ee1dc37e8360c330739343798fc6dcae5a5947a65f53ebc63cf4d96cbd76159`.
- Every rank recorded 11 Critic checkpoint invocations per block, 220 selected
  contexts, 440 original FA2 saves, and exactly 440 replay reuses.  No physical
  block or Score call was bypassed.
- No CUDA/cgroup OOM.  Peak allocated was `24.965996 GiB/rank`; peak reserved
  reached `38.533203 GiB/rank`, and the smallest end receipt reported only
  `0.012085 GiB` driver free.  This is a capacity PASS but not enough headroom
  to justify automatically doubling retention to stride 1.

## Formal timing

- Run: `iter355_gan_cuda_selectiveflash20_timing_r1_20260829`
- Summary SHA-256:
  `567a2caf438bf80924e1b504078d6588b565aca5579fef6178b8a4acc951e004`
- Unprofiled Full5: `142.295341636 s`
- Versus C4: `-0.911371474 s` (`-0.6364%`)
- Versus C0: `-30.719141368 s` (`-17.7552%`, `1.215883x`)
- Requested threshold: `121.110138103 s`
- Remaining gap: `21.185203533 s`

## Decision

Promote C5 as the fastest correctness-qualified substrate, but classify the
gain as small.  The memory receipt rejects an unmeasured stride-1 expansion by
default.  The next action is a fresh diagnostic profile of C5 and an
evidence-driven kernel/system candidate; exposed-NCCL-only tuning and known
packing failures cannot close the remaining gap.
