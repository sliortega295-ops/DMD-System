# Helios GAN-on: N70-OFF native C2 retime

Run ID: `gan_nomerge_down40_native_c2_world16_20261013_r1`  
Source exact clean HEAD: `ba52a2446b7f090cd31af9504679e68770d32603`  
Branch: `agent/helios-generator-merge-relaxed-numerics-20261003`  
Status: **native timers complete and independently offline revalidated; original post-timer finalization FAIL preserved; strict numerical REVIEW**.

| Item | Native result |
|---|---:|
| C1, warm-up only | 117.716096268 s |
| C2, steady-state max of 16 original CUDA-fenced rank walls | 116.253523015 s |
| Time ratio against matched FSDP2 reference 164.30 s | 1.413291× |
| Time ratio against this round's FSDP2 reference 163.90 s | 1.409850× |
| Difference from historical N70-ON 113.589670249 s | +2.663853 s (+2.345%) |
| Application/finite/version gate | 16/16 PASS, both cycles |
| Actual N70 merge count | 0 on every rank |
| Peak allocated / reserved across cycle receipts | 36.810488 / 38.050781 GiB |
| New CUDA / cgroup OOM | 0 / 0 |
| Optional same-session N70-ON B | NOT_RUN |
| Profiler | NOT_RUN |

## Frozen workload and real no-merge proof

Only `--generator-nograd-merged-execution` changed to `--no-generator-nograd-merged-execution`, alongside its matching process-local OFF/strict environment settings. No source checkout/default was changed. Explicit capsule/shape-qualification rebinding and source/launcher/worker/helper SHA pins are recorded in `manifests/packet_diff_r1.json`, `packet/configs/manifest_A_r1.json`, and `reports/diff_*.patch`.

All original helpers retained, including down40 (body byte-identical), pure pinned SM80 FA3, local CUDA EMA, both residuals cuda_out_of_place, world16 FSDP2 FULL_SHARD, rollout B1, Score packing OFF, no parameter offload/streaming/model swap. Generator persistent-unshard and all accepted epilogue/norm/history/cross-KV helpers remained enabled exactly as the original capsule. Critic official B3 GAN/R1 graph is unchanged; this does not batch Generator rollouts.

Bootstrap logical package SHA: `c66ff180fa4d00c78ba05ffd1826b7e358e0ab38d2f3f68f5f92dad5fe42dd44`. Rank16 operation trace SHA: `d31d4d473a3574b5b39e50ad2b5e72e17aee7baf707130ea0513092553f7cd15`. Original same source runtime/inputs/RNG and initialization were delegated, with no extra RNG calls or tensor value reads added by the no-merge audit.

Actual per-rank observations: two disabled installer calls, two disabled no-grad scope calls, zero enabled installations/scopes, zero N70 window constructions, zero PEFT merge/unmerge calls. All 406 LoRA layers in each model were unmerged before and after both timers; N70 model state was absent. Original source receipt reports `requested=false`, `enabled=false`, `scope_count=0`, `scopes=[]`, `execution_shadow=false`. Once-refresh and merged shadow were not enabled/present. The existing source `merged_execution_receipt_errors(...requested=False)` gate was executed on the real saved receipt during offline revalidation and returned no errors.

Down40 is independent of N70 and worked: 800 original Down saves and 800 replay reuses over two cycles, 20 graphs, 10 ordered C updates, no extra FA3 calls, no added copies/fences/collectives. Original source FA3 public call delta=9440 and original checkpoint counters match their pinned exact expectations.

Per cycle: G forward31, C forward14, G backward1, C backward10, G optimizer/commit1, C optimizer/commit5, EMA1. Final physical versions G4/C20/EMA802, logical G202/C1010/EMA802. Three grad-enabled G pyramid selections and all finite loss gates PASS on 16 ranks. This is the original GAN contract, not the No-GAN EMA5 contract.

## Important failure and numerical boundary

My added **post-timer** observer used `result['policy']` instead of the actual schema key `result['p0_policy']`. This raised KeyError only after both source timers, all updates, the source native result validator, FA3 counters and down40 completion checks had finished. All 16 raw top-level receipts therefore remain FAIL and both node owners retain their failure status; none was rewritten or relabeled PASS.

An independent CPU-only revalidation verifies the exact KeyError stack, prior `native_completed=true`/completion-drain witness, both cycle PASS/finite/census/version records, timer validity, source no-merge receipt, actual negative-guard counters, down40 replay, per-rank argument hashes and clean teardown. This supports reporting the **completed native C2 measurement**, but it is not a successful end-to-end runtime-finalization claim. `scripts/fixes/nomerge_audit_schema_fix_r2.py` contains the one-key fix and a CPU regression proof; it was NOT used to rerun GPUs.

Strict paired full-state numerical equivalence of the inherited FA3 vs FA2 path remains **REVIEW / NOT_PROVEN**. Finite/census, CPU synthetic tests and negative merge guards cannot replace that numerical admission. No threshold was relaxed. `formal_speedup_eligible=false` remains false. The two ratios above are native C2 time ratios, not promotion to a validated paper claim. Historical N70-ON delta is descriptive, not a same-session paired estimate of N70's isolated cost.

Two node cleanup checks PASS: all16 GPUs idle, both locks reacquired/released, worktree clean, cgroup OOM delta0 on both nodes. Node1's historical oom_kill=3 was retained; no new kill occurred. Exactly **one** GPU experiment was launched; optional B and all other experiments NOT_RUN.

## Read-only Excel recommendation

Workbook `/home/lyy/Downloads/DMD_System_MLSys_Paper_Tables.xlsx` was read only to locate cells; before/after SHA identical. Main first sheet, GAN-on row:

- `C8` (DMD-System seconds): 113.59 → **116.25**.
- `D8` (ratio to 164.30): 1.44643 → **1.413291**.
- Annotate `N70-OFF; completed timing / post-timer audit error; numerical REVIEW`, rather than marking formal correctness PASS.
- Secondary summaries `D21` and `D36` also contain113.59; if manually updating, use the same 116.25. `G36` total saved time becomes 48.046477 s. Do not invent a new state/scheduling causal split from this retime.

Excel was **not modified**; no push, no main/default/environment changes.

## Evidence

Local persistent root: `/home/lyy/Helios/research_artifacts/DMD-System/gan_nomerge_retime_20261013`. Raw logs, source launch manifests, per-rank receipts and ownership audit remain in remote persistent root `/share/project/shuchenweng/yongyan_liu/research_artifacts/DMD-System/gan_nomerge_retime_20261013/packet/ownership_r1/runs/gan_nomerge_down40_native_c2_world16_20261013_r1`. Small native/argv/backend/no-merge receipts are under `receipts/`; report, exact timing CSV, preserved failure readbacks and offline revalidation are under `reports/` and `scripts/`. Setup and cleanup are outside the source C2 timer. No Nsight wall was used.
