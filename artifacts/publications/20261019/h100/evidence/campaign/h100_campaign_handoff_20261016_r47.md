# R47 — SSH recovered; W2 completed; W1 canary import repaired; foreign GPU use blocks launch

UTC 2026-10-16T02:53:42.900Z. Campaign remains incomplete. No controller restart, foreign signal, shared environment change, push or PR.

## New measured W2 result
Exact source8fc70e8ce9f6e0243604812120119c9f4c078e55, same-head retained-G control versus only internal rollout-fence elision:

| Arm | Native Full5 |
|---|---:|
| Retained-G control | 95.534631493s |
| Same + interior fence elision | 94.311432633s |

Observed single-pair saving1.223198860s, ratio1.012969783x. Actual both validation/correctness/native receipts PASS,8physical/16independent logicalslots, G1/C5/EMA5, finalG3/C15/EMA15, no new CUDA/cgroup OOM and original owned teardown empty. Retained41units restore inbothB1scopes, no unitsremainfull; pre/post retentionallocatorflush and outertimerfences kept.
Full-state numerical parity NOT_RUN; formal claim false. Descriptive ratio to older official-swap138.371948s is1.467181x, crossrevision—not a newplacement-only causal comparison. The legacy status.json remainedRUNNING after completion; actual result+all runreceipts govern.
Small evidence: artifacts/helios_h100_rollout_fence_probe_20261015_r1/reports/completed_fence_pair_r1/.

## W1 actual cheap failure, not new model/OOM/numeric failure
At00:20UTC all8rank private-cache canary failed during native package import with OfflineModeIsEnabled: frozen package initialization auto-selectedFA3 onH100, attempted a Hub request while offline. No model loaded.44realGPU tests+9nativewidth1536forward/gradientlayouts had already passed at3bb2e8, retained.
The canary now reuses the existing production configure_offline_helios() binding. Additional diagnostic fix unwraps the unchanged torch.compile-disable decorator before pinning the native source identity; initialCPUcheck incorrectly hashed the Torch wrapper file and is preserved.
Current local+remote clean source97b8ecc9cf7747d93619acad1a6083f98faae084, two small scoped local commits only.10focusedCPU helper tests pass; selected remote54tests=18PASS+36CUDA explicitlyskipped. A fresh real nativeRMSNorm import with no visibleGPU PASS, nativefileSHA25b8bc708e0b7c1f41d49b27e2a645817fb603423390d8d83a689c8777501cb7, CUDA_initialized=false. Worker/successloop/operator source12files byteunchanged.
New053B is CPUadmitted; GPU NOT_RUN. Reuse the hash-bound44GPUtests/9layouts, then exactlyone remaining cheap8GPUcompilecanary, fresh two-armcorrectness, original160rollout/sampled-state gate, onlythen single nativeC2 perarm. Expensive norm attempt2/max2, max4remaining modelarms; no budget reset from canary/import fixes.
Path: artifacts/wan13b_h100_norm_graph_20261015_r1/norm_offline_import_recovery_r3/.

## Actual resource blocker after reconnect
R6 PID683126/startticks644241187 is alive, held on053 failure review; no ownGPUworker remains. At02:52:44Z NVML listed15contexts (rollingaround16), roughly1GiB each and all8GPUs busy. A separate visible StreamingSparsev02 workflow under /home/jiangzhikun/Self-Forcing/ has16evaluationworkers, CUDA_VISIBLE_DEVICES explicitlytwo perGPU, supervisor776194/workflow776274 and auto-resume watcher. These are NOT ourR6 tasks. AllLinuxusersshowroot, so actual human identity NOT_ESTABLISHED; NVML↔OSPID exact join stillUNRESOLVED acrossnamespaces. Read-only evidence at artifacts/gpu_ownership_readonly_20261016_r1/ownership.json. No signals sent.
Do not release053 disposition whileforeignGPUcontextsremain: R6's nextaudit would correctly rejectthem. Keep solecontroller; no duplicate/fakeoccupancy.

## Next authorized serial work
1. On realallGPUrelease, finish053review then053Bcorrectedcanary/normpair.
2.054B W5 second/final capture with actualallworkerStartStop+endnone canaryproven. Keep originalpartial profileFAIL, require8ranklatephase/600BWminimum/copyreconciliation before anyanalysisclaim.
3.055 W4 first corrected profile; aa21GPUworker unchanged, allworkerAPI alreadyexists, onlyCLIflushadapter. Finalanalysis1b exacteventcounts/repeatedranges.
4. Continue evidence-guided optimization and honest remainingmatrixaudit, no automatic wholecampaign completion.
W6 same0419 native matrix308.670470/320.666055/278.886442s retained; FA2<=280met; full-modelFA3 gates remainFAIL/unwaived and MFU notmeasured. W5Grid remainsapproximate, never appliedtoW6.
