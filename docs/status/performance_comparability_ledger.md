# Performance comparability ledger

This file is the fail-closed authority for which preserved DistillSchedule
measurements may form a speedup pair. A functional PASS or a valid standalone
measurement is not automatically a comparable performance denominator.

| Evidence | Contract/status | Comparable use |
|---|---|---|
| Isolated Critic-only Sequential S1/S2/S3 | trace hash 9e282f4b..., same layout/state/transport; later same-layout numerical contract binding PASS; 354.892/365.309/348.746 s | May compare only with the corresponding isolated Pipeline P1/P2/P3 |
| Isolated Critic-only Pipeline P1/P2/P3 | same contract; 258.777/254.510/270.570 s | Median 258.777 versus 354.892 s: 1.371x throughput-equivalent speedup, 27.08% makespan reduction, isolated Critic-only scope only |
| RA16-F5-GLP-SCORE4 | world16/MB1/GAS1/global16, checkpoint-10, steps 10--14, trace v3, ZeRO-1 CPU optimizer, Generator g12 plus LP-state phase offload, A32 activation offload, Score tail blocks 36--39 streamed; functional PASS; one diagnostic-instrumented run: 404.041535 s | Valid official-shape legacy-v3 Full5 standalone result. May be compared only diagnostically with the same-v3 historical all-40 reference: 41.33% shorter / 1.704x throughput-equivalent. Formal repeated paired speedup is Not Run; never compare with strict-v4. |
| Historical exact-B0 all-40 reference | world16/MB1/GAS1/global16, checkpoint-10, steps 10--14, trace v3, Generator and Score all-40 shared-immutable streaming window 1, A32 activation offload, ZeRO-1 CPU optimizer; 688.725684 s | Diagnostic denominator for the legacy-v3 GLP-SCORE4 observation only; it is not a strict-v4 baseline and is not a repeated formal trial. |
| Current strict-v4 Step10-A | official-normal-path trace hash 654d4901..., DP8/MB1/GAS2/global16; about 123 s standalone low-overhead observation | No matched optimization/baseline pair yet |
| Historical Step15-A 48.551620 s | different control/gradient lineage and other contract differences | Historical anchor only; never a strict-v4 denominator |
| RA16-F5-FS40 Attempt005-002 | world16/MB1/GAS1/global16, trace v3, diagnostic Full5 767.167 s and 153.433 s weighted | Functional legacy-v3 typed revalidation only; not comparable to strict-v4. Its 767.167 s also must not replace the frozen exact-B0 688.726 s denominator without an explicitly paired timing contract. |
| Route-B continuous Full5 correctness Attempt003 | strict-v4, Step10-A global16 but target Critic DP12/global12, detailed correctness instrumentation, 344.846 s | Functional feasibility only; not comparable to Route-A and not formal timing |

## Current claim boundary

There is exactly one preserved **formal repeated** causal timing comparison:
isolated Critic-only Sequential versus Pipeline under their shared frozen
contract. There is also one preserved legacy-v3 Full5 **single-run diagnostic**
comparison: `RA16-F5-GLP-SCORE4` at 404.041535 s versus the same-v3 historical
all-40 reference at 688.725684 s. The latter is useful mechanism evidence but
is not a formal speedup because no randomized repeated paired trials were run.

There is no matched official-normal-path strict-v4 Step10-A or Full5
baseline/candidate performance pair yet. A future official Full5 comparison
must freeze the same strict-v4 trace, logical checkpoint, 16 logical samples
per compared update, precision, loss, optimizer/version/EMA semantics and
timing mode on both sides. Correctness repairs such as trace synchronization
are never credited as speedup.
