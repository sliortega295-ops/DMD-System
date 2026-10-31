# Performance model: quantities that must not be conflated

## 1. Objective and measurement identity

Let T0 be the frozen, accepted baseline latency for one declared workload window,
Tc the latest accepted control in the SAME window, and S* the target speedup over T0.
Then T*=T0/S*, achieved S=T0/Tc, remaining saving=max(0,Tc−T*), and required next speedup=Tc/T*.
Always report time reduction and throughput increase separately. A 1.5× speedup means 33.33% less latency at fixed work/resources; 50% less latency means 2×.

For multiple workloads, freeze the requested aggregate and weights. Do not substitute mean seconds for geometric-mean speedup, or optimize only the easiest row. More GPUs may improve time but is a different resource comparison; also report GPU-hours per valid window/sample. Preserve the original baseline while control advances.

A measurement identity covers source+dirty-diff, resolved plan, assets, semantic contract,
window/update counts, precision, hardware/topology, instrumentation and relevant software.
A filename, commit name, kernel count or process exit code alone is not an identity.

## 2. Observed activity, dependency, and causality

Use four distinct labels:
1. **Summed event duration**: includes concurrent overlap; can exceed wall time.
2. **Activity union**: real timeline coverage for one resource/window.
3. **Non-overlapped activity**: e.g. U(copy)\U(model compute). A diagnostic, NOT a causal marginal saving.
4. **Counterfactual E2E saving**: T(schedule)−T(changed schedule), supported by dependency/resource reasoning or a controlled probe.

Construct only the graph needed to decide the next experiment. Nodes include role calls,
GPU/CPU computations, copies, collectives, joins and version synchronization. Edges cover
true tensor/gradient/version dependencies plus known stream/resource ordering. Model
shared HBM bandwidth, SM, copy engines, PCIe root complexes, NICs, host DRAM and capacity.
NCCL kernels may wait for late ranks; their whole duration is not transferable link cost.
Some communications may use engines not identified by an NCCL-kernel-name heuristic.

Amdahl is a **conditional local model**, not automatic arithmetic on profiler percentages.
For a non-double-counted critical portion t of Tc, local speedup s and added exposed overhead h:

    ΔT = t(1−1/s) − h
    S_vs_control = Tc / (Tc−ΔT)
    S_vs_original = T0 / (Tc−ΔT)

With h=0 and t fully removable, the conditional ceiling is Tc/(Tc−t).
Use it only while other durations, critical dependencies and contention are unchanged.
With parallel/near-critical paths, recompute the DAG: shortening one path may expose another.
Do not add independent candidate gains. Removing an off-path operation can still help by
freeing a contested resource; conversely non-overlapped work may have slack and save nothing.

`scripts/dmd_perf.py dag` evaluates an explicitly supplied acyclic fixed-order model.
It does not discover dependencies, optimize a schedule, account for omitted capacity/links,
or establish a hardware upper bound. Never call its output an actual measured trace result.

## 3. Physical relaxation versus practical target

For an operation with required FLOPs F, required traffic B and matched capacity Pcap/BWcap,
its optimistic duration is at least max(F/Pcap, B/BWcap, mandatory_dependency_floor).
At system level combine:

    T_physical_LB = max(longest path of valid node lower bounds,
                        each shared-resource required demand / capacity)

The capacities must match hardware, dense/sparse execution, precision, topology, direction,
and resource multiplicity. Per-GPU bandwidth is not aggregate PCIe/NIC bandwidth. Required
traffic must exclude what the candidate could legitimately cache/eliminate. If a bound
relaxes a real constraint it can be weak; never add overlapping resource bounds.

- Certified/defensible physical LB > target: this route, under its stated assumptions,
  cannot close the target alone. Change the cost model or its role in a portfolio.
- Matched primitive benchmarks: evidence for **practical achievable efficiency**, not a
  proof of physical impossibility.
- A scheduled candidate latency interval: forecast, not a lower bound or guaranteed gain.
- A measured viable configuration: achievable incumbent, not the optimum.

Do not conclude the whole research goal is impossible from a weak prototype or a practical
forecast. Say which route has been ruled out, which assumptions matter, and what remains unknown.

## 4. FLOP accounting and MFU for compound DMD

Count at the **logical workload-window** level. Inventory every required call and shape.
G no-grad rollout, selected differentiable rollout, Real/Fake Score, backward, Teacher/GAN
and EMA frequencies vary by workload. LoRA-only training avoids many weight gradients but
may still propagate gradients through a large frozen backbone. Do not apply vanilla 6NT to
all calls, multiply logical TP work by TP ranks, or use trainable parameter count as total F.

For homogeneous dense-BF16 compute with a declared convention:

    MFU_system = F_useful / (T_window × Σ_reserved_GPU P_peak_dense_BF16)
    HFU_system = F_executed / (T_window × Σ_reserved_GPU P_peak_dense_BF16)

F_useful excludes activation recomputation; F_executed includes it and measured redundant
work. If shared/cached work changes the numerator convention, declare and retain both
reference-work efficiency and actual achieved FLOP/s rather than quietly improving MFU by
counting eliminated work as executed. CPU work belongs in wall time, not GPU FLOPs.
Reserved GPU capacity stays in the denominator during role-idle windows.

Mixed-precision heterogeneous kernels need per-precision/resource breakdowns; no single
BF16 peak meaningfully describes FP8, TF32, FP32 and CPU work together. Explicitly label
any normalized capacity proxy. No advertised sparse peak without actual sparse computation.

For locating idle-versus-compute loss, also measure:

    A = Σ_model-compute GPU active union seconds / Σ_reserved_GPU window seconds
    U_active = F_executed,covered / Σ_i(P_peak_i × t_model-active,covered,i)

`U_active` is an **active-model-kernel-window utilization proxy**, not a standard second MFU.
Use the same covered operations, devices and precision in numerator and denominator.
Prefer role-level achieved FLOP/s and representative kernel roofline too. Low system MFU
with higher U_active suggests scheduling/exposure opportunities, but does not prove them;
large kernel active coverage with low U_active may indicate memory/launch/stall limitations.
High occupancy or busy GPU is not proof of a good implementation or a saturated system.

A counter that misses custom CUDA/FlashAttention/backward is partial. Record unsupported
ops and estimate bounds or use manual shape counts; do not invent a percentage. Run expensive
FLOP discovery once per changed workload shape in a separate diagnostic run, never each timed run.
Values above plausible limits trigger a scope/accounting audit, not clipping to100%.

## 5. Opportunity portfolio and value of information

Candidate record: mechanism; M/L/Q/K layer; affected calls; source evidence; low/central/high
GROSS saving; low/central/high added exposed cost; semantic/capacity risks; engineer-hours,
GPU-hours, full-runs and queue/wall time; alternatives; dependency/enabler relationship.

Net interval = [gross_low−overhead_high, gross_mid−overhead_mid, gross_high−overhead_low].
Do not fabricate probabilities. First compare material E2E leverage and confidence, then
cost/risk and feasibility; show savings per engineering hour only when estimates are credible.
Use a cheap discriminating probe when uncertainty could reverse the ranking. An unknown
large opportunity should not be ranked below a tiny known win merely because it is unknown.

Check route portfolios against remaining gap without double counting. A small memory-saving
kernel can enable residency/less recompute and deserves an ENABLER probe with a capacity
calculation and downstream route, even if its direct latency saving is tiny. For gate admission, mark its estimate.scope=enabled_portfolio and forecast the joint
mechanism including the enabler cost. Its direct kernel-time gain may be zero or negative;
the joint net return, capacity evidence and downstream validation must justify it. Do not
sum the same enabled benefit into both the enabler and downstream candidate. Ordinary tiny
kernels remain deferred while material M/L/Q uncertainty can be resolved cheaply.

After a promotion, re-profile only the relevant changed boundary and refresh the portfolio.
Amdahl fractions and old cost forecasts are stale after major topology/lifecycle changes.
