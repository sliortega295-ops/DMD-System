# W5 H100: preserved three-arm re-audit, no new GPU experiment

Approximate Grid1000, authorized common H100 bootstrap; not A100 slot/RNG replay, not W6 continuous AR8. Timer: one Student1+Fake5 outer cycle,96rank-qualified consumed samples,16independent logicalslots/8physicalGPUowners.

| Arm | Source | Native s | Allocated GiB | Reserved GiB | Activation policy |
|---|---|---:|---:|---:|---|
| resident_boundary_CPU | d81ebb2 | 637.204307036 | 59.670 | 63.125 | checkpoint_boundary_cpu |
| FSDP_control | 9bfd3ed | 638.522138278 | 41.603 | 45.516 | none |
| FSDP_GPU_attention_cache | 9bfd3ed | 539.718383034 | 64.116 | 71.082 | none |

All24rankreceipts rechecked: perphysicalrank forward48/12/2, grad2/10/0, backwards2/10; Student1/Fake5 optimizer/schedulercommits, AdaLNparams0/dropped13,010,457,600,1200replayhits/0miss and600wraps. Rank-qualified source sample identities exactly match acrossarms. All finitechecks occur aftertimer; no newhostOOM; original ownedteardowns empty.

Named Resident requires600CPUboundarycopies/rank and is NOT a pure GPU no-offload baseline. Its d81source is a separately registered capacitycontract, not the9bfdsource of bothFSDParms. PureResident Fake2backwardOOM remains preserved. The CPUcopy bytecensus denotes requestedlogicalbytes, NOT measured PCIe D2H/H2D.

FSDPcontrol / GPUattentioncache single-trial ratio 1.1830653880804722x (same9bfdsource). Full-model cross-arm numericstate parity NOT_RUN; formal strict speedup eligibility FALSE. No median/CV/significanceclaim. Hostpeak is10second cgroup.current sampling, not perrankRSS; driverfree minimum notcontinuouslymeasured. Native times frozen.054B profile must pass real8ranklatephase andBWcensus gates before attribution; profilewall is not a performance denominator.

Initial R1 postprocessor incorrectly treated profiler statusstring NOT_RUN asboolean; its CPU failure is preserved, no modelrerun/rawmutation.
