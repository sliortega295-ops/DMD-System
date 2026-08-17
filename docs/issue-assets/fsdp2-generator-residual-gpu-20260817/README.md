# Generator residual GPU-resident policy

This bundle records the already completed Iteration 306 No-GAN measurement that
motivates the fixed P0 default in this branch. It is a policy result, not a new
profiling run.

## Policy

The only intended policy change was Generator residual storage:

```text
control/previous fixed path:  Generator cpu_inplace; Critic cuda_out_of_place
Iteration 306 variant:         Generator cuda_out_of_place; Critic cuda_out_of_place
```

The residual is a chunked activation/residual used by the Helios block. This is
not frozen-backbone parameter swapping, FSDP parameter offload, or model
placement migration. The generic `runtime.critic_pool_helios` default remains
unchanged; only the static FSDP2 P0 entry points now default to the validated
GPU-resident policy. Historical CPU-staging controls remain available through
the explicit `--*-residual-storage` flags and environment variables.

## Measured evidence

Iteration 306 ran the real No-GAN world16 Full5 workload on the official
384x640 contract:

| Policy | Full5 | Correctness |
|---|---:|---|
| Generator CPU / Critic GPU, three-run median | 141.546299770 s | 16/16, G3/C15/EMA15 |
| Generator GPU / Critic GPU, one run | 118.546756894 s | 16/16, G3/C15/EMA15 |

The single-run difference is 22.999542876 s (16.2488% lower wall time, or
1.1940x throughput-equivalent). The artifact is strong evidence that Generator
CPU staging is costly. It is not presented as a new three-run formal speedup;
the native source artifact is preserved at:

`/home/lyy/Helios/research_artifacts/DMD-System/fsdp2_p0/iteration_306_generator_residual_gpu_20260817/`

No Nsight capture was used for this comparison, and no new timing was launched
by this commit.

## Default-policy change

The static FSDP2 P0 launcher, controller, worker, and No-GAN/GAN baseline plans
now default both residual policies to `cuda_out_of_place`. An explicit CPU
policy still reproduces the historical control when needed for a bounded A/B.
Existing Iteration 303--305 CPU-residual profiles are historical evidence and
are not overwritten or relabeled.

## Claim boundary

This bundle does not claim that all prior D2H disappeared, nor does it identify
every remaining FSDP/autograd transfer. The current Nsight hardware-metrics
capture lacked GPU-counter tables (`ERR_NVGPUCTRPERM`), so exact hardware-level
residual byte attribution remains a separate diagnostic question.
