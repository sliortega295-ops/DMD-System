# vLLM-Omni 0.19 Helios latent rollout backend

## Pinned service source

The BAAI rollout service does **not** target vLLM/vLLM-Omni 0.20. Its source
identity is the pair:

```text
vLLM-Omni base tag:    v0.19.0rc1
vLLM-Omni base commit: 191b9a8dbf22fc494b7db34ea5ae04153f44f94d
vLLM family:           0.19
Helios backport SHA:   84c9a914184373a771a5839bd6198d05ba8f981154732fc6f7bfbf5feafb3d99
```

Stock vLLM-Omni 0.19 already accepts `output_type="latent"` and returns
`real_history_latents`, but it still executes `vae.decode()` before returning.
For training rollouts that decode is unused. DistillSchedule therefore carries
one minimal, auditable backport:

```text
third_party/patches/vllm_omni_v0_19_helios_latent_no_vae.patch
```

The patch only places the existing VAE decode/history-video construction under
`if output_type != "latent"`. It does not import the unrelated performance
changes from upstream PR #3800. PR #3800 is provenance for the no-VAE guard,
not the deployed revision.

Prepare a clean checkout with:

```bash
scripts/distillschedule/prepare_vllm_omni_019_helios.sh \
  /home/lyy/Helios/vllm-omni-019 \
  /home/lyy/Helios/vllm-omni-019-source-receipt.json
```

The preparation command fails unless the checkout is exactly the pinned base,
is clean, the patch SHA matches, `git apply --check` succeeds, and the patched
source proves that VAE decode is dominated by the non-latent branch. The
receipt records base identity, patch identity, and patched pipeline SHA256.
Do not hand-edit an installed wheel.

## Why the training runtime does not change

The vLLM-Omni version is isolated behind the rollout adapter:

```text
adapter.capabilities()
adapter.sync_generator(...)
adapter.rollout(...)
adapter.close()
```

Changing the service implementation from the PR-3800 prototype to 0.19 plus a
backport does not change the Full5 schedule, trace, optimizer, Critic, or EMA.
A remote service capability mapping must explicitly report:

```text
upstream_revision = 191b9a8...
compatibility_patch_sha256 = 84c9a914...
vllm_omni_version_family = 0.19
vllm_version_family = 0.19
```

Missing or mismatched source identity fails before the first external rollout.
The service must additionally prove latent output, zero VAE decode, exact trace
replay, hot Generator-LoRA synchronization, and dedicated rollout residency.

## Where the engine is legal

Only post-update, no-grad Generator rollouts move to vLLM-Omni.

### No-GAN Full5

```text
Step10 differentiable Generator update G2 -> G3       native training runtime
Steps10-14 Critic-input rollouts using G3             vLLM-Omni 0.19 service
Critic commits C10 -> C15                              native training runtime
EMA10 -> EMA15                                        native training runtime
```

An isolated critic-only subgraph starting from G3/C11 may serve steps11-14 with
a pinned G3 identity.

### GAN-on Full5

```text
Step1000 DMD + GAN-G Generator update G200 -> G201    native training runtime
Steps1000-1004 Critic-input rollouts using G201       vLLM-Omni 0.19 service
Fake/Real Score and GAN-G                             native training runtime
GAN-D/R1 Critic commits C1000 -> C1005                native training runtime
EMA800 -> EMA801                                      native training runtime
```

An isolated post-step1000 critic-only subgraph may serve steps1001-1004 with a
pinned G201 identity.

The external engine is never used for the differentiable Generator-update
rollout, Score/GAN calls, Critic backward/update, EMA, or the native B1
correctness reference.

## Exact rollout contract

A training rollout returns the BF16 tensor consumed by Critic:

```text
[B, 16, 9, 48, 80]
```

Skipping VAE is necessary but not sufficient. The service must consume the
immutable rank-qualified entry noise, prompt embedding, short/mid/long history,
all frame-index tensors, and frozen pyramid-stage schedule. Its physical DiT
`call_count` and `stage_call_count` are compared with the immutable trace.

After the training Generator commits, the adapter hot-loads G3 or G201. The
sync receipt must prove:

```text
source_trainable_sha256 == loaded_trainable_sha256
```

across all training ranks. A version label without byte identity is not enough.

## First deployment canary

The first real service experiment deliberately does not search parallelism:

```text
TP=1
replicas=1
logical request B1
Score packing off
```

Run five post-update rollout requests, compare every returned latent against the
native exact B1 reference, then execute the unchanged five Critic commits. This
isolates three questions: zero VAE calls, vLLM-vs-native numerical parity, and
G3/G201 LoRA synchronization.

Only after TP1x1 B1 passes should the rollout pool search replica/TP/batching
layouts such as TP1xN, TP2xN, or TP4xN. PP is not part of the current rollout
service canary.

## Correctness gate

Every external row is compared against native exact B1 with:

```text
finite and same shape/dtype
mean-relative <= 5e-3
normalized L2 <= 5e-3
cosine >= 0.9999
max-absolute <= 5e-2 + 1e-2 * reference peak
```

The ordinary Full5 audit remains active for Generator/Critic gradient shards,
final trainable parameters, and Adam moments. Timing requires a pinned passing
correctness summary.

## Deployment environment

The training launcher expects a thin client to a dedicated rollout service:

```bash
export DISTILLSCHEDULE_VLLM_OMNI_ADAPTER_FACTORY='my_client:create_adapter'
export DISTILLSCHEDULE_VLLM_OMNI_ENDPOINT='grpc://rollout-service:50051'
export DISTILLSCHEDULE_VLLM_OMNI_POOL_ID='helios-rollout-pool'
export DISTILLSCHEDULE_VLLM_OMNI_DEDICATED_POOL=1
export DISTILLSCHEDULE_VLLM_OMNI_EXPECTED_REVISION='191b9a8dbf22fc494b7db34ea5ae04153f44f94d'
export DISTILLSCHEDULE_VLLM_OMNI_EXPECTED_PATCH_SHA256='84c9a914184373a771a5839bd6198d05ba8f981154732fc6f7bfbf5feafb3d99'
```

Each training rank may own a client, but production deployment is a shared
rollout pool, not one full Helios engine per training rank. The in-process
executor in this repository is a reference adapter, not the world16 production
deployment.

Training-node Nsight sees client wait and returned-latent transfer only. Kernel,
TP, and request-batching behavior must be profiled separately on the rollout
service, and performance results must disclose the extra rollout GPUs.

## Claim boundary

The 0.19 source/backport contract is implemented and CI-testable. A real
vLLM-Omni 0.19 service on the A100 cluster, hot LoRA synchronization, numerical
parity, throughput, and end-to-end Full5 speedup remain `NOT RUN / NOT CLAIMED`
until cluster evidence is collected.
