# Construction and state-retention contracts

These explicit opt-ins change physical ownership/lifetime, not DMD mathematics.
They do not make CPU fixtures evidence of large-model CUDA fit or speedup.
Source, capacity, gradient, RNG and checkpoint gates remain mandatory.

<a id="frozen_teacher_construction_handoff_v1"></a>
## Frozen Teacher construction handoff v1

Schema/opt-in: `model.parameters.construction = "cpu-fsdp2-frozen-teacher-v1"`.
This is a narrow experimental CPU construction/lifecycle contract, not a
large-model, CUDA, pretrained-quality, performance or release qualification.

### Supported request

The registered FLUX.2 klein Base adapter uses its existing pinned modern-flow
stack, real local native assets, same-family Student/Fake LoRA and frozen
unadapted Teacher, FP32, equal rank-local B1, and the existing common flow-DMD
Program. Launch and initialize a full-world CPU/Gloo group with at least two
ranks; use explicit `trainer.distributed.backend = "gloo"` and
`data.sharding = "rank_strided"`.

The only execution selection is:

```json
{
  "mode": "manual",
  "implementation": {
    "backend": "torch-callable-replicated-v1",
    "reduction": "fsdp2_reshard",
    "bucket_cap_mb": 0,
    "local": {
      "backend": "torch-callable-resident-v1",
      "share_immutable": false,
      "checkpoint_bindings": []
    }
  }
}
```

No automatic-before-construction, saved plan, calibration/trial clone,
CUDA/NCCL, HSDP, keep-forward, role pool, immutable sharing, mixed precision,
full-trainable construction root, LoRA Teacher or other family is admitted.
The existing per-component parameter budgets and asset/hash admission remain
mandatory. This recipe does not supply a new capacity budget or relax an
existing native owner, source, purity, cache or generation gate.

### Ownership and construction order

1. The existing `factory.admit_construction` hook validates the request before
   assets, then owns the factory's optional `construct(factory, parameters)`
   stage. Factories without that method keep the original call path.
2. Media, dense Student, Student LoRA, dense Fake and Fake LoRA retain their
   original order. Teacher is constructed last through the unchanged internal
   `_load_native_frozen_shards` seam: meta native Parameters, native FSDP,
   `to_empty`, validated local safetensor slices. There is no dense Teacher
   constructor, full Teacher payload or Teacher parameter all-gather during
   this construction stage.
3. One exact ordered full-world 1D mesh object is retained per live process
   group, including sequential fresh constructions. Equivalent foreign mesh
   objects are not substituted for it. This preserves the loader's exact mesh
   identity checks despite native DTensor caching.
4. The new native child is bound to the actual frozen outer Teacher object,
   its `native` child, actual Parameter/storage/constant objects and values,
   generated/original classes, methods/MRO and native FSDP state. A per-
   construction collective cohort distinguishes separate owners, even with
   identical shapes, weights and source. A metadata flag is not a certificate.
   A forward before adoption expires the new owner before native tensor
   collectives.
5. The existing `ModelComponents` retains an ephemeral
   `sharded_construction` provider, never checkpoint state. It reuses the
   existing registered `dmd_forward`, frozen-root, parameter-shadow and child-
   boundary declarations; no alternate trainer or model-name planner branch
   is installed.
6. The existing `ShardedLifecycle` alone adopts the root once. It does not
   re-shard the already-owned native child, rebind its Parameters or create
   Teacher optimizer/EMA state. Student/Fake optimizer migration and complete
   Student EMA use the existing implementation. Both Teacher outer root and
   native child close to shards at the existing completion boundary.

Caller-pre-sharded roots, copied/forged certificates, root/child/method/
Parameter/storage/constant rebinding, foreign meshes, expired owners, changed
execution choices, cross-rank owner mixes and asymmetric handoffs reject
collectively before Session tensor signatures or any update. Ordinary dense
factories still have to provide dense contiguous Parameters. Closing a provider
expires its own lifetime and removes its registry entry; it does not rebind or
poison an existing optimizer, EMA or published parameter payload.

### Initialization, source and persistence

The construction scope restores caller Python/NumPy/Torch RNG, including on
failure. Student/Fake dense constructors and LoRA draws retain their original
order and values. The omitted dense Teacher constructor would have consumed
additional construction RNG: **its internal RNG stream is not declared
equivalent**. The opt-in has a distinct model metadata/config identity; no
training RNG change or silent old-checkpoint reuse is allowed.

Generated FSDP subclasses are not generally accepted as inspectable sources.
Only a live construction certificate may bind the exact issued native class,
original model class/source, actual native FSDP class definition, methods/MRO
and native state. Foreign/impostor generated classes remain rejected. Dense
model-code serialization is unchanged.

An explicit manual implementation skips unused default inventory only AFTER
collective native-shard certificate validation. Ordinary dense manual,
reference, automatic and native atomic-owner paths retain their original
inventory, source, descriptor-purity, pairing and capacity admission. The
selection is still validated by the existing Session; no reference fallback
or automatic candidate is added.

Same-topology restore reconstructs a fresh controlled Teacher, installs the
existing saved lifecycle, then uses the existing staged local-shard checkpoint
codec for modules, Adam, schedulers, full EMA, clocks, data position and training
RNG. The ephemeral certificate/cohort is renewed, not deserialized or reused.
Failure before publication discards/expires only that newly constructed owner.

### Evidence boundaries

`tests/test_training_sharded_construction.py` includes actual tiny FLUX.1
compatibility probes and separate actual FLUX.2 cases. Missing FLUX.2 is an
explicit deferred target, never replaced by FLUX.1. The production FLUX.2
factory test exercises real native media/data, public `train`, and fresh-
process resume on the pinned Torch 2.10 modern-flow lane; it is not satisfied
by the local fixture-role compatibility tests.

Student/Fake are still dense at factory time and their initial full-parameter
EMA is still dense. Other native lifecycle integrity signatures may still
materialize full tensors. Thus **three-role large-model construction,
bounded end-to-end integrity/export, LTX construction, CUDA/GPU memory,
throughput and quality remain unqualified** by this contract.

<a id="wan_rollout_shared_scratch_v1"></a>
## Experimental Wan Generator independent graph scratch pool

Default `isolated.v1` remains unchanged. The opt-in typed strategy
`rollout_graph_pool_policy=shared-scratch.v1` requires the original Resident
incumbent bundle and exactly retained Critic blocks26–29. Existing tail6/tail8
OOM evidence is not overridden and those combinations are rejected.

Only independent no-grad Generator rollout graph temporary storage is shared.
Capture uses one stream and one explicit pool. Replay is restricted to one host
thread/device/CUDA stream and rejects grad-enabled or nested-capture execution.
Original inputs are copied into each signature's static inputs; each output is
freshly cloned before another replay can overwrite graph-private storage.
Critic forward/backward graph pools, reduction, optimizer, EMA, LoRA precision,
parameters, sample/RNG streams, update versions and workload calls are unchanged.

An actual receipt must bind the selected pool policy and verify every graph's
real pool identity. No memory or speed claim follows from CPU policy tests.
Requires a real original-model CUDA output/input/version/lifetime primitive
before matched native two-cycle testing. Default main/profile settings unchanged.

<a id="wan_tail6_retention_v1"></a>
## Wan Critic tail6 autograd retention, experimental v1

This optional policy admits exactly Critic24–29 for original autograd storage.
It does not change model, precision, loss, RNG, updates, EMA, application calls,
input trace or globalB16. Existing default retain4.v1 still rejects >4 blocks;
tail8.v1 remains distinct and its actual CUDA OOM evidence is retained.

Only explicit manual native strategy mode is allowed, never inherited old
calibration or an automatic planner optimum claim. Compare on the same source
against retained26–29, two continuous cycles, C2 only. Keep original node owner,
2GiB allocated margin and64GiB host reserve. Actual capacity/semantic failure
ends this bounded direction; do not retry other block counts by changing IDs.
CPU output/gradient/Adam/RNG tests are synthetic and do not prove realGPU parity.
No canonical/default configuration is changed. Pending target/GPU validation.

<a id="wan_tail8_retention_v1"></a>
## Experimental Resident Critic tail8 state retention

Explicit `retention_policy=tail8.v1`, exact blocks22..29 only. The original
retain4.v1 default, dictionary bytes, IDs and rejection of >4 blocks remain.
Same audited normal-autograd restoration wrapper as retain4; no new kernels,
math, checkpoint/data/dtype/global-batch/optimizer/RNG changes. Application
calls remain unchanged; only eight Critic blocks avoid checkpoint recompute.
No automatic search admission or historical cost reuse is enabled.

The earlier actual retain4 pair20.324208→20.002455s and allocated23.409→26.692GiB
motivate one bounded tail8 pair, after covering all six workloads. Capacity is
only a prediction (roughly30GiB allocated), not a fit claim. Original2GiB
allocated headroom and64GiB startup host reserve remain. Existing2 W1 launches
stay charged; at most2 additional same-source control/child native C2 launches
within W1cap4 and campaigncap24. Full-state comparison retains REVIEW and the
original thresholds; no strict/long-trajectory equivalence claim.
