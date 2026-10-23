# Troubleshooting and evidence boundaries

Start with the configuration, dependency lane and last complete run receipt.
Do not replace the model, data, precision, backend or source pin merely to make
an error disappear. Keep failed output and use a new run directory for a retry.

| Symptom | What to check | Safe next action |
| --- | --- | --- |
| `doctor` reports missing prepared input | Resolved config paths and actual manifest | Prepare/import the required genuine inputs; do not fill missing conditions with random tensors |
| Optional package/import failure | [Environment lane](installation.md#separate-original-model-dependency-lanes), actual import/version and original recipe | Reuse a matching environment; do not combine incompatible Diffusers pins |
| Recipe has placeholders | Required model/source/input/capacity/plan fields | Supply your explicitly owned assets before launch; no automatic download or guessed historical path |
| Source/layout/hash mismatch | Request, installed factory bytes, source index, saved plan and artifact hashes | Use the exact admitted source or create a separately reviewed new configuration/plan; do not edit receipts/goldens |
| Missing shard or `_SUCCESS` | Manifest, every named rank payload and complete-cycle marker | Preserve the partial directory; resume only a preceding complete checkpoint |
| Resume changes topology/data/horizon | Original request, scheduler contract, sample inventory, seed and rank layout | Use a supported same-topology complete-cycle boundary; arbitrary topology migration is unsupported |
| Owner is poisoned after a partial update | `failure.json`, last valid checkpoint and native step receipt | Stop that owner; reconstruct from the last complete checkpoint, not a partially committed in-memory model |
| Video generation rejects a native latent result | Export factory, `generate`/`decode` capability and output kind | Add an actual model decoder, or explicitly request `--output-kind latents`; latents are not generated media |
| Planning selects an unsupported capability | Measured candidate/source identity, effective policy and executed operations | Reject rather than falling back to a fixed recipe or reference backend |
| Peer exits or hangs in a distributed CPU check | First failing rank, process exit/signal, Gloo stage and framework/Torch/Python identity | Collect the original error on all ranks; a barrier/sleep or skipped test is not a demonstrated fix |

## What to retain

A Program run writes configuration/source identity, `effective-policy.json`,
`metrics.jsonl`, `summary.json` and complete checkpoint manifests; failures use
`failure.json` where the owning runtime can publish one. Native workers retain
their own rank/node receipts and actual source/asset/capacity gates. Check the
recipe before assuming a field or capability exists.

Keep the exact command (without credentials), source revision/diff, resolved
configuration, imported runtime, input identities, final versions and raw error.
When sharing a problem, small JSON/CSV/log excerpts are usually enough. Model
weights, full datasets, profilers and optimizer shards should remain in the
operator's persistent storage unless transfer is actually required.

## A known distributed CPU qualification limit

Hosted CI has observed a native child `SIGABRT` with
`terminate called without an active exception`. It appeared in an HSDP parity
test and, on another revision, a metadata-only collective worker. Unchanged local
runs with native observation passed, but did **not** identify the hosted cause.
This is not a verified framework fix, a proven infrastructure flake or evidence
that a failed rank completed its assertions. No test is disabled to hide it.

A useful diagnosis needs the first failing rank's stage and a native fatal-signal
backtrace on the reproducing runtime. Python exit code0 elsewhere, normal local
GDB runs, a different Torch version or an additional cleanup/barrier does not
establish the cause. Preserve the original failed job and source identity.
Published [acceptance reports](../status/FRAMEWORK_ORIGINAL_CPU_ACCEPTANCE_20261022.md)
retain this limit rather than claiming universal hosted-CI success.

## Checkpoint versus inference weights

A training checkpoint includes optimizer/scheduler/EMA, RNG, data progress and
logical versions under the exact restore contract. An inference export instead
contains the declared model state and source/config binding for new inputs.
They are not interchangeable. Full distributed export can require substantial
CPU materialization; tiny CPU qualification does not establish large-model host
capacity. See [training and output contracts](training.md#generate-and-export).
