# Experimental Wan Generator independent graph scratch pool

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
