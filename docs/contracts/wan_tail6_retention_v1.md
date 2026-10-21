# Wan Critic tail6 autograd retention, experimental v1

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
