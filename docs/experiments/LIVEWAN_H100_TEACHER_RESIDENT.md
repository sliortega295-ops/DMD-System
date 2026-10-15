# H100 C34 pipeline: Teacher residency probe

Status: CPU-tested candidate, **GPU capacity/parity/timing NOT_RUN**. Do not
promote it over the measured 45.278492 s Resident or 45.937355 s pipeline.

The H100 C34 pipeline currently uses four Generator ranks and four Score
ranks. Teacher is FSDP4/reshard while the strong baseline has all roles fully
resident on all eight GPUs. The observed Score max allocated is 21.562 GiB.
Ideal extra static Teacher storage is 19.961 GiB, not a measured capacity bound.

The existing launcher accepts the optional child-process setting:

```
DISTILLSCHEDULE_HARDWARE_PROFILE=h100-8-b16
LIVEWAN_PIPELINE_TEACHER_PLACEMENT=full-resident
```

Use the existing H100 campaign's one-node launcher, not the legacy two-host
Python controller. This change does not expand that controller's topology.
Default `legacy-sharded` leaves both A100 and existing H100 plans unchanged.
The new choice only changes Teacher parameter placement. It preserves C34
four-wave routing, independent global B16, Student/Critic placement, BF16
Teacher weights, loss, precision, trace, RNG, update order, physical B2 Critic
batching, all-gather-independent model call counts and the existing timer.
No parameter/activation offload, attention backend change or new prefetch.

Run validate -> capacity/correctness or two-cycle trajectory -> existing
comparison -> `build_livewan_two_cycle_timing_gate.py` -> one native C2 timing.
The gate additionally binds all eight actual storage receipts and the
effective plan. A gate for sharded Teacher cannot authorize resident Teacher,
or vice versa. All four Score replicas must contain frozen non-DTensor CUDA
BF16 parameters and zero Teacher FSDP groups; G ranks must not own Teacher.
Unbound generic timing/profile paths are rejected until separately admitted.

Use a new immutable run ID and exact source HEAD for every arm. Existing
strict/review numerical semantics are unchanged; CPU fixtures are not GPU
parity or full-model correctness. Hold owned GPU locks through teardown and
never run while foreign reservations are active. Current W6 R6 queued source
and admission remain unchanged.
