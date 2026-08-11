# Bounded recovery contract: RA16-F5-PR4-EC

`RA16-F5-PR4-EC` is one fail-closed recovery attempt for the measured
`RA16-F5-PR4` Step10-A CUDA OOM. It preserves the entire PR4 algorithm, trace,
checkpoint, topology, ZeRO-1 backend, batch/update semantics, four streamed
Generator blocks `[0,24,1,5]`, Score policy, Critic-rollout phase residency and
five EMA updates.

The only added physical action is the already implemented
`DISTILLSCHEDULE_EMPTY_CACHE_BEFORE_BACKWARD=1`, which calls
`torch.cuda.empty_cache()` immediately before Generator and Critic backward.
The falsifiable question is whether releasing allocator-owned unused segments
can satisfy the 216-MiB allocation that failed with only 102--210 MiB driver
free while 205--371 MiB was allocator-reserved but unused.

This is not a timing optimization claim. It receives exactly one real Full5
attempt. If the same Step10-A OOM recurs, this path is closed; no further cache
or allocator patch escalation is allowed. Any future increase in streamed
blocks must receive a new experiment ID.
