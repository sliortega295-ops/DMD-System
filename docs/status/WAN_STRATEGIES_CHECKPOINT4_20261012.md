# Checkpoint 4: original engine factory and native observer binding

The parameterized original-Wan path now binds the selected implementation before
model construction, calling the existing fair worker's original `_build_engines`
and P0 handlers. It selects the actual FSDP2 or GradReadyAsyncResidentEngine,
checks effective chunks/precision/fusion and uses the existing synchronized
observer on the parameterized graph. Original losses, optimizer settings,
adapter switching and sampler implementation are retained.

28 targeted CPU tests passed locally (factory-binding tests, 17 planning/oracle
cases and four original-function argument tests with a stub sampler). These
confirm interface selection, not full model/GPU execution. A hash-bound one-time
workflow publishes the two small native invocation edits and their exact source
provenance; its results must be checked before claiming remote integration.

Next: aggregate complete rank16 calibration/reference artifacts, wire the
original task into the existing native command, and run supported-version CI.
No GPU server connection, model assets or new performance results are available
in this session. Raw native execution is not a numerical parity claim.
