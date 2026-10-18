# C53: selected Critic block autograd retention, default off

Parent: exact frozen C51 plan and all its dependencies. This candidate does not
combine the C52 frozen FFN-down reuse change. Canonical C51/C52 rejection cases
remain unchanged; only the versioned C53 child may select this policy.

Workload: world16, Generator8 + Score8, 368x640x81, actual global B16,
continuous sigma, G1/C5/EMA0 per cycle, two native consecutive cycles. The
sample, trace, bootstrap, seed, precision, loss, optimizer and version rules
remain those of C51. No merged LoRA weights, CPU offload or new kernel.

Critic blocks26--29 execute the same original block function with native
non-checkpointed autograd. Blocks0--25 keep the same pinned FA3 SAC checkpoint.
All 30 blocks still execute their original forward once per grad DiT. Retained
activations are owned by that update's graph and released by normal backward;
there is no cross-sample or cross-version result cache.

For each Score rank over two cycles, the runtime must observe:

* 10 grad DiT calls, 300 original block calls (10 per block);
* 260 checkpoint scopes, original SAC FA3 calls, and replay cache reads;
* 40 directly retained block forwards and 40 backward hook observations;
* failures=0, exact selected block histogram, original application/update census.

Generator ranks are NOT_APPLICABLE. Canonical C51 still requires 300 replay
cache reads; its census is not reduced. Child data cannot use an old C51/C52
conditional gate. The node header, worker and summaries bind the same source,
variant, arm, asset hashes, seed, shape and effective plan identity.

CPU proof uses a scaled real custom operator, actual SAC/non-reentrant
checkpoint, original module hooks, backward, RNG and Adam. Synthesized rank
receipts in chain tests are SYNTHETIC_TEST_ONLY, never GPU evidence. Target
Torch2.10 CPU checks, complete world16 qualification, capacity and numerical
comparison are mandatory before a matched unprofiled C2 pair. Numerical REVIEW
is not strict equivalence. GPU execution: NOT_RUN at source preparation.
