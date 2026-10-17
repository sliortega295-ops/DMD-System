# Helios No-GAN tail4 retention (experimental v1)

Opt-in implementation policy `tail4.v1` selects exactly Critic blocks36–39.
The historical `bounded2.v1` default still rejects more than2 retained blocks.
The native application/workload contract is unchanged: same HSDP/world16, input
trace/bootstrap, B1/global16, dtypes, G1/C5/EMA5, losses and update/version order.
Selected blocks execute the original module with autograd retained, rather than
non-reentrant checkpoint replay. No detach, weight merge, activation offload,
attention-result substitution, or altered random draw is introduced.

The policy is serialized in candidate ID and assignment, reaches the existing
block retention dispatcher, and is checked through actual native per-block call
receipts. Two cycles require40 retained calls, ten each selected block.
No previous calibration/correctness receipt is inherited. Use same-source
qualification then native two-cycle control retained38–39 vs tail4 candidate.
Original capacity, semantic and numerical admission rules remain unchanged.
No canonical profile/default or main is changed; GPU status Pending validation.
