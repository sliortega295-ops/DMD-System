# Native PP graph retirement

The native pyramid solver can execute a selected grad forward, followed by
a no-grad denoising update at the next scale. The latter severs the earlier
autograd graph. Application call and grad-forward counts still include that
forward, but native backward must not invent a gradient for it.

The external-loss PP schedule previously retained this unreachable inner
graph and failed the pending-backward cycle gate. Its outer autograd context
is now observed with a weak reference. Only after that context is destroyed
can the corresponding stage cache be retired. A live or unregistered context
is never retired. No additional backward, collective, parameter update or
model call is issued. Discards and completed backwards are separate receipts.

A real CPU/Gloo PP2 regression checks output/parameter-gradient parity through
three selected forwards with one native no-grad cut; a live graph remains
pending. Original chained-gradient tests remain in force. Target GPU status
is pending until the new exact-head world16 receipt passes; the original
failed b222 artifact is retained, not rewritten.
