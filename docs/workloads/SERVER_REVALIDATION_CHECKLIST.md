# Server revalidation checklist

1. Checkout the exact integrated main SHA and require a clean worktree.
2. Initialize pinned gitlinks recursively and verify their revisions.
3. Fill an assets JSON with absolute asset paths and SHA256 identity files.
4. Run `validate`, then a correctness run. Bind timing to the exact correctness receipt.
5. Confirm 16 ranks, update/version census, OOM counters and native summaries.
6. Only then run unprofiled repetitions. Historical branch timings are not reasserted by this checklist.

Current integration-branch GPU/server status: **NOT_RUN_ON_INTEGRATION_BRANCH**.
