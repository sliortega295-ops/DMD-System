# Optional round note

Append this to the project's WORK_LOG, or keep its existing format. A few
sentences are enough; unknown estimates may stay unknown. No approval is implied.

- **Why this change:** current bottleneck; useful prior implementation or failure.
- **What changed:** source/config or patch; expected benefit and main risk/cost.
- **What happened:** actual command and output paths; checks and their coverage;
  all measured times, or explicitly not measured. Approximate effort is enough
  unless the user/project requires exact accounting.
- **Decision:** keep, optional, promising/unresolved, revert or incomplete; what
  new evidence would justify more work. Identify the best verified version.

At handoff, add the remaining checks and any stop/resource limit. Saving an
incomplete experiment does not require another run or a campaign database.
