# R48 — Explicit user stop before machine reboot

The user cancelled queueing. This supersedes the R47 continuation instructions.
Do not restart R6, re-enable cancelled entries, or run further GPU work without a new user instruction.

## Actual verified stop
- Original R6 controller PID 683126 / start ticks 644241187: exited, /proc entry absent.
- No own training workers, profilers, or scoped launchers remained.
- No process retained a file descriptor to the three campaign/node/asset lock files at verification.
- Only the exact PID/start-tick/cmdline-qualified R6 controller received SIGTERM.
- The three not-yet-executed queue entries (053B W1, 054B W5, 055 W4) were renamed to .cancelled.json with identical content and SHA256.
- STOP_AFTER_CURRENT persists; auto-resume is disabled.
- No foreign task was signalled. This workflow did not reboot the machine.
- Read-only nvidia-smi query returned exit code 0 and zero compute contexts at the verification instant; this is not a continuing global-idle guarantee.
- All previous successful/failed runs, raw evidence, code, environments, weights, and caches were preserved.
- Local matching automation metadata and local scoped watcher process checks found none. No new automation was created.

## Unfinished work is cancelled, not completed
W1 corrected no-weight canary/norm pair; W5 final corrected profile; W4 first corrected profile remain NOT_RUN.
Their failures, admission records, search budgets, and original source heads are retained for an explicitly requested future resume.
No pending item is relabelled PASS.

## Evidence
Actual verification UTC: 2026-10-16T03:12:43.822615+00:00
Remote: /home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1/artifacts/user_stop_before_reboot_20261016T031200Z_r1/
Local: /home/lyy/Helios/research_artifacts/DMD-System/five_workload_h100_20260930/h100_resident_baselines_20261014_r1/artifacts/user_stop_before_reboot_20261016T031200Z_r1
See manifests/stop_result.json, pre_stop.json, cancelled_queue.json, controller_final_status.json.
