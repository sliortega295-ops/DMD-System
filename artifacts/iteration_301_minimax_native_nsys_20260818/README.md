# Iteration 301 MiniMax-H3 Native Nsight attribution

Diagnostic-only evidence for the selective checkpoint-boundary-offload Native workload.
The formal Native denominator remains 917.993973 s; the Nsight profile is not a speed claim.

The analyzer and profiling source are already tracked in:
- scripts/distillschedule/analyze_lightx2v_minimax_nsys.py
- runtime/lightx2v_minimax_nsys.py
- src/distillschedule/profiling/minimax_nsys_attribution.py
- scripts/distillschedule/profile_lightx2v_minimax_nsys_world16_node.sh

This curated directory contains the report, attribution JSON, small tables, plot, manifest and research log. Raw .nsys-rep/SQLite files remain in the external evidence directory and are intentionally not committed.

Key diagnostic result: NCCL union 2914.200 GPU-s, 2556.088 hidden by compute, 358.112 exposed; boundary-offload candidate union 244.745 GPU-s, 244.035 exposed; movement-exposed union 502.237 GPU-s. These are overlap attributions, not removable wall time.
