# GAN Nsight Systems capture

Nsight profiling is diagnostic only and never supplies the formal speedup numerator/denominator.

Each node runs one `nsys profile` session around its Accelerate process tree with CUDA/NVTX/OSRT tracing and `cudaProfilerApi` capture range. The training wrapper synchronizes world16 before step1000 model compute, then global ranks0 and8 (local-rank0 on each node) call `cudaProfilerStart`; a second barrier releases all ranks. After every rank completes the step1004 Critic commit, another world16 barrier executes before ranks0 and8 call `cudaProfilerStop`.

All 16 ranks write exact-range receipts. A node profile is accepted only if the Nsight report contains CUDA/NVTX/API activity and the shared receipt validator sees the exact `[1000,1004]` logical window with controller ranks `[0,8]`.
