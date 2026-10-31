# Primary-only inference export staging

`training.inference.publish_export(..., primary_only=True)` is an internal,
explicit output-transport option for a native owner. Existing callers default
to the original all-owner CPU staging. It does not change training, the
`distillschedule.inference_export.v1` format, or the inference loader.

All ranks agree on the option and module layouts before materialization. Every
owner calls the same `DTensor.full_tensor` collectives, visiting named modules
and tensor keys in deterministic order. Only rank zero converts and retains
the dense tensors as CPU export state. Other owners release each transient
dense result without making a CPU state copy. A primary CPU-copy failure is
exchanged before any peer starts the next tensor collective. Only rank zero
writes weights and publishes the existing manifest and completion marker.

The primary still needs enough host memory for the complete exported state.
Every owner still needs capacity for its transient full tensor; no collective
is skipped and no bound on model-sized CUDA peak is implied. The calling owner
must separately admit host/cgroup and device capacity. This option is not a
capacity-policy override and does not enable any new model/backend combination.
Custom adapter exporters own their own staging and cannot request this option.

Tensor values, dtypes, layouts, nested safe extra state, live parameter identity,
trainability, module mode and RNG are preserved. Unsupported objects, divergent
rank options/layouts, corrupt exports and publication failures still fail.
Restore remains inference-only, not a training checkpoint resume.

The source/CPU proof uses small synthetic modules with actual two-process
Gloo/DTensor collectives. Pretrained 14B export, CUDA capacity and full native
world16 output require their separate integration acceptance.
