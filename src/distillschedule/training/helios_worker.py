"""Explicit W2 reference or versioned SafeBundle stream into original P0."""
from __future__ import annotations

import importlib.util
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import repository_root, load_profile, load_assets, verify_assets, file_sha256
from .helios_inputs import HeliosCohortTask, load_stream
from .helios_reference_assets import (SCHEMA as REFERENCE_ASSETS, asset_config as reference_assets,
    verify_pins as verify_reference_pins, load_bootstrap as load_reference_bootstrap)

_OFF = ("pack-real-score-pair", "generator-nograd-lora-fused-epilogue", "all-role-lora-scale1-add",
    "optimizer-dispatch-warmup", "generator-nograd-residual-fusion", "exact-grad-residual-fusion",
    "exact-modulated-norm-epilogue", "exact-modulated-norm-materialization", "exact-modulated-norm-backward",
    "exact-modulated-norm-layernorm", "exact-modulated-norm-fused-backward", "exact-history-current-fusion",
    "generator-nograd-cross-kv-cache", "generator-nograd-static-context-cache", "generator-nograd-persistent-unshard",
    "cuda-allocator-expandable-segments", "elide-explicit-rollout-sync", "elide-critic-scalar-logs",
    "elide-official-generator-free-memory", "lora-checkpoint-replay-scope", "lora-b-gemm-epilogue")


def reference_argv(assets, output, *, stream=None, logical_hash=None):
    """Complete explicit original-parser controls; no inherited winner flags."""
    portable = assets.get("schema_version") == REFERENCE_ASSETS
    if portable:
        require(stream is not None and logical_hash is not None,
                "direct reference requires admitted bootstrap and first fresh cohort", path="helios.training")
        trace = stream["cohorts"][0]
        expected_logical, expected_trace = logical_hash, trace["trace_hash"]
    else:
        env = load_profile("helios14b_nogan_best")["runtime"]["environment"]
        expected_logical, expected_trace = env["DISTILLSCHEDULE_LOGICAL_STEP10_HASH"], env["DISTILLSCHEDULE_TRACE_BUNDLE_HASH"]
    args = ["--variant", "no-gan", "--run-id", "original-helios-training-v1", "--output-dir", str(output),
        "--expected-logical-hash", expected_logical,
        "--expected-trace-hash", expected_trace,
        "--mesh-policy", "world16-fsdp", "--rollout-batch-size", "1", "--rollout-backend", "native-fsdp2-batch",
        "--attention-backend", "sdpa", "--modulation-chunk-tokens", "512", "--selective-flash-checkpoint-stride", "0",
        "--ema-device", "cuda", "--generator-residual-storage", "cuda_out_of_place",
        "--critic-residual-storage", "cuda_out_of_place", "--timing-mode", "--no-profile-diagnostics",
        "--generator-nograd-persistent-unshard-flush-allocator-cache"]
    for role, option in (("generator_model", "generator-root"), ("generator_precast", "precast-generator-root"),
        ("score_model", "score-root"), ("score_precast", "precast-score-root"),
        ("logical_checkpoint", "logical-package"), ("trace", "trace-bundle"), ("negative_prompt", "negative-prompt")):
        args.extend(("--" + option, trace["trace_bundle"] if portable and role == "trace" else assets["assets"][role]["path"]))
    return args + ["--no-" + name for name in _OFF]


def _admit_inputs(config, task_value, assets_value, paths):
    from distillschedule.search.helios_request import OriginalHeliosRequest
    request = OriginalHeliosRequest.from_dict(task_value)
    HeliosCohortTask(0, request)
    portable = assets_value.get("schema_version") == REFERENCE_ASSETS
    if portable:
        assets = reference_assets(json.loads(Path(paths["assets"]).read_text()))
        require(assets == assets_value, "assets changed during admission", path="helios.training")
        observed = verify_reference_pins(assets)
        logical, _ = load_reference_bootstrap(assets, verify_payloads=False)
    else:
        profile = load_profile("helios14b_nogan_best")
        assets = load_assets(paths["assets"], profile)
        require(assets == assets_value, "assets changed during admission", path="helios.training")
        observed = verify_assets(profile, assets)
    stream_path = Path(config.data["prepared"])
    stream = load_stream(stream_path, cycles=config.trainer["cycles"], expected_kind=config.data["input_kind"])
    require(stream["identity"]["request"] == request.to_dict(), "stream/task differs", path="helios.training")
    resume = config.trainer.get("resume_from")
    if resume:
        from distillschedule.workloads.best_known.registry import file_sha256
        root = Path(resume)
        require((root / "manifest.json").is_file() and (root / "_SUCCESS").is_file()
                and (root / "_SUCCESS").read_text().strip() == file_sha256(root / "manifest.json"),
                "complete native resume checkpoint required", path="helios.training")
    admitted = {"task": request.to_dict(), "assets": assets, "stream": stream,
        "selected": {"schema_version": "distillschedule.helios_reference_policy.v1", "mesh": "world16-fsdp",
                     "rollout_window": 1, "chunk_tokens": 512, "optimizer": "original_restored_AdamW",
                     "checkpoint": "native_recompute", "attention": "sdpa"},
        "plan": None, "argv": reference_argv(assets, paths["output"], stream=stream,
            logical_hash=logical.manifest_hash if portable else None), "asset_observations": observed,
        "bootstrap": ({"logical_manifest_hash":logical.manifest_hash, "trace_bundle_hash":stream["cohorts"][0]["trace_hash"]}
                      if portable else None),
        "validation_scope": "CPU_W2_SOURCE_ASSET_STREAM_METADATA; ORIGINAL_PAYLOAD_AND_PHYSICAL_GATES_ON_WORKER"}
    return admitted


def admit_reference(config, task_value, assets_value, paths):
    admitted = _admit_inputs(config, task_value, assets_value, paths)
    stream = admitted["stream"]
    portable = admitted["assets"].get("schema_version") == REFERENCE_ASSETS
    if "native_history_policy" in stream["identity"]:
        from distillschedule.runtime.helios_native_history import descriptor
        require(portable and config.execution.get("mode", "reference") == "reference"
                and not any(key in config.execution for key in
                            ("preset", "plan", "implementation", "transport", "capacity_policy")),
                "native history requires explicit portable W2 reference assets/policy", path="helios.training")
        admitted["selected"] = {**admitted["selected"], "native_history":descriptor()}
    return admitted


def admit_safe_stream(config, task_value, assets_value, paths):
    """New direct strategy, not a widened saved benchmark plan or reference."""
    from . import helios_safe_stream as safe
    require(assets_value.get("schema_version") == REFERENCE_ASSETS,
            "safe stream requires explicit local versioned assets, not historical benchmark defaults", path="helios.training")
    admitted = admit_reference(config, task_value, assets_value, paths)
    return _lower_safe_stream(admitted, safe=safe, run_id="original-helios-safe-stream-v1",
        validation_scope="CPU_W2_SOURCE_ASSET_FRESH_STREAM_SAFE_POLICY; NATIVE_KERNEL_CAPACITY_NUMERICAL_GATES_NOT_RUN")


def admit_native_safe_stream(config, task_value, assets_value, paths):
    from . import helios_native_safe_stream as combined
    combined.validate_config(config)
    require(assets_value.get("schema_version") == REFERENCE_ASSETS,
            "native SafeBundle requires explicit local versioned assets", path="helios.training")
    admitted = _admit_inputs(config, task_value, assets_value, paths)
    require("native_history_policy" in admitted["stream"]["identity"],
            "native SafeBundle history policy missing", path="helios.training")
    return _lower_safe_stream(admitted, safe=combined, run_id="original-helios-native-safe-stream-v1",
        validation_scope="CPU_NATIVE_HISTORY_FULL_SAFE_POLICY; GPU_CAPACITY_KERNEL_NUMERICAL_NOT_RUN")


def admit_checkpoint_cpu_stream(config, task_value, assets_value, paths):
    from . import helios_checkpoint_cpu as candidate
    candidate.validate_config(config)
    require(assets_value.get("schema_version") == REFERENCE_ASSETS,
            "checkpoint CPU requires explicit local versioned assets", path="helios.training")
    admitted = _admit_inputs(config, task_value, assets_value, paths)
    require("native_history_policy" in admitted["stream"]["identity"],
            "checkpoint CPU native history policy missing", path="helios.training")
    admitted["checkpoint_input_capacity"] = candidate.load_capacity(config)
    return _lower_safe_stream(admitted, safe=candidate, run_id="original-helios-native-safe-checkpoint-cpu-v1",
        validation_scope="CPU_NATIVE_HISTORY_FULL_SAFE_CHECKPOINT_INPUT_POLICY; TARGET_2_10_NATIVE_GPU_CAPACITY_NOT_RUN")


def _lower_safe_stream(admitted, *, safe, run_id, validation_scope):
    args = admitted["argv"]
    for flag, value in (("--mesh-policy", "node-hsdp"), ("--attention-backend", "local-fa2"),
                        ("--run-id", run_id)):
        args[args.index(flag) + 1] = value
    for name, enabled in safe.effective_flags().items():
        if enabled:
            args[args.index("--no-" + name.replace("_", "-"))] = "--" + name.replace("_", "-")
    admitted.update(selected=safe.descriptor(), implementation=safe.implementation(), validation_scope=validation_scope)
    return admitted


def _safe_policy(config):
    from . import helios_safe_stream, helios_native_safe_stream, helios_checkpoint_cpu
    if config.execution.get("preset") == helios_checkpoint_cpu.PRESET_ID:
        return helios_checkpoint_cpu
    return (helios_native_safe_stream if config.execution.get("preset") == helios_native_safe_stream.PRESET_ID
            else helios_safe_stream)


def _backend():
    root = repository_root()
    source = root / "third_party/helios_compat/source"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    name = "_distillschedule_helios_training_original_backend"
    if name not in sys.modules:
        path = root / "scripts/distillschedule/probe_static_fsdp2_p0_world16_hardened.py"
        spec = importlib.util.spec_from_file_location(name, path)
        require(spec is not None and spec.loader is not None, "original hardened worker missing", path="helios.training")
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def immutable_assets(assets):
    """Pin real frozen payload bytes for restart, outside all training work.

    The historical builder's cheap manifest/config/index/size checks remain
    unchanged. A resumable run additionally verifies every pre-cast shard hash
    and stores the actually verified manifests/negative payload identities.
    """
    import torch
    from runtime.precast_checkpoint import (
        validate_precast_checkpoint, HELIOS_KEEP_IN_FP32_MODULES, PRECAST_MANIFEST_NAME, DEFAULT_INDEX_NAME,
    )
    from runtime.gst_pool_contract import NegativePromptAsset
    rows = assets["assets"]
    portable = assets.get("schema_version") == REFERENCE_ASSETS
    if portable:
        verify_reference_pins(assets)
    identity = {}
    for role, subfolder in (("generator", "transformer_ode"), ("score", "transformer")):
        directory = Path(rows[role + "_precast"]["path"])
        manifest = validate_precast_checkpoint(directory, expected_target_dtype=torch.bfloat16,
            expected_keep_in_fp32_modules=HELIOS_KEEP_IN_FP32_MODULES,
            expected_source_dir=Path(rows[role + "_model"]["path"]) / subfolder, verify_shard_hashes=True)
        if portable:
            source = Path(rows[role + "_model"]["path"]) / subfolder
            require(file_sha256(source / "config.json") == manifest["source"]["config_sha256"]
                    and file_sha256(source / DEFAULT_INDEX_NAME) == manifest["source"]["index_sha256"],
                    "precast source config/index identity changed", path="helios.training")
        identity[role + "_precast"] = {"manifest_sha256": file_sha256(directory / PRECAST_MANIFEST_NAME),
            "shards": {row["file"]: row["sha256"] for row in manifest["artifact"]["shards"]},
            "payload_hashes_verified": True}
    negative = NegativePromptAsset.load(rows["negative_prompt"]["path"])
    identity["negative_prompt"] = {"manifest_hash": negative.manifest["manifest_hash"],
        "payload_sha256": negative.manifest["payload"]["file_sha256"],
        "tensor_sha256": negative.manifest["payload"]["tensor_sha256"]}
    if portable:
        _, identity["logical_checkpoint"] = load_reference_bootstrap(assets)
    return identity


@contextmanager
def reference_process_group():
    """Own fresh NCCL/Gloo groups; never borrow or destroy a caller's group."""
    from .distributed import world16_reference_process_group
    with world16_reference_process_group("W2") as ownership:
        yield ownership


def run_reference(config, admitted=None, source=None, *, admit=None, native_output=None):
    """Public execution admits all local source/config files collectively first."""
    return _run(config, admitted, source, admit=admit, optimized=False, native_output=native_output)


def run_safe_stream(config, admitted=None, source=None, *, admit=None, native_output=None):
    """Same original process/setup owner; explicit opt-in optimized controls."""
    return _run(config, admitted, source, admit=admit, optimized=True, native_output=native_output)


def _run(config, admitted, source, *, admit, optimized, native_output=None):
    require(native_output is None or config.trainer.get("resume_from") and not config.trainer.get("checkpoint_every"),
            "native output requires complete checkpoint and no training saves", path="helios.training")
    if admit is None:
        import torch.distributed as dist
        if dist.is_initialized():
            return _run_admitted(config, admitted, source, optimized=optimized, native_output=native_output)
        with reference_process_group() as ownership:
            return _run_admitted(config, admitted, source, ownership=ownership, optimized=optimized, native_output=native_output)
    from . import distributed
    with reference_process_group() as ownership:
        _, source, _, admitted = distributed.checked(admit, "Helios original source/config/input admission")
        distributed.checked(lambda: require(not source["identity"]["dirty"],
            "execution requires a clean committed source checkout", path="helios.training"), "Helios original source cleanliness")
        if optimized:
            from .helios_safe_stream import validate_environment
            distributed.checked(lambda: validate_environment(require_kernel=True), "Helios safe stream dependency/environment")
        else:
            distributed.checked(lambda: require(not any(os.environ.get(k) for k in
                ("LOCAL_KERNELS", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")),
                "reference attention/allocator environment differs", path="helios.training"), "Helios original environment")
        backend = distributed.checked(_backend, "Helios original import/dependency admission")
        return _run_admitted(config, admitted, source, ownership=ownership, backend=backend, optimized=optimized,
                             native_output=native_output)


def _run_admitted(config, admitted, source, *, ownership=None, backend=None, optimized=False, native_output=None):
    """Entered on every existing torchrun rank; never launches/reserves nodes."""
    from .helios_lifecycle import original_training_scope, admit_original_output
    if ownership is None:
        names = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
        require(all(os.environ.get(k, "").isdigit() for k in names), "explicit torchrun ranks required", path="helios.training")
        world, rank, local, local_world = (int(os.environ[k]) for k in names)
        require(world == 16 and local_world == 8 and 0 <= rank < 16 and local == rank % 8,
                "W2 requires two nodes of eight ranks", path="helios.training")
    # Public admission supplies metadata only. Rank-local input/frozen payloads
    # must wait for the original worker's Gloo plane, or one failed rank could
    # leave its peers entering the original NCCL initialization alone.
    stream = admitted["stream"]
    portable = admitted["assets"].get("schema_version") == REFERENCE_ASSETS
    if optimized:
        safe = _safe_policy(config)
        require(config.execution.get("mode") == "manual" and config.execution.get("preset") == safe.PRESET_ID
                and portable and admitted.get("implementation") == safe.implementation()
                and admitted["selected"] == safe.descriptor(),
                "safe stream typed implementation/source binding changed", path="helios.training")
        # Repeat at the direct API seam, before importing the original backend.
        from . import distributed
        distributed.checked(lambda: safe.validate_environment(require_kernel=True), "Helios safe stream dependency/environment")
    else:
        require("implementation" not in admitted, "reference cannot use optimized stream policy", path="helios.training")
    bootstrap = {} if portable else load_profile("helios14b_nogan_best")["runtime"]["environment"]
    identity = {"schema_version": "distillschedule.helios_training_identity.v1",
        "source_head": source["identity"]["head"], "source_fingerprint": stream["identity"]["source_fingerprint"],
        "stream_sha256": stream["stream_sha256"], "assets": admitted["assets"]["assets"],
        "reference_policy": admitted["selected"],
        "bootstrap_logical_hash": admitted["bootstrap"]["logical_manifest_hash"] if portable else bootstrap["DISTILLSCHEDULE_LOGICAL_STEP10_HASH"],
        "bootstrap_trace_hash": stream["cohorts"][0]["trace_hash"] if portable else bootstrap["DISTILLSCHEDULE_TRACE_BUNDLE_HASH"]}
    from distillschedule.data.helios_epoch_inputs import identity_binding
    identity.update(identity_binding(stream))
    if "native_history_policy" in stream["identity"]:
        from .helios_inputs import NATIVE_KIND, NATIVE_SCHEMA
        from . import helios_native_safe_stream as combined
        if optimized:
            combined = safe
            combined.validate_config(config)
            descriptor = combined.history_descriptor
        else:
            from distillschedule.runtime.helios_native_history import descriptor
        require((not optimized or safe is combined) and config.data.get("input_kind") == NATIVE_KIND
                and stream["schema_version"] == NATIVE_SCHEMA
                and admitted["selected"].get("native_history") == descriptor(),
                "native history phase implementation not bound", path="helios.training")
        identity.update(schema_version="distillschedule.helios_native_history_training_identity.v1",
                        native_history=descriptor())
    if optimized:
        identity["schema_version"] = (safe.IDENTITY if "native_history_policy" in stream["identity"]
                                      else "distillschedule.helios_safe_stream_identity.v1")
        identity["execution_policy"] = identity.pop("reference_policy")
    binding = {"config": config, "stream": stream, "identity": identity, "assets": admitted["assets"],
               "portable_reference":portable, **(ownership or {})}
    if native_output is not None:
        require(config.trainer.get("resume_from") and not config.trainer.get("checkpoint_every"),
                "native output requires complete checkpoint and no training saves", path="helios.training")
        binding["native_output"] = native_output
    if optimized:
        binding["implementation"] = admitted["implementation"]
        from . import helios_checkpoint_cpu as checkpoint_cpu
        if safe is checkpoint_cpu:
            identity["checkpoint_input_capacity"] = admitted["checkpoint_input_capacity"]
    old = sys.argv
    try:
        sys.argv = ["original-helios-training-v1", *admitted["argv"]]
        with original_training_scope(binding):
            # Reject collectively outside P0's catch-all receipt writer. Record
            # publication eligibility only after all ranks admit their paths.
            admit_original_output(rank=int(os.environ["RANK"]))
            if native_output is not None:
                from .helios_lifecycle import _plane
                with _plane(binding):
                    native_output.admit()
            return (backend or _backend()).main()
    finally:
        sys.argv = old


def prepare_main(argv=None):
    """Compatibility entry; the discoverable public data command owns parsing."""
    from distillschedule.data.cli import main
    return main(["prepare-helios-stream", *(sys.argv[1:] if argv is None else argv)])
