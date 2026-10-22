"""Public training requests lowered to the *existing* original-model runtimes.

This is deliberately not a model-name -> best-profile dispatcher. A plan-mode
request must supply the selected, source-bound native plan. The original
worker still owns physical assets, rank/trace tensors, capacity, optimizer and
numerical gates. CPU admission never claims those GPU gates passed.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from .config import TrainingConfig


_FAMILIES = {
    "W1": ("original_wan_ar_dmd", None),
    "W2": ("original_helios_nogan", "helios14b_nogan_best"),
    "W3": ("original_helios_gan", "helios14b_gan_best"),
    "W4": ("original_self_forcing_c51", "livewan_crossscale_pipeline_best"),
    "W5": ("original_h3_fullseq_grid1000", "minimax_h3_grid1000_best"),
    "W6": ("original_diffsynth_h3_ar", None),
}


def _require(condition, message):
    if not condition:
        raise ValueError("native training: " + message)


def _workload(config):
    _require(isinstance(config, TrainingConfig), "TrainingConfig is required")
    _require(config.execution.get("adapter") == "native", "execution.adapter must be native")
    workload = config.model.get("workload")
    _require(workload in _FAMILIES, "model.workload must be W1-W6")
    return workload


def describe_native(config: TrainingConfig) -> dict[str, Any]:
    """Describe support without importing Torch, CUDA helpers or model code."""
    workload = _workload(config)
    catalog = json.loads((Path(__file__).parents[1] / "workloads/catalog.json").read_text())
    row = next(item for item in catalog["workloads"] if item["id"] == workload)
    worker = workload in ("W1", "W6")
    helios_stream = _helios_stream(config, workload)
    result = dict(
        workload=workload, name=row["name"], protocol=row["protocol"],
        algorithm=_FAMILIES[workload][0], config_source=str(config.source),
        native_api=(f"distillschedule.strategies.{'wan' if workload == 'W1' else 'h3_ar'}_cli:main"
                    if worker else "distillschedule.search.execution:run_plan"),
        source=dict(catalog_index=catalog["source_index"],
                    native_code=row["code"], selected_plan_is_authoritative=True),
        launch=("enter on each world16 torchrun rank; this API does not reserve or start nodes"
                if worker else "existing two-node dispatcher; explicit saved-plan asset/transport owners"),
        capabilities=dict(
            modes=["reference", "manual", "plan"] if worker else ["plan"],
            cycles=(dict(min=2, max=64, fresh_inputs="original_wan_cohort_stream.v1",
                         reference_manual_fixed_trace=2, incumbent_bundle=2)
                    if workload == "W1" else dict(exact=2, arbitrary_training=False)),
            checkpoint=dict(supported=workload == "W1", scope=(
                "fresh native cohort stream only; trainable weights, named Adam, scheduler, EMA, clocks and per-rank RNG"
                if workload == "W1" else "not exposed by this binding"), verification="CPU_SOURCE_AND_LIFECYCLE_ONLY"),
            resume=dict(supported=workload == "W1", scope=(
                "complete native cohort boundary, same source/stream/topology/strategy" if workload == "W1"
                else "not exposed by this binding"), verification="CPU_SOURCE_AND_LIFECYCLE_ONLY"),
            generate=dict(supported=False, reason="no native generation adapter exposed by this binding"),
            export=dict(supported=False, reason="no portable inference-export adapter exposed by this binding"),
            final_state_evidence=(dict(supported=True,
                api="distillschedule.strategies.wan_worker:save_final_state",
                contents="trainables, Adam evidence and EMA tensors; NOT a restart checkpoint")
                if workload == "W1" else dict(supported=False,
                    reason="native sampled/fingerprint receipts are not tensor checkpoints"))),
        historical_note=row["note"], gpu_execution="NOT_RUN", claimable_speedup=False)
    if helios_stream:
        result.update(native_api="distillschedule.training.helios_worker:run_reference",
                      launch="enter on each existing world16 torchrun rank; no node reservation/launch")
        result["source"]["selected_plan_is_authoritative"] = False
        result["capabilities"].update(modes=["reference"],
            bootstrap=dict(global_step=10, versions=dict(generator=2, critic=10, ema=10),
                cold_start_supported=False,
                asset_contract="distillschedule.helios_reference_assets.v1 or unchanged legacy native assets"),
            cycles=dict(min=2, max=64, fresh_inputs="original_helios_cohort_stream.v1",
                        scope="five actual step-qualified per-rank source features per cohort"),
            checkpoint=dict(supported=True, scope="original G/C trainables, Adam, scheduler, EMA, clocks and per-rank RNG",
                            verification="CPU_ORIGINAL_HELPERS_AND_STATE_ONLY"),
            resume=dict(supported=True, scope="complete native cohort boundary; same source/stream/assets/topology/reference",
                        verification="CPU_ORIGINAL_HELPERS_AND_STATE_ONLY"))
    if _h3_grid_training(config, workload):
        result.update(native_api="distillschedule.training.h3_grid_training:run_reference",
                      launch="enter on each original world16 torchrun rank; no node reservation/launch")
        result["source"]["selected_plan_is_authoritative"] = False
        result["capabilities"].update(modes=["reference"],
            cycles=dict(min=2, max=64, fresh_inputs="original_h3_grid_latent_dataset.v1",
                        scope="finite original latent_dataset; 96 distinct cached records per complete G1/F5 cycle"),
            checkpoint=dict(supported=True, scope="two named LoRAs, original Adam/schedulers, clocks, sampler cursor and per-rank RNG; EMA0",
                            verification="CPU_SOURCE_AND_LIFECYCLE_ONLY"),
            resume=dict(supported=True, scope="versioned complete-cycle checkpoint; exact same source/input/config/horizon/topology",
                        verification="CPU_SOURCE_AND_LIFECYCLE_ONLY"))
    return result


def _helios_stream(config, workload):
    return workload == "W2" and config.data.get("input_kind") == "original_helios_cohort_stream.v1"


def _h3_grid_training(config, workload):
    return workload == "W5" and config.data.get("input_kind") == "original_h3_grid_latent_dataset.v1"


def _path(config, section, key, *, required=True):
    value = getattr(config, section).get(key)
    _require(not required or isinstance(value, str) and bool(value), f"{section}.{key} is required")
    if not value:
        return None
    _require(isinstance(value, str), f"{section}.{key} must be a path")
    _require("REPLACE_WITH" not in value and "${" not in value and "<" not in value,
             f"replace the explicit placeholder in {section}.{key}: {value}")
    path = Path(value).expanduser()
    return (config.source.parent / path).resolve()


def _object(path, label, *, placeholders=True):
    _require(path is not None and path.is_file(), f"missing {label}: {path}")
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), f"{label} must contain a JSON object")
    if placeholders:
        _placeholders(value, label)
    return value


def _placeholders(value, label):
    if isinstance(value, dict):
        for key, item in value.items():
            _placeholders(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _placeholders(item, f"{label}[{index}]")
    elif isinstance(value, str):
        _require("REPLACE_WITH" not in value and "${" not in value
                 and value not in ("0" * 40, "0" * 64), f"replace the explicit placeholder in {label}")


def _options(config, workload):
    """Do not accept a public knob that the native call would silently ignore."""
    for section, allowed in (
        ("model", {"workload"}), ("algorithm", {"name"}),
        ("execution", {"adapter", "mode", "task", "assets", "plan", "implementation",
                       "capacity_policy", "transport", "expected_head"}),
        ("data", {"prepared", "input_kind", "replay_fixed_trace"}),
        ("trainer", {"cycles", "output_dir", "checkpoint_every", "resume_from", "seed", "generate", "export"}),
    ):
        _require(not set(getattr(config, section)) - allowed,
                 f"unsupported {section} fields: {sorted(set(getattr(config, section)) - allowed)}; "
                 "original shape/math belong in execution.task")
    _require(not config.algorithm.get("name") or config.algorithm["name"] == _FAMILIES[workload][0],
             "algorithm.name does not match the original workload")
    _require(not (workload == "W4" and config.data.get("input_kind") == "original_livewan_teacher_cohorts.v1"),
             "W4 fresh inputs are usable by the native InputLoader only; the C51 fresh-training lifecycle "
             "and drained complete-cycle restart seam are not implemented")
    for field in ("generate", "export"):
        _require(not config.trainer.get(field), f"trainer.{field} is not supported by this native adapter")
    _require(_h3_grid_training(config, workload) or "seed" not in config.trainer,
             "trainer.seed override is unsupported; native RNG is bound by the prepared trace/bootstrap")
    if _h3_grid_training(config, workload):
        _require(config.execution.get("mode", "reference") == "reference"
                 and not any(config.execution.get(k) for k in ("plan", "implementation", "transport", "capacity_policy")),
                 "W5 upstream training supports explicit reference only; optimized selected plans are not admitted")
        _require(set(config.data) == {"input_kind", "prepared"} and bool(config.data.get("prepared")),
                 "W5 requires its explicit native latent inventory; benchmark replay is not fresh input")
        _require(2 <= config.trainer["cycles"] <= 64 and "seed" in config.trainer
                 and "checkpoint_every" in config.trainer,
                 "W5 original training requires cycles 2..64 and explicit matching native seed/checkpoint_every")
        return "reference"
    if _helios_stream(config, workload):
        _require(config.execution.get("mode", "reference") == "reference"
                 and not any(config.execution.get(k) for k in ("plan", "implementation", "transport", "capacity_policy")),
                 "W2 fresh stream currently supports explicit reference only; manual/selected plans are not ignored")
        _require(set(config.data) == {"input_kind", "prepared"} and bool(config.data.get("prepared")),
                 "W2 fresh stream requires explicit prepared data; fixed replay is not substituted")
        _require(2 <= config.trainer["cycles"] <= 64, "W2 fresh stream cycles must be 2..64")
        return "reference"
    _require(not (config.trainer.get("checkpoint_every", 0) or config.trainer.get("resume_from"))
             or workload == "W1" and config.data.get("input_kind") == "original_wan_cohort_stream.v1"
                and bool(config.data.get("prepared")),
             "native checkpoint_every must be 0; final evidence is not a restart checkpoint")
    cycles = config.trainer["cycles"]
    _require((workload == "W1" and 2 <= cycles <= 64) or (workload != "W1" and cycles == 2),
             f"{workload} cycles must be {'2..64' if workload == 'W1' else 'exactly 2'}; requested count is never ignored")
    mode = config.execution.get("mode", "reference")
    _require(mode in (("reference", "manual", "plan") if workload in ("W1", "W6") else ("plan",)),
             f"{workload} supports {describe_native(config)['capabilities']['modes']}; no implicit best/auto recipe")
    _require(mode != "plan" or not config.execution.get("implementation"),
             "plan mode uses the selected plan, not an implementation override")
    _require(mode == "plan" or not config.execution.get("plan"), "reference/manual mode cannot ignore execution.plan")
    _require(workload == "W6" or not config.execution.get("capacity_policy"),
             "execution.capacity_policy is W6-only; other native capacities are already bound by the selected plan")
    _require(workload not in ("W1", "W6") or not config.execution.get("transport"),
             "W1/W6 execute inside existing torchrun ranks, not a transport override")
    if workload != "W1":
        _require(not config.data, "native data for W2-W6 is bound by execution.assets; generic prepared data is not substituted")
    return mode


def _source(config):
    from distillschedule.workloads.best_known.registry import repository_root
    from distillschedule.search.resources import observe_source_identity
    result = observe_source_identity(repository_root())
    _require(result["status"] == "PASS", "unable to observe current source identity")
    expected = config.execution.get("expected_head")
    if expected is not None:
        _placeholders(expected, "execution.expected_head")
        _require(expected == result["identity"]["head"], "execution.expected_head differs from current source")
    return result


def _wan(config, mode, task_value, assets_value, paths):
    from distillschedule.strategies.wan_task import OriginalWanTask
    from distillschedule.strategies.wan_strategy import WanStrategy
    from distillschedule.strategies.wan_worker import OriginalWanAssets
    from distillschedule.strategies.wan_planning import validate_plan, source_fingerprint
    from distillschedule.runtime.wan13b_upper_bound import Wan13BBootstrap
    from distillschedule.workloads.best_known.registry import file_sha256

    task = OriginalWanTask.from_dict(task_value)
    assets = OriginalWanAssets.from_dict(assets_value)
    for key in ("model_root", "bootstrap", "trace_bundle", "negative_prompt"):
        path = Path(getattr(assets, key))
        _require(path.is_absolute() and path.is_dir(), f"Wan asset {key} must name an existing absolute directory: {path}")
    bootstrap = Wan13BBootstrap.load(assets.bootstrap, expected_hash=assets.expected_bootstrap_hash,
                                    expected_lora_rank=task.lora_rank, verify_payloads=True)
    _require(Path(bootstrap.manifest["model_root"]).resolve() == Path(assets.model_root).resolve(),
             "Wan bootstrap/model_root mismatch")
    if mode == "plan":
        plan = validate_plan(task, paths["plan"])
        strategy = WanStrategy.from_dict(plan["selected"]["strategy"])
        _require(plan["identity"]["source_fingerprint"] == source_fingerprint(), "Wan plan source changed")
        identity = plan["identity"].get("assets", {})
        _require(identity.get("bootstrap_hash") == assets.expected_bootstrap_hash
                 and identity.get("trace_bundle_hash") == assets.expected_trace_hash,
                 "Wan plan bootstrap/trace differs from execution.assets")
        model = Path(assets.model_root) / "transformer"
        _require((model / "config.json").is_file() and any(model.glob("*.safetensors")), "original Wan model files missing")
        files = [model / "config.json", *sorted(model.glob("*.safetensors")), *sorted(model.glob("*.index.json"))]
        _require(identity.get("model_files") == {p.name: file_sha256(p) for p in files}, "Wan model file identity differs from plan")
    else:
        _require(config.trainer["cycles"] == 2 or config.data.get("prepared"),
                 "Wan reference/manual supports exactly 2 cycles without a fresh native stream")
        plan = None
        strategy = WanStrategy.from_dict(_object(paths["implementation"], "execution.implementation"))
        strategy.validate_for(task)
        _require(mode != "reference" or strategy == WanStrategy("fsdp", 512, False),
                 "reference mode requires the explicit original FSDP strategy, not an optimized bundle")
    _require(not strategy.retained_critic_blocks or mode == "manual"
             or strategy.retention_execution == "planner-retain4.v1", "uncalibrated experimental Wan retention")
    prepared = _path(config, "data", "prepared", required=False)
    if prepared:
        _require(config.data.get("input_kind") == "original_wan_cohort_stream.v1"
                 and not config.data.get("replay_fixed_trace"),
                 "data.prepared must explicitly be an original_wan_cohort_stream.v1, not generic prepared data")
        # Metadata/source admission is CPU-only. The unmodified native
        # load_stream verifies every original rank trace before model use.
        from distillschedule.strategies.wan_stream import SCHEMA, WanCohortSources, task_at
        from distillschedule.workloads.best_known.registry import fingerprint
        stream = _object(prepared, "data.prepared")
        unsigned = dict(stream); sha = unsigned.pop("stream_sha256", None)
        _require(stream.get("schema_version") == SCHEMA and sha == fingerprint(unsigned)
                 and stream.get("status") == "READY_FRESH_COHORT_STREAM" and stream.get("task") == task.to_dict(),
                 "changed original Wan stream/task")
        source = stream["identity"]
        cohorts = WanCohortSources.from_dict(source["sources"])
        _require(source["task"] == task.to_dict() and source["source_fingerprint"] == source_fingerprint()
                 and len(stream["cohorts"]) == len(cohorts.cohorts) == len(source["inventory"]) == config.trainer["cycles"],
                 "Wan stream source or requested cycles differ")
        for index, entry in enumerate(stream["cohorts"]):
            _require(entry["index"] == index and entry["task"] == task_at(task, index).to_dict(),
                     "Wan stream cohort order/task changed")
        command = "train-stream"
    else:
        _require(config.data.get("replay_fixed_trace") is True and not config.data.get("input_kind"),
                 "fixed trace is replay, not fresh training: set data.replay_fixed_trace=true or provide an explicit native cohort stream")
        command = "run" if mode == "plan" else "benchmark-manual"
    _require(not strategy.candidate_bundle.enabled or (config.trainer["cycles"] == 2 and prepared is None),
             "Wan incumbent literal bundle only supports 2 fixed-trace cycles")
    args = [command, "--task", str(paths["task"]), "--assets", str(paths["assets"]),
            "--output", str(paths["output"]), "--execute"]
    if mode == "plan":
        args += ["--plan", str(paths["plan"])]
    else:
        args += ["--implementation", str(paths["implementation"])]
    args += ["--stream", str(prepared)] if prepared else ["--cycles", str(config.trainer["cycles"])]
    if config.trainer.get("checkpoint_every", 0):
        args += ["--checkpoint-every", str(config.trainer["checkpoint_every"])]
    if config.trainer.get("resume_from"):
        resume = _path(config, "trainer", "resume_from")
        _require((resume / "manifest.json").is_file() and (resume / "_SUCCESS").is_file(),
                 "native resume requires a complete checkpoint directory")
        args += ["--resume-from", str(resume)]
    return dict(task=task.to_dict(), selected=strategy.to_dict(), plan=plan, argv=args,
                validation_scope="CPU_TASK_PLAN_BOOTSTRAP_MODEL_IDENTITIES; ORIGINAL_TRACE_TENSORS_DEFERRED_TO_WORKER")


def _h3_ar(config, mode, task_value, assets_value, paths):
    from distillschedule.strategies.h3_ar_task import H3ARTask, H3ARImplementation
    from distillschedule.strategies.h3_ar_assets import H3ARAssets
    from distillschedule.strategies.h3_ar_planning import validate_plan, validate_policy
    task = H3ARTask.from_dict(task_value)
    policy = _object(paths["capacity_policy"], "execution.capacity_policy")
    validate_policy(policy)
    assets = H3ARAssets.from_dict(assets_value)
    admitted = assets.validate()
    if mode == "plan":
        plan, _ = validate_plan(task, paths["plan"])
        _require(plan["identity"]["assets_sha256"] == assets.digest, "W6 plan/assets changed")
        _require(plan["capacity_policy"] == policy, "W6 selected plan/capacity_policy changed")
        impl = H3ARImplementation.from_dict(plan["selected"]["implementation"])
    else:
        plan = None
        impl = H3ARImplementation(config.execution.get("implementation"))
        _require(mode != "reference" or impl.id == "zero3-native", "W6 reference requires zero3-native")
    args = ["run" if mode == "plan" else "benchmark-manual", "--task", str(paths["task"]),
            "--assets", str(paths["assets"]), "--capacity-policy", str(paths["capacity_policy"]),
            "--output", str(paths["output"]), "--execute"]
    args += ["--plan", str(paths["plan"])] if mode == "plan" else ["--implementation", impl.id]
    return dict(task=task.to_dict(), selected=impl.to_dict(), plan=plan, argv=args,
                asset_admission=admitted, capacity_policy=policy,
                validation_scope="CPU_NATIVE_SOURCE_CHECKPOINT_AUDIT_INPUT_CONTRACT_AND_SELECTED_COSTS; LIVE_CAPACITY_DEFERRED_TO_WORKER")


def _search(config, workload, task_value, assets_value, paths):
    from distillschedule.search.execution import validated_profile
    from distillschedule.workloads.best_known.registry import (
        repository_root, git_head, load_assets, _resolve_configuration, _portable_request_bindings, fingerprint)
    root = repository_root()
    # An immutable LiveWan lowering intentionally includes the unresolved
    # source template beside fully resolved node plans. Its ${ENV:...}
    # expressions are compiler inputs, not unfilled user asset placeholders.
    plan = _object(paths["plan"], "execution.plan", placeholders=False)
    _require(plan.get("profile_id") == _FAMILIES[workload][1], "saved plan belongs to another workload")
    _require(plan.get("integration_head") == git_head(root), "saved plan belongs to another source HEAD")
    transport = _object(paths["transport"], "execution.transport") if paths.get("transport") else None
    if transport is not None:
        from distillschedule.search.r9_contract import validate_controller_envelope, SCHEMA
        from distillschedule.search.r9_native import validate_deployment
        profile = validate_controller_envelope(plan)
        validate_deployment(transport)
        marker = dict(schema_version=SCHEMA, deployment_sha256=fingerprint(transport))
        _require(plan.get("execution_transport", marker) == marker, "saved plan transport binding changed")
    else:
        _require("execution_transport" not in plan, "saved plan requires its explicit execution.transport")
        profile = validated_profile(plan, root=root)
    _require(profile["contract"].get("continuous_cycles") == config.trainer["cycles"],
             "requested cycles differ from the saved native plan")
    assets = load_assets(paths["assets"], profile)
    _require(assets == assets_value == plan.get("asset_config"), "saved-plan execution.assets/gates changed; regenerate the plan")
    rebuilt = _resolve_configuration(profile, root=root, asset_config=assets, run_id=plan["run_id"],
        mode=plan["mode"], request_bindings=_portable_request_bindings(plan, profile, assets))
    for key, value in rebuilt.items():
        if key not in {"canonical_status", "canonical_scope", "workload_semantics_hash"}:
            _require(plan.get(key) == value, "saved native launch configuration changed: " + key)
    candidate = plan["search_candidate"]["id"]
    if workload in ("W2", "W3"):
        from distillschedule.search.helios_request import OriginalHeliosRequest
        from distillschedule.search.helios_implementation import HeliosImplementation
        task = OriginalHeliosRequest.from_dict(task_value)
        impl = HeliosImplementation.parse(candidate)
        _require(task.variant == ("no-gan" if workload == "W2" else "gan"), "Helios task variant differs from workload")
        selected_task = impl.request or OriginalHeliosRequest(variant=impl.variant)
    elif workload == "W4":
        from distillschedule.search.livewan_request import LiveWanRequest
        from distillschedule.search.livewan_implementation import LiveWanImplementation
        task = LiveWanRequest.from_dict(task_value)
        impl = LiveWanImplementation.parse(candidate)
        _require(impl.bundle == "c51_full_v1", "W4 requires the explicit C51 full bundle; C34 is not substituted")
        selected_task = impl.request
    else:
        from distillschedule.search.h3_request import H3Request
        from distillschedule.search.h3_implementation import H3Implementation
        task = H3Request.from_dict(task_value)
        impl = H3Implementation.parse(candidate)
        selected_task = impl.request
    _require(task == selected_task, "execution.task shape/protocol differs from the selected native plan")
    return dict(task=task.to_dict(), selected=deepcopy(plan["search_candidate"]), plan=plan,
                transport=transport, hardware=profile["hardware"],
                validation_scope=("CPU_PORTABLE_R9_ENVELOPE; CALIBRATION_AND_PHYSICAL_GATES_ON_RECEIPT_HOST"
                    if transport else "CPU_SELECTED_PLAN_SOURCE_ASSET_REQUEST_LAUNCH_BINDINGS; PHYSICAL_GATES_ON_NATIVE_WORKERS"))


def _admit(config):
    workload = _workload(config)
    mode = _options(config, workload)
    source = _source(config)
    paths = {key: _path(config, "execution", key, required=(key in ("task", "assets")
               or key == "plan" and mode == "plan" or key == "capacity_policy" and workload == "W6"))
             for key in ("task", "assets", "plan", "capacity_policy", "transport")}
    if workload == "W1" and mode != "plan":
        paths["implementation"] = _path(config, "execution", "implementation")
    paths["output"] = _path(config, "trainer", "output_dir")
    from distillschedule.workloads.best_known.registry import repository_root
    checkout = repository_root().resolve()
    _require(paths["output"] not in (Path("/"), Path.home(), checkout)
             and checkout not in paths["output"].parents,
             "trainer.output_dir must be a project run directory outside the source checkout, not a home/filesystem root")
    task_value = _object(paths["task"], "execution.task")
    assets_value = _object(paths["assets"], "execution.assets")
    if workload == "W1":
        admitted = _wan(config, mode, task_value, assets_value, paths)
    elif _helios_stream(config, workload):
        from .helios_worker import admit_reference
        admitted = admit_reference(config, task_value, assets_value, paths)
    elif _h3_grid_training(config, workload):
        from .h3_grid_training import admit_reference
        admitted = admit_reference(config, task_value, assets_value, paths)
    elif workload == "W6":
        admitted = _h3_ar(config, mode, task_value, assets_value, paths)
    else:
        admitted = _search(config, workload, task_value, assets_value, paths)
    plan = admitted.get("plan")
    if plan is not None and workload in ("W1", "W6"):
        identity = plan["identity"]
        head = identity.get("source_head") if workload == "W1" else identity["source"]["source_head"]
        _require(head == source["identity"]["head"], "selected native plan belongs to another source HEAD")
    return workload, source, paths, admitted


def _report(config, workload, source, paths, admitted):
    result = describe_native(config)
    result.update(status="ADMITTED_CPU_NATIVE_BINDING_GPU_NOT_RUN", source_observation=source,
                  config_identity=config.identity, task=admitted["task"], selected=admitted["selected"],
                  validation_scope=admitted["validation_scope"], output_dir=str(paths["output"]),
                  native_physical_admission="REQUIRED_ON_ORIGINAL_WORKER_BEFORE_MODEL_CONSTRUCTION",
                  native_completion="NOT_RUN; process success alone is not numerical parity")
    if "argv" in admitted:
        result["native_argv"] = admitted["argv"]
        result["hardware"] = dict(world_size=16, nodes=2, gpus_per_node=8,
                                  current_runtime="NOT_PROBED_BY_CPU_DOCTOR")
    else:
        result.update(hardware=admitted["hardware"], native_run_id=admitted["plan"]["run_id"],
                      native_mode=admitted["plan"]["mode"], selected_plan=str(paths["plan"]))
    if "capacity_policy" in admitted:
        result["capacity_policy"] = admitted["capacity_policy"]
    if "lifecycle" in admitted:
        result["lifecycle"] = admitted["lifecycle"]
    return result


def validate_native(config: TrainingConfig) -> dict[str, Any]:
    """Run CPU source/schema/selected-plan admission; no reservation or CUDA."""
    workload, source, paths, admitted = _admit(config)
    return _report(config, workload, source, paths, admitted)


def run_native(config: TrainingConfig, *, dry_run: bool = False) -> dict[str, Any] | int:
    """Invoke the real native runtime, without rewriting/reselecting its plan."""
    if _helios_stream(config, _workload(config)) and not dry_run:
        # All rank-local source/asset/stream metadata belongs under the fresh
        # training wrapper's owned Gloo plane, before entering original P0.
        from .helios_worker import run_reference
        return run_reference(config, admit=lambda: _admit(config))
    if _h3_grid_training(config, _workload(config)) and not dry_run:
        # This lifecycle enters a metadata plane before rank-local admission or
        # model imports. A bad source/input on one rank must fail its peers too.
        from .h3_grid_training import run_reference
        return run_reference(config, admit=lambda: _admit(config))
    workload, source, paths, admitted = _admit(config)
    if dry_run:
        result = _report(config, workload, source, paths, admitted)
        result["status"] = "DRY_RUN_NATIVE_BINDING_NOT_EXECUTED"
        return result
    _require(not source["identity"]["dirty"], "execution requires a clean committed source checkout")
    if workload == "W1":
        from distillschedule.strategies.wan_cli import main
        return main(admitted["argv"])
    if workload == "W6":
        from distillschedule.strategies.h3_ar_cli import main
        return main(admitted["argv"])
    from distillschedule.search.execution import run_plan
    return run_plan(admitted["plan"], coordinator_output=paths["output"],
                    transport_config=admitted["transport"])
