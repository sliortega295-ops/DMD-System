"""Measured native actions -> existing frontier planner -> real native handlers.

Calibration, search and execution have distinct artifacts. Measured CPU tensor
costs are never called GPU data; the random test model is never called a Wan
checkpoint. This adapter lowers serial single-device orders only, so no
predicted inter-role GPU overlap is silently mapped to sequential execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping

from distillschedule.runtime.parameterized_cycle import build_parameterized_cycle_graph
from distillschedule.search.cost_model import ComponentContract, CostModel, OperationCost
from distillschedule.search.frontier_schedule import schedule_frontier
from distillschedule.search.plan_ir import BindingRule, Lifetime, StateAllocation
from distillschedule.workloads.best_known.registry import fingerprint, file_sha256 as _file_sha256, repository_root, write_new_json
from .task import NativeTask, ensure

CALIBRATION = "distillschedule.native_parameter_calibration.v1"
PLAN = "distillschedule.native_parameter_plan.v1"


def file_sha256(path) -> str:
    return _file_sha256(Path(path))


def _load(path):
    with Path(path).open() as f:
        value = json.load(f)
    ensure(isinstance(value, dict), "expected a JSON object")
    return value


def source_fingerprint() -> str:
    root = repository_root()
    paths = list((root / "src/distillschedule/native").glob("*.py"))
    paths += [root / p for p in ("src/distillschedule/runtime/parameterized_cycle.py",
        "src/distillschedule/runtime/role_runner.py", "src/distillschedule/runtime/version_sync.py",
        "src/distillschedule/runtime/wan13b_lora_epilogue.py", "src/distillschedule/search/frontier_schedule.py",
        "src/distillschedule/search/plan_ir.py", "src/distillschedule/search/cost_model.py", "runtime/dmd_boundary.py")]
    return fingerprint({str(p.relative_to(root)): file_sha256(p) for p in sorted(paths)})


def runtime_identity(engine) -> dict[str, Any]:
    import torch
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
               "device": str(engine.device), "threads": torch.get_num_threads(),
               "interop_threads": torch.get_num_interop_threads(),
               "matmul_precision": torch.get_float32_matmul_precision(),
               "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
               "cudnn_tf32": torch.backends.cudnn.allow_tf32,
               "deterministic": torch.are_deterministic_algorithms_enabled()}
    for name in ("diffusers", "triton", "safetensors"):
        try: runtime[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: runtime[name] = None
    hardware = {"machine": platform.machine(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
                "platform": platform.platform(), "node": platform.node()}
    if engine.device.type == "cuda":
        prop = torch.cuda.get_device_properties(engine.device)
        hardware["gpu"] = {"name": prop.name, "uuid": str(getattr(prop, "uuid", "UNAVAILABLE")),
                           "total_memory": prop.total_memory, "capability": [prop.major,prop.minor]}
    return {"source_fingerprint": source_fingerprint(), "task_sha256": engine.task.digest,
            "model_sha256": engine.adapter.initial_model_sha256,
            "frozen_sha256": engine.adapter.base_sha256,
            "conditions_sha256": engine.conditions.identity,
            "runtime": runtime, "hardware": hardware}


def cost_binding(identity):
    try:
        head = subprocess.check_output(["git","rev-parse","HEAD"], cwd=repository_root(),text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("native calibration requires a source checkout") from exc
    return {"source_head": head, "hardware_fingerprint": fingerprint(identity["hardware"]),
            "environment_fingerprint": fingerprint({"runtime": identity["runtime"], "code": identity["source_fingerprint"]}),
            "asset_fingerprint": fingerprint({k: identity[k] for k in ("model_sha256","frozen_sha256","conditions_sha256")}),
            "semantic_fingerprint": identity["task_sha256"],
            "precision_fingerprint": fingerprint(identity["runtime"])}


def capabilities(task: NativeTask) -> dict[str, Any]:
    # No Torch import/model construction for a dry-run.
    variants = [{"checkpoint": ckpt, "epilogue": "reference"} for ckpt in (False, True)]
    try: importlib.metadata.version("triton"); triton = True
    except importlib.metadata.PackageNotFoundError: triton = False
    fused = task.device.startswith("cuda") and task.dtype == "bfloat16" and triton
    if fused:
        variants += [{"checkpoint": c, "epilogue": "wan-fused"} for c in (False, True)]
    return {"backend": task.backend, "protocol": task.protocol, "task_sha256": task.digest,
            "latent_shape": list(task.latent_shape), "logical_batch": task.batch,
            "fake_updates": task.fake_updates, "denoising_steps": task.denoising_steps,
            "rollout_bytes": task.rollout_bytes, "variants": variants,
            "fallbacks": [] if fused else [{"capability": "wan-fused-epilogue",
                "reason": "requires explicit CUDA/BF16 execution and Triton", "selected_reference": "torch-pointwise"}],
            "parameter_source": "TASK_NOT_CANONICAL_RECIPE", "physical_placement": "single-device-resident",
            "unsupported": ["DiffSynth-H3 protocol", "previous/current history", "GAN", "FSDP/HSDP/pools",
                            "automatic physical-backbone sharing", "CPU activation offload"],
            "status": "CAPABILITY_DESCRIPTION_NOT_MODEL_VALIDATION"}


def _contract(task, action, implementation):
    return ComponentContract(family="parameterized_native."+action[0],
        work={"task_sha256":task.digest, "action":action, "microbatch":task.microbatch,
              "accumulation":task.accumulation,"latent_shape":list(task.latent_shape)},
        precision={"base":task.dtype,"lora":"float32","rounding":"reference-or-explicit-matched-epilogue"},
        placement={"world_size":1,"device":task.device,"residency":"separate-role-resident"},
        implementation=implementation,
        context={"cycle":"after_one_complete_update_cycle","concurrency":"serial-device",
                 "protocol":task.protocol,"rng":"operation-qualified","inputs":"retained-through-C-return"})


def calibrate(task: NativeTask, output: str | Path) -> dict[str, Any]:
    from .engine import NativeEngine, compare_states
    from .models import tensors_digest
    ensure(task.rollout_bytes <= task.max_retained_bytes, "even one rollout exceeds retained budget")
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=False)
    engine = NativeEngine(task)
    identity, binding = runtime_identity(engine), None
    binding = cost_binding(identity)
    initial = engine.snapshot()
    try:
        reference_losses = [engine.run_reference_cycle() for _ in range(2)]
        reference = engine.signature_tensors()
        reference_versions = dict(engine.versions)
        variants = []
        for index, impl in enumerate(capabilities(task)["variants"]):
            engine.restore(initial)
            engine.adapter.set_implementation(checkpoint_enabled=impl["checkpoint"], epilogue=impl["epilogue"])
            cold = engine.run_cycle(calibration=True)
            warm = engine.run_cycle(calibration=True)
            gate = compare_states(reference, engine.signature_tensors())
            ensure(gate["status"] == "PASS" and engine.versions == reference_versions,
                   f"implementation failed sequential reference equivalence: {gate}")
            ensure(engine.adapter.base_digest() == identity["frozen_sha256"], "frozen weights mutated")
            records = []
            for observation in warm["observations"]:
                action = observation["action"]
                contract = _contract(task, action, impl)
                raw = {"schema_version":"distillschedule.operation_calibration.v1","binding":binding,
                       "key":contract.digest,"samples_ns":[observation["duration_ns"]],
                       "workspace_bytes_per_rank":observation["workspace_bytes"],"status":"PASS",
                       "profiled":False,"kind":"native_operation","operation_contract_sha256":contract.digest,
                       "component_contract":contract.to_dict(),"device":str(engine.device),
                       "fixture":task.fixture,"memory_scope":warm["memory_scope"],
                       "output_bytes":observation["output_bytes"],"reference_gate":gate}
                path = root / f"variant{index}-{action}.json"; write_new_json(path,raw)
                evidence = {k: raw[k] for k in ("status","profiled","kind","operation_contract_sha256","component_contract")}
                evidence.update(source_receipt=str(path),source_receipt_sha256=file_sha256(path))
                records.append(OperationCost(contract.digest,tuple(raw["samples_ns"]),raw["workspace_bytes_per_rank"],evidence))
            costs = CostModel(binding,tuple(records),scope="MEASURED"); costs.verify_receipts()
            cost_path=root/f"costs-{index}.json";write_new_json(cost_path,costs.to_dict())
            variants.append({"implementation":impl,"cost_model":str(cost_path),"cost_model_sha256":file_sha256(cost_path),
                             "reference_gate":gate,"static_bytes":engine.static_bytes(),
                             "cold":cold,"warm":warm})
        ensure(runtime_identity(engine) == identity, "runtime changed during calibration")
        manifest={"schema_version":CALIBRATION,"task":task.to_dict(),"identity":identity,"binding":binding,
                  "capabilities":capabilities(task),"model_sources":engine.adapter.sources,"variants":variants,"reference_losses":reference_losses,
                  "reference_state_sha256":tensors_digest(reference.items()),"reference_versions":reference_versions,
                  "status":"PASS_MEASURED_NATIVE_COMPONENTS","device":str(engine.device),"fixture":task.fixture,
                  "gpu_execution":"NOT_RUN" if engine.device.type == "cpu" else "CALIBRATION_ONLY",
                  "claimable_speedup":False,"historical_workload_parity":"NOT_CLAIMED_NEW_PROTOCOL"}
        write_new_json(root/"calibration.json",manifest)
        return manifest
    except BaseException as exc:
        write_new_json(root/"failure.json",{"status":"FAIL","error":str(exc),"task_sha256":task.digest})
        raise
    finally:
        engine.restore(initial)


def validate_calibration(task: NativeTask, path: str | Path, *, load_engine: bool = False):
    from .engine import NativeEngine
    manifest = _load(path)
    ensure(manifest.get("schema_version") == CALIBRATION and manifest.get("status") == "PASS_MEASURED_NATIVE_COMPONENTS",
           "not a successful native calibration")
    ensure(manifest.get("task") == task.to_dict(), "task changed: recalibrate affected native work")
    ensure(source_fingerprint() == manifest["identity"]["source_fingerprint"],
           "source changed: calibration is stale")
    ensure(cost_binding(manifest["identity"]) == manifest["binding"], "calibration source binding changed")
    engine = NativeEngine(task) if load_engine else None
    if engine is not None:
        ensure(runtime_identity(engine) == manifest["identity"],
               "model/data/device/environment changed: calibration is stale")
    else:
        # Offline CPU planning validates bytes and task/code identity. Device and actual loaded
        # model identities are checked again by run(); planning must never allocate a GPU model.
        if not task.fixture:
            ensure(file_sha256(task.conditions_path) == manifest["identity"]["conditions_sha256"],
                   "conditions changed: calibration is stale")
            for role, source in manifest["model_sources"].items():
                if role in {"generator", "critic", "teacher"}:
                    root = Path(task.checkpoints[role])
                    observed = {p.name: file_sha256(p) for p in [root/"config.json", *sorted(root.glob("*.safetensors"))]}
                    ensure(observed == source, "checkpoint changed: calibration is stale")
    ensure(manifest["variants"], "empty calibration")
    for v in manifest["variants"]:
        ensure(v["implementation"] in capabilities(task)["variants"], "implementation no longer admitted")
        ensure(v["reference_gate"]["status"] == "PASS", "missing reference correctness gate")
        ensure(file_sha256(v["cost_model"]) == v["cost_model_sha256"], "cost-model artifact changed")
        costs=CostModel.from_dict(_load(v["cost_model"])); costs.verify_receipts()
        costs.bind(manifest["binding"],problem_scope="MEASURED")
        graph=build_parameterized_cycle_graph(source_steps=tuple(range(task.fake_updates)),
                 initial_versions={"generator":1,"critic":task.fake_updates,"ema":len(task.ema_after)},ema_after=task.ema_after)
        ensure({r.key for r in costs.records} == {_contract(task,op.operation_id,v["implementation"]).digest
                                                 for op in graph.operations}, "cost inventory does not match actual task graph")
        if engine is not None:
            ensure(v["static_bytes"] == engine.static_bytes(), "native static storage differs from calibration")
    return manifest,engine


def plan(task: NativeTask, calibration_path: str | Path, *, max_states=50000, beam_width=0, timeout_seconds=60):
    manifest, engine = validate_calibration(task,calibration_path)
    graph=build_parameterized_cycle_graph(source_steps=tuple(range(task.fake_updates)),
             initial_versions={"generator":1,"critic":task.fake_updates,"ema":len(task.ema_after)},ema_after=task.ema_after)
    ranked=[]
    for variant in manifest["variants"]:
        impl=variant["implementation"]
        costs=CostModel.from_dict(_load(variant["cost_model"]))
        rules={op.operation_id:BindingRule(op.operation_id,{},_contract(task,op.operation_id,impl).digest,
                 (0,),("gpu:0",),("native-state",),_contract(task,op.operation_id,impl).digest)
               for op in graph.operations}
        # Resource 0 is the sole native device; retained limits are independent of workspace.
        max_workspace=max(r.workspace_bytes_per_rank for r in costs.records)
        model_limit=variant["static_bytes"]+max_workspace+task.max_retained_bytes
        if task.device.startswith("cuda"):
            # This is a conservative model capacity ceiling, not a current free-memory observation.
            model_limit=min(model_limit,manifest["identity"]["hardware"]["gpu"]["total_memory"]-task.gpu_reserve_bytes)
        ensure(model_limit>0,"no available device capacity after reserve")
        lives=tuple(Lifetime(f"output-r{i}",f"r{i}",(f"c{i}",),(0,),task.rollout_bytes,"producer_exit")
                    for i in range(task.fake_updates))
        states=(StateAllocation("native-state",(0,),variant["static_bytes"],("native_runtime",),False,
                                manifest["identity"]["model_sha256"]),)
        result=schedule_frontier(graph,rules,costs,states=states,lifetimes=lives,
                    memory_limits={0:model_limit},retained_limits={0:task.max_retained_bytes},
                    max_states=max_states,beam_width=beam_width,timeout_seconds=timeout_seconds)
        result["implementation"]=impl
        if result["status"]=="PASS_CPU_SCHEDULE":ranked.append(result)
    ensure(ranked,"no native schedule within calibrated capability/capacity/search budget")
    ranked.sort(key=lambda p:(p["predicted_makespan_ns"],p["memory"][0]["peak_bytes"],fingerprint(p["implementation"])))
    selected=ranked[0]
    result={"schema_version":PLAN,"task":task.to_dict(),"identity":manifest["identity"],
            "calibration_path":str(Path(calibration_path).resolve()),"calibration_sha256":file_sha256(calibration_path),
            "selected":selected,"candidate_count":len(ranked),"graph":graph.to_dict(),
            "status":"PASS_NATIVE_PLAN_LOWERING","placement":"single-device-resident",
            "device":task.device,"fixture":task.fixture,"gpu_execution":"NOT_RUN_BY_SEARCH","claimable_speedup":False,
            "native_execution_model":"SERIAL_COMMON_PREPARED_ROLE_PLAN",
            "execution_identity_recheck":"REQUIRED_AT_NATIVE_RUN","lowered_order":selected["order"],
            "limitations":["no multi-GPU layout search","no inter-role concurrent dispatch","new explicit flow protocol"]}
    result["plan_sha256"]=fingerprint(result)
    return result


def run(task: NativeTask, plan_path: str | Path, output: str | Path, *, cycles: int = 2):
    from .engine import compare_states
    ensure(type(cycles) is int and 2<=cycles<=100000,"native run needs at least two consecutive cycles")
    document=_load(plan_path)
    unsigned=dict(document); sha=unsigned.pop("plan_sha256",None)
    ensure(document.get("schema_version")==PLAN and sha==fingerprint(unsigned),"native plan changed")
    ensure(document["task"]==task.to_dict(),"plan/task mismatch")
    ensure(file_sha256(document["calibration_path"])==document["calibration_sha256"],"calibration artifact changed")
    manifest,engine=validate_calibration(task,document["calibration_path"],load_engine=True)
    ensure(document["identity"]==manifest["identity"],"plan identity changed")
    impl=document["selected"]["implementation"]
    ensure(any(v["implementation"]==impl for v in manifest["variants"]),"uncalibrated implementation")
    ensure(document["placement"]=="single-device-resident" and document["lowered_order"]==document["selected"]["order"],
           "unsupported placement or order mismatch")
    graph=build_parameterized_cycle_graph(source_steps=tuple(range(task.fake_updates)),
             initial_versions={"generator":1,"critic":task.fake_updates,"ema":len(task.ema_after)},ema_after=task.ema_after)
    ensure(document["graph"]==json.loads(json.dumps(graph.to_dict())),"plan graph differs from native protocol")
    # Before any model updates, the actual common runner validates dependencies and release points.
    order=tuple(document["lowered_order"])
    from distillschedule.runtime.role_runner import StaticRoleRunner
    from distillschedule.runtime.version_sync import VersionLedger
    dummy={op.kind.value:lambda ctx:None for op in graph.operations}
    StaticRoleRunner("native_runtime",dummy,versions=VersionLedger(graph.metadata["initial_versions"])).prepare(graph,order=order)
    variant=next(v for v in manifest["variants"] if v["implementation"]==impl)
    costs=CostModel.from_dict(_load(variant["cost_model"]))
    ensure(document["selected"]["cost_model_sha256"]==costs.digest and
           document["selected"]["graph_sha256"]==fingerprint(graph.to_dict()),"selected cost/graph identity mismatch")
    elapsed,live,peak,retained_peak=0,0,engine.static_bytes(),0
    expected_timeline=[]
    for key in order:
        cost=costs.get_component(_contract(task,key,impl))
        peak=max(peak,engine.static_bytes()+live+cost.workspace_bytes_per_rank)
        expected_timeline.append({"operation":key,"start_ns":elapsed,"end_ns":elapsed+cost.duration_ns,
                                  "resources":["gpu:0"]})
        elapsed+=cost.duration_ns
        if key.startswith("r"):live+=task.rollout_bytes
        elif key.startswith("c"):live-=task.rollout_bytes
        retained_peak=max(retained_peak,live)
        peak=max(peak,engine.static_bytes()+live)
    ensure(live==0 and retained_peak<=task.max_retained_bytes,"selected retained capacity violation")
    ensure(document["selected"]["timeline"]==expected_timeline and
           document["selected"]["predicted_makespan_ns"]==elapsed,"selected timing does not reconcile with costs")
    memory=document["selected"]["memory"]
    ensure(len(memory)==1 and memory[0]["rank"]==0 and memory[0]["peak_bytes"]==peak
           and memory[0]["static_bytes"]==engine.static_bytes() and peak<=memory[0]["limit_bytes"],
           "selected memory does not reconcile with actual protocol")
    if engine.device.type=="cuda":
        import torch
        free,_=torch.cuda.mem_get_info(engine.device)
        ensure(peak<=free+torch.cuda.memory_allocated(engine.device)-task.gpu_reserve_bytes,
               "current GPU headroom does not admit selected plan")
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    initial=engine.snapshot()
    try:
        # Fresh matched two-cycle correctness, outside native timing; full state, not only a loss scalar.
        for _ in range(2):engine.run_reference_cycle()
        reference=engine.signature_tensors()
        engine.restore(initial)
        engine.adapter.set_implementation(checkpoint_enabled=impl["checkpoint"],epilogue=impl["epilogue"])
        for _ in range(2):engine.run_cycle(order)
        gate=compare_states(reference,engine.signature_tensors())
        ensure(gate["status"]=="PASS","selected order failed full-state reference gate")
        ensure(engine.adapter.base_digest()==manifest["identity"]["frozen_sha256"],"frozen parameters changed")
        engine.restore(initial)
        receipts=[engine.run_cycle(order) for _ in range(cycles)]
        ensure(runtime_identity(engine)==manifest["identity"],"runtime identity changed during execution")
        ensure(engine.adapter.base_digest()==manifest["identity"]["frozen_sha256"],"frozen parameters changed")
        import torch
        summary={"schema_version":"distillschedule.native_parameter_execution.v1","status":"PASS_NATIVE_EXECUTION",
                 "plan_sha256":sha,"task_sha256":task.digest,"identity":manifest["identity"],
                 "implementation":impl,"reference_gate":gate,"cycles":receipts,
                 "final_versions":dict(engine.versions),"device":str(engine.device),"fixture":task.fixture,
                 "gpu_execution":"NOT_RUN" if engine.device.type=="cpu" else "NATIVE_RUN",
                 "timed_cycles":list(range(1,cycles)),"warmup_cycles":[0],"claimable_speedup":False,
                 "quality_validation":"NOT_RUN","historical_workload_parity":"NOT_CLAIMED_NEW_PROTOCOL"}
        # weights_only-compatible tensor artifact; not an advertised resumable training checkpoint.
        torch.save(engine.signature_tensors(),root/"final-state.pt")
        summary["final_state_sha256"]=file_sha256(root/"final-state.pt")
        write_new_json(root/"execution.json",summary)
        return summary
    except BaseException as exc:
        write_new_json(root/"failure.json",{"status":"FAIL","error":str(exc),"plan_sha256":sha})
        raise
