"""New W3 lifecycle seams; CPU/source proofs are not native kernel acceptance."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from helios_reference_fixtures import frozen_assets, original_logical, portable_asset_document


def test_stream_scope_is_versioned_without_widening_legacy_cache_keys():
    from distillschedule.runtime.helios_gan_manual.down_contract import ScopeKey
    from distillschedule.runtime.helios_gan_manual.stream import StreamScopeKey
    key = StreamScopeKey(rank=0, cycle=100, index=4, logical_step=1499,
                         version=509, graph="B3_GAN_R1", block=39)
    assert key.validate() is key
    with pytest.raises(ValueError):
        ScopeKey(**key.__dict__).validate()
    for changed in ({"logical_step": 1004}, {"version": 19}, {"cycle": True},
                    {"graph": "B4"}, {"block": 40}):
        with pytest.raises(ValueError):
            replace(key, **changed).validate()


def test_public_stream_admits_100_complete_cycles_not_legacy_two_cycle(tmp_path):
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training.native import _options, describe_native
    raw = {"schema_version": "distillschedule.training.v1", "model": {"workload": "W3"},
           "algorithm": {"name": "original_helios_gan"},
           "data": {"input_kind": "original_helios_gan_cohort_stream.v1", "prepared": "stream.json"},
           "execution": {"adapter": "native", "mode": "manual", "task": "task.json",
                         "assets": "assets.json", "preset": "helios_gan.fa3_down40_nomerge_stream.v1"},
           "trainer": {"cycles": 100, "checkpoint_every": 50, "output_dir": "run"}}
    config = TrainingConfig.from_dict(raw, source=tmp_path / "training.json")
    assert _options(config, "W3") == "manual"
    report = describe_native(config)
    assert report["native_api"].endswith("helios_gan_training:run_training")
    assert report["capabilities"]["resume"]["supported"] is True
    assert report["gpu_execution"] == "NOT_RUN"


def test_fresh_gan_clocks_keep_physical_and_curriculum_namespaces():
    from distillschedule.training.helios_gan_training import expected_versions, logical_versions
    assert expected_versions(100) == {"generator": 102, "critic": 510, "ema": 900}
    assert logical_versions(100) == {"generator": 300, "critic": 1500, "ema": 900}


def test_original_gan_generator_helper_uses_actual_new_entry_step(monkeypatch):
    """Argument characterization of the original helper, not a fake loss proof."""
    from contextlib import nullcontext
    import torch
    import ast
    from pathlib import Path
    # Execute this exact source function without importing the unavailable
    # native dependency group. This tests argument binding, NOT loss math or
    # successful original-backend dependency import (recorded separately).
    path = Path(__file__).resolve().parents[1] / "scripts/distillschedule/probe_static_fsdp2_full5_world16.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_run_gan_generator_update")
    namespace = {"torch": torch, "TIMING_OMITTED": "OMITTED"}
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
    original = SimpleNamespace(**namespace)
    engine = SimpleNamespace(global_steps=3, backward=lambda loss: loss.backward(), step=lambda: None)
    entry = SimpleNamespace(logical_step=1005, operation_trace=object())
    trace = SimpleNamespace(load_generator_entry=lambda **kw: entry, source=lambda **kw: {})
    namespace["run_generator_update_trace_rollout"] = lambda **kw: (
        torch.ones(1, requires_grad=True), None, {"test": "argument_only"})
    seen = []
    def official(config, operation_trace, source, step, negative, device):
        seen.append(step)
        return {}
    namespace["score_official_kwargs"] = official
    namespace["_gan_score_boundary_gradient"] = lambda **kw: (torch.ones(1), {})
    result = original._run_gan_generator_update(args=SimpleNamespace(elide_explicit_rollout_sync=True),
        rank=0, device="cpu", generator_engine=engine, critic_engine=SimpleNamespace(module=object()),
        trace=trace, cfg=SimpleNamespace(training_config=SimpleNamespace(gan_g_weight=.05)),
        negative=torch.zeros(1), collect_diagnostics=False,
        diagnostics=SimpleNamespace(scope=lambda *a: nullcontext(), add_tensor_tree=lambda *a, **kw: None))
    assert seen == [1005]
    assert result["logical_step"] == 1005


def metadata_cohort(root, index, *, sources=None, config_sha="c" * 64, source_sha="d" * 64):
    """Metadata-only fixture: workers MUST reject its missing tensor payloads.

    Unlike a capture fixture this must not be called actual native input data.
    It exercises the real new metadata/row/source admission and negative gates.
    """
    from distillschedule.training import helios_gan_inputs as inputs
    from runtime.operation_trace import _manifest_hash, write_operation_trace_bundle_manifest
    from distillschedule.workloads.best_known.registry import fingerprint
    task = inputs.GANCohortTask(index)
    directory = Path(root) / f"cohort-{index:04d}"
    rows = [{"rank": rank, "step": step, "path": str(Path(root) / f"feature-{index}-{step}-{rank}.pt"),
             "sha256": fingerprint([index, step, rank]), "latent_shape": [1], "text_shape": [2,4096]}
            for step in task.source_steps for rank in range(16)]
    sources = sources or {"seed": 41 + index, "samples": [[r["path"] for r in rows if r["step"] == step] for step in task.source_steps]}
    ranks = []
    for rank in range(16):
        path = directory / f"rank_{rank:02d}"; path.mkdir(parents=True)
        manifest = {"schema_version":"distillschedule.operation_trace.v1", "classification":"CPU_METADATA_ONLY_NOT_NATIVE_INPUT_PRODUCTION",
            "source":{"source_fingerprint":source_sha, "training_config_sha256":config_sha,
                "rank":rank, "samples":[r for r in rows if r["rank"] == rank], "seed":sources["seed"], "cohort":index,
                "rng_scope":"new_native_CPU_streams_not_prior_training_rng_continuation"},
            "contract":{inputs.TRACE_KEY:task.to_dict(), "resolution":[384,640], "full_latent_shape":[1,16,9,48,80],
                "stage_spatial_shapes":[[12,20],[24,40],[48,80]], "dtype":"bfloat16", "dfake_gen_update_ratio":5},
            "tensors":{}, "operations":[]}
        def tensor(key, shape, dtype="bfloat16"):
            manifest["tensors"][key] = {"shape":shape, "dtype":dtype, "sha256":fingerprint([key, rank])}
        def operation(key, control):
            manifest["operations"].append({"key":key, "control":control})
        for step in task.source_steps:
            operation(f"step/{step}/branch", {"global_step":step, "dfake_gen_update_ratio":5,
                "train_generator":step == task.first_step, "train_critic":True, "gan":True, "reward":False, "vae_decode":False})
            tensor(f"step/{step}/sample/text_embedding", [2,4096])
            tensor(f"step/{step}/sample/attention_mask", [2], "bool")
            for source in ("primary", "secondary"):
                for role, frames in (("x0",1), ("history",19), ("target",9)):
                    tensor(f"step/{step}/sample/{source}/{role}", [16,frames,48,80])
            for phase in (("generator","critic") if step == task.first_step else ("critic",)):
                prefix = f"step/{step}/{phase}"
                for role, key in inputs.keys(step, phase).items():
                    if role in ("text_embedding", "attention_mask"):
                        continue
                    shape, dtype = [1,16,9,48,80], "bfloat16"
                    if role.startswith("indices_"):
                        shape, dtype = [1, {"indices_hidden_states":9,"indices_history_short":2,
                            "indices_history_mid":2,"indices_history_long":16}[role]], "int64"
                    elif role.endswith("_timestep"):
                        shape, dtype = [1], "int64"
                    elif role.endswith("_crop"):
                        shape, dtype = [1,2], "int64"
                    elif role == "block_noise_stage1":
                        shape[-2:] = [24,40]
                    elif role in ("history_short","history_mid","history_long","continuous_history"):
                        shape[2] = {"history_short":2,"history_mid":2,"history_long":16,"continuous_history":19}[role]
                    tensor(key, shape, dtype)
                schedule = [{"stage":s, "call_count":1, "denoising_exit":0, "denoising_step_budget":2,
                    "latent_shape":[1,16,9,*hw], "timestep":[float(900-s*300)]} for s,hw in enumerate(task.stage_spatial_shapes)]
                operation(prefix + "/rollout", {"pyramid_stage_call_sequence":schedule,
                    "initial_synchronized_pyramid_stage_exit":[0,0,0], "call_count":3,
                    "stage_schedule_scope":"global_rank0_broadcast_equivalent"})
                operation(prefix + ("/dmd" if phase == "generator" else "/train"), {"global_step":step})
                operation(prefix + ("/gan_g" if phase == "generator" else "/gan_d"), {"global_step":step})
        manifest["trace_hash"] = _manifest_hash(manifest)
        (path / "trace.json").write_text(json.dumps(manifest))
        ranks.append({"rank":rank, "path":path.name, "trace_hash":manifest["trace_hash"]})
    bundle = write_operation_trace_bundle_manifest(directory, ranks)
    return {"index":index, "task":task.to_dict(), "trace_bundle":str(directory), "trace_hash":bundle["bundle_hash"]}, rows, sources


def metadata_stream(root, cycles=3, *, config_sha="c" * 64):
    from distillschedule.training import helios_gan_inputs as inputs
    from distillschedule.workloads.best_known.registry import fingerprint
    source_sha = inputs.source_fingerprint()
    values = [metadata_cohort(root, i, source_sha=source_sha, config_sha=config_sha) for i in range(cycles)]
    result = {"schema_version":inputs.SCHEMA, "status":"READY_FRESH_COHORT_STREAM",
        "identity":{"source_fingerprint":source_sha, "training_config_sha256":config_sha,
            "request":inputs.REQUEST.to_dict(), "inventory":[r[1] for r in values], "sources":[r[2] for r in values]},
        "cohorts":[r[0] for r in values], "gpu_execution":"NOT_RUN", "claimable_speedup":False}
    result["stream_sha256"] = fingerprint(result)
    (Path(root) / "stream.json").write_text(json.dumps(result))
    return result


def test_source_roles_admit_exact_original_oneframe_history_and_target(tmp_path):
    """Metadata admission only; actual producer/adapter execution is separate."""
    from distillschedule.training import helios_gan_inputs as inputs
    entry, _, _ = metadata_cohort(tmp_path, 1)
    trace = inputs.load_cohort(inputs.GANCohortTask(1), entry["trace_bundle"], expected_hash=entry["trace_hash"])
    for control in trace._control.values():
        for step in trace.logical_steps:
            for source in ("primary", "secondary"):
                for role, frames in (("x0",1), ("history",19), ("target",9)):
                    row = control.manifest["tensors"][f"step/{step}/sample/{source}/{role}"]
                    assert row["shape"] == [16,frames,48,80]
                    assert row["dtype"] == "bfloat16"


@pytest.mark.parametrize("source", ["primary", "secondary"])
@pytest.mark.parametrize("role,frames", [("x0",0), ("x0",2), ("x0",28), ("history",18),
                                        ("history",20), ("target",8), ("target",10)])
def test_source_roles_reject_nonoriginal_time_geometry(tmp_path, source, role, frames):
    from distillschedule.training import helios_gan_inputs as inputs
    from runtime.operation_trace import OperationTrace
    entry, _, _ = metadata_cohort(tmp_path, 1)
    directory = Path(entry["trace_bundle"])
    controls = {rank: OperationTrace(root=directory / f"rank_{rank:02d}",
        manifest=json.loads((directory / f"rank_{rank:02d}" / "trace.json").read_text()), tensors={})
        for rank in range(16)}
    controls[0].manifest["tensors"][f"step/1005/sample/{source}/{role}"]["shape"][1] = frames
    with pytest.raises(ValueError, match="step-qualified original GAN source missing/changed"):
        inputs.validate_controls(inputs.GANCohortTask(1), controls)


def test_real_metadata_admission_and_rows_bind_current_cohort_not_frozen_paths(tmp_path):
    from distillschedule.training import helios_gan_inputs as inputs
    from distillschedule.runtime.helios_gan_manual.stream import qualify_rows
    doc = metadata_stream(tmp_path)
    assert inputs.load_stream(tmp_path / "stream.json", cycles=3) == doc
    entry = doc["cohorts"][1]
    trace = inputs.load_cohort(inputs.GANCohortTask(1), entry["trace_bundle"], expected_hash=entry["trace_hash"])
    rows, qualification = qualify_rows(trace, 9)
    assert set(rows) == {(s,g) for s in range(1005,1010) for g in ("B1_DENOISE","B3_GAN_R1")}
    assert set(rows.values()) == {11040,13680}
    assert qualification["source_entry_path"].startswith(str(tmp_path))
    assert qualification["bundle_hash"] == entry["trace_hash"]
    assert qualification["task"]["source_steps"] == list(range(1005,1010))
    # Metadata-only admission is not actual payload qualification.
    with pytest.raises(FileNotFoundError):
        inputs.load_stream(tmp_path / "stream.json", cycles=3, verify_rank=9)
    source = Path(qualification["source_entry_path"])
    source.write_text(source.read_text() + " ")
    changed = json.loads(source.read_text()); changed["source"]["seed"] += 1; source.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="FRESH_RANK_SOURCE_CHANGED"):
        qualify_rows(trace, 9)


@pytest.mark.parametrize("mutation", ["shape", "wrong_cohort", "missing_rank", "schedule", "old_marker"])
def test_fresh_gan_shape_source_rank_and_schedule_fail_closed(tmp_path, mutation):
    from distillschedule.training import helios_gan_inputs as inputs
    from runtime.operation_trace import OperationTrace
    entry, _, _ = metadata_cohort(tmp_path, 1)
    trace = inputs.load_cohort(inputs.GANCohortTask(1), entry["trace_bundle"], expected_hash=entry["trace_hash"])
    controls = {r:OperationTrace(root=t.root, manifest=deepcopy(t.manifest), tensors={}) for r,t in trace._control.items()}
    row = controls[0].manifest
    if mutation == "shape": row["tensors"]["step/1005/critic/history_mid"]["shape"][2] = 4
    elif mutation == "wrong_cohort": row["contract"][inputs.TRACE_KEY]["index"] = 0
    elif mutation == "missing_rank": controls.pop(15)
    elif mutation == "schedule": row["operations"][1]["control"]["pyramid_stage_call_sequence"][0]["call_count"] = 2
    else: row["contract"]["distillschedule_original_helios_request"] = inputs.REQUEST.to_dict()
    with pytest.raises((ValueError,KeyError)):
        inputs.validate_controls(inputs.GANCohortTask(1), controls)


def test_new_stream_source_gate_uses_existing_reviewed_overlay_legacy_pin_unchanged():
    from distillschedule.runtime import fsdp2_static
    from distillschedule.runtime.helios_gan_manual.stream import validate_fsdp_source, INTEGRATED_FSDP_SHA
    from distillschedule.runtime.helios_gan_manual.down_replay import FSDP_SHA, NativeDownSAC
    assert validate_fsdp_source(fsdp2_static)["integrated_sha256"] == INTEGRATED_FSDP_SHA
    assert FSDP_SHA != INTEGRATED_FSDP_SHA
    with pytest.raises(ValueError, match="EXACT_NATIVE_PUBLIC_SAC_SOURCE"):
        NativeDownSAC.validate_sources(None, fsdp2_static, SimpleNamespace(__file__=fsdp2_static.__file__))


def test_new_scope_keeps_actual_down_storage_dtype_and_frozen_selection():
    from distillschedule.runtime.helios_gan_manual.stream import StreamScopeKey
    from distillschedule.runtime.helios_gan_manual.down_contract import TensorMeta, Leaf, select_down
    key = StreamScopeKey(0,100,4,1499,509,"B1_DENOISE",39)
    def meta(shape, stride, ptr, grad=False):
        return TensorMeta(shape,stride,"torch.bfloat16","cuda:0",grad,ptr,0)
    x = meta((11040,13824),(13824,1),1,True)
    weight = meta((5120,13824),(13824,1),2)
    bias = meta((5120,),(1,),3)
    mat2 = meta((13824,5120),(1,13824),2)
    leaf = Leaf("critic.blocks.39.ffn.net.2.base_layer",x,weight,bias)
    assert select_down(key,leaf,(bias,x,mat2),{},rows=11040)
    for args in ((bias,x,replace(mat2,storage_id=4)), (bias,replace(x,device="cpu"),mat2),
                 (bias,x,replace(mat2,requires_grad=True))):
        with pytest.raises(ValueError): select_down(key,leaf,args,{},rows=11040)


def test_stream_cache_boundary_rejects_live_fifo_wrong_versions_and_extra_calls(tmp_path):
    import torch
    from distillschedule.runtime.helios_gan_manual.stream import NativeDownStreamSAC
    from distillschedule.training import helios_gan_inputs as inputs
    entry, _, _ = metadata_cohort(tmp_path, 99)
    trace = inputs.load_cohort(inputs.GANCohortTask(99), entry["trace_bundle"], expected_hash=entry["trace_hash"])
    core = NativeDownStreamSAC(torch=torch,rank=0)
    core.generator,core.critic,core.ema = SimpleNamespace(global_steps=101),SimpleNamespace(global_steps=505),SimpleNamespace(optimization_step=899)
    core.begin_cycle(99,trace)
    core.generator.global_steps += 1
    for index in range(5):
        expected = (100,index,1495+index,505+index)
        assert core.next_critic_scope({"step":1495+index,"rank":0}) == expected
        core.calls += 1; core.critic.global_steps += 1
        if index == 0: core.ema.optimization_step += 1
    with pytest.raises(ValueError, match="EXACT5"):
        core.next_critic_scope({"step":1500,"rank":0})
    core.state.update(combined_context_count=400,extra_down=dict(original=400,replay=400),
                      graph_drains=[{} for _ in range(10)])
    core.stores["live"] = {"queue":[object()]}
    with pytest.raises(ValueError, match="DRAINED"):
        core.cycle_boundary(100)
    core.stores.clear(); core.cycle_boundary(100); core.finish()
    assert not core.rows and not core.stores
    core.stores["failed"] = {"policy_only":[object()]}; core.abort(); core.restored()
    assert core.state["failed"] is True and not core.stores


def gan_assets(tmp_path, frozen, logical):
    import shutil
    import yaml
    from distillschedule.workloads.best_known.registry import fingerprint, file_sha256, repository_root
    from distillschedule.training.helios_gan_assets import SCHEMA
    root = tmp_path / "systems-bootstrap"
    shutil.copytree(logical, root)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["ema"]["optimization_step"] = 800
    manifest["systems_only_control"] = {"classification":"systems_only_gan_bootstrap",
        "logical_schedule_versions":{"generator":200,"critic":1000,"ema":800},
        "logical_control_window":list(range(1000,1005)), "official_checkpoint_1000":False}
    manifest.pop("manifest_hash"); manifest["manifest_hash"] = fingerprint(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    ema_file = root / "ema/sparse_ema_manifest.json"
    ema = json.loads(ema_file.read_text()); ema["optimization_step"] = 800; ema_file.write_text(json.dumps(ema))
    document = portable_asset_document(frozen, root)
    config = yaml.safe_load((repository_root() / "configs/distillschedule/templates/helios_stage3_gan_384x640_systems.yaml").read_text())
    config["training_config"].update(is_random_drop=False,noise_corrupt_clean_prob_history=1.)
    config_path = tmp_path / "gan.yaml"; config_path.write_text(yaml.safe_dump(config))
    document["schema_version"] = SCHEMA
    document["assets"]["gan_config"] = {"path":str(config_path),"sha256":file_sha256(config_path)}
    from distillschedule.runtime.helios_gan_manual.backend_library import FA3_PINS
    kernel=tmp_path/"DECLARED_KERNEL_DIRECTORY_NOT_A_NATIVE_BINARY"; kernel.mkdir()
    document["assets"]["fa3_kernel"]={"path":str(kernel),"files_sha256":dict(FA3_PINS)}
    return document


def test_actual_public_admission_uses_explicit_local_gan_assets_not_historical_paths(
        tmp_path, frozen_assets, original_logical, monkeypatch):
    from distillschedule.training import helios_gan_training as training, helios_gan_inputs as inputs
    from distillschedule.training import helios_gan_assets as assets_io, native
    from distillschedule.training.config import TrainingConfig
    assets = gan_assets(tmp_path, frozen_assets, original_logical)
    logical, identity = assets_io.load_bootstrap(assets)
    assert identity["payload_hashes_verified"] and identity["physical_initial"] == training.expected_versions(0)
    assert logical.manifest["systems_only_control"]["official_checkpoint_1000"] is False
    # Actual frozen CPU payloads were verified above; declared kernel pins
    # cannot masquerade as an admitted local FA3 binary package.
    with pytest.raises(FileNotFoundError): assets_io.immutable_assets(assets)
    stream_root = tmp_path / "inputs"; stream_root.mkdir()
    doc = metadata_stream(stream_root, config_sha=assets["assets"]["gan_config"]["sha256"])
    assets_path = tmp_path / "assets.json"; assets_path.write_text(json.dumps(assets))
    task_path = tmp_path / "task.json"; task_path.write_text(json.dumps(inputs.REQUEST.to_dict()))
    raw = {"schema_version":"distillschedule.training.v1", "model":{"workload":"W3"}, "algorithm":{},
        "data":{"input_kind":inputs.KIND,"prepared":str(stream_root/"stream.json")},
        "execution":{"adapter":"native","mode":"manual","preset":training.PRESET_ID,
            "task":str(task_path),"assets":str(assets_path)},
        "trainer":{"cycles":3,"checkpoint_every":1,"output_dir":str(tmp_path/"run")}}
    config = TrainingConfig.from_dict(raw, source=tmp_path / "training.json")
    from distillschedule.search.helios_implementation import HeliosImplementation
    monkeypatch.setattr(HeliosImplementation, "environment", lambda *a,**k: pytest.fail("inherited historical plan/assets"))
    result = native.validate_native(config)
    assert result["selected"] == training.descriptor()
    assert result["capabilities"]["bootstrap"]["physical_versions"] == training.expected_versions(0)
    assert native.run_native(config,dry_run=True)["status"] == "DRY_RUN_NATIVE_BINDING_NOT_EXECUTED"
    assert not (tmp_path/"run").exists()
    admitted = training.admit_training(config,inputs.REQUEST.to_dict(),assets,
        {"assets":assets_path,"output":tmp_path/"run"})
    assert admitted["stream"] == doc
    args = admitted["argv"]
    assert args[args.index("--variant")+1] == "gan"
    assert args[args.index("--expected-trace-hash")+1] == doc["cohorts"][0]["trace_hash"]
    assert args[args.index("--gan-config")+1] == assets["assets"]["gan_config"]["path"]
    assert "--generator-nograd-persistent-unshard" in args and "--generator-nograd-static-context-cache" not in args
    assert "/share/" not in " ".join(args)


def _set_effect_counters(build, values):
    for key,value in values.items():
        if key == "public_backend_call_count": build["gan_manual_runtime"][key] = value
        elif key.startswith("down_"): build["gan_manual_runtime"]["down40"]["extra_down"][key.removeprefix("down_")] = value
        else:
            prefix,name = key.split(".",1)
            if prefix in ("generator","critic"):
                owner = build[prefix]["all_role_lora_scale1_add"]
                if name.startswith("shape."): owner["lora_b_gemm_selected_shape_counts"][name.removeprefix("shape.")] = value
                else: owner[name] = value
            else:
                role,owner = {"critic_checkpoint":("critic","checkpoint_policy"), "norm":("generator","exact_modulated_norm_epilogue"),
                    "history":("generator","exact_history_current_fusion"), "kv":("generator","generator_nograd_cross_kv_cache"),
                    "residual":("generator","generator_nograd_residual_fusion")}[prefix]
                build[role][owner][name] = value


def _cpu_lifecycle_run(root, output, *, resume=None, stop=None):
    """Real Program/observer/Adam/EMA/state IO, SYNTHETIC P0 operations/counters.

    This intentionally does not claim full original GAN math, kernels, source
    payload capture, FSDP16 or physical capacity. Those are distinct target gates.
    """
    import random
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Replicate
    from contextlib import nullcontext
    from distillschedule.training import helios_gan_training as training, helios_gan_inputs as inputs
    from distillschedule.training.native_state import engine_state
    from distillschedule.training.checkpoint import pack_state, rng_state
    from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter, FSDP2ShardEMA
    from distillschedule.runtime import helios_planner_cycle as cycle
    from distillschedule.runtime.helios_gan_manual.stream import NativeDownStreamSAC
    from runtime.operation_trace import OperationTrace
    root,output = Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    random.seed(71); np.random.seed(71); torch.manual_seed(71)
    patch = pytest.MonkeyPatch()
    dist.init_process_group("gloo",init_method="file://"+str(output/"rendezvous"),rank=0,world_size=1)
    try:
        mesh = init_device_mesh("cpu",(1,),mesh_dim_names=("CPU_FIXTURE_NOT_FSDP16",))
        class Root(torch.nn.Module):
            def __init__(self, role):
                super().__init__(); self.role=role
                parameter = torch.tensor([.2 if role == "generator" else .3])
                self.weight = torch.nn.Parameter(distribute_tensor(parameter,mesh,[Replicate()]) if role == "generator" else parameter)
                if role == "critic": self.gan_head = torch.nn.Parameter(torch.tensor([.07]))
                self.register_buffer("base",torch.tensor(.1)); self.adapters_enabled=True
            def enable_adapters(self): self.adapters_enabled=True
            def disable_adapters(self): self.adapters_enabled=False
            def forward(self,hidden_states,encoder_hidden_states=None,gan_mode=False,**kw):
                weight = self.weight.to_local() if hasattr(self.weight,"to_local") else self.weight
                value = weight if self.adapters_enabled else self.base
                flow = (hidden_states.float()*.01+value.reshape(())).to(torch.bfloat16)
                return (flow,flow.float().mean()*self.gan_head) if gan_mode else (flow,)
        engines=[]
        for role,step in (("generator",2),("critic",10)):
            model=Root(role); optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,betas=(0.,.999))
            for parameter in model.parameters():
                optimizer.state[parameter] = {"step":torch.tensor(float(step)),"exp_avg":torch.zeros_like(parameter),"exp_avg_sq":torch.zeros_like(parameter)}
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.); scheduler.last_epoch=step
            engines.append(FSDP2EngineAdapter(model,optimizer,scheduler,physical_step=step,global_batch=16,dp_world_size=1))
        generator,critic=engines
        ema=FSDP2ShardEMA.__new__(FSDP2ShardEMA)
        ema._names=["weight"]; ema._parameters=[generator.module.weight]; ema._shadows=[torch.tensor([.19])]
        ema.device="cpu"; ema.nvtx_enabled=False; ema.last_transfer_bytes=0
        ema.decay=.99; ema.min_decay=0.; ema.update_after_step=750; ema.optimization_step=800; ema.cur_decay_value=None
        stream=json.loads((root/"stream.json").read_text())
        binding={"config":SimpleNamespace(trainer={"output_dir":str(output),"cycles":len(stream["cohorts"]),"checkpoint_every":1,
            **({"resume_from":str(resume)} if resume else {})}), "stream":stream,
            "identity":{"scope":"CPU_REAL_LIFECYCLE_SYNTHETIC_P0_NOT_GAN_OR_GPU_PROOF","stream":stream["stream_sha256"]}}
        binding["checkpoint_policy"]=training._checkpoint_policy(binding["identity"])
        core=NativeDownStreamSAC(torch=torch,rank=0); core.generator,core.critic,core.ema=generator,critic,ema
        binding["gan_core"]=core
        metadata_load=inputs.load_cohort
        def load(task,directory,*,expected_hash):
            trace=metadata_load(task,directory,expected_hash=expected_hash)
            # Explicit payload fixture, NOT OperationTrace hash verification.
            # Production validate_rank_payload is never patched or relaxed.
            tensors={}
            for key,row in trace._control[0].manifest["tensors"].items():
                dtype={"bfloat16":torch.bfloat16,"int64":torch.int64,"bool":torch.bool}[row["dtype"]]
                tensors[key]=torch.tensor(task.index+1,dtype=dtype).expand(*row["shape"])
            trace._loaded[0]=OperationTrace(root=trace._control[0].root,manifest=trace._control[0].manifest,tensors=tensors)
            return trace
        patch.setattr(inputs,"load_cohort",load)
        patch.setattr(cycle._Observer,"memory",lambda self,**kw:{"status":"SYNTHETIC_CPU_TELEMETRY_NOT_GPU"})
        fixture=json.loads((Path(__file__).parent/"fixtures/helios_gan_manual_counters.json").read_text())["build"]
        full=training.implementation().candidate_bundle.effect_snapshot(fixture)
        delta={k:v//2 for k,v in full.items()}; counters=dict.fromkeys(full,0)
        _set_effect_counters(fixture,counters)
        def record(entry):
            identity = entry.pool_entry.identity_sha256 if hasattr(entry,"pool_entry") else entry.identity_sha256
            return {"call_count":3,"stage_call_count":{0:1,1:1,2:1},"synchronize_after_rollout":False,
                    "empty_cache_before_rollout":False,"trace_identity_sha256":identity}
        def forwards(engine,entry):
            value=None
            for hw in ((12,20),(24,40),(48,80)):
                hidden=entry.tensors["entry_noise"][...,:hw[0],:hw[1]]
                value=engine.module(hidden_states=hidden,encoder_hidden_states=entry.tensors["text_embedding"])[0]
            return value
        def gupdate(**kw):
            trace=kw["trace"]; entry=trace.load_generator_entry(rank=0,device="cpu")
            source=trace.source(rank=0,device="cpu")
            value=forwards(generator,entry)
            critic.module.enable_adapters()
            critic.module(hidden_states=value.detach())
            critic.module.disable_adapters()
            critic.module(hidden_states=value.detach()); critic.module(hidden_states=value.detach())
            critic.module.enable_adapters(); critic.module(hidden_states=value.detach()[...,:24,:40],gan_mode=True)
            draw=random.random()+float(np.random.random())+float(torch.rand(()))
            generator.backward(value.float().mean()*(draw+float(source["primary_target"].flatten()[0])))
            generator.step()
            return {"logical_step":entry.logical_step,"source_rank":0,"rollout":record(entry)}
        def rollout(**kw):
            values,records={},[]
            for step in kw["steps"]:
                entry=kw["trace"].load_critic_entry(rank=0,step=step,device="cpu")
                with torch.no_grad(): values[step]=forwards(generator,entry)
                row=record(entry); row.update(member_trace_identities=[entry.pool_entry.identity_sha256],
                    logical_steps=[step],batch_size=1)
                records.append(row)
            return values,records,{"enabled":True,"restored":True,"rollout_batch_count":5,
                "set_no_reshard_count":41,"explicit_reshard_count":41,"unsharded_after_exit_count":0}
        samples=[]
        def cupdate(**kw):
            step=kw["step"]; core.next_critic_scope({"step":step,"rank":0})
            kw["trace"].load_critic_entry(rank=0,step=step,device="cpu")
            source=kw["trace"].source(rank=0,device="cpu"); samples.append((step,float(source["secondary_target"].flatten()[0])))
            critic.module.enable_adapters()
            first=critic.module(hidden_states=kw["x0"])[0]
            critic.set_gradient_accumulation_boundary(is_boundary=False); critic.backward(first.float().square().mean())
            second=critic.module(hidden_states=kw["x0"][...,:24,:40].expand(3,-1,-1,-1,-1),gan_mode=True)[1]
            critic.set_gradient_accumulation_boundary(is_boundary=True); critic.backward(second.square().mean()); critic.step()
            core.calls+=1; core.state["combined_context_count"]+=80
            for mode in ("original","replay"): core.state["extra_down"][mode]+=80
            core.state["graph_drains"].extend([{"fixture":"SYNTHETIC_NOT_SAC_EXECUTION"}]*2)
            if step%5 == 4:
                for key in counters: counters[key]+=delta[key]
                _set_effect_counters(fixture,counters)
            return {"logical_step":step,"source_rank":0}
        p0=SimpleNamespace(torch=torch,base=SimpleNamespace(_run_gan_generator_update=gupdate),
                           _precompute_critic_rollouts=rollout,_run_gan_critic_precomputed=cupdate)
        def progress(value):
            if stop and value["completed_cycles"] == stop: raise InterruptedError("TEST_COMPLETE_CYCLE_STOP")
        try:
            result=training.execute_cycles(p0=p0,args=SimpleNamespace(timing_mode=True,profile_mode=False,profile_diagnostics=False,ema_device="cpu"),
                policy=SimpleNamespace(elide_critic_scalar_logs=True,elide_official_generator_free_memory=True),
                result={"build":fixture},rank=0,device="cpu",generator=generator,critic=critic,ema=ema,diagnostics=SimpleNamespace(add_bytes=lambda *a,**kw:None),
                negative=None,cfg=object(),generator_free_memory={},binding=binding,progress=progress)
            assert result["native_GPU_continuation"] == "NOT_RUN"
        except InterruptedError:
            assert stop is not None
        snapshot={"state":pack_state(engine_state({"generator":generator,"critic":critic},ema=ema)),
                  "rng":rng_state(),"samples":samples}
        torch.save(snapshot,output/"final.pt")
        return snapshot
    finally:
        patch.undo(); dist.destroy_process_group()


def _cpu_lifecycle_child_env():
    import os
    paths = [str(Path(__file__).parents[1] / "src"), str(Path(__file__).parent)]
    if inherited := os.environ.get("PYTHONPATH"):
        paths.append(inherited)
    return dict(os.environ, PYTHONPATH=os.pathsep.join(paths),
                CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")


def test_fresh_process_env_keeps_admitted_dependency_overlay(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    overlays = [tmp_path / name for name in ("dependency-a", "dependency-b")]
    for directory in overlays:
        directory.mkdir()
    (overlays[1] / "r30_admitted_dependency.py").write_text("TOKEN = 'external-overlay-only'\n")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(map(str, overlays)))
    env = _cpu_lifecycle_child_env()
    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(Path(__file__).parents[1] / "src"), str(Path(__file__).parent), *map(str, overlays)]
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["PYTHONDONTWRITEBYTECODE"] == "1"
    # No site packages: the subprocess must consume the inherited dependency
    # path, not an installed duplicate. This is not native-helper qualification.
    child = subprocess.run([sys.executable, "-S", "-c",
        "import r30_admitted_dependency as dependency; print(dependency.TOKEN)"],
        env=env, capture_output=True, text=True, timeout=10)
    assert child.returncode == 0, child.stdout + child.stderr
    assert child.stdout.strip() == "external-overlay-only"


def test_new_cycle_runner_real_program_state_checkpoint_and_fresh_process_resume(tmp_path):
    import subprocess
    import sys
    import torch
    from test_training_end_to_end import equal
    root=tmp_path/"inputs"; root.mkdir(); metadata_stream(root,cycles=4)
    full=_cpu_lifecycle_run(root,tmp_path/"full")
    _cpu_lifecycle_run(root,tmp_path/"first",stop=2)
    checkpoint=tmp_path/"first/checkpoints/cycle-000002"
    assert (checkpoint/"_SUCCESS").is_file()
    code="from test_training_helios_gan_stream import _cpu_lifecycle_run; import sys; _cpu_lifecycle_run(*sys.argv[1:3],resume=sys.argv[3])"
    env = _cpu_lifecycle_child_env()
    child=subprocess.run([sys.executable,"-c",code,str(root),str(tmp_path/"resumed"),str(checkpoint)],env=env,capture_output=True,text=True,timeout=90)
    assert child.returncode == 0, child.stdout+child.stderr
    resumed=torch.load(tmp_path/"resumed/final.pt",weights_only=False)
    assert equal(full["state"],resumed["state"])
    assert equal(full["rng"],resumed["rng"])
    assert resumed["samples"] == full["samples"][10:]
    # A missing local shard must be rejected before another live update.
    shards=list(checkpoint.glob("*.pt")); assert shards
    shards[0].unlink()
    with pytest.raises((ValueError,FileNotFoundError)):
        _cpu_lifecycle_run(root,tmp_path/"incomplete",resume=checkpoint)


def test_stream_source_gate_rejects_wrong_current_hash_and_unreviewed_overlay(tmp_path,monkeypatch):
    import shutil
    from distillschedule.runtime import fsdp2_static
    from distillschedule.runtime.helios_gan_manual.stream import validate_fsdp_source
    from distillschedule.workloads.best_known import registry
    real=registry.repository_root()
    path=tmp_path/"src/distillschedule/runtime/fsdp2_static.py"; path.parent.mkdir(parents=True)
    shutil.copyfile(fsdp2_static.__file__,path)
    lock_path=tmp_path/"docs/workloads/source_layout.lock.json"; lock_path.parent.mkdir(parents=True)
    shutil.copyfile(real/"docs/workloads/source_layout.lock.json",lock_path)
    monkeypatch.setattr(registry,"repository_root",lambda:tmp_path)
    module=SimpleNamespace(__file__=str(path))
    assert validate_fsdp_source(module)
    original=path.read_bytes(); path.write_bytes(original+b"\n# UNREVIEWED\n")
    with pytest.raises(ValueError,match="EXACT_REVIEWED_STREAM_FSDP_SOURCE"): validate_fsdp_source(module)
    path.write_bytes(original)
    lock=json.loads(lock_path.read_text()); lock["files"]["src/distillschedule/runtime/fsdp2_static.py"]["source_commit"]="0"*40
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError,match="EXISTING_REVIEWED"): validate_fsdp_source(module)


def test_step_qualified_source_cache_reuses_only_g_then_c_and_drains(tmp_path):
    from distillschedule.training import helios_gan_inputs as inputs
    entry,_,_=metadata_cohort(tmp_path,1)
    trace=inputs.load_cohort(inputs.GANCohortTask(1),entry["trace_bundle"],expected_hash=entry["trace_hash"])
    materialized=[]
    def payload(rank,step,device):
        value=object(); materialized.append((rank,step,device,value)); return value
    # No CUDA call: this is only a cache/ownership protocol fixture.
    trace._materialize_source=payload
    trace._selected_step,trace._selected_phase=1005,"generator"
    first=trace.source(rank=0,device="cuda:0")
    with pytest.raises(ValueError,match="not drained"): trace.validate_boundary()
    with pytest.raises(ValueError,match="already consumed"): trace.source(rank=0,device="cuda:0")
    trace._selected_phase="critic"
    assert trace.source(rank=0,device="cuda:0") is first
    trace._selected_step=1006
    assert trace.source(rank=0,device="cuda:0") is not first
    assert len(materialized)==2 and [r[1] for r in materialized]==[1005,1006]
    trace.validate_boundary(); trace.close()
    assert not trace._source_cache and not trace._source_uses


def test_explicit_local_fa3_keeps_pinned_bytes_and_legacy_path_guard(tmp_path):
    from distillschedule.runtime.helios_gan_manual import backend_library as library
    with pytest.raises(ValueError,match="PINNED_EXISTING_FA3_PATH"):
        library.verify_files(tmp_path)
    for name in library.FA3_PINS:
        (tmp_path/name).write_bytes(b"SYNTHETIC_NOT_A_PINNED_KERNEL")
    with pytest.raises(ValueError,match="PINNED_EXISTING_FA3_FILE"):
        library.verify_explicit_files(tmp_path)
    # Both loaders share the same bytecode/schema/loaded-module/maps checker.
    # No binary is imported or hardware queried by this source test.
    import ast
    parsed=ast.parse(Path(library.__file__).read_text())
    funcs={n.name:n for n in parsed.body if isinstance(n,ast.FunctionDef)}
    for name in ("load_actual","load_explicit"):
        assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="_load_verified"
                   for n in ast.walk(funcs[name]))
    assert {"attest_function","validate_maps","read_maps"} <= {n.func.id for n in ast.walk(funcs["_load_verified"])
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}


def test_gan_boundary_refuses_disabled_adapters_pending_gradients_and_noncycle_restart():
    import torch
    from distillschedule.training import helios_gan_training as training
    from distillschedule.training.native_state import engine_state
    from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter
    engines=[]
    for _ in range(2):
        model=torch.nn.Linear(1,1); model.adapters_enabled=True
        optimizer=torch.optim.AdamW(model.parameters()); scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        engines.append(FSDP2EngineAdapter(model,optimizer,scheduler,physical_step=10,dp_world_size=1))
    g,c=engines; state=engine_state({"generator":g,"critic":c})
    trace=SimpleNamespace(validate_boundary=lambda:None)
    training.validate_boundary(state,g,c,trace)
    c.module.adapters_enabled=False
    with pytest.raises(ValueError,match="enabled-adapter"): training.validate_boundary(state,g,c,trace)
    c.module.adapters_enabled=True; next(c.module.parameters()).grad=torch.ones_like(next(c.module.parameters()))
    with pytest.raises(ValueError,match="unconsumed gradients"): training.validate_boundary(state,g,c,trace)
    binding={"stream":{"cohorts":[{}, {}, {}]},"checkpoint_policy":{"policy":"test"}}
    header={"cycle":1,"cursor":1,"versions":training.expected_versions(1),"policy":binding["checkpoint_policy"]}
    training.validate_progress(header,binding)
    for changed in ({"cursor":2},{"versions":{"generator":3,"critic":15,"ema":805}},{"cycle":3,"cursor":3}):
        with pytest.raises(ValueError,match="G1/C5/EMA1"): training.validate_progress(dict(header,**changed),binding)


_IGNORED_CONFIG_LOCATORS = {
    "model_config": {"pretrained_model_name_or_path", "transformer_model_name_or_path",
                     "real_score_model_name_or_path", "reward_model_name_or_path"},
    "data_config": {"gan_data_root"},
}


def _canonical_gan_config():
    import yaml
    from distillschedule.workloads.best_known.registry import repository_root
    return yaml.safe_load((repository_root() / "configs/distillschedule/templates/helios_stage3_gan_384x640_systems.yaml").read_text())


def test_portable_config_helper_removes_only_five_unused_system_locators():
    from distillschedule.training.helios_gan_inputs import portable_config
    expected = _canonical_gan_config()
    for section, names in _IGNORED_CONFIG_LOCATORS.items():
        for name in names:
            expected[section].pop(name)
    actual = portable_config()
    assert actual == expected
    assert not any(s in json.dumps(actual) for s in ("/share/", "/home/shuchenweng/", "/mnt/bn/"))
    actual["training_config"]["learning_rate"] = 123
    assert portable_config() == expected  # Never mutate/cache the canonical owner.


@pytest.mark.parametrize("locator_mode", ["omit", "replace", "none"])
def test_native_config_accepts_only_optional_system_locator_changes(tmp_path, locator_mode):
    import yaml
    from distillschedule.training.helios_gan_inputs import read_config
    value = _canonical_gan_config()
    expected = deepcopy(value["training_config"])
    for section, names in _IGNORED_CONFIG_LOCATORS.items():
        for name in names:
            if locator_mode == "omit": value[section].pop(name)
            elif locator_mode == "none": value[section][name] = None
            else: value[section][name] = ["/public/example/features"] if name == "gan_data_root" else "/public/example/model"
    path = tmp_path / "portable.yaml"; path.write_text(yaml.safe_dump(value))
    assert read_config(path) == expected


@pytest.mark.parametrize("section,name,value", [
    ("model_config", "lora_rank", 128),
    ("model_config", "upcast_vae", False),
    ("model_config", "unrecognized_name_or_path", "/ignored-but-not-registered"),
    ("data_config", "dataset_sampling_ratios", [1.]),
    ("data_config", "single_height", 192),
    ("training_config", "learning_rate", .001),
    ("training_config", "mixed_precision", "no"),
    ("training_config", "gan_g_weight", .1),
    ("training_config", "is_use_gan_final", False),
    ("training_config", "generator_dynamic_alpha", 5.),
    ("validation_config", "stage2_simulated_inference_steps", [1,1,1]),
])
def test_portable_config_still_rejects_unknown_math_precision_sampling_and_heads(tmp_path,section,name,value):
    import yaml
    from distillschedule.training.helios_gan_inputs import portable_config, read_config
    config=portable_config(); config[section][name]=value
    path=tmp_path/"changed.yaml"; path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match="config changed|algorithm/optimizer"):
        read_config(path)


@pytest.mark.parametrize("section,name,value", [
    ("model_config", "pretrained_model_name_or_path", {"not":"a locator"}),
    ("data_config", "gan_data_root", [5]),
])
def test_portable_config_rejects_invalid_unused_locator_types(tmp_path,section,name,value):
    import yaml
    from distillschedule.training.helios_gan_inputs import portable_config, read_config
    config=portable_config(); config[section][name]=value
    path=tmp_path/"changed.yaml"; path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match="invalid unused"):
        read_config(path)


def test_original_native_capture_and_new_cohort_official_helper_bindings(tmp_path):
    """Target dependency gate; NOT_RUN locally when native extras are missing.

    One new rank/cohort producer + original argument adapters, not a rerun of
    the completed baseline/full-model/math suites or a GPU acceptance claim.
    """
    import importlib.util
    missing=[name for name in ("omegaconf","einops","peft") if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip("NOT_RUN_NATIVE_DEPENDENCIES: "+", ".join(missing))
    import shutil
    import torch
    import yaml
    from distillschedule.training import helios_gan_inputs as inputs
    from distillschedule.training.helios_inputs import _inventory
    from distillschedule.workloads.best_known.registry import repository_root, file_sha256
    from runtime.operation_trace import _manifest_hash, write_operation_trace_bundle_manifest
    from runtime.gan_rolephase_official import load_official_config, score_official_kwargs, critic_official_kwargs
    training=inputs.portable_config()
    training["training_config"].update(is_random_drop=False,noise_corrupt_clean_prob_history=1.)
    config=tmp_path/"gan.yaml"; config.write_text(yaml.safe_dump(training)); controls=inputs.read_config(config)
    samples=[]; task=inputs.GANCohortTask(1); seed=42
    for offset in range(5):
        names=[]
        for rank in range(16):
            local=seed+task.index*1000000+rank*5+offset
            sections=next(n for n in range(2,32) if all(torch.randint(0,n,(1,),generator=torch.Generator().manual_seed(s)).item()>0
                for s in (local,local+999999)))
            path=tmp_path/f"feature-{offset}-{rank}.pt"
            value=.2+.001*(offset+rank)
            torch.save({"vae_latent":torch.tensor(value,dtype=torch.bfloat16).expand(sections,16,9,48,80),
                "prompt_embed":torch.full((2,4096),value,dtype=torch.bfloat16),"attention_mask":torch.ones(2,dtype=torch.bool)},path)
            names.append(str(path))
        samples.append(names)
    source=inputs.HeliosCohortSources.from_dict({"seed":seed,"samples":samples})
    inventory=_inventory(task,source)
    provenance={"training_config_sha256":file_sha256(config),"source_fingerprint":inputs.source_fingerprint()}
    actual=inputs.capture_rank(task,source,inventory,controls,rank=0,output=tmp_path/"actual",provenance=provenance)
    # Other ranks are metadata replicas ONLY. Only rank zero is read/executed;
    # this fixture is not a qualified all-rank producer or native GPU input run.
    bundle=tmp_path/"cohort"; ranks=[]
    for rank in range(16):
        directory=bundle/f"rank_{rank:02d}"; directory.mkdir(parents=True)
        manifest=deepcopy(actual.manifest)
        manifest["source"].update(rank=rank,samples=[r for r in inventory if r["rank"]==rank])
        manifest["trace_hash"]=_manifest_hash(manifest)
        (directory/"trace.json").write_text(json.dumps(manifest))
        if rank==0: shutil.copyfile(actual.root/"tensors.pt",directory/"tensors.pt")
        ranks.append({"rank":rank,"path":directory.name,"trace_hash":manifest["trace_hash"]})
    header=write_operation_trace_bundle_manifest(bundle,ranks)
    trace=inputs.load_cohort(task,bundle,expected_hash=header["bundle_hash"])
    inputs.validate_rank_payload(trace,0,source)
    g=trace.load_generator_entry(rank=0,device="cpu"); first=trace.source(rank=0,device="cpu")
    cfg=load_official_config(config)
    assert all(getattr(cfg.model_config,name) is None for name in _IGNORED_CONFIG_LOCATORS["model_config"])
    assert list(cfg.data_config.gan_data_root) == []
    assert not any(s in config.read_text() for s in ("/share/", "/home/shuchenweng/", "/mnt/bn/"))
    bound=score_official_kwargs(cfg,g.operation_trace,first,1005,torch.zeros(1,2,4096,dtype=torch.bfloat16),torch.device("cpu"))
    assert bound["global_step"]==1005 and bound["trace_prefix"]=="step/1005/generator"
    c=trace.load_critic_entry(rank=0,step=1006,device="cpu"); second=trace.source(rank=0,device="cpu")
    args=critic_official_kwargs(cfg,c.operation_trace,second,1006,torch.ones(task.latent_shape,dtype=torch.bfloat16),SimpleNamespace())
    assert args["global_step"]==1006 and args["trace_prefix"]=="step/1006/critic"
    assert not torch.equal(first["primary_target"],second["primary_target"])
    assert "gan_g_noise" in g.tensors and "r1_epsilon" in c.tensors
    trace.close()


@pytest.fixture
def cpu_native_cycle_admission(tmp_path, monkeypatch):
    """Real runner admission, with CPU stand-ins for DTensor/Gloo ownership."""
    from contextlib import nullcontext
    pytest.importorskip("torch", reason="NOT_RUN_CPU_CHARACTERIZATION: torch unavailable")
    from distillschedule.training import helios_gan_training as training, helios_lifecycle

    layout = {"topology": deepcopy(training.TOPOLOGY), "trainables_sha256": "a" * 64}
    binding = {"gan_stream": True, "gan_core": object(), "metadata_group": object(),
        "config": SimpleNamespace(trainer={"resume_from": "CPU_CHECKPOINT_STAND_IN"}),
        "identity": {"stream": "CPU_LAYOUT_CHARACTERIZATION", "native_owner_layout": deepcopy(layout)},
        "checkpoint_policy": {"old_policy_must_be_replaced": True}}
    generator, critic = SimpleNamespace(), SimpleNamespace()
    kwargs = dict(p0=SimpleNamespace(base=SimpleNamespace(_atomic_json=lambda *args: None)),
        args=object(), policy=object(), result={"build": {}}, output=tmp_path / "unused.json",
        rank=0, device="cpu", logical=object(), generator_engine=generator,
        critic_engine=critic, ema=SimpleNamespace(optimization_step=800), diagnostics=object(),
        negative=object(), cfg=SimpleNamespace(training_config=SimpleNamespace(is_separate_gan_grad=False)),
        generator_free_memory={})
    # No group initialization or DTensor/model construction is requested.
    monkeypatch.setattr(helios_lifecycle, "_plane", lambda value: nullcontext())
    monkeypatch.setattr(training, "owner_layout", lambda generator, critic: deepcopy(layout))
    return binding, layout, kwargs


@pytest.mark.parametrize("mutation", ["none", "not_mapping", "missing", "extra", "mesh",
    "mesh_names", "device", "placements", "short_digest", "uppercase_digest", "nonstring_digest"])
def test_stream_layout_record_rejects_changed_native_ownership(mutation):
    from distillschedule.training import helios_gan_training as training
    value = {"topology": deepcopy(training.TOPOLOGY), "trainables_sha256": "a" * 64}
    if mutation == "none": value = None
    elif mutation == "not_mapping": value = []
    elif mutation == "missing": value.pop("trainables_sha256")
    elif mutation == "extra": value["unbound"] = True
    elif mutation == "mesh": value["topology"]["mesh"] = list(range(8))
    elif mutation == "mesh_names": value["topology"]["mesh_dim_names"] = ["other"]
    elif mutation == "device": value["topology"]["device_type"] = "cpu"
    elif mutation == "placements": value["topology"]["placements"] = [{"kind": "replicate"}]
    elif mutation == "short_digest": value["trainables_sha256"] = "a" * 63
    elif mutation == "uppercase_digest": value["trainables_sha256"] = "A" * 64
    else: value["trainables_sha256"] = 1
    with pytest.raises(ValueError, match="GAN checkpoint native owner layout missing/changed"):
        training.validate_layout_record(value)


@pytest.mark.parametrize("saved", [None, "different_digest", "different_topology"])
def test_native_cycles_layout_mismatch_stops_before_delegation(cpu_native_cycle_admission, monkeypatch, saved):
    from distillschedule.training import helios_gan_training as training, helios_lifecycle
    binding, layout, kwargs = cpu_native_cycle_admission
    if saved is None: binding["identity"].pop("native_owner_layout")
    else:
        changed = deepcopy(layout)
        if saved == "different_digest": changed["trainables_sha256"] = "b" * 64
        else: changed["topology"]["placements"] = [{"kind": "shard", "dim": 1}]
        binding["identity"]["native_owner_layout"] = changed
    identity, policy = binding["identity"], binding["checkpoint_policy"]
    before = deepcopy(identity)
    calls = []
    monkeypatch.setattr(training, "execute_cycles", lambda **kw: calls.append(kw))
    with helios_lifecycle.original_training_scope(binding):
        with pytest.raises(ValueError, match="rebuilt native GAN owner layout differs"):
            training.run_native_cycles(**kwargs)
    assert not calls  # No update/checkpoint owner can run after the refusal.
    assert binding["identity"] is identity and identity == before
    assert binding["checkpoint_policy"] is policy
    assert kwargs["result"] == {"build": {}}


@pytest.mark.parametrize("resume", [False, True])
def test_native_cycles_layout_binds_identity_and_checkpoint_policy(cpu_native_cycle_admission, monkeypatch, resume):
    from distillschedule.training import helios_gan_training as training, helios_lifecycle
    from distillschedule.workloads.best_known.registry import fingerprint
    binding, layout, kwargs = cpu_native_cycle_admission
    if not resume:
        binding["config"].trainer.pop("resume_from")
        binding["identity"].pop("native_owner_layout")
    training.validate_layout_record(layout)
    previous = binding["identity"]
    calls, returned = [], object()
    def execute(**kw):
        assert kw["binding"] is binding
        assert binding["identity"] == {**previous, "native_owner_layout": layout}
        assert binding["checkpoint_policy"] == {
            "schema_version": "distillschedule.helios_gan_stream_checkpoint_policy.v1",
            "identity_sha256": fingerprint(binding["identity"])}
        calls.append(kw)
        return returned
    monkeypatch.setattr(training, "execute_cycles", execute)
    with helios_lifecycle.original_training_scope(binding):
        assert training.run_native_cycles(**kwargs) is returned
    assert len(calls) == 1 and binding["identity"] is not previous
    for name in ("p0", "args", "policy", "result", "diagnostics", "negative", "cfg", "generator_free_memory"):
        assert calls[0][name] is kwargs[name]
    assert calls[0]["generator"] is kwargs["generator_engine"]
    assert calls[0]["critic"] is kwargs["critic_engine"]


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_installed_stream_wrapper_restores_real_hooks_and_policy_caches(
        cpu_native_cycle_admission, monkeypatch, outcome):
    """Real installer/runner/SAC bind and unwind; NO native math/kernel proof.

    Hardware, FA3 loading, PEFT types, DTensor layout and Gloo are explicit
    CPU stand-ins. Real meta Linear/Module hooks, inventory/merge guards,
    source checks (except installed CPU checkpoint), ExitStack and SAC
    abort/restored remain in the chain. Only the downstream cycle body is
    replaced; neither the installed wrapper nor core.bind is mocked.
    """
    import ast
    import sys
    from types import ModuleType
    import torch
    import torch.utils.checkpoint as checkpoint
    from distillschedule.training import helios_gan_training as training, helios_lifecycle
    from distillschedule.runtime import fsdp2_static as fsdp
    from distillschedule.runtime.helios_gan_manual import runtime, stream, backend_abi
    from distillschedule.workloads.best_known.registry import repository_root

    binding, layout, kwargs = cpu_native_cycle_admission
    binding.pop("gan_core")
    binding["implementation"] = training.implementation()
    binding["assets"] = {"assets": {"fa3_kernel": {"path": "CPU_FA3_LOAD_STUB_NOT_A_BINARY"}}}
    external = repository_root() / "third_party/helios_compat/source"
    modules = {}
    for name in ("helios", "helios.modules", "helios.modules.helios_kernels",
                 "helios.modules.helios_kernels.attention_dispatch", "helios.modules.transformer_helios",
                 "helios.utils", "helios.utils.utils_helios_post"):
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent: setattr(modules[parent], child, module)
    dispatch = modules["helios.modules.helios_kernels.attention_dispatch"]
    transformer = modules["helios.modules.transformer_helios"]
    dispatch.__file__ = str(external / "helios/modules/helios_kernels/attention_dispatch.py")
    transformer.__file__ = str(external / "helios/modules/transformer_helios.py")
    def no_kernel(*args, **kw):
        pytest.fail("CPU wrapper characterization must not execute native kernels")
    dispatch.flash_attn_func = no_kernel
    dispatch.flash_attn_varlen_func = no_kernel
    dispatch.attn_varlen_func = transformer.attn_varlen_func = no_kernel
    # Preserve the exact original helper source attestation without importing
    # its native dependency group or executing any of its loss math.
    helper_path = external / "helios/utils/utils_helios_post.py"
    helper = next(node for node in ast.parse(helper_path.read_text()).body
                  if isinstance(node, ast.FunctionDef) and node.name == "_critic_loss")
    code = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), helper], type_ignores=[])
    namespace = modules["helios.utils.utils_helios_post"].__dict__
    exec(compile(ast.fix_missing_locations(code), str(helper_path), "exec"), namespace)

    class LoraLayer(torch.nn.Module):
        lora_A = lora_B = object()
        merged = False
        merged_adapters = ()
        def merge(self): return "CPU_ORIGINAL_MERGE"
    class Linear(LoraLayer):
        pass
    class Conv3d(LoraLayer):
        pass
    generator_model, critic_model = torch.nn.Module(), torch.nn.Module()
    generator_model.layers = torch.nn.ModuleList(Linear() for _ in range(406))
    for name in ("patch_short", "patch_mid", "patch_long"): setattr(generator_model, name, Conv3d())
    critic_model.blocks = torch.nn.ModuleList()
    for _ in range(40):
        block = torch.nn.Module(); block.ffn = torch.nn.Module()
        down = Linear()
        down.base_layer = torch.nn.Linear(13824, 5120, device="meta")
        down.base_layer.requires_grad_(False)
        block.ffn.net = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity(), down])
        critic_model.blocks.append(block)
    critic_model.layers = torch.nn.ModuleList(Linear() for _ in range(366))
    generator = SimpleNamespace(module=generator_model, global_steps=2)
    critic = SimpleNamespace(module=critic_model, global_steps=10, backward=no_kernel, step=no_kernel)
    kwargs.update(generator_engine=generator, critic_engine=critic)
    monkeypatch.setattr(runtime, "_peft_lora_types", lambda: (Linear, Conv3d, LoraLayer))
    device_calls = []
    fake_torch = SimpleNamespace(__version__="2.10.0+cu128", version=SimpleNamespace(cuda="12.8"),
        nn=torch.nn, Tensor=torch.Tensor, cuda=SimpleNamespace(
            set_device=lambda rank: device_calls.append(rank),
            get_device_capability=lambda rank: (8, 0), get_device_name=lambda rank: "CPU_STUB_A100"))
    p0 = kwargs["p0"]
    p0.torch, p0._run_gan_critic_precomputed = fake_torch, no_kernel
    p0._attention_backend_receipt = lambda requested: {"status": "CPU_REFERENCE_STUB"}
    p0.configure_fsdp2_checkpointing = lambda: {"selective_flash_op": backend_abi.FA3_FW,
        "scope": "CPU_CHECKPOINT_STUB", "selective_flash_enabled": True}
    for key in tuple(runtime.os.environ):
        if key.startswith("DISTILLSCHEDULE_") or key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            monkeypatch.delenv(key)
    for key, value in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "16"}.items():
        monkeypatch.setenv(key, value)
    loads = []
    def load_backend(torch_arg, path):
        assert torch_arg is fake_torch and path == binding["assets"]["assets"]["fa3_kernel"]["path"]
        loads.append(path)
        return no_kernel, {"forward": backend_abi.FA3_FW, "backward": backend_abi.FA3_BW}, {
            "scope": "CPU_LOADER_STUB_NOT_NATIVE_ATTESTATION"}
    monkeypatch.setattr(runtime.backend_library, "load_explicit", load_backend)
    def cpu_checkpoint_source(core, fsdp_arg, checkpoint_arg):
        assert fsdp_arg is fsdp and checkpoint_arg is checkpoint
        core.state["current_fsdp_source"] = stream.validate_fsdp_source(fsdp_arg)
        core.state["checkpoint_source"] = "CPU_CHECKPOINT_STUB_NOT_NATIVE_SOURCE_ADMISSION"
    monkeypatch.setattr(stream.NativeDownStreamSAC, "validate_sources", cpu_checkpoint_source)
    aborts, captured = [], []
    original_abort = stream.NativeDownStreamSAC.abort
    def record_abort(core):
        assert core.current is not None and core.stores and core.rows
        aborts.append(core)
        original_abort(core)
    monkeypatch.setattr(stream.NativeDownStreamSAC, "abort", record_abort)
    targets = [(training, "run_native_cycles"), (dispatch, "flash_attn_func"),
        (dispatch, "flash_attn_varlen_func"), (transformer, "attn_varlen_func"),
        (fsdp, "SELECTIVE_FLASH_ATTN_OP"), (fsdp, "fsdp2_metadata_tolerant_checkpoint"),
        (fsdp, "_selective_flash_checkpoint_contexts"), (p0, "_attention_backend_receipt"),
        (p0, "configure_fsdp2_checkpointing"), (p0, "_run_gan_critic_precomputed"),
        (critic, "backward"), (critic, "step")]
    originals = [(obj, name, name in vars(obj), vars(obj).get(name), getattr(obj, name)) for obj, name in targets]
    loras = [m for model in (generator_model, critic_model) for m in model.modules() if isinstance(m, LoraLayer)]
    merges = [(layer, layer.merge) for layer in loras]
    all_modules = list(generator_model.modules()) + list(critic_model.modules())
    error, returned = RuntimeError("INJECTED_AFTER_REAL_STREAM_CORE_BINDING"), object()
    def execute(**kw):
        core = binding["gan_core"]
        captured.append(core)
        assert core.bound and core.generator is generator and core.critic is critic
        assert len(core._restore_targets) == 5
        assert len(loras) == 815 and all("merge" in vars(layer) for layer in loras)
        for obj, name, _, _, old in originals: assert getattr(obj, name) != old
        assert all(len(block.ffn.net[2].base_layer._forward_pre_hooks) == 1
                   and len(block.ffn.net[2].base_layer._forward_hooks) == 1 for block in critic_model.blocks)
        assert len(critic_model._forward_pre_hooks) == 1
        with pytest.raises(ValueError, match="forbids every merged-LoRA execution"): loras[0].merge()
        assert p0._attention_backend_receipt("local-fa2")["numerical_status"] == "REVIEW"
        assert p0.configure_fsdp2_checkpointing()["scope"] == "SELECTIVE_PINNED_FA3_WHOLE_RESULT_REPLAY"
        if outcome == "failure":
            core.stores["CPU_POLICY_REFERENCE"] = {"queue": [object()]}
            core.current, core.cohort, core.rows = {"CPU_ACTIVE_GRAPH": True}, 0, {(1000, "B1_DENOISE"): 11040}
            raise error
        # Synthetic drained completion exercises the real finish guard only;
        # it is not a completed training cycle or an executed-effects receipt.
        core.started_at, core.completed = 0, 1
        return returned
    monkeypatch.setattr(training, "execute_cycles", execute)
    def assert_restored():
        assert "gan_core" not in binding
        for obj, name, own, raw, old in originals:
            assert (name in vars(obj)) == own
            assert vars(obj).get(name) is raw if own else getattr(obj, name) == old
        assert all("merge" not in vars(layer) and layer.merge == merge for layer, merge in merges)
        assert all(not m._forward_pre_hooks and not m._forward_hooks for m in all_modules)
        assert captured and not captured[0].stores and not captured[0].rows
        assert captured[0].current is captured[0].cohort is None
        assert captured[0].state["restored"] is True
    with helios_lifecycle.original_training_scope(binding):
        try:
            with runtime.manual_runtime_scope(p0) as receipt:
                assert receipt["restored"] is False and "gan_core" not in binding
                try:
                    assert training.run_native_cycles(**kwargs) is returned
                finally:
                    # Check before the outer context can unwind its ExitStack.
                    assert_restored()
        except RuntimeError as raised:
            assert outcome == "failure" and raised is error
        else:
            assert outcome == "success"
    assert len(captured) == len(loads) == 1 and device_calls == [0]
    assert kwargs["result"]["build"]["gan_manual_runtime"] is receipt
    assert receipt["down40"] is captured[0].state and receipt["runner_calls"] == 1
    assert receipt["restored"] is (outcome == "success")
    assert receipt.get("failed", False) is (outcome == "failure")
    assert captured[0].state.get("failed", False) is (outcome == "failure")
    assert aborts == ([captured[0]] if outcome == "failure" else [])
    assert receipt["numerical_status"] == "REVIEW" and receipt["formal_speedup_eligible"] is False
    assert helios_lifecycle.current_training() is None


@pytest.mark.parametrize("mutation,message", [
    ("inventory", "effect inventory changed"), ("stale", "mechanisms missing/stale"),
    ("zero", "mechanisms missing/stale"), ("negative", "mechanisms missing/stale"),
    ("down_count", "mechanisms missing/stale"), ("down_replay", "mechanisms missing/stale"),
    ("lora_attempt", "LoRA execution/replay counter mismatch"),
    ("lora_completed", "LoRA execution/replay counter mismatch"),
    ("lora_kernel", "LoRA execution/replay counter mismatch"),
    ("lora_grad", "LoRA execution/replay counter mismatch"),
    ("fa3_replay", "FA3/norm replay counter mismatch"),
    ("norm_backward", "FA3/norm replay counter mismatch"),
])
def test_stream_executed_effects_rejects_stale_and_mismatched_counters(mutation, message):
    from distillschedule.training import helios_gan_training as training
    build = json.loads((Path(__file__).parent / "fixtures/helios_gan_manual_counters.json").read_text())["build"]
    full = training.implementation().candidate_bundle.effect_snapshot(build)
    before, delta = dict.fromkeys(full, 0), {key: value // 2 for key, value in full.items()}
    _set_effect_counters(build, delta)
    valid = training.executed_effects(build, before)
    assert valid["status"] == "PASS_EXECUTED_FRESH_GAN_BUNDLE"
    assert valid["call_deltas"] == delta and valid["numerical_status"] == "REVIEW"
    if mutation == "inventory": before.pop("public_backend_call_count")
    elif mutation == "stale": before = dict(delta)
    elif mutation == "zero": delta["public_backend_call_count"] = 0
    elif mutation == "negative": before["public_backend_call_count"] = delta["public_backend_call_count"] + 1
    elif mutation == "down_count": delta["down_original"] = delta["down_replay"] = 399
    elif mutation == "down_replay": delta["down_replay"] = 399
    else:
        key = {"lora_attempt": "generator.lora_b_gemm_selected_attempt_count",
            "lora_completed": "critic.lora_b_gemm_completed_invocation_count",
            "lora_kernel": "generator.lora_b_gemm_kernel_call_count",
            "lora_grad": "critic.lora_b_gemm_selected_grad_invocation_count",
            "fa3_replay": "critic_checkpoint.selective_flash_recompute_reuse_count",
            "norm_backward": "norm.backward_kernel_call_count"}[mutation]
        delta[key] += 1
    _set_effect_counters(build, delta)
    with pytest.raises(ValueError, match=message): training.executed_effects(build, before)
