"""Original Helios parameter binding; CPU metadata fixtures are not GPU evidence."""
from copy import deepcopy
from dataclasses import replace
import builtins
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace as NS

import pytest

from distillschedule.search.helios_implementation import HeliosImplementation
from distillschedule.search.helios_request import OriginalHeliosRequest, TRACE_KEY, TRACE_HASH_KEY
from distillschedule.search.contracts import semantic_contract
from distillschedule.workloads.best_known.registry import fingerprint, load_profile


def implementation(request=None):
    return HeliosImplementation(request.variant if request else "no-gan", "world16-fsdp", 512, 5, False,
                                request=request)


def controls(request):
    from runtime.static_world16_trace import _generator_keys, _critic_keys
    from runtime.operation_trace import OperationTrace
    operations, tensors = [], {}
    for step in request.source_steps:
        operations.append(dict(key=f"step/{step}/branch", control=dict(global_step=step,
            dfake_gen_update_ratio=5, train_generator=step == request.source_steps[0],
            train_critic=True, gan=request.variant == "gan", reward=False, vae_decode=False)))
        phases = [("critic", _critic_keys(step, gan=request.variant == "gan"))]
        if step == request.source_steps[0]:
            phases.append(("generator", _generator_keys(step, gan=request.variant == "gan")))
        for phase, keys in phases:
            for key in keys.values():
                tensors[key] = {"shape": list(request.latent_shape), "dtype": "bfloat16"}
            sequence = [dict(stage=i, call_count=1, denoising_exit=0, timestep=[1000.],
                denoising_step_budget=request.stage2_inference_steps[i],
                timestep_grid=[1000., 500.][:request.stage2_inference_steps[i]],
                latent_shape=[*request.latent_shape[:3], *hw])
                for i, hw in enumerate(request.stage_spatial_shapes)]
            operations.append(dict(key=f"step/{step}/{phase}/rollout", control=dict(call_count=3,
                pyramid_stage_call_sequence=sequence, initial_synchronized_pyramid_stage_exit=[0, 0, 0],
                stage_schedule_scope="global_rank0_broadcast_equivalent")))
            operations.append(dict(key=f"step/{step}/{phase}/{'train' if phase == 'critic' else 'dmd'}", control={}))
            if request.variant == "gan":
                operations.append(dict(key=f"step/{step}/{phase}/{'gan_d' if phase == 'critic' else 'gan_g'}", control={}))
    contract = request.trace_contract()
    return {rank: OperationTrace(Path(f"/fixture/rank{rank}"),
                deepcopy(dict(contract=contract, tensors=tensors, operations=operations, trace_hash="0"*64)), {})
            for rank in range(16)}


@pytest.mark.parametrize("variant", ["no-gan", "gan"])
@pytest.mark.parametrize("frames,horizon", [(33, 2), (17, 2), (33, 1), (17, 1)])
def test_request_roundtrip_lowering_graph_and_cost_identity(variant, frames, horizon):
    from distillschedule.search.execution import resolve_candidate, validated_profile
    from distillschedule.search.helios_action_costs import action_graph, component_contract
    from distillschedule.workloads.best_known.registry import repository_root
    request = OriginalHeliosRequest(variant=variant, video_frames=frames,
                                   stage2_inference_steps=(horizon,)*3)
    impl = implementation(request)
    assert OriginalHeliosRequest.from_dict(request.to_dict()) == request
    assert HeliosImplementation.parse(impl.id) == impl
    assert HeliosImplementation.from_assignment(impl.assignment) == impl
    profile = impl.lower_profile()
    assert profile["contract"][TRACE_KEY] == request.to_dict()
    assert profile["contract"]["full_latent_shape"] == list(request.latent_shape)
    assert semantic_contract(profile) != semantic_contract(load_profile(impl.workload))
    plan = resolve_candidate(impl.workload, impl.id, mode="correctness")
    assert validated_profile(plan, root=repository_root()) == profile
    graph = action_graph(impl)
    assert graph.metadata[TRACE_KEY] == request.to_dict()
    assert graph.metadata[TRACE_HASH_KEY] == request.digest
    legacy = replace(impl, request=None)
    work = dict(physical_batch=1, logical_batch=16)
    assert component_contract(graph.operations[0], impl, work).digest != component_contract(
        action_graph(legacy).operations[0], legacy, work).digest
    assert impl.version_mapping(profile["contract"]) == legacy.version_mapping(legacy.lower_profile()["contract"])


def test_default_identity_and_serialization_are_untouched():
    impl = implementation()
    assert impl.id == "joint-helios1-nogan-fsdp-c512-p0-w5"
    assert set(impl.assignment) == {"variant", "mesh", "chunk_tokens", "rollout_window", "persistent_unshard"}
    assert TRACE_KEY not in impl.lower_profile()["contract"]
    assert semantic_contract(impl.lower_profile()) == semantic_contract(load_profile(impl.workload))
    assert "DISTILLSCHEDULE_HELIOS_REQUEST" not in impl.environment()


@pytest.mark.parametrize("change", [dict(video_frames=1), dict(video_frames=21), dict(world_size=8),
    dict(fake_updates=1), dict(physical_batch=8), dict(global_batch=8), dict(video_height=320),
    dict(stage2_inference_steps=(1, 2, 1)), dict(stage2_inference_steps=(0, 0, 0)),
    dict(stage2_inference_steps=(True, True, True)), dict(variant="wan")])
def test_unimplemented_workload_dimensions_fail_closed(change):
    with pytest.raises(ValueError):
        OriginalHeliosRequest(**change)
    with pytest.raises(ValueError):
        OriginalHeliosRequest.from_dict({"unknown": 1})


@pytest.mark.parametrize("variant", ["no-gan", "gan"])
def test_exact_request_trace_binding_and_stale_f33_rejection(variant):
    request = OriginalHeliosRequest(variant=variant, video_frames=17, stage2_inference_steps=(1,)*3)
    rows = controls(request)
    assert request.validate_rank_controls(rows)["status"] == "PASS"
    stale = controls(replace(request, video_frames=33))
    with pytest.raises(ValueError, match="request|shape"):
        request.validate_rank_controls(stale)
    # A rehashed/relabelled manifest is still rejected by tensor/stage metadata.
    for row in stale.values():
        row.manifest["contract"] = request.trace_contract()
    with pytest.raises(ValueError, match="shape|geometry"):
        request.validate_rank_controls(stale)


@pytest.mark.parametrize("mutation", ["digest", "horizon", "budget", "rank", "branch", "local", "unknown"])
def test_trace_control_tampering_is_rejected(mutation):
    request = OriginalHeliosRequest(video_frames=17, stage2_inference_steps=(1,)*3)
    rows = controls(request)
    row = rows[0]
    if mutation == "digest": row.manifest["contract"][TRACE_HASH_KEY] = "f"*64
    if mutation == "horizon": row.manifest["contract"][TRACE_KEY]["stage2_inference_steps"] = [2]*3
    if mutation == "unknown": row.manifest["contract"][TRACE_KEY]["ignored"] = True
    if mutation == "rank": rows.pop(15)
    if mutation == "branch": row.operation("step/10/branch")["control"]["dfake_gen_update_ratio"] = 1
    control = row.operation("step/10/critic/rollout")["control"]
    if mutation == "budget": control["pyramid_stage_call_sequence"][0]["denoising_step_budget"] = 2
    if mutation == "local": control["stage_schedule_scope"] = "rank_local"
    with pytest.raises(ValueError): request.validate_rank_controls(rows)


def test_explicit_runtime_identity_requires_matching_request_env_and_config(monkeypatch):
    from distillschedule.search.helios_implementation import current_implementation
    request = OriginalHeliosRequest(variant="gan", video_frames=17, stage2_inference_steps=(1,)*3)
    impl = implementation(request)
    env = impl.environment() | {"DISTILLSCHEDULE_HELIOS_PLANNER_MODE": "native"}
    assert current_implementation(env) == impl
    assert json.loads(env["DISTILLSCHEDULE_HELIOS_REQUEST"]) == request.to_dict()
    for field in ("DISTILLSCHEDULE_HELIOS_REQUEST", "DISTILLSCHEDULE_HELIOS_REQUEST_SHA256"):
        with pytest.raises(ValueError): current_implementation({k: v for k, v in env.items() if k != field})
    with pytest.raises(ValueError): current_implementation(implementation().environment() | {
        "DISTILLSCHEDULE_HELIOS_PLANNER_MODE": "native", "DISTILLSCHEDULE_HELIOS_REQUEST": "{}"})
    cfg = NS(training_config=NS(num_critic_input_frames=9, is_enable_stage2=True, stage2_num_stages=3),
             validation_config=NS(stage2_simulated_inference_steps=[2]*3))
    bound = request.bind_official_config(cfg)
    assert cfg.training_config.num_critic_input_frames == 9
    assert bound.training_config.num_critic_input_frames == 5
    assert bound.validation_config.stage2_simulated_inference_steps == [1]*3
    request.validate_official_config(bound)
    with pytest.raises(ValueError): request.validate_official_config(cfg)


def test_original_helper_geometry_is_helios_not_a_wan_model():
    from distillschedule.runtime.rollout_geometry import rollout_geometry
    request = OriginalHeliosRequest(video_frames=17, stage2_inference_steps=(1,)*3)
    trace = controls(request)[0]
    geometry = rollout_geometry(trace)
    assert geometry == request
    assert geometry.model_family == "helios14b"
    trace.manifest["contract"][TRACE_HASH_KEY] = "stale"
    with pytest.raises(ValueError): rollout_geometry(trace)


def test_real_fake_tensor_metadata_rejects_stale_f33():
    torch = pytest.importorskip("torch")
    from distillschedule.runtime.helios_planner_cycle import _Observer
    request = OriginalHeliosRequest(video_frames=17)
    observer = _Observer(torch=torch, device="meta", rank=0, impl=implementation(request),
        generator=NS(global_steps=2), critic=NS(global_steps=10), ema=NS(optimization_step=10),
        mode="native", version_offsets=dict(generator=0, critic=0, ema=0))
    observer.current = dict(action_id="r0", produced=[])
    observer.actions = [observer.current]
    with pytest.raises(ValueError, match="rollout"):
        observer.produced(10, torch.empty((1, 16, 9, 48, 80), device="meta", dtype=torch.bfloat16))
    observer.produced(10, torch.empty(request.latent_shape, device="meta", dtype=torch.bfloat16))
    assert observer.live[10]["shape"] == [1, 16, 5, 48, 80]


def test_p0_request_binding_precedes_trace_construction_and_gan_call():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py").read_text()
    assert source.index("helios_request =") < source.index("trace = StaticWorld16Trace(")
    assert "request=helios_request" in source
    assert "helios_request.bind_official_config(cfg)" in source
    for path in ("runtime/gst_pool_helios.py", "runtime/static_fsdp2_p0_helios.py"):
        assert "from distillschedule.runtime.rollout_geometry import rollout_geometry" in (root/path).read_text()


def write_bundle(request, root):
    from runtime.operation_trace import _manifest_hash, write_operation_trace_bundle_manifest
    entries = []
    for rank, trace in controls(request).items():
        path = root / f"rank_{rank:02d}"
        path.mkdir(parents=True)
        trace.manifest["trace_hash"] = _manifest_hash(trace.manifest)
        (path / "trace.json").write_text(json.dumps(trace.manifest))
        entries.append(dict(rank=rank, path=path.name, trace_hash=trace.trace_hash))
    return write_operation_trace_bundle_manifest(root, entries)


@pytest.mark.parametrize("variant", ["no-gan", "gan"])
def test_compiler_exact_fresh_trace_hash_binding_and_default_regression(tmp_path, monkeypatch, variant):
    from distillschedule.search.execution import resolve_candidate, validated_profile
    from distillschedule.workloads.best_known.registry import file_sha256, repository_root
    request = OriginalHeliosRequest(variant=variant, video_frames=17, stage2_inference_steps=(1,)*3)
    impl = implementation(request)
    bundle = write_bundle(request, tmp_path / "trace")
    assets = json.loads((repository_root()/f"configs/distillschedule/best_known/assets_examples/{impl.workload}.json").read_text())
    assets["assets"]["trace"] = dict(path=str(tmp_path/"trace"), identity_file="bundle.json",
                                     sha256=file_sha256(tmp_path/"trace/bundle.json"))
    path = tmp_path/"assets.json"
    path.write_text(json.dumps(assets))
    plan = resolve_candidate(impl.workload, impl.id, mode="correctness", assets_path=path)
    env_key = "DISTILLSCHEDULE_GAN_TRACE_HASH" if variant == "gan" else "DISTILLSCHEDULE_TRACE_BUNDLE_HASH"
    assert plan["nodes"][0]["environment"][env_key] == bundle["bundle_hash"]
    assert plan["original_helios_trace_binding"] == dict(bundle_hash=bundle["bundle_hash"],
        bundle_file_sha256=assets["assets"]["trace"]["sha256"], request_sha256=request.digest)
    validated_profile(plan, root=repository_root())
    # Exercise the real compiler -> local-launch admission seam. Only missing
    # external assets and the intentionally dirty host checkout are fixtures;
    # effective environment/request/trace gates below are not stubbed.
    from distillschedule.workloads.best_known import dispatch
    monkeypatch.setattr(dispatch, "source_layout", lambda *a, **k: {})
    monkeypatch.setattr(dispatch, "verify_assets", lambda *a, **k: [])
    real_output = dispatch.subprocess.check_output
    def checkout_output(argv, **kwargs):
        if "status" in argv:
            return ""
        return real_output(argv, **kwargs)
    monkeypatch.setattr(dispatch.subprocess, "check_output", checkout_output)
    dispatch.validate_local(plan, root=repository_root(), node_rank=0)
    old = resolve_candidate(impl.workload, replace(impl, request=None).id, mode="correctness", assets_path=path)
    assert old["nodes"][0]["environment"][env_key] == load_profile(impl.workload)["runtime"]["environment"][env_key]
    assert "original_helios_trace_binding" not in old
    bad = deepcopy(plan)
    bad["nodes"][0]["environment"][env_key] = "0"*64
    with pytest.raises(ValueError): validated_profile(bad, root=repository_root())


@pytest.mark.parametrize("mutation", ["file_sha", "bundle_hash", "old_request", "wrong_identity"])
def test_compiler_rejects_trace_sha_bundle_hash_or_request_drift(tmp_path, mutation):
    from distillschedule.search.execution import resolve_candidate
    from distillschedule.workloads.best_known.registry import file_sha256, repository_root
    request = OriginalHeliosRequest(video_frames=17, stage2_inference_steps=(1,)*3)
    write_bundle(replace(request, video_frames=33) if mutation == "old_request" else request, tmp_path/"trace")
    if mutation == "bundle_hash":
        path = tmp_path/"trace/bundle.json"
        raw = json.loads(path.read_text()); raw["bundle_hash"] = "f"*64
        path.write_text(json.dumps(raw))
    impl = implementation(request)
    assets = json.loads((repository_root()/f"configs/distillschedule/best_known/assets_examples/{impl.workload}.json").read_text())
    assets["assets"]["trace"] = dict(path=str(tmp_path/"trace"), identity_file="bundle.json",
        sha256="0"*64 if mutation == "file_sha" else file_sha256(tmp_path/"trace/bundle.json"))
    if mutation == "wrong_identity": assets["assets"]["trace"]["identity_file"] = "IDENTITY.sha256"
    path = tmp_path/"assets.json"; path.write_text(json.dumps(assets))
    with pytest.raises(ValueError): resolve_candidate(impl.workload, impl.id, mode="correctness", assets_path=path)


def test_benchmark_scope_preserves_rank_native_grids_and_strict_call_parity():
    request = OriginalHeliosRequest(video_frames=17, stage2_inference_steps=(1,)*3,
                                   stage_schedule_policy="benchmark_v4_strict_global")
    rows = controls(request)
    for rank, trace in rows.items():
        for op in trace.manifest["operations"]:
            control = op["control"]
            if "pyramid_stage_call_sequence" not in control: continue
            control.update(stage_schedule_policy=request.stage_schedule_policy, all_history_zero=rank % 2 == 0)
            for stage in control["pyramid_stage_call_sequence"]:
                stage.update(denoising_step_budget=2 if rank % 2 == 0 else 1,
                             timestep_grid=[1000., 500.] if rank % 2 == 0 else [1000.])
    result = request.validate_rank_controls(rows)
    assert result["cross_rank_call_count_parity"]
    assert result["upstream_local_amplified_exit_distribution"] == "NOT_CLAIMED"
    assert "-b4" in implementation(request).id
    assert HeliosImplementation.parse(implementation(request).id).request == request
    # A rank-local extra call is still rejected even on a doubled native grid.
    stage = rows[0].operation("step/10/critic/rollout")["control"]["pyramid_stage_call_sequence"][0]
    stage.update(call_count=2, denoising_exit=1, timestep=[1000., 500.])
    with pytest.raises(ValueError): request.validate_rank_controls(rows)


def test_request_native_grid_recomputed_from_original_source_and_history_predicate(monkeypatch):
    """CPU-only original scheduler math; optional text utilities are not imported."""
    import ast
    import importlib.util
    import math
    import sys
    import types
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from distillschedule.workloads.best_known.registry import repository_root
    root = repository_root()/"third_party/helios_compat/source"
    schedpath = root/"helios/scheduler/scheduling_helios.py"
    spec = importlib.util.spec_from_file_location("helios.scheduler.scheduling_helios", schedpath)
    scheduler = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, scheduler)
    spec.loader.exec_module(scheduler)
    # Execute unchanged source definitions, not a numerical substitute. The
    # runtime imports the complete native module in its validated environment.
    basepath = root/"helios/utils/utils_base.py"
    base = types.ModuleType("helios.utils.utils_base"); base.__file__ = str(basepath)
    base.__dict__.update(torch=torch, math=math, Literal=__import__("typing").Literal)
    selected = [node for node in ast.parse(basepath.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name in ("calculate_shift", "apply_schedule_shift")]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(basepath), "exec"), base.__dict__)
    monkeypatch.setitem(sys.modules, base.__name__, base)
    request = OriginalHeliosRequest(video_frames=17, stage2_inference_steps=(1,)*3,
                                   stage_schedule_policy="benchmark_v4_strict_global")
    trace = controls(request)[0]
    for op in trace.manifest["operations"]:
        c = op["control"]
        if "pyramid_stage_call_sequence" not in c: continue
        c.update(stage_schedule_policy=request.stage_schedule_policy, all_history_zero=True)
        for stage in c["pyramid_stage_call_sequence"]:
            s = scheduler.HeliosScheduler(shift=1., stages=3, stage_range=[0, 1/3, 2/3, 1], gamma=1/3)
            index = stage["stage"]; s.set_timesteps(3, index, device="cpu")
            s.sigmas = torch.cat([s.sigmas[:-2], s.sigmas[-1:]])
            sigmas, _ = base.apply_schedule_shift(s.sigmas,
                torch.empty(stage["latent_shape"], device="meta"), sigmas_two=s.sigmas_per_stage[index],
                base_seq_len=256, max_seq_len=4096, base_shift=.5, max_shift=1.15, time_shift_type="linear")
            t = s.timesteps_per_stage[index]
            grid = (t.min()+sigmas[:-1]*(t.max()-t.min())).tolist()
            stage.update(denoising_step_budget=2, timestep_grid=grid, timestep=grid[:1])
    assert request.validate_native_stage_grids(trace)["status"] == "PASS"
    trace.operation("step/10/critic/rollout")["control"]["pyramid_stage_call_sequence"][0]["timestep_grid"][1] += 10.
    with pytest.raises(ValueError, match="native.*grid|grid.*native"):
        request.validate_native_stage_grids(trace)
    # Actual tensor sums, not an unaudited control label, decide the native
    # amplified budget. These are small test tensors, not real video features.
    for op in trace.manifest["operations"]:
        if not op["key"].endswith("/rollout"): continue
        prefix = op["key"].removesuffix("/rollout")
        for role in ("history_short", "history_mid", "history_long"):
            trace.tensors[prefix+"/"+role] = torch.zeros((1,))
    request.validate_loaded_history(trace)
    trace.tensors["step/10/critic/history_short"].fill_(1.)
    with pytest.raises(ValueError, match="history"):
        request.validate_loaded_history(trace)


def test_request_inventory_and_real_typed_component_problem(tmp_path, monkeypatch):
    from distillschedule.search import helios_model_adapter as m
    from distillschedule.search.workflow_cli import candidate_inventory
    from distillschedule.search.helios_action_costs import anchors, PRECISION
    from test_refactor_helios_search_bridge import native_rows
    from test_refactor_model_planning import BINDING
    request = OriginalHeliosRequest(video_frames=17, stage2_inference_steps=(1,)*3)
    impl = implementation(request)
    inventory = candidate_inventory(impl.workload, request=request)
    assert inventory["workload_request"] == request.to_dict()
    assert all(HeliosImplementation.parse(i["id"]).request == request for i in inventory["candidates"])
    assert inventory["native_adapter"][TRACE_KEY] == request.to_dict()
    rows, _ = native_rows(monkeypatch, window=5)
    for row in rows:
        for mem in [row["post_setup_memory"], *[a[k] for c in row["cycles"] for a in c["actions"]
                    for k in ("memory_before", "memory_after")]]:
            mem.update(driver_total_bytes=40*2**30, driver_free_bytes=35*2**30)
    pair = anchors(impl.workload, request=request)
    identity = {**BINDING, "source_head": rows[0]["source_head"], "precision_fingerprint": PRECISION,
                "semantic_fingerprint": fingerprint(semantic_contract(impl.lower_profile()))}
    monkeypatch.setattr(m, "_read_fragment", lambda ref, root: ({"binding": identity}, pair[ref["index"]], deepcopy(rows)))
    path = tmp_path/"layout.json"
    path.write_text(json.dumps(dict(schema_version=m.SCHEMA, fragments=[{"index": 0}, {"index": 1}],
                                   **{TRACE_KEY: request.to_dict()})))
    problem = m.build_problem(path)
    assert problem.graph.metadata[TRACE_HASH_KEY] == request.digest
    assert len(problem.decisions["original_request"]) == 1
    assert all(HeliosImplementation.from_assignment(seed).request == request for seed in problem.seeds)
    data = json.loads(path.read_text()); data.pop(TRACE_KEY); path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="request"): m.build_problem(path)


def test_effective_request_receipt_cannot_reuse_another_launch_trace(monkeypatch):
    from test_refactor_helios_planner_native import _fixture
    f = _fixture(monkeypatch)
    request = OriginalHeliosRequest(video_frames=17)
    impl = implementation(request)
    env = impl.environment() | {"DISTILLSCHEDULE_HELIOS_PLANNER_MODE": "native",
                               impl.trace_hash_environment: "b"*64}
    f.kwargs["args"].original_helios_request = request
    f.kwargs["args"].expected_trace_hash = "b"*64
    binding = impl.receipt_binding(args=f.kwargs["args"], policy=f.kwargs["policy"], environment=env)
    assert binding["expected_trace_hash"] == "b"*64
    f.kwargs["args"].expected_trace_hash = "c"*64
    with pytest.raises(ValueError, match="trace hash"):
        impl.receipt_binding(args=f.kwargs["args"], policy=f.kwargs["policy"], environment=env)


@pytest.mark.parametrize('variant', ['no-gan', 'gan'])
@pytest.mark.parametrize('mutation', ['digest', 'path', 'request'])
def test_direct_launcher_binds_admitted_request_trace(tmp_path, monkeypatch, variant, mutation):
    from distillschedule.search import helios_native as n
    from distillschedule.workloads.best_known.registry import file_sha256
    request = OriginalHeliosRequest(variant=variant, video_frames=17)
    impl = implementation(request)
    trace_key = load_profile(impl.workload)['required_assets']['trace']['environment']
    plan = dict(integration_head='a'*40, search_candidate={'id':impl.id},
        asset_config={'assets':{'trace':{'path':'/asset-host/admitted-trace'}}},
        original_helios_trace_binding=dict(bundle_hash='b'*64, request_sha256=request.digest))
    path = tmp_path/'plan.json'; path.write_text(json.dumps(plan))
    gate = dict(execution_receipt=dict(path=str(path), sha256=file_sha256(path)))
    gate_path = tmp_path/'gate.json'; gate_path.write_text(json.dumps(gate))
    env = impl.environment() | {n.MODE_ENV:'native', impl.trace_hash_environment:'b'*64,
        trace_key:'/asset-host/admitted-trace',
        'DISTILLSCHEDULE_FSDP2_CORRECTNESS_SUMMARY':str(gate_path),
        'DISTILLSCHEDULE_FSDP2_CORRECTNESS_SHA256':file_sha256(gate_path)}
    monkeypatch.setattr(n, 'git_head', lambda root:'a'*40)
    # Isolate the direct-launch join: the source/rank-comparison gate already
    # has its own real negative coverage; it is not a physical asset check.
    admitted=[]; monkeypatch.setattr(n, 'validate_admission', lambda *args:admitted.append(args))
    assert n.validate_launcher_admission(env)['status']=='PASS_NATIVE_SOURCE_BOUND_ADMISSION'
    assert len(admitted)==1
    if mutation=='digest': env[impl.trace_hash_environment]='c'*64
    if mutation=='path': env[trace_key]='/asset-host/other-trace'
    if mutation=='request':
        plan['original_helios_trace_binding']['request_sha256']='d'*64
        path.write_text(json.dumps(plan)); gate['execution_receipt']['sha256']=file_sha256(path)
        gate_path.write_text(json.dumps(gate)); env['DISTILLSCHEDULE_FSDP2_CORRECTNESS_SHA256']=file_sha256(gate_path)
    with pytest.raises(ValueError, match='admitted.*trace|trace.*admitted'):
        n.validate_launcher_admission(env)


def test_request_assembly_preserves_request_and_rejects_mixed_cohorts(tmp_path, monkeypatch):
    from distillschedule.search import helios_model_adapter as m
    from distillschedule.search import component_calibration
    from test_refactor_model_planning import BINDING
    request = OriginalHeliosRequest(video_frames=17)
    impl = implementation(request)
    result = dict(status="PASS_NATIVE_COMPONENT_CALIBRATION", source_head=m.git_head(m.repository_root()),
                  implementation_id=impl.id, layout_fragment={"path": "fixture", "sha256": "0"*64},
                  operation_costs={"path": "fixture-costs", "sha256": "1"*64})
    path = tmp_path/"result.json"; path.write_text(json.dumps(result))
    model = NS(verify_receipts=lambda: None, bind=lambda *a, **k: None, to_dict=lambda: {})
    monkeypatch.setattr(m, "_read_fragment", lambda *a: ({"binding": BINDING}, impl, []))
    monkeypatch.setattr(m, "bound", lambda *a: {})
    monkeypatch.setattr(m.CostModel, "from_dict", lambda *a: model)
    monkeypatch.setattr(component_calibration, "pool_component_models", lambda *a, **k: model)
    seen = []
    def build(path, **kwargs):
        seen.append(json.loads(path.read_text()))
        return NS(binding=BINDING, evidence_scope="CPU_FIXTURE_ONLY", to_dict=lambda: {})
    monkeypatch.setattr(m, "build_problem", build)
    m.assemble_calibrations([path], output=tmp_path/"assembled", expected_head=result["source_head"], request=request)
    assert seen[0][TRACE_KEY] == request.to_dict()
    with pytest.raises(ValueError, match="request"):
        m.assemble_calibrations([path], output=tmp_path/"mixed", expected_head=result["source_head"],
                                request=replace(request, video_frames=33))


@pytest.mark.parametrize("mutation", ["stage", "budget", "timestep"])
def test_persisted_actual_pyramid_metadata_cannot_relabel_stale_work(monkeypatch, mutation):
    from test_refactor_helios_planner_native import _fixture
    from distillschedule.runtime.helios_planner_cycle import run_two_cycles, _validate_action
    f = _fixture(monkeypatch)
    run_two_cycles(**f.kwargs)
    request = OriginalHeliosRequest(video_frames=17, stage2_inference_steps=(1,)*3)
    impl = implementation(request)
    row = deepcopy(f.result["cycles"][0]["actions"][0])
    schedule = controls(request)[0].operation("step/10/generator/rollout")["control"]["pyramid_stage_call_sequence"]
    row["trace_entries"][0].update({TRACE_HASH_KEY: request.digest, "stage_schedule": schedule,
                                  "stage_schedule_policy": request.stage_schedule_policy, "all_history_zero": False})
    row["helper_record"]["rollout"]["output_shape"] = list(request.latent_shape)
    generator_calls = []
    for call in row["forwards"]:
        hidden = call["inputs"]["hidden_states"]
        hidden["shape"] = list(request.latent_shape)
        if call["root_module"] == "generator": generator_calls.append(call)
    for call, stage in zip(generator_calls, schedule, strict=True):
        call["inputs"]["hidden_states"]["shape"] = list(stage["latent_shape"])
        call["inputs"]["timestep"] = dict(shape=[1], dtype="torch.bfloat16")
        call["timestep"] = stage["timestep"][0]
    _validate_action(row, impl)
    if mutation == "stage":
        schedule[0]["latent_shape"][2] = 9
        generator_calls[0]["inputs"]["hidden_states"]["shape"][2] = 9
    if mutation == "budget":
        schedule[0].update(denoising_step_budget=2, timestep_grid=[1000., 500.])
    if mutation == "timestep": generator_calls[0]["timestep"] = 123.
    with pytest.raises(ValueError): _validate_action(row, impl)


def _persisted_request_rows(monkeypatch):
    """Qualify CPU metadata fixtures, not observations from native training."""
    from test_refactor_helios_search_bridge import native_rows
    from distillschedule.runtime.helios_planner_cycle import _cycle_graph

    rows, legacy = native_rows(monkeypatch)
    request = OriginalHeliosRequest(stage2_inference_steps=(1,)*3)
    impl = replace(legacy, request=request)
    schedule = controls(request)[0].operation("step/10/critic/rollout")["control"]["pyramid_stage_call_sequence"]
    raw = (998.4970114182303, 897.5469382610918, 742.7147682074888)
    for stage, timestep in zip(schedule, raw, strict=True):
        stage.update(timestep_grid=[timestep], timestep=[timestep])
    for row in rows:
        row["candidate_id"] = impl.id
        row["planner_binding"].update(planner_implementation=impl.assignment,
            expected_trace_hash="b"*64, **{TRACE_KEY: request.to_dict(), TRACE_HASH_KEY: request.digest})
        row.setdefault("identities", {}).update(trace_bundle_hash="b"*64, original_helios_native_grid=dict(
            status="PASS", request_sha256=request.digest, operation_count=6))
        for cycle in row["cycles"]:
            graph, _ = _cycle_graph(impl, request.source_steps, cycle["initial_versions"])
            cycle["graph"] = graph.to_dict()
            for action in cycle["actions"]:
                retained = []
                for entry in action["trace_entries"]:
                    entry.update(expected_call_count=3, expected_stage_call_count={0: 1, 1: 1, 2: 1},
                        stage_schedule=deepcopy(schedule), stage_schedule_policy=request.stage_schedule_policy,
                        all_history_zero=False, **{TRACE_HASH_KEY: request.digest})
                    calls = [call for call in action["forwards"] if call["application_forward"]
                        and call["root_module"] == "generator"
                        and call["trace"]["rollout_operation"] == entry["rollout_operation"]][:3]
                    for call, stage, timestep in zip(calls, schedule if calls else [],
                                                    (1000., 896., 744.) if calls else [], strict=True):
                        call["inputs"]["hidden_states"]["shape"] = stage["latent_shape"]
                        call["inputs"]["timestep"] = dict(shape=[1], dtype="torch.bfloat16")
                        call["timestep"] = timestep
                    retained.extend(calls)
                action["forwards"] = [call for call in action["forwards"]
                    if not (call["application_forward"] and call["root_module"] == "generator")
                    or any(call is kept for kept in retained)]
                action["counts"]["generator_forward"] = len(retained)
                records = ([action["helper_record"]["rollout"]] if action["action_id"] == "g"
                           else action.get("rollout_records", []))
                for record in records:
                    record.update(call_count=3, stage_call_count={0: 1, 1: 1, 2: 1},
                        logical_call_count=3, logical_stage_call_count={0: 1, 1: 1, 2: 1}, calls=[{} for _ in range(3)])
    return json.loads(json.dumps(rows)), impl


@pytest.mark.parametrize("request_qualified", [False, True])
@pytest.mark.parametrize("scope", ["action", "summarize"])
def test_persisted_timestep_audit_and_summary_require_no_torch(monkeypatch, request_qualified, scope):
    from test_refactor_helios_search_bridge import native_rows
    from distillschedule.runtime.helios_planner_cycle import _validate_action
    from distillschedule.search.helios_native import summarize

    rows, impl = (_persisted_request_rows(monkeypatch) if request_qualified
                  else native_rows(monkeypatch))
    rows = json.loads(json.dumps(rows))
    original_import = builtins.__import__
    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("Torch deliberately unavailable on lean CPU controller")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_torch)
    with pytest.raises(ModuleNotFoundError, match="deliberately unavailable"):
        __import__("torch")
    if scope == "action":
        action = rows[0]["cycles"][0]["actions"][0]
        _validate_action(action, impl)
        if request_qualified:
            action["forwards"][0]["timestep"] += 1e-7
            with pytest.raises(ValueError, match="timestep"):
                _validate_action(action, impl)
    else:
        report = summarize(rows, expected_head=rows[0]["source_head"], candidate_id=impl.id)
        assert report["status"] == "PASS" and report["rank_count"] == 16
        assert report["calibration_only"] and not report["formal_timing_eligible"]


def test_timestep_projection_matches_torch_at_bf16_and_fp32_rounding_boundaries():
    torch = pytest.importorskip("torch")
    from distillschedule.runtime.helios_planner_cycle import _bf16_timestep

    values = [998.4970114182303, 897.5469382610918, 742.7147682074888,
              623.1242597886123, 385.3111935203593, 305.52198382516355]
    # Both parities of the BF16 tie; include the much narrower FP32 rounding
    # cells, whose double rounding differs from a direct FP64-to-BF16 round.
    for midpoint in (251.5, 252.5, 305., 307., 385., 387., 501., 503.,
                     622., 626., 742., 746., 894., 898., 998., 1002.):
        half_fp32_ulp = math.ldexp(1., math.frexp(midpoint)[1]-25)
        for boundary in (midpoint-half_fp32_ulp, midpoint, midpoint+half_fp32_ulp):
            values.extend((math.nextafter(boundary, -math.inf), boundary,
                           math.nextafter(boundary, math.inf)))
    expected = torch.tensor(values, dtype=torch.float64).to(torch.bfloat16).tolist()
    assert [_bf16_timestep(value) for value in values] == expected
    assert _bf16_timestep(998.) == 1000.  # Odd lower BF16 mantissa rounds up.
    assert _bf16_timestep(1002.) == 1000.  # Even lower BF16 mantissa stays down.
    assert _bf16_timestep(math.nextafter(998., -math.inf)) == 1000.
    assert _bf16_timestep(math.nextafter(1002., math.inf)) == 1000.
    assert _bf16_timestep(998.-math.ldexp(1., -14)) == 996.
    assert _bf16_timestep(1002.+math.ldexp(1., -14)) == 1004.


def test_timestep_projection_matches_torch_float_range_and_nonfinite_values():
    torch = pytest.importorskip("torch")
    from distillschedule.runtime.helios_planner_cycle import _bf16_timestep

    # Native grid validation rejects non-finite source values and its actual
    # scheduler stays near [0, 1000]; the scalar helper still mirrors IEEE range.
    values = [0., -0., math.ldexp(1., -134), math.ldexp(1., -133),
              math.ldexp(3., -134), math.ldexp(1., -126),
              struct.unpack(">f", struct.pack(">I", 0x7F7FFFFF))[0],
              1e100, -1e100, math.inf, -math.inf, math.nan]
    expected = torch.tensor(values, dtype=torch.float64).to(torch.bfloat16).tolist()
    for value, native in zip(values, expected, strict=True):
        actual = _bf16_timestep(value)
        assert math.isnan(actual) if math.isnan(native) else actual == native
        if native == 0.:
            assert math.copysign(1., actual) == math.copysign(1., native)


def _real_timestep_observer(*, raw_timesteps=(998.4970114182303, 897.5469382610918, 742.7147682074888),
                            native_cast=True, input_dtype="float64", timestep_shape=(1,), omit_timestep=False):
    """CPU hook/finish evidence with real tensors, not a distributed training run."""
    torch = pytest.importorskip("torch")
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from distillschedule.runtime.helios_planner_cycle import _Observer, _COUNTS

    request = OriginalHeliosRequest(video_frames=17, stage2_inference_steps=(1,)*3)
    impl = implementation(request)
    schedule = controls(request)[0].operation("step/10/critic/rollout")["control"]["pyramid_stage_call_sequence"]
    for stage, raw in zip(schedule, raw_timesteps, strict=True):
        stage.update(timestep_grid=[raw], timestep=[raw])
    entry = dict(source_rank=0, source_step=10, trace_identity_sha256="cpu-real-timestep",
                 rollout_operation="step/10/critic/rollout", expected_call_count=3,
                 expected_stage_call_count={0: 1, 1: 1, 2: 1}, stage_schedule=schedule,
                 stage_schedule_policy=request.stage_schedule_policy, all_history_zero=False,
                 **{TRACE_HASH_KEY: request.digest})
    row = dict(action_id="r0", source_steps=[10], source_rank=0, counts=dict.fromkeys(_COUNTS, 0),
               trace_entries=[entry], forwards=[], produced=[], losses=[], rollout_records=[dict(
                   call_count=3, stage_call_count={0: 1, 1: 1, 2: 1},
                   output_shape=list(request.latent_shape), logical_steps=[10], batch_size=1,
                   member_trace_identities=[entry["trace_identity_sha256"]])])

    class Model(torch.nn.Module):
        adapters_enabled = True

        def enable_adapters(self): self.adapters_enabled = True
        def disable_adapters(self): self.adapters_enabled = False
        def forward(self, *, hidden_states, timestep=None): return (hidden_states,)

    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    assert policy.cast_forward_inputs is True

    def cast_inputs(module, args, kwargs):
        # FSDP2's prepended pre-hook casts floating inputs before the observer.
        return args, {key: value.to(policy.param_dtype) if torch.is_floating_point(value) else value
                      for key, value in kwargs.items()}

    def engine():
        model = Model()
        if native_cast:
            model.register_forward_pre_hook(cast_inputs, prepend=True, with_kwargs=True)
        return NS(module=model, global_steps=0, optimizer=NS(step=lambda: None),
                  backward=lambda loss: None, step=lambda: None)

    generator, critic = engine(), engine()
    observer = _Observer(torch=torch, device="cpu", rank=0, impl=impl, generator=generator,
                         critic=critic, ema=NS(optimization_step=0), mode="native",
                         version_offsets=dict(generator=0, critic=0, ema=0))
    observer.actions.append(row)
    observer.current, observer.current_entry = row, entry
    with observer.installed():
        for stage, raw in zip(schedule, raw_timesteps, strict=True):
            kwargs = dict(hidden_states=torch.empty(stage["latent_shape"], dtype=torch.bfloat16))
            if not omit_timestep:
                kwargs["timestep"] = torch.full(timestep_shape, raw, dtype=getattr(torch, input_dtype))
            generator.module(**kwargs)
    observer.current, observer.current_entry = None, None
    # Supply the unrelated finish finite-loss precondition without a training step.
    observer.losses.append((row, "generator", torch.tensor(1.)))
    return observer, row, impl


@pytest.mark.parametrize("raw_timesteps", [
    (998.4970114182303, 897.5469382610918, 742.7147682074888),
    (623.1242597886123, 385.3111935203593, 305.52198382516355),
    (1000., 500., 250.),
])
def test_actual_helios_timestep_audit_projects_fp64_grid_to_native_bf16(raw_timesteps):
    torch = pytest.importorskip("torch")
    observer, row, _ = _real_timestep_observer(raw_timesteps=raw_timesteps)
    assert observer.finish()["census"] == "PASS"
    assert [call["timestep"] for call in row["forwards"]] == torch.tensor(
        raw_timesteps, dtype=torch.float64).to(torch.bfloat16).tolist()
    assert all(call["inputs"]["timestep"]["dtype"] == "torch.bfloat16" for call in row["forwards"])
    assert [stage["timestep_grid"][0] for stage in row["trace_entries"][0]["stage_schedule"]] == list(raw_timesteps)
    assert not observer._timesteps


@pytest.mark.parametrize("input_dtype", ["float16", "float32", "float64", "int64"])
def test_actual_helios_timestep_audit_rejects_wrong_dtype_even_at_exact_grid_values(input_dtype):
    observer, _, _ = _real_timestep_observer(raw_timesteps=(1000., 500., 250.), native_cast=False,
                                            input_dtype=input_dtype)
    with pytest.raises(ValueError, match="timestep"):
        observer.finish()


@pytest.mark.parametrize("shape", [(), (1, 1), (0,), (2,), (1, 2)])
def test_actual_helios_timestep_audit_requires_native_b1_vector_shape(shape):
    with pytest.raises(ValueError, match="timestep"):
        observer, _, _ = _real_timestep_observer(raw_timesteps=(1000., 500., 250.), timestep_shape=shape)
        observer.finish()


def test_actual_helios_timestep_audit_requires_observed_tensor():
    with pytest.raises(ValueError, match="timestep"):
        _real_timestep_observer(omit_timestep=True)


@pytest.mark.parametrize("mutation", ["raw", "next_ulp", "previous_ulp", "sub_ulp", "nan", "inf",
                                    "missing_value", "missing_metadata", "wrong_metadata_dtype"])
def test_persisted_helios_model_timestep_audit_is_exact_and_fail_closed(mutation):
    torch = pytest.importorskip("torch")
    from distillschedule.runtime.helios_planner_cycle import _validate_action
    observer, row, impl = _real_timestep_observer()
    observer.finish()
    call = row["forwards"][0]
    if mutation == "raw":
        call["timestep"] = row["trace_entries"][0]["stage_schedule"][0]["timestep"][0]
    elif mutation in ("next_ulp", "previous_ulp"):
        direction = float("inf") if mutation == "next_ulp" else -float("inf")
        call["timestep"] = torch.nextafter(torch.tensor(call["timestep"], dtype=torch.bfloat16),
                                         torch.tensor(direction, dtype=torch.bfloat16)).item()
    elif mutation == "sub_ulp": call["timestep"] += 1e-7
    elif mutation == "nan": call["timestep"] = float("nan")
    elif mutation == "inf": call["timestep"] = float("inf")
    elif mutation == "missing_value": call.pop("timestep")
    elif mutation == "missing_metadata": call["inputs"].pop("timestep")
    elif mutation == "wrong_metadata_dtype": call["inputs"]["timestep"]["dtype"] = "torch.float64"
    with pytest.raises(ValueError, match="timestep"):
        _validate_action(row, impl)
