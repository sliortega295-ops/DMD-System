"""Independent opt-in W3 native-history ABI; local fixtures are not GPU proof."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest

POLICY = "source_native_amplified_v1"
KIND = "original_helios_gan_native_history_cohort_stream.v1"


def task(index=0):
    from distillschedule.training.helios_gan_inputs import GANCohortTask
    return GANCohortTask(index, native_policy=POLICY)


def controls(value):
    """Exact synthetic metadata only, independently retaining GAN-only roles."""
    from test_training_helios_native_history import controls as nogan_controls
    from distillschedule.training.helios_inputs import TRACE_KEY as W2_KEY
    from distillschedule.training import helios_gan_inputs as inputs
    rows = nogan_controls(value)
    for row in rows.values():
        row.manifest["contract"].pop(W2_KEY)
        row.manifest["contract"][inputs.TRACE_KEY] = value.to_dict()
        for step in value.source_steps:
            row.operation(f"step/{step}/branch")["control"]["gan"] = True
            for group in ("primary", "secondary"):
                for role, frames in (("x0", 1), ("history", 19), ("target", 9)):
                    row.manifest["tensors"][f"step/{step}/sample/{group}/{role}"] = dict(
                        shape=[16, frames, 48, 80], dtype="bfloat16")
            for phase in (("generator", "critic") if step == value.first_step else ("critic",)):
                prefix = f"step/{step}/{phase}"
                noise = "gan_g_noise" if phase == "generator" else "r1_epsilon"
                row.manifest["tensors"][prefix + "/" + noise] = dict(shape=list(value.latent_shape), dtype="bfloat16")
                crops = ("gan_g_crop",) if phase == "generator" else ("gan_d_fake_crop", "gan_d_real_crop", "gan_d_r1_crop")
                for crop in crops:
                    row.manifest["tensors"][prefix + "/" + crop] = dict(shape=[1, 2], dtype="int64")
                row.manifest["operations"].append(dict(key=prefix + ("/gan_g" if phase == "generator" else "/gan_d"),
                    control=dict(global_step=step)))
        indexed = {op["key"]: op for op in row.manifest["operations"]}
        row.operation = lambda key, rows=indexed: rows[key]
    return rows


def test_independent_gan_schema_normal_default_and_w2_rejection():
    from distillschedule.training import helios_gan_inputs as inputs
    from distillschedule.runtime import helios_gan_native_history as abi
    from distillschedule.runtime.helios_native_history import SEMANTICS as W2_SEMANTICS
    from distillschedule.training.helios_inputs import HeliosCohortTask
    old = inputs.GANCohortTask(0)
    assert old.to_dict() == dict(schema_version="distillschedule.helios_gan_training_cohort.v1", index=0,
        request=inputs.REQUEST.to_dict(), source_steps=[1000, 1001, 1002, 1003, 1004])
    new = task(1)
    assert new.stage_schedule_policy == POLICY and new.request == inputs.REQUEST
    assert new.source_steps == (1005, 1006, 1007, 1008, 1009)
    assert inputs.GANCohortTask.from_dict(new.to_dict()) == new and new.digest != old.digest
    assert inputs.NATIVE_KIND == KIND and abi.descriptor()["semantics"] == abi.SEMANTICS != W2_SEMANTICS
    assert abi.descriptor()["gan"] is True and abi.descriptor()["n70"] is False
    with pytest.raises(ValueError): abi.require_task(HeliosCohortTask(0, native_policy=POLICY))
    with pytest.raises(ValueError): inputs.GANCohortTask(0, native_policy="benchmark_v4_strict_global")
    bad = new.to_dict(); bad["schema_version"] = old.to_dict()["schema_version"]
    with pytest.raises(ValueError): inputs.GANCohortTask.from_dict(bad)


def test_rank_local_gan_census_is_independent_and_keeps_all_operations():
    from distillschedule.training import helios_gan_inputs as inputs
    from distillschedule.runtime.helios_gan_native_history import SEMANTICS
    rows = controls(task()); value = inputs.validate_controls(task(), rows)
    assert value["semantics"] == SEMANTICS and value["rank_count"] == 16
    assert value["local_exception_entry_count"] == 48 and value["cross_rank_call_count_parity"] is False
    assert value["execution_requirement"] == "GAN_G_ALIGNED_COMPLETE_BLOCK_PHASE_AND_ORIGINAL_PERSISTENT_R_ALL"
    with pytest.raises(ValueError): inputs.validate_controls(inputs.GANCohortTask(0), rows)


@pytest.mark.parametrize("mutation", ["w2", "branch", "budget", "rank", "initial", "normal_exit", "total", "gan_noise", "crop", "r1", "geometry", "extra_contract"])
def test_native_gan_does_not_broaden_existing_reject_cases(mutation):
    from distillschedule.training import helios_gan_inputs as inputs
    rows = controls(task()); row = rows[1]; c = row.operation("step/1000/generator/rollout")["control"]
    if mutation == "w2": row.manifest["contract"]["distillschedule_helios_training_cohort"] = {}
    if mutation == "branch": row.operation("step/1000/branch")["control"]["gan"] = False
    if mutation == "budget": c["pyramid_stage_call_sequence"][0]["denoising_step_budget"] = 4
    if mutation == "rank": c["rank"] = 0
    if mutation == "initial": c["initial_synchronized_pyramid_stage_exit"] = [1, 0, 0]
    if mutation == "normal_exit": c["pyramid_stage_call_sequence"][0]["denoising_exit"] = 1
    if mutation == "total": c["call_count"] = 4
    if mutation == "gan_noise": row.manifest["tensors"].pop("step/1000/generator/gan_g_noise")
    if mutation == "crop": row.manifest["tensors"]["step/1000/critic/gan_d_r1_crop"]["shape"] = [1, 1]
    if mutation == "r1": row.manifest["tensors"]["step/1001/critic/r1_epsilon"]["dtype"] = "float32"
    if mutation == "geometry": row.manifest["contract"]["resolution"] = [64, 64]
    if mutation == "extra_contract": row.manifest["contract"]["distillschedule_original_helios_request_sha256"] = "a" * 64
    with pytest.raises((ValueError, KeyError)): inputs.validate_controls(task(), rows)


def public_config(tmp_path, kind=KIND):
    from distillschedule.training.config import TrainingConfig
    return TrainingConfig.from_dict(dict(schema_version="distillschedule.training.v1", model=dict(workload="W3"),
        algorithm=dict(name="original_helios_gan"), data=dict(input_kind=kind, prepared="stream.json"),
        execution=dict(adapter="native", mode="manual", task="task.json", assets="assets.json",
                       preset="helios_gan.fa3_down40_nomerge_stream.v1"),
        trainer=dict(cycles=2, checkpoint_every=1, output_dir="run")), source=tmp_path / "training.json")


def test_public_kind_requires_full_original_gan_bundle_and_cli_explicit_opt_in(tmp_path, monkeypatch, capsys):
    from distillschedule.training import native, helios_gan_inputs as inputs
    from distillschedule.training.config import TrainingConfig
    from distillschedule.cli import main
    conf = public_config(tmp_path)
    assert native._options(conf, "W3") == "manual"
    described = native.describe_native(conf)
    assert described["capabilities"]["native_history"]["gan"] is True
    assert described["capabilities"]["cycles"]["fresh_inputs"] == KIND
    old = native.describe_native(public_config(tmp_path, inputs.KIND))
    assert "native_history" not in old["capabilities"] and old["capabilities"]["cycles"]["fresh_inputs"] == inputs.KIND
    for changed in (dict(preset="helios_nogan.safe_stream_ed8.v1"), dict(mode="reference"), dict(plan="old.json")):
        raw = conf.to_dict(); raw["execution"].update(changed)
        with pytest.raises(ValueError): native._options(TrainingConfig.from_dict(raw, source=conf.source), "W3")
    source = tmp_path / "sources.json"
    source.write_text(json.dumps([dict(seed=42, samples=[[f"/features/{i}/{s}/{r}.pt" for r in range(16)]
                                                        for s in range(5)]) for i in range(2)]))
    seen = []
    def producer(sources, config, output, *, input_policy, native_policy=None):
        seen.append(native_policy); return dict(status="SYNTHETIC_TEST_ONLY", gpu_execution="NOT_RUN")
    monkeypatch.setattr(inputs, "prepare_stream", producer)
    argv = ["data", "prepare-helios-gan-stream", "--sources", str(source), "--training-config", str(tmp_path / "gan.yaml")]
    assert main([*argv, "--output", str(tmp_path / "native"), "--history-policy", POLICY]) == 0
    assert main([*argv, "--output", str(tmp_path / "old")]) == 0
    assert seen == [POLICY, None] and "NOT_RUN" in capsys.readouterr().out


@pytest.fixture
def phase_protocol(monkeypatch):
    from test_training_helios_native_history import phase_protocol as existing
    return existing.__wrapped__(monkeypatch)


@pytest.mark.parametrize("fail", [False, True])
def test_original_gan_update_gets_whole_grad_phase_not_a_new_trainer(phase_protocol, fail):
    import ast
    import torch
    from pathlib import Path
    from distillschedule.runtime.helios_planner_cycle import _prepare_cycle
    from distillschedule.training.helios_gan_training import implementation
    from distillschedule.runtime.role_runner import OperationContext
    from distillschedule.runtime.version_sync import VersionLedger
    from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
    engine, _, _ = phase_protocol
    # Execute the actual unchanged original GAN helper, with source-preserving
    # dependency stand-ins at its rollout/Score seam. This is argument/phase,
    # not GAN loss, actual native imports, FSDP or numerical parity evidence.
    path = Path(__file__).parents[1] / "scripts/distillschedule/probe_static_fsdp2_full5_world16.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_run_gan_generator_update")
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    namespace = dict(torch=torch, TIMING_OMITTED="OMITTED")
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
    selected = task(); seen = []
    class Observer:
        mode, rank, device = "native", 0, "cpu"
        impl = implementation()
        generator, critic = engine, NS(module=object())
        rows = []
        def versions(self): return dict(generator=200 + engine.global_steps, critic=1000, ema=800)
        @contextmanager
        def action(self, op, *, source_steps):
            row = dict(action_id=op.operation_id, source_steps=source_steps, memory_after={})
            self.rows.append(row); yield row
        def produced(self, step, output): pass
        def entry(self, value, *, generator, rank, native_task):
            assert value is entry and generator is True and rank == 0 and native_task == selected
    observer = Observer()
    entry = NS(logical_step=1000, operation_trace=object())
    trace = NS(task=selected, load_generator_entry=lambda **kw: entry, source=lambda **kw: {"GAN_SOURCE": True})
    def rollout(**kw):
        view = kw["engine"]
        assert view is not engine and all(u.group.is_unsharded for u in engine.module.units)
        for i in range(8): view(i)
        return torch.ones(1, requires_grad=True), None, dict(call_count=8)
    def score(**kw):
        seen.append(("original_GAN_Score_boundary", kw["gan_g_weight"]))
        assert all(u.group.is_unsharded for u in engine.module.units)
        if fail: raise RuntimeError("original_GAN_Score_failure")
        return torch.ones(1), {"GAN_R1_math": "NOT_RUN"}
    namespace.update(run_generator_update_trace_rollout=rollout, _gan_score_boundary_gradient=score,
                     score_official_kwargs=lambda cfg, trace, source, step, negative, device: {"source": source, "step": step})
    detached_calls = []
    def detached(**kw):
        # Preserve the exact original detached persistence owner; nesting a
        # whole-generator phase here would fail the actual scope admission.
        assert kw["generator_engine"] is engine and not any(u.group.is_unsharded for u in engine.module.units)
        detached_calls.append(tuple(kw["steps"]))
        return ({s: object() for s in selected.source_steps}, [{} for _ in selected.source_steps],
                dict(enabled=True, restored=True, rollout_batch_count=5, set_no_reshard_count=41,
                     explicit_reshard_count=41, unsharded_after_exit_count=0))
    p0 = NS(base=NS(_run_gan_generator_update=namespace["_run_gan_generator_update"]), _precompute_critic_rollouts=detached)
    prepared, graph, _ = _prepare_cycle(observer=observer, p0=p0, args=NS(timing_mode=True, profile_mode=False, profile_diagnostics=False),
        policy=StaticFSDP2P0Policy(rollout_batch_size=1, generator_nograd_persistent_unshard=True,
            elide_official_generator_free_memory=True), steps=selected.source_steps, trace=trace, negative=object(),
        cfg=NS(training_config=NS(gan_g_weight=.05)), diagnostics=NS(scope=lambda *a: nullcontext(), add_tensor_tree=lambda *a, **kw: None),
        generator_free_memory={})
    from distillschedule.core.enums import OperationKind as Kind
    assert [sum(op.kind == kind for op in graph.operations) for kind in
            (Kind.GENERATOR_UPDATE, Kind.CRITIC_UPDATE, Kind.EMA_UPDATE, Kind.CRITIC_ROLLOUT)] == [1, 5, 1, 1]
    assert tuple(op.logical_step for op in graph.operations if op.kind == Kind.CRITIC_UPDATE) == selected.source_steps
    op, callback = next((op, fn) for op, fn in prepared._bound if op.operation_id == "g")
    ctx = OperationContext(op, VersionLedger(observer.versions()))
    if fail:
        with pytest.raises(RuntimeError, match="original_GAN_Score_failure"): callback(ctx)
        assert engine.global_steps == 0
    else:
        result = callback(ctx)
        assert result["score_forwards"] == 4 and result["logical_step"] == 1000
        assert engine.events[-1] == ("original_clip_adam_scheduler_zero_grad",)
        op, callback = next((op, fn) for op, fn in prepared._bound if op.operation_id == "r_all")
        callback(OperationContext(op, VersionLedger(observer.versions())))
        assert detached_calls == [selected.source_steps] and "native_history_phase" not in observer.rows[-1]
    phase = observer.rows[0]["native_history_phase"]
    assert phase["schema_version"] == "distillschedule.helios_gan_native_history_execution.v1"
    assert phase["unshard_count"] == phase["reshard_count"] == 41 and phase["restored"]
    assert phase["backward_count"] == phase["commit_count"] == int(not fail)
    assert not any(u.group.is_unsharded for u in engine.module.units)


def native_metadata_stream(root):
    """Hash-authenticated metadata, not a native payload or input-production proof."""
    from pathlib import Path
    from test_training_helios_gan_stream import metadata_stream
    from distillschedule.training import helios_gan_inputs as inputs
    from distillschedule.workloads.best_known.registry import fingerprint
    from runtime.operation_trace import _manifest_hash, write_operation_trace_bundle_manifest
    root = Path(root); root.mkdir()
    doc = metadata_stream(root, cycles=2)
    doc["schema_version"] = inputs.NATIVE_SCHEMA
    doc["identity"]["native_history_policy"] = POLICY
    for index, cohort in enumerate(doc["cohorts"]):
        value = task(index); cohort["task"] = value.to_dict(); expected = controls(value)
        directory = Path(cohort["trace_bundle"]); ranks = []
        for rank in range(16):
            path = directory / f"rank_{rank:02d}" / "trace.json"
            manifest = json.loads(path.read_text())
            manifest["contract"] = expected[rank].manifest["contract"]
            manifest["operations"] = expected[rank].manifest["operations"]
            manifest["trace_hash"] = _manifest_hash(manifest); path.write_text(json.dumps(manifest))
            ranks.append(dict(rank=rank, path=path.parent.name, trace_hash=manifest["trace_hash"]))
        (directory / "bundle.json").unlink()
        cohort["trace_hash"] = write_operation_trace_bundle_manifest(directory, ranks)["bundle_hash"]
    doc.pop("stream_sha256"); doc["stream_sha256"] = fingerprint(doc)
    (root / "stream.json").write_text(json.dumps(doc))
    return doc


def test_native_stream_load_current_source_down40_rows_and_old_loader_reject(tmp_path):
    from distillschedule.training import helios_gan_inputs as inputs
    from distillschedule.runtime.helios_gan_manual.stream import qualify_rows
    from distillschedule.workloads.best_known.registry import fingerprint
    doc = native_metadata_stream(tmp_path / "input"); path = tmp_path / "input/stream.json"
    loaded = inputs.load_stream(path, cycles=2, expected_kind=KIND)
    assert loaded == doc and inputs.task_for_stream(doc, 1) == task(1)
    with pytest.raises(ValueError, match="stream changed/incomplete"): inputs.load_stream(path, cycles=2)
    entry = doc["cohorts"][0]
    trace = inputs.load_cohort(task(), entry["trace_bundle"], expected_hash=entry["trace_hash"])
    assert trace.rank_control_validation["cross_rank_call_count_parity"] is False
    rows, binding = qualify_rows(trace, 0)
    assert set(rows.values()) == {11040, 13680} and binding["task"] == task().to_dict()
    assert len(rows) == 10 and binding["source_entry_sha256"]
    from distillschedule.runtime.rollout_geometry import rollout_geometry
    assert rollout_geometry(trace._control[0]) == inputs.REQUEST
    bad = deepcopy(doc); bad["identity"]["source_fingerprint"] = "0" * 64
    bad.pop("stream_sha256"); bad["stream_sha256"] = fingerprint(bad); path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="cycles/source/request changed"):
        inputs.load_stream(path, cycles=2, expected_kind=KIND)
    trace.close()


@pytest.mark.parametrize("epoch", [False, True])
def test_public_preparation_propagates_gan_history_abi_and_epoch_draws(tmp_path, monkeypatch, epoch):
    """Real public orchestration with SYNTHETIC metadata capture, not payloads."""
    from pathlib import Path
    import yaml
    from distillschedule.training import helios_gan_inputs as inputs
    from distillschedule.data import helios_epoch_inputs as epochs
    from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
    from runtime.operation_trace import _manifest_hash
    count = 80 if epoch else 160; files = []
    for i in range(count):
        path = tmp_path / f"metadata-not-tensor-{i}.pt"; path.write_text(f"SYNTHETIC_METADATA_ONLY_{i}")
        files.append(str(path))
    config = tmp_path / "gan.yaml"; config.write_text(yaml.safe_dump(inputs.portable_config()))
    if epoch:
        source = dict(schema_version=epochs.SOURCE_SCHEMA, seed=42, cycles=2, evidence_kind="CPU_SYNTHETIC_TEST_ONLY",
                      records=[dict(id=f"record-{i}", path=p, sha256=file_sha256(Path(p))) for i,p in enumerate(files)])
        # No tensor payload is substituted into production loading; only this
        # explicit metadata test's source-record boundary is a test double.
        def records(value, task_value):
            assert task_value.native_policy == POLICY and task_value.request == inputs.REQUEST
            return [dict(**r, content_sha256=fingerprint(r), latent_shape=[2,16,9,48,80], text_shape=[2,4096])
                    for r in value["records"]]
        monkeypatch.setattr(epochs, "_source_records", records)
    else:
        source = [inputs.HeliosCohortSources(42, tuple(tuple(files[i*80+s*16+r] for r in range(16)) for s in range(5)))
                  for i in range(2)]
        def inventory(task_value, sources):
            return [dict(rank=r, step=step, path=sources.samples[s][r], sha256=file_sha256(Path(sources.samples[s][r])),
                         latent_shape=[2,16,9,48,80], text_shape=[2,4096])
                    for s,step in enumerate(task_value.source_steps) for r in range(16)]
        monkeypatch.setattr(inputs, "_inventory", inventory)
    seen = []
    def capture(task_value, sources, inventory, config_value, *, rank, output, provenance):
        seen.append((task_value.index, rank, task_value.native_policy))
        row = controls(task_value)[rank]; manifest = row.manifest
        manifest.update(schema_version="distillschedule.operation_trace.v1", classification="CPU_METADATA_ONLY_NOT_NATIVE_INPUT_PRODUCTION",
            source={**provenance, "rank": rank, "samples": [r for r in inventory if r["rank"] == rank],
                    "seed": sources.seed, "cohort": task_value.index,
                    "rng_scope": "new_native_CPU_streams_not_prior_training_rng_continuation"})
        for key, descriptor in manifest["tensors"].items(): descriptor["sha256"] = fingerprint([key, rank, task_value.index])
        manifest["trace_hash"] = _manifest_hash(manifest)
        output = Path(output); output.mkdir(parents=True); (output / "trace.json").write_text(json.dumps(manifest))
        return NS(trace_hash=manifest["trace_hash"])
    monkeypatch.setattr(inputs, "capture_rank", capture)
    output = tmp_path / "prepared"
    result = inputs.prepare_stream(source, config, output, native_policy=POLICY,
                                   input_policy=epochs.EPOCH_POLICY if epoch else None)
    assert result["schema_version"] == inputs.NATIVE_SCHEMA and result["identity"]["native_history_policy"] == POLICY
    assert len(seen) == 32 and all(row[2] == POLICY for row in seen)
    assert inputs.load_stream(output / "stream.json", cycles=2, expected_kind=KIND) == result
    if epoch:
        assert result["identity"]["input_policy"]["draw_count"] == 160
        assert result["identity"]["sources"][0]["seed"] == result["identity"]["sources"][1]["seed"] == 42
        assert {r["record_id"] for r in result["identity"]["inventory"][0]} == {r["record_id"] for r in result["identity"]["inventory"][1]}
    with pytest.raises(ValueError):
        inputs.prepare_stream(source, config, tmp_path / "bad", native_policy="W2_IS_NOT_GAN_CAPABILITY")
    assert not (tmp_path / "bad").exists()


def test_native_checkpoint_identity_progress_exact_clocks_and_original_state_restore(tmp_path):
    """Actual shared checkpoint/native state views; tiny CPU tensors, no GAN math."""
    import random
    import torch
    from distillschedule.training import helios_gan_training as training
    from distillschedule.runtime.helios_gan_native_history import descriptor
    from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter
    from distillschedule.training.native_state import engine_state
    from distillschedule.training.checkpoint import save_checkpoint, restore_checkpoint, read_manifest, pack_state, rng_state
    from test_training_end_to_end import equal
    class Model(torch.nn.Module):
        def __init__(self, gan):
            super().__init__(); self.weight = torch.nn.Parameter(torch.tensor([.2]))
            if gan: self.gan_head = torch.nn.Parameter(torch.tensor([.07]))
    def engines():
        values = []
        for gan, steps in ((False, 3), (True, 15)):
            model = Model(gan); opt = torch.optim.AdamW(model.parameters(), lr=.0002)
            for parameter in model.parameters():
                opt.state[parameter] = dict(step=torch.tensor(float(steps)), exp_avg=torch.full_like(parameter,.01),
                                            exp_avg_sq=torch.full_like(parameter,.02))
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.); scheduler.last_epoch = steps
            values.append(FSDP2EngineAdapter(model, opt, scheduler, physical_step=steps, global_batch=16, dp_world_size=1))
        # The actual native view serializes EMA's clock and local shadow. No
        # alternative EMA recurrence is implemented or exercised in this test.
        ema = NS(_names=["weight"], _parameters=[values[0].module.weight], _shadows=[torch.tensor([.19])],
                 decay=.99, min_decay=0., update_after_step=750, device="cpu", optimization_step=801, cur_decay_value=.8)
        return values, ema
    identity = dict(schema_version="distillschedule.helios_gan_native_history_training_identity.v1",
        source_head="f85dfda02b7958a383a3391205f4df588ab1dd5b",
        source_fingerprint=training.inputs.source_fingerprint(), stream_sha256="a" * 64,
        execution_policy=training.descriptor(POLICY), gan_native_history=descriptor())
    policy = training._checkpoint_policy(identity)
    assert policy["schema_version"] == "distillschedule.helios_gan_native_history_checkpoint_policy.v1"
    values, ema = engines(); original = engine_state(dict(zip(("generator", "critic"), values)), ema=ema)
    before = pack_state(original); saved_rng = rng_state()
    path = save_checkpoint(tmp_path / "checkpoint", original, identity=identity, cycle=1, cursor=1,
                           versions=training.expected_versions(1), policy=policy)
    assert (path / "_SUCCESS").is_file()
    header = read_manifest(path, identity=identity)
    binding = dict(stream=dict(identity={}, cohorts=[{}, {}]), checkpoint_policy=policy)
    training.validate_progress(header, binding)
    for changed in (dict(cursor=0), dict(versions=dict(generator=3, critic=15, ema=805)), dict(policy={})):
        with pytest.raises(ValueError): training.validate_progress({**header, **changed}, binding)
    for key in ("source_head", "source_fingerprint", "stream_sha256"):
        with pytest.raises(ValueError): read_manifest(path, identity={**identity, key: "changed"})
    with pytest.raises(ValueError): training._checkpoint_policy({**identity, "gan_native_history": {"W2_alias": True}})
    random.random(); torch.rand(2)
    restored_values, restored_ema = engines()
    for engine in restored_values:
        with torch.no_grad():
            for p in engine.module.parameters(): p.add_(7)
        engine.global_steps += 20
    restored = engine_state(dict(zip(("generator", "critic"), restored_values)), ema=restored_ema)
    restore_checkpoint(path, restored, identity=identity)
    assert equal(before, pack_state(restored)) and equal(saved_rng, rng_state())
    assert "gan_head" in restored["critic.weights"].state_dict()["tensors"]


def test_old_w3_flags_recipe_and_unphased_dispatch_unchanged(phase_protocol):
    from distillschedule.training import helios_gan_training as training, helios_gan_inputs as inputs
    from distillschedule.runtime.helios_planner_cycle import _prepare_cycle
    from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
    from distillschedule.runtime.role_runner import OperationContext
    from distillschedule.runtime.version_sync import VersionLedger
    engine, _, _ = phase_protocol
    class Observer:
        mode, rank, device = "native", 0, "cpu"
        impl = training.implementation(); generator, critic = engine, object()
        def versions(self): return dict(generator=200, critic=1000, ema=800)
        @contextmanager
        def action(self, op, *, source_steps):
            self.row = dict(memory_after={}); yield self.row
    observer = Observer(); seen = []
    def helper(**kw):
        assert kw["generator_engine"] is engine
        seen.append(True); return {"old_W3_helper": True}
    prepared, _, _ = _prepare_cycle(observer=observer, p0=NS(base=NS(_run_gan_generator_update=helper)),
        args=NS(timing_mode=True, profile_mode=False, profile_diagnostics=False),
        policy=StaticFSDP2P0Policy(rollout_batch_size=1, generator_nograd_persistent_unshard=True),
        steps=inputs.GANCohortTask(0).source_steps, trace=NS(task=inputs.GANCohortTask(0)),
        negative=None, cfg=object(), diagnostics=object(), generator_free_memory={})
    op, callback = next((op, fn) for op, fn in prepared._bound if op.operation_id == "g")
    assert callback(OperationContext(op, VersionLedger(observer.versions()))) == {"old_W3_helper": True}
    assert seen == [True] and "native_history_phase" not in observer.row and not engine.events
    impl = training.implementation()
    assert (impl.variant, impl.mesh, impl.chunk_tokens, impl.rollout_window, impl.persistent_unshard) == ("gan", "world16-fsdp", 2048, 5, True)
    assert impl.candidate_bundle.environment()["DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_MERGED_EXECUTION"] == "off"
    assert impl.candidate_bundle.cost_identity()["merged_lora"] is False
    assert training.expected_versions(2) == dict(generator=4, critic=20, ema=802)
    assert training.logical_versions(2) == dict(generator=202, critic=1010, ema=802)


def test_actual_observer_binds_independent_gan_semantics_and_rejects_w2_phase():
    import torch
    from distillschedule.runtime.helios_planner_cycle import _Observer
    from distillschedule.training.helios_gan_training import implementation
    from distillschedule.runtime import helios_gan_native_history as abi
    row = controls(task())[0]; row.trace_hash = "a" * 64
    observer = _Observer(torch=torch, device="cpu", rank=0, impl=implementation(), generator=object(), critic=object(),
                         ema=object(), mode="native", version_offsets=dict(generator=198, critic=990, ema=0))
    observer.current = dict(action_id="g", source_steps=[1000], source_rank=0, trace_entries=[])
    entry = NS(source_id=0, logical_step=1000, identity_sha256="b" * 64,
               rollout_operation="step/1000/generator/rollout", operation_trace=row)
    observer.entry(entry, generator=True, rank=0, native_task=task())
    actual = observer.current
    assert actual["trace_entries"][0]["native_history_semantics"] == abi.SEMANTICS
    actual["native_history_phase"] = dict(schema_version=abi.SCHEMA, action_id="g", execution_requirement=abi.REQUIREMENT,
        with_grad=True, task_sha256=task().digest, unshard_count=41, reshard_count=41, restored=True, backward_count=1, commit_count=1)
    abi.validate_action(actual)
    for change in (dict(schema_version="distillschedule.helios_native_history_execution.v1"), dict(task_sha256="W2"),
                   dict(unshard_count=40), dict(backward_count=0), dict(commit_count=0), dict(restored=False)):
        bad = deepcopy(actual); bad["native_history_phase"].update(change)
        with pytest.raises(ValueError): abi.validate_action(bad)


def test_gan_wrapper_reuses_exact_complete_block_checkpoint_validator(phase_protocol):
    from distillschedule.runtime import helios_gan_native_history as abi
    engine, Unit, _ = phase_protocol
    capability = abi.admit_engine(engine)
    assert capability["schema_version"] == abi.SCHEMA and capability["checkpoint_blocks"] == 40 and capability["fsdp_units"] == 41
    with pytest.raises(ValueError, match="G-only"):
        with abi.generator_phase(engine, task=task(), with_grad=False, action_id="r_all"): pass
    original = engine.module._gradient_checkpointing_func
    engine.module._gradient_checkpointing_func = lambda *args: None
    with pytest.raises(ValueError, match="checkpoint function"): abi.admit_engine(engine)
    engine.module._gradient_checkpointing_func = original
    engine.module.units[-1] = Unit(engine.events, "interior-linear-is-not-whole-block")
    with pytest.raises(ValueError, match="whole checkpoint blocks"): abi.admit_engine(engine)


def test_original_detached_persistence_preserves_all_rank_local_calls_without_nested_phase(phase_protocol):
    """Actual original scope, synthetic 41-unit forwards; NOT actual FSDP proof."""
    from distillschedule.runtime.fsdp2_nograd_persistence import generator_nograd_persistent_unshard_scope
    from distillschedule.runtime.helios_gan_native_history import SEMANTICS, validate_action
    engine, Unit, _ = phase_protocol; rows = controls(task())
    for rank in range(16):
        entries = []; calls = 0; flush = []
        assert not any(u.group.is_unsharded for u in engine.module.units)
        with generator_nograd_persistent_unshard_scope(engine.module, enabled=True, module_type=Unit,
                empty_cache=lambda: flush.append(True), memory_snapshot=lambda: {}) as receipt:
            for step in task().source_steps:
                control = rows[rank].operation(f"step/{step}/critic/rollout")["control"]
                entries.append(dict(source_rank=rank, native_history_policy=POLICY, native_history_semantics=SEMANTICS,
                    stage_schedule=control["pyramid_stage_call_sequence"], all_history_zero=control["all_history_zero"]))
                for _ in range(control["call_count"]):
                    # A first complete FSDP forward materializes all 41 units;
                    # the unchanged original scope retains them thereafter.
                    for unit in engine.module.units:
                        if not unit.group.is_unsharded: unit.unshard(async_op=False)
                    calls += 1
                receipt["rollout_batch_count"] += 1
        assert calls == sum(e["stage_schedule"][i]["call_count"] for e in entries for i in range(3))
        assert receipt["fully_unsharded_before_exit_count"] == 41 and flush == [True, True]
        action = dict(action_id="r_all", source_rank=rank, trace_entries=entries, persistence=receipt)
        validate_action(action)
        for change in (dict(explicit_reshard_count=40), dict(fully_unsharded_before_exit_count=40), dict(restored=False)):
            bad = deepcopy(action); bad["persistence"].update(change)
            with pytest.raises(ValueError): validate_action(bad)
        assert not any(u.group.is_unsharded for u in engine.module.units)


def test_target_original_gan_zero_history_capture_exact_rng_crops_and_r1(tmp_path):
    """Capture sampling preserves caller RNG after original helper loading.

    Cold transitive import initialization is a separate observed boundary:
    Rich style.py consumes Python getrandbits. As in the original R199 fixture,
    load those helpers before measuring actual capture-derived sampling.
    """
    import importlib.util
    missing = [name for name in ("ftfy", "omegaconf", "peft") if importlib.util.find_spec(name) is None]
    if missing: pytest.skip("NOT_RUN_NATIVE_DEPENDENCIES: " + ", ".join(missing))
    import random
    import torch
    import yaml
    from runtime.operation_trace import OperationTrace
    from distillschedule.training import helios_gan_inputs as inputs
    from distillschedule.workloads.best_known.registry import file_sha256
    cfg = inputs.portable_config()
    cfg["training_config"]["noise_corrupt_clean_prob_history"] = 1.
    config_path = tmp_path / "gan.yaml"; config_path.write_text(yaml.safe_dump(cfg)); config = inputs.read_config(config_path)
    value = task(); rank = 0; samples = []; inventory = []
    for offset, step in enumerate(value.source_steps):
        names = []
        for r in range(16):
            source = tmp_path / f"feature-{offset}-{r}.pt"; names.append(str(source))
            if r == rank:
                # Zero source history is genuine native input, not a forced
                # control flag, seed selection, normal exit, or resampling.
                torch.save(dict(vae_latent=torch.zeros(1, dtype=torch.bfloat16).expand(2, 16, 9, 48, 80),
                    prompt_embed=torch.full((2,4096), .01 * (offset+1), dtype=torch.bfloat16),
                    attention_mask=torch.ones(2, dtype=torch.bool)), source)
                inventory.append(dict(rank=r, step=step, path=str(source), sha256=file_sha256(source)))
        samples.append(tuple(names))
    sources = inputs.HeliosCohortSources(42, tuple(samples))
    from distillschedule.strategies.wan_capture import input_stream, native_primitives, stage_control
    # Admit the real CPU helper closure before native capture reuses it; a
    # plain import would leave an unadmitted namespace for later CPU proofs.
    native = native_primitives(source_only=True)  # Cold imports may consume Python RNG.
    before, py_before = torch.get_rng_state().clone(), random.getstate()
    result = inputs.capture_rank(value, sources, inventory, config, rank=rank, output=tmp_path / "actual",
                                 provenance=dict(evidence_kind="CPU_SYNTHETIC_TEST_ONLY"))
    assert torch.equal(before, torch.get_rng_state()) and py_before == random.getstate()
    loaded = OperationTrace.load(result.root, verify=True)
    trace = NS(task=value, logical_steps=value.source_steps, _load=lambda _: loaded)
    inputs.validate_rank_payload(trace, rank, sources)
    rank_calls = []
    for r in range(16):
        control = stage_control(native, value, 42, value.index, value.first_step, "generator", all_history_zero=True, rank=r)
        assert control["initial_synchronized_pyramid_stage_exit"] == loaded.operation("step/1000/generator/rollout")["control"]["initial_synchronized_pyramid_stage_exit"]
        assert control["rank"] == r and all(s["denoising_step_budget"] == 4 for s in control["pyramid_stage_call_sequence"])
        rank_calls.append(tuple(s["call_count"] for s in control["pyramid_stage_call_sequence"]))
    assert len(set(rank_calls)) > 1  # One fixed seed, no seed search or resampling.
    for step in value.source_steps:
        for phase in (("generator", "critic") if step == value.first_step else ("critic",)):
            prefix = f"step/{step}/{phase}"; c = loaded.operation(prefix + "/rollout")["control"]
            assert c["all_history_zero"] is True and c["rank"] == rank
            assert all(s["denoising_step_budget"] == 4 for s in c["pyramid_stage_call_sequence"])
            for role in (("gan_g_noise",) if phase == "generator" else ("r1_epsilon",)):
                with input_stream(42, value.index, rank, step, phase, role):
                    expected = torch.randn(value.latent_shape, dtype=torch.bfloat16)
                assert torch.equal(loaded.tensor(prefix + "/" + role), expected)
            for crop in (("gan_g_crop",) if phase == "generator" else ("gan_d_fake_crop", "gan_d_real_crop", "gan_d_r1_crop")):
                with input_stream(42, value.index, rank, step, phase, crop):
                    expected = torch.stack((torch.randint(0,25,(1,)), torch.randint(0,41,(1,))), dim=1)
                assert torch.equal(loaded.tensor(prefix + "/" + crop), expected)
    from runtime.operation_trace import STAGE3_SOURCE_KEYS
    from runtime.gan_rolephase_official import load_official_config, score_official_kwargs, critic_official_kwargs
    source = {name: loaded.tensor(key.replace("sample/0", "step/1000/sample")).unsqueeze(0)
              for name, key in STAGE3_SOURCE_KEYS.items()}
    original_config = load_official_config(config_path)
    g = score_official_kwargs(original_config, loaded, source, 1000, torch.zeros(1,2,4096,dtype=torch.bfloat16), torch.device("cpu"))
    c = critic_official_kwargs(original_config, loaded, source, 1000, torch.zeros(value.latent_shape,dtype=torch.bfloat16), NS())
    assert g["is_use_gan"] is c["is_use_gan"] is True and c["is_separate_gan_grad"] is False
    assert g["global_step"] == c["global_step"] == 1000
    for key in ("aprox_r1", "aprox_r2", "r1_weight", "r2_weight", "r1_sigma", "r2_sigma", "gan_d_weight"):
        assert c[key] == getattr(original_config.training_config, key)
    # The pre-existing W3 producer still rejects the exact same zero history.
    with pytest.raises(ValueError, match="unsupported rank-local amplified branch"):
        inputs.capture_rank(inputs.GANCohortTask(0), sources, inventory, config, rank=rank,
                            output=tmp_path / "old-default-rejected", provenance={})
    loaded.tensor("step/1000/generator/history_short", clone=False).flatten()[0] = 1.
    with pytest.raises(ValueError, match="stage grid/RNG exits"): inputs.validate_rank_payload(trace, rank, sources)
    assert not torch.cuda.is_initialized()
