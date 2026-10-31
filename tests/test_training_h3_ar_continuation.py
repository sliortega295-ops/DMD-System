"""R32 CPU protocol evidence; never native CUDA/ZeRO-3 qualification.

The engine below is SYNTHETIC_TEST_ONLY. Its module and alternating Adam are
the licensed original native classes, but its shard writer is not DeepSpeed.
"""
import contextlib
import copy
import ast
import json
import math
import os
from pathlib import Path
import random
import sys
import subprocess
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from distillschedule.strategies.h3_ar_task import H3ARTask, H3ARImplementation, SOURCE_RECEIPT
from distillschedule.training import checkpoint as cp
from distillschedule.training.h3_ar_inputs import (
    KIND, SCHEMA as INPUT_SCHEMA, H3ARInputStream, input_metadata, stream_graph,
)
from distillschedule.training.h3_ar_state import NativeCheckpoint, Progress
from distillschedule.training.h3_ar_training import checkpoint_policy, training_identity
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def stream(tmp_path, cycles=3, seed=17):
    from distillschedule.strategies.h3_ar_inputs import expected_input
    tmp_path.mkdir(parents=True, exist_ok=True)
    task = H3ARTask()
    rows = []
    for cycle in range(1, cycles+1):
        for rank in range(16):
            index = (cycle-1)*16+rank
            raw = tmp_path/f"raw-{index}.pt"
            torch.save({"fresh_record": index}, raw)
            origin = tmp_path/f"source-{index}.txt"
            origin.write_text(str(index))
            metadata = {k: v for k, v in expected_input(task, rank).items() if k not in ("rank", "input_identity")}
            rows.append(dict(cycle=cycle, rank=rank, record_id=f"record-{index}",
                path=str(raw), sha256=file_sha256(raw), metadata=metadata,
                sources={"original_cached_record": dict(path=str(origin), sha256=file_sha256(origin))}))
    value = dict(schema_version=INPUT_SCHEMA, input_kind=KIND, cycles=cycles, seed=seed,
                 native_source_receipt_sha256=SOURCE_RECEIPT,
                 input_contract_sha256=task.input_contract_sha256, records=rows)
    value["input_sha256"] = fingerprint(value)
    return write(tmp_path/"stream.json", value), value


def test_finite_fresh_order_rank6_geometry_and_legacy_graph_guard(tmp_path):
    path, value = stream(tmp_path)
    admitted = input_metadata(path, cycles=3, seed=17)
    assert admitted == value
    cursor = H3ARInputStream(path, rank=6, cycles=3, seed=17)
    assert [cursor.record(c)["record_id"] for c in (1, 2, 3)] == ["record-6", "record-22", "record-38"]
    assert cursor.record(1)["metadata"]["input_shapes"]["audio"] == [2, 32, 94]
    assert H3ARInputStream(path, rank=0, cycles=3, seed=17).record(1)["metadata"]["input_shapes"]["audio"] == [2, 32, 93]
    graph = stream_graph(H3ARTask(), cycle=3, horizon=3, input_sha256=value["input_sha256"])
    assert graph.operations[0].required_versions == dict(student=2, fake=2, ema=0)
    from distillschedule.strategies.h3_ar_task import native_graph
    with pytest.raises(Exception, match="only two"):
        native_graph(H3ARTask(), cycle=3)
    from distillschedule.strategies.h3_ar_inputs import prepare_rank_input
    with pytest.raises(Exception, match="manifest identity"):
        prepare_rank_input(H3ARTask(), dict(path=str(path), sha256=file_sha256(path)), None, 0, torch)


@pytest.mark.parametrize("bad", ["duplicate", "order", "metadata", "horizon", "source", "raw"])
def test_stream_refuses_drift_before_parser_or_model(tmp_path, bad):
    path, value = stream(tmp_path)
    if bad == "duplicate":
        value["records"][16]["sha256"] = value["records"][0]["sha256"]
    elif bad == "order":
        value["records"][0], value["records"][1] = value["records"][1], value["records"][0]
    elif bad == "metadata":
        value["records"][6]["metadata"]["input_shapes"]["audio"][-1] = 93
    elif bad == "horizon":
        value["cycles"] = 4
    elif bad == "source":
        Path(value["records"][0]["sources"]["original_cached_record"]["path"]).write_text("changed")
    else:
        Path(value["records"][0]["path"]).write_bytes(b"corrupt")
    value["input_sha256"] = fingerprint({k: v for k, v in value.items() if k != "input_sha256"})
    write(path, value)
    with pytest.raises((ValueError, RuntimeError)):
        input_metadata(path, cycles=3, seed=17, verify_files=True)


class ProtocolEngine:
    """SYNTHETIC_TEST_ONLY save/load API and native class/Adam exercise."""
    def __init__(self):
        # Exact licensed class ASTs, not rewritten class doubles. Loading the
        # full model provider would require einops, absent in the local CPU
        # environment. No missing dependencies are installed for this unit.
        from distillschedule.reference.diffsynth_h3_cached import source_identity
        source_identity()
        root = Path(__file__).parents[1]/"src/distillschedule/reference/diffsynth_h3_cached"
        namespace = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, math=math, dataclass=dataclass)
        schedule = ast.parse((root/"diffusion/flow_match.py").read_text())
        scheduler = next(n for n in schedule.body if isinstance(n, ast.ClassDef) and n.name == "FlowMatchScheduler")
        scheduler.body = [n for n in scheduler.body if isinstance(n, ast.FunctionDef) and n.name == "set_timesteps_minimax_h3"]
        exec(compile(ast.Module(body=[scheduler], type_ignores=[]), str(root/"diffusion/flow_match.py"), "exec"), namespace)
        for path, names in ((root/"h3_dmd2.py", {"DMD2Config", "DualLoRALinear", "H3Adapter", "DMD2Trainer",
                "make_schedule", "add_noise", "prediction_to_x0", "dmd_gradient", "exact_gradient_surrogate", "modality_sum", "rollout"}),
                            (root/"dmd2_zero3.py", {"DMD2Module", "AlternatingAdamW", "step"})):
            tree = ast.parse(path.read_text())
            tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
            exec(compile(tree, str(path), "exec"), namespace)
        H3Adapter, DMD2Config = namespace["H3Adapter"], namespace["DMD2Config"]
        DMD2Module, AlternatingAdamW = namespace["DMD2Module"], namespace["AlternatingAdamW"]
        model = torch.nn.Module()
        model.linear = torch.nn.Linear(2, 2)
        adapter = H3Adapter(model, DMD2Config(lora_rank=2, lora_alpha=2, lora_targets="linear", critic_updates=1))
        head = torch.nn.Linear(2, 1)
        self.module = DMD2Module(adapter, head)
        self.native_api = SimpleNamespace(**namespace)
        self.optimizer = AlternatingAdamW([
            dict(params=adapter.role_parameters("student"), role="student", lr=.01),
            dict(params=adapter.role_parameters("fake_score"), role="critic", lr=.01),
            dict(params=list(head.parameters()), role="critic", head=True, lr=.01),
        ])
        self.global_steps, self.calls, self.fail_save = 0, [], False
        self.save_non_zero_checkpoint = self.save_zero_checkpoint = True
        self.config = {"SYNTHETIC_TEST_ONLY": True}

    def zero_load_from_fp32_weights(self):
        return True

    def use_node_local_storage(self):
        return False

    def _get_ckpt_name(self, root, tag):
        from distillschedule.training.distributed import rank_world
        return str(Path(root)/tag/f"zero_pp_rank_{rank_world()[0]}_mp_rank_00_model_states.pt")

    def _get_zero_ckpt_name(self, root, tag):
        from distillschedule.training.distributed import rank_world
        return str(Path(root)/tag/f"bf16_zero_pp_rank_{rank_world()[0]}_mp_rank_00_optim_states.pt")

    def zero_optimization_stage(self):
        return 3  # Protocol marker, NOT a real ZeRO-3 engine.

    def _get_zero_param_shapes(self):
        names = {id(p): n for n, p in self.module.named_parameters()}
        return [{names[id(p)]: p.shape for p in group["params"]} for group in self.optimizer.param_groups]

    def save_checkpoint(self, save_dir, **kwargs):
        self.calls.append(("save", kwargs))
        if self.fail_save:
            raise RuntimeError("injected save failure")
        folder = Path(save_dir)/kwargs["tag"]
        folder.mkdir(parents=True, exist_ok=True)
        named = {n: p.detach().clone() for n, p in self.module.named_parameters() if p.requires_grad}
        torch.save(dict(module=named, param_shapes=self._get_zero_param_shapes(), global_steps=self.global_steps,
                        buffer_names=[], frozen_param_shapes=None, shared_params={}, frozen_param_fragments=None,
                        global_samples=self.global_steps, ds_config=self.config, ds_version="0.18.4",
                        dp_world_size=cp.distributed.rank_world()[1], mp_world_size=1,
                        **kwargs["client_state"]),
                   self._get_ckpt_name(save_dir, kwargs["tag"]))
        torch.save(self.optimizer.state_dict(), self._get_zero_ckpt_name(save_dir, kwargs["tag"]))
        return True

    def load_checkpoint(self, load_dir, **kwargs):
        self.calls.append(("load", kwargs))
        folder = Path(load_dir)/kwargs["tag"]
        value = torch.load(self._get_ckpt_name(load_dir, kwargs["tag"]), weights_only=True)
        with torch.no_grad():
            for name, p in self.module.named_parameters():
                if name in value["module"]:
                    p.copy_(value["module"][name])
        self.optimizer.load_state_dict(torch.load(self._get_zero_ckpt_name(load_dir, kwargs["tag"]), weights_only=True))
        self.global_steps = value["global_steps"]
        return str(folder), {k: v for k, v in value.items() if k not in ("module", "global_steps", "dp_world_size", "mp_world_size")}


def session(seed=7):
    from collections import Counter
    from distillschedule.strategies.h3_ar_session import H3ARSession
    from distillschedule.runtime.version_sync import VersionLedger
    torch.manual_seed(seed)
    engine = ProtocolEngine()
    adapter, discriminator = engine.module.adapter, engine.module.discriminator
    discriminator.requires_grad_(False)
    result = H3ARSession.__new__(H3ARSession)  # SYNTHETIC_TEST_ONLY construction seam, not a 50-block model.
    result.__dict__.update(engine=engine, optimizer=engine.optimizer, adapter=adapter, discriminator=discriminator,
        rank=0, task=H3ARTask(), pool=None, view=None, leaf=None, segment_origin=0, local_completed=0,
        device=torch.device("cpu"), torch=torch, census=Counter(), updates=Counter(), adam=Counter(),
        implementation=H3ARImplementation("zero3-native"), config=adapter.config,
        replay_compatibility={"counters": Counter(detached_frozen_recompute_tensors=1)},
        ledger=VersionLedger(dict(student=0, fake=0, ema=0)), observer_baseline={}, pool_ready=False)
    def predict(latents, sigmas, inputs, checkpoint=False):
        role = adapter.layers[0][1].role
        result.census[role] += 1
        if torch.is_grad_enabled(): result.census[role+"_grad"] += 1
        # Synthetic two-feature backbone, but unchanged native rollout/loss/
        # alternating-step classes below. This is not H3 numerical evidence.
        return tuple(adapter.dit.linear(x) + inputs["offset"] for x in latents)
    adapter.predict = predict
    def backward(loss):
        loss.backward()
        for group in engine.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: p.grad = torch.zeros_like(p)
        engine.optimizer.step()
        engine.optimizer.zero_grad(set_to_none=True)
        engine.global_steps += 1
    def step(role, loss):
        engine.native_api.step(SimpleNamespace(backward=backward), engine, engine.optimizer, discriminator, role, loss, adapter.config)
        result.updates[role] += 1
        result.adam[role] += 1
        result.census[role+"_backward"] += 1
    result.trainer = engine.native_api.DMD2Trainer(adapter, discriminator, adapter.config, step)
    return result


def advance(s, cycle):
    s.before = s.census.copy(), s.updates.copy(), s.adam.copy()
    real = (torch.tensor([[cycle/100, 1.0]]), torch.tensor([[1.0, -cycle/100]]))
    s.metrics = s.trainer.update(real, {"offset": cycle/1000})
    return s.finish(cycle)


def components(tmp_path, cycles=3, start=0):
    path, _ = stream(tmp_path, cycles=cycles)
    s = session()
    cursor = H3ARInputStream(path, rank=0, cycles=cycles, seed=17, completed=start)
    s.input_provider = cursor
    owner = NativeCheckpoint(s)
    progress = Progress(s, cursor, owner)
    return s, cursor, owner, {"native_progress": progress}


def complete_cycle(s, cursor, cycle):
    cursor.pending = cycle
    return advance(s, cycle)


def test_native_api_all_permanent_trainables_inactive_head_and_flag_finally(tmp_path):
    s, cursor, owner, state = components(tmp_path)
    complete_cycle(s, cursor, 1)
    flags = {n: p.requires_grad for n, p in s.engine.module.named_parameters()}
    target = cp.save_checkpoint(tmp_path/"ckpt", state, identity={"R32": True}, cycle=1, cursor=1,
        versions=s.versions(), native_checkpoint=owner)
    call = s.engine.calls[-1]
    assert call[0] == "save" and call[1]["exclude_frozen_parameters"] is True and call[1]["save_latest"] is False
    data = torch.load(target/"native/state/zero_pp_rank_0_mp_rank_00_model_states.pt", weights_only=True)
    assert any("fake_score" in name for name in data["module"])
    assert any("student" in name for name in data["module"])
    assert any("discriminator" in name for name in data["module"])
    assert not any("base" in name for name in data["module"])
    assert {n: p.requires_grad for n, p in s.engine.module.named_parameters()} == flags
    assert all(not s.optimizer.state[p] for p in s.discriminator.parameters())
    s.engine.fail_save = True
    with pytest.raises(RuntimeError, match="injected"):
        cp.save_checkpoint(tmp_path/"failed", state, identity={}, cycle=1, cursor=1,
                           versions=s.versions(), native_checkpoint=owner)
    assert {n: p.requires_grad for n, p in s.engine.module.named_parameters()} == flags
    assert not (tmp_path/"failed/_SUCCESS").exists()


@pytest.mark.parametrize("bad", ["missing", "corrupt", "identity", "layout", "used_engine"])
def test_restore_rejects_before_any_owner_mutation(tmp_path, bad):
    s, cursor, owner, state = components(tmp_path)
    complete_cycle(s, cursor, 1)
    cp.save_checkpoint(tmp_path/"ckpt", state, identity={"source": "same"}, cycle=1, cursor=1,
                       versions=s.versions(), native_checkpoint=owner)
    restored = session()
    target_owner = NativeCheckpoint(restored)
    target_cursor = H3ARInputStream(cursor.path, rank=0, cycles=3, seed=17, completed=1)
    restored.input_provider = target_cursor
    target_state = {"native_progress": Progress(restored, target_cursor, target_owner)}
    if bad == "used_engine":
        restored.engine.global_steps = 2
    before = {n: p.detach().clone() for n, p in restored.engine.module.named_parameters()}
    shard = tmp_path/"ckpt/native/state/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
    if bad == "missing":
        shard.unlink()
    elif bad == "corrupt":
        shard.write_bytes(b"corrupt")
    elif bad == "layout":
        target_owner.layout["invalid"] = {}
    with pytest.raises((ValueError, RuntimeError)):
        cp.restore_checkpoint(tmp_path/"ckpt", target_state,
            identity={"source": "changed" if bad == "identity" else "same"}, native_checkpoint=target_owner)
    assert not restored.engine.calls
    assert all(torch.equal(before[n], p) for n, p in restored.engine.module.named_parameters())


def test_native_restart_exact_adam_rng_cursor_and_skip_semantics(tmp_path):
    s, cursor, owner, state = components(tmp_path)
    complete_cycle(s, cursor, 1)
    cp.save_checkpoint(tmp_path/"ckpt", state, identity={"R32": True}, cycle=1, cursor=1,
                       versions=s.versions(), native_checkpoint=owner)
    for cycle in (2, 3):
        complete_cycle(s, cursor, cycle)
    expected = copy.deepcopy(s.engine.module.state_dict()), copy.deepcopy(s.optimizer.state_dict()), torch.get_rng_state()
    restored = session()
    target_owner = NativeCheckpoint(restored)
    target_cursor = H3ARInputStream(cursor.path, rank=0, cycles=3, seed=17, completed=1)
    restored.input_provider = target_cursor
    cp.restore_checkpoint(tmp_path/"ckpt", {"native_progress": Progress(restored, target_cursor, target_owner)},
                          identity={"R32": True}, native_checkpoint=target_owner)
    assert restored.engine.calls[-1][1]["load_optimizer_states"] is True
    assert restored.engine.global_steps == 2 and restored.trainer.generator_step == 1
    assert restored.segment_origin == 1 and target_cursor.completed == 1
    for cycle in (2, 3):
        complete_cycle(restored, target_cursor, cycle)
    def equal(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a: equal(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b): equal(x, y)
        else:
            assert a == b
    equal(expected, (restored.engine.module.state_dict(), restored.optimizer.state_dict(), torch.get_rng_state()))
    assert restored.engine.global_steps == 6 and target_cursor.completed == 3


def test_pending_transient_cache_refuses_publication(tmp_path):
    s, cursor, owner, state = components(tmp_path)
    complete_cycle(s, cursor, 1)
    s.pool = SimpleNamespace(selected=None, rollout_plan=None, group=None, prefetched={"pending": 1},
        prefetch_groups={}, immutable_entries=[], consumer_queues=[], buffers={}, pending={})
    with pytest.raises((ValueError, RuntimeError), match="cache|prefetch|pending|drain"):
        cp.save_checkpoint(tmp_path/"ckpt", state, identity={}, cycle=1, cursor=1,
                           versions=s.versions(), native_checkpoint=owner)
    assert not s.engine.calls and not (tmp_path/"ckpt/_SUCCESS").exists()


def public_config(tmp_path, monkeypatch, *, reference=False):
    from distillschedule.training.config import TrainingConfig
    from distillschedule.strategies.h3_ar_assets import H3ARAssets
    path, _ = stream(tmp_path/"data", cycles=100)
    fields = {name: {} for name in H3ARAssets.__dataclass_fields__}
    fields.update(schema_version="distillschedule.original_h3_ar_assets.v1", source_root="/SYNTHETIC_TEST_ONLY/source",
        checkpoint_root="/SYNTHETIC_TEST_ONLY/checkpoint", cache_manifest=dict(path=str(path), sha256=file_sha256(path)))
    write(tmp_path/"assets.json", fields)
    write(tmp_path/"task.json", H3ARTask().to_dict())
    write(tmp_path/"capacity.json", dict(gpu_limit_bytes=1000, gpu_reserve_bytes=10, host_limit_bytes=3000, host_reserve_bytes=100))
    # Only actual large asset admission is substituted. The public config,
    # source-layout, stream bytes/order, policy and native lowering remain real.
    monkeypatch.setattr(H3ARAssets, "validate", lambda _: {"evidence_kind": "SYNTHETIC_TEST_ONLY_ASSET_ADMISSION"})
    value = dict(schema_version="distillschedule.training.v1", model=dict(workload="W6"),
        algorithm=dict(name="original_diffsynth_h3_ar"), data=dict(input_kind=KIND, prepared=str(path)),
        execution=dict(adapter="native", mode="reference" if reference else "manual", task=str(tmp_path/"task.json"),
                       assets=str(tmp_path/"assets.json"), capacity_policy=str(tmp_path/"capacity.json")),
        trainer=dict(cycles=100, seed=17, checkpoint_every=50, stop_after_cycle=50, output_dir=str(tmp_path/"run")))
    value["execution"].update(dict(implementation="zero3-native") if reference else dict(preset="h3_ar.zero3_ap_stream.v1"))
    return TrainingConfig.from_dict(value, source=tmp_path/"training.json")


@pytest.mark.parametrize("reference", [False, True])
def test_public_stream_config_lowers_to_actual_new_native_owner_not_historical_script(tmp_path, monkeypatch, reference):
    from distillschedule.training.native import describe_native, validate_native, run_native
    import distillschedule.training.h3_ar_training as native
    config = public_config(tmp_path, monkeypatch, reference=reference)
    report = validate_native(config)
    assert report["native_worker"] == "distillschedule.training.h3_ar_training:run_training"
    assert report["selected"]["id"] == ("zero3-native" if reference else "zero3-ap")
    assert report["capabilities"]["cycles"]["max"] == 100
    assert report["capabilities"]["checkpoint"]["supported"] is True
    assert "native_argv" not in report and not Path(config.trainer["output_dir"]).exists()
    called = []
    monkeypatch.setattr(native, "run_training", lambda conf, *, admit: called.append((conf, admit()[3])) or 31)
    assert run_native(config) == 31
    assert called[0][1]["lifecycle"]["owner"].startswith("ORIGINAL_H3ARSession")
    assert describe_native(config)["gpu_execution"] == "NOT_RUN"
    first = training_identity("a"*40, called[0][1], config)
    raw = config.to_dict()
    raw["trainer"].update(stop_after_cycle=100, resume_from=str(tmp_path/"old-checkpoint"), output_dir=str(tmp_path/"continued"))
    from distillschedule.training.config import TrainingConfig
    resumed = TrainingConfig.from_dict(raw, source=config.source)
    assert training_identity("a"*40, called[0][1], resumed) == first


@pytest.mark.parametrize("bad", ["horizon", "no_seed", "input_kind", "stop", "implementation", "capacity"])
def test_public_stream_bad_controls_fail_before_construction(tmp_path, monkeypatch, bad):
    from distillschedule.training.native import validate_native
    from distillschedule.training.config import TrainingConfig
    config = public_config(tmp_path, monkeypatch)
    raw = config.to_dict()
    if bad == "horizon": raw["trainer"]["cycles"] = 101
    elif bad == "no_seed": raw["trainer"].pop("seed")
    elif bad == "input_kind": raw["data"]["input_kind"] = "historical"
    elif bad == "stop": raw["trainer"]["stop_after_cycle"] = 49
    elif bad == "implementation": raw["execution"]["implementation"] = "zero3-native"
    else: raw["execution"].pop("capacity_policy")
    with pytest.raises(ValueError):
        validate_native(TrainingConfig.from_dict(raw, source=config.source))
    assert not Path(config.trainer["output_dir"]).exists()


def test_actual_epoch_helper_cold_50_to_51_then_drained_warm_52(tmp_path):
    """Real frozen epoch helper, SYNTHETIC_TEST_ONLY pool/CUDA/model body."""
    from collections import Counter
    from distillschedule.strategies.h3_ar_assets import HELPER_PINS
    source = os.environ.get("DISTILLSCHEDULE_H3_AR_EPOCH_HELPER")
    if not source:
        pytest.skip("explicit SHA-bound native epoch helper source required; no fetch or replacement")
    assert file_sha256(Path(source)) == HELPER_PINS["native_selected_cycle_scope_r1"]
    namespace = {}
    exec(compile(Path(source).read_text(), source, "exec"), namespace)
    epoch = SimpleNamespace(**namespace)
    s, cursor, _, _ = components(tmp_path, cycles=100)
    # Install the ORIGINAL read-only view against trainer zero, as production
    # does; only then restore global native clocks and rebase the view origin.
    s.view = epoch.RelativeVersionView(s.trainer)
    s.pool = SimpleNamespace(selected=None, rollout_plan=None, group=None, prefetched={}, prefetch_groups={},
        immutable_entries=[], consumer_queues=[], buffers={}, pending={}, sources={}, entries={}, routing={},
        last_consumed={}, prefetch_stats={}, counters=Counter(pinned_pool_logical_bytes=0), cpu_test=True,
        receipt=lambda: {"evidence_kind": "SYNTHETIC_TEST_ONLY_POOL"})
    s.trainer.generator_step = s.trainer.critic_step = cursor.completed = 50
    s.engine.global_steps = 100
    s.rebase_segment(50)
    begins = []
    def begin(pool, view):
        begins.append(view.origin)
        return epoch.begin_next_cycle(pool, view)
    s.epoch = SimpleNamespace(finish_cycle=epoch.finish_cycle, begin_next_cycle=begin)
    s.torch = SimpleNamespace(Tensor=torch.Tensor, bfloat16=torch.bfloat16,
                              autocast=lambda **kw: contextlib.nullcontext())
    s.build = {}
    s.assets = SimpleNamespace(cache_manifest=cursor.reference)
    s.native = None
    def load(task, cache, native, rank, t, **kw):
        cycle = cursor.completed+1
        cursor.pending = cycle
        row = cursor.record(cycle)
        return dict(input_latents=torch.ones(1, 2), audio_input_latents=torch.ones(1, 2), offset=cycle/1000), None, {
            k: row[k] for k in ("path", "sha256", "record_id", "cycle")}, {"input_shapes": task.input_shapes(rank)}
    cursor.prepare_rank_input = load  # Data/CUDA substitute is explicit; original helper and Session are real.
    s.input_identity = {}
    s.prepare(51).run()
    assert begins == []  # The fresh resumed pool was empty, not a completed old epoch.
    s.pool.selected = object()
    first = s.finish(51)
    assert s.engine.global_steps == 102 and not first["metrics"]["gan_active"]
    cache = first["selected_cache"]
    assert cache["source_student_version"] == cache["cache_epoch"] == cache["segment_origin"] == 50
    assert cache["source_helper_epoch_metadata"]["cold_cache_staging_inside_timer"] is False
    assert cache["cold_cache_staging_inside_timer"] is True and cache["same_process_pinned_pool_reuse"] is False
    s.drain_for_checkpoint()
    assert begins == [50] and s.view.origin == 51 and s.pool.selected is None
    s.prepare(52).run()  # A drained pool must not receive a second begin_next_cycle.
    assert begins == [50]
    s.pool.selected = object()
    second = s.finish(52)
    assert second["selected_cache"]["process_local_cycle"] == 2
    assert second["selected_cache"]["cold_cache_staging_inside_timer"] is False
    assert s.engine.global_steps == 104 and cursor.completed == 52 and not second["metrics"]["gan_active"]


def test_process_local_fused_observers_do_not_expect_51_cycles_of_work(tmp_path):
    from collections import Counter
    s, cursor, _, _ = components(tmp_path, cycles=100)
    s.trainer.generator_step = s.trainer.critic_step = cursor.completed = 50
    s.engine.global_steps = 100
    s.leaf = {"counters": Counter(), "replay_lifetime_counters": Counter(max_pending_leaf_blocks=0)}
    s.previous_leaf = s.previous_lifetime = Counter()
    s.leaf_api = SimpleNamespace(finalize_lifetime=lambda dit, view: {"SYNTHETIC_TEST_ONLY": True})
    s.dit = None
    s.swiglu = {"counters": Counter(physical_swiglu_forward=9, physical_swiglu_backward=7)}
    s.cutlass = {"counters": Counter(physical_B_module_forward=3, fused_custom_backward=2),
                 "operator_counters": Counter(physical_CUDA_B_GEMM_round_then_add=5)}
    s.rms = {"counters": Counter()}
    s.pointwise = {"counters": Counter(_modulate_scale_shift=1, _modulate_gate=1, _apply_rope=1)}
    s.mechanism = SimpleNamespace(counters=Counter(lora_inplace_completed_sums=1))
    s.prefetch_api = SimpleNamespace(receipt=lambda _: {"SYNTHETIC_TEST_ONLY": True})
    s.prefetch = None
    s.rebase_segment(50)
    s.swiglu["counters"].update(physical_swiglu_forward=724, physical_swiglu_backward=104)
    s.cutlass["counters"].update(physical_B_module_forward=2600, fused_custom_backward=400)
    s.cutlass["operator_counters"].update(physical_CUDA_B_GEMM_round_then_add=2600)
    s.rms["counters"].update(physical_norm_module_forward=56*50, logical_mod_calls=28*50,
        logical_rope_calls=28*50, fused_mod_forward=28*50, fused_rope_forward=28*50,
        fused_mod_backward=4*50, fused_rope_backward=4*50)
    checks = complete_cycle(s, cursor, 51)
    assert checks["backend_effects"]["process_local_completed"] == 1
    assert checks["final_versions"] == dict(student=51, fake=51, ema=0) and s.engine.global_steps == 102


def _restart_child(checkpoint, manifest, output):
    s = session()
    cursor = H3ARInputStream(manifest, rank=0, cycles=100, seed=17, completed=50)
    s.input_provider = cursor
    owner = NativeCheckpoint(s)
    random.seed(999)
    torch.manual_seed(999)
    cp.restore_checkpoint(checkpoint, {"native_progress": Progress(s, cursor, owner)},
                          identity={"R32": "new-process-100"}, native_checkpoint=owner)
    assert s.segment_origin == 50 and s.engine.global_steps == 100
    for cycle in range(51, 101):
        checks = complete_cycle(s, cursor, cycle)
        assert checks["metrics"]["gan_active"] is False
    torch.save(dict(module=s.engine.module.state_dict(), optimizer=s.optimizer.state_dict(), rng=cp.rng_state(),
                    cursor=cursor.completed, versions=s.versions(), global_steps=s.engine.global_steps), output)
    print(json.dumps(dict(evidence_kind="SYNTHETIC_TEST_ONLY_ENGINE_ACTUAL_LICENSED_TRAINER_ADAM", cycles=100,
                          native_global_steps=s.engine.global_steps, CUDA_initialized=torch.cuda.is_initialized())))


def test_new_os_process_50_to_100_matches_full_native_class_adam_rng_cursor(tmp_path):
    s, cursor, owner, state = components(tmp_path, cycles=100)
    for cycle in range(1, 51): complete_cycle(s, cursor, cycle)
    checkpoint = cp.save_checkpoint(tmp_path/"cycle50", state, identity={"R32": "new-process-100"},
        cycle=50, cursor=50, versions=s.versions(), native_checkpoint=owner)
    for cycle in range(51, 101):
        assert complete_cycle(s, cursor, cycle)["metrics"]["gan_active"] is False
    expected = dict(module=s.engine.module.state_dict(), optimizer=s.optimizer.state_dict(), rng=cp.rng_state(),
                    cursor=cursor.completed, versions=s.versions(), global_steps=s.engine.global_steps)
    output = tmp_path/"restored.pt"
    script = "from test_training_h3_ar_continuation import _restart_child; import sys; _restart_child(*sys.argv[1:])"
    paths = [str(Path(__file__).parent), str(Path(__file__).parents[1]/"src")]
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths),
               CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    process = subprocess.run([sys.executable, "-B", "-c", script, str(checkpoint), str(cursor.path), str(output)],
                             env=env, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["native_global_steps"] == 200
    actual = torch.load(output, weights_only=True)
    def equal(a, b):
        if isinstance(a, torch.Tensor): assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a: equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for left, right in zip(a, b): equal(left, right)
        else: assert a == b
    equal(expected, actual)
    assert all(not s.optimizer.state[p] for p in s.discriminator.parameters())


def test_installed_deepspeed_serialization_methods_include_inactive_roles_without_backbone():
    """Actual installed methods only; no DeepSpeed engine or CUDA construction."""
    engine_module = pytest.importorskip("deepspeed.runtime.engine", reason="target-only installed DeepSpeed0.18.4 method proof")
    from distillschedule.training.h3_ar_training import DEEPSPEED_ENGINE_SHA256
    assert file_sha256(Path(engine_module.__file__)) == DEEPSPEED_ENGINE_SHA256
    s = session()
    owner = NativeCheckpoint(s)
    parameters = dict(s.engine.module.named_parameters())
    shim = SimpleNamespace(module=s.engine.module, random_ltd_enabled=lambda: False,
        optimizer=SimpleNamespace(fp16_groups=[list(reversed(list(owner.parameters.values())))]),
        bfloat16_enabled=lambda: True, zero_optimization_stage=lambda: 3,
        param_names={p: n for n, p in parameters.items()})
    method = engine_module.DeepSpeedEngine
    flags = {n: p.requires_grad for n, p in parameters.items()}
    with owner.registered_trainables():
        weights = method.module_state_dict(shim, exclude_frozen_parameters=True)
        assert set(weights) == set(owner.parameters)
        assert not any(".base." in n for n in weights)
        shapes = method._get_zero_param_shapes(shim)
        assert list(shapes[0]) == list(reversed(list(owner.parameters)))
    assert {n: p.requires_grad for n, p in parameters.items()} == flags
    assert not torch.cuda.is_initialized()


def test_public_discovery_remains_torch_and_model_import_free():
    script = """
import sys
from distillschedule.training.native_presets import describe_preset
from distillschedule.training.h3_ar_training import descriptor
assert describe_preset('W6', 'h3_ar.zero3_ap_stream.v1')['continuous_training']
assert descriptor()['horizon']['max'] == 100
assert 'torch' not in sys.modules
assert not any(k.startswith(('deepspeed', 'diffsynth', 'accelerate')) for k in sys.modules)
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("bad", ["module_key", "partition_order", "native_config", "native_clock"])
def test_hashed_but_malformed_native_metadata_is_staged_before_restore_mutation(tmp_path, bad):
    s, cursor, owner, state = components(tmp_path)
    complete_cycle(s, cursor, 1)
    target = cp.save_checkpoint(tmp_path/"ckpt", state, identity={"R32": True}, cycle=1, cursor=1,
        versions=s.versions(), native_checkpoint=owner)
    path = target/"native/state/zero_pp_rank_0_mp_rank_00_model_states.pt"
    value = torch.load(path, weights_only=True)
    if bad == "module_key": value["module"].pop(next(n for n in value["module"] if "fake_score" in n))
    elif bad == "partition_order": value["param_shapes"][0] = dict(reversed(list(value["param_shapes"][0].items())))
    elif bad == "native_config": value["ds_config"]["changed"] = True
    else: value["global_steps"] = 4
    torch.save(value, path)
    # Re-sign to exercise semantic preflight rather than the ordinary byte-hash
    # rejection. A real publisher would refuse these payloads before _SUCCESS.
    manifest = json.loads((target/"manifest.json").read_text())
    manifest["native_checkpoint"]["files"][str(path.relative_to(target))] = file_sha256(path)
    write(target/"manifest.json", manifest)
    (target/"_SUCCESS").write_text(file_sha256(target/"manifest.json")+"\n")
    restored = session()
    restored.input_provider = H3ARInputStream(cursor.path, rank=0, cycles=3, seed=17, completed=1)
    native = NativeCheckpoint(restored)
    before = {n: p.detach().clone() for n, p in restored.engine.module.named_parameters()}
    with pytest.raises((ValueError, RuntimeError)):
        cp.restore_checkpoint(target, {"native_progress": Progress(restored, restored.input_provider, native)},
                              identity={"R32": True}, native_checkpoint=native)
    assert restored.engine.calls == []
    assert all(torch.equal(p, before[n]) for n, p in restored.engine.module.named_parameters())


def test_native_load_exception_restores_all_temporary_trainability_flags(tmp_path, monkeypatch):
    s, cursor, owner, state = components(tmp_path)
    complete_cycle(s, cursor, 1)
    cp.save_checkpoint(tmp_path/"ckpt", state, identity={}, cycle=1, cursor=1,
                       versions=s.versions(), native_checkpoint=owner)
    restored = session()
    restored.input_provider = H3ARInputStream(cursor.path, rank=0, cycles=3, seed=17, completed=1)
    native = NativeCheckpoint(restored)
    flags = {n: p.requires_grad for n, p in restored.engine.module.named_parameters()}
    def fail(*args, **kwargs):
        assert all(p.requires_grad for p in native.parameters.values())
        raise RuntimeError("injected native load failure")
    monkeypatch.setattr(restored.engine, "load_checkpoint", fail)
    with pytest.raises(RuntimeError, match="injected native load"):
        cp.restore_checkpoint(tmp_path/"ckpt", {"native_progress": Progress(restored, restored.input_provider, native)},
                              identity={}, native_checkpoint=native)
    assert flags == {n: p.requires_grad for n, p in restored.engine.module.named_parameters()}


def test_initial_parser_result_is_staged_once_then_consumed_by_same_constructor_hook(tmp_path, monkeypatch):
    import distillschedule.training.h3_ar_inputs as inputs
    path, value = stream(tmp_path, cycles=2)
    metadata = value["records"][0]["metadata"]
    def meta(value):
        if isinstance(value, dict) and set(value) == {"shape", "dtype"}:
            return torch.empty(value["shape"], dtype=getattr(torch, value["dtype"].split(".")[-1]), device="meta")
        if isinstance(value, dict): return {k: meta(v) for k, v in value.items()}
        if isinstance(value, list): return [meta(v) for v in value]
        return value
    raw = meta(metadata["raw_tensors"])
    torch.save(raw, value["records"][0]["path"])
    value["records"][0]["sha256"] = file_sha256(Path(value["records"][0]["path"]))
    value["input_sha256"] = fingerprint({k: v for k, v in value.items() if k != "input_sha256"})
    write(path, value)
    calls = []
    positive = {**meta(metadata["positive_tensors"]), **metadata["positive_scalars"]}
    def parse(raw, cfg):
        calls.append("SYNTHETIC_TEST_ONLY_META_PARSER")
        return positive, None
    monkeypatch.setattr(inputs, "validate_parser", lambda native: None)
    native = SimpleNamespace(prepare_batch=parse)
    cursor = H3ARInputStream(path, rank=0, cycles=2, seed=17)
    witness = cursor.stage_initial(H3ARTask(), cursor.reference, native, 0, torch)
    prepared = cursor.prepare_rank_input(H3ARTask(), cursor.reference, native, 0, torch)
    assert len(calls) == 1 and prepared[0] is positive and prepared[3] == witness
    assert cursor.completed == 0 and cursor.pending == 1 and cursor._prepared is None


def test_public_data_cli_forwards_explicit_finite_preparation_controls(tmp_path, monkeypatch, capsys):
    from distillschedule.data.cli import main
    import distillschedule.data.h3_ar_stream as data
    called = []
    monkeypatch.setattr(data, "prepare_inputs", lambda *args, **kwargs: called.append((args, kwargs))
        or {"status": "SYNTHETIC_TEST_ONLY_CLI"})
    assert main(["prepare-h3-ar-stream", "--sources", str(tmp_path/"sources.json"), "--assets", str(tmp_path/"assets.json"),
                 "--cycles", "100", "--seed", "42", "--output", str(tmp_path/"prepared.json")]) == 0
    assert called == [((tmp_path/"sources.json", tmp_path/"prepared.json"),
                       dict(assets_path=tmp_path/"assets.json", cycles=100, seed=42))]
    assert json.loads(capsys.readouterr().out)["status"] == "SYNTHETIC_TEST_ONLY_CLI"


def test_new_owner_dispatch_and_common_native_publication_are_one_reachable_chain(tmp_path, monkeypatch):
    """SYNTHETIC_TEST_ONLY CUDA/other ranks; actual native class update owner."""
    from distillschedule.training.h3_ar_training import execute_segment
    from distillschedule.training import distributed
    s, cursor, _, _ = components(tmp_path/"inputs", cycles=3)
    s.torch = SimpleNamespace(Tensor=torch.Tensor, bfloat16=torch.bfloat16,
        autocast=lambda **kw: contextlib.nullcontext(), cuda=SimpleNamespace(synchronize=lambda _: None))
    s.build, s.native, s.input_identity = {}, None, {}
    s.assets = SimpleNamespace(cache_manifest=cursor.reference)
    def load(task, cache, native, rank, t, **kw):
        cycle = cursor.completed+1
        cursor.pending = cycle
        row = cursor.record(cycle)
        return dict(input_latents=torch.ones(1, 2), audio_input_latents=torch.ones(1, 2), offset=cycle/1000), None, {
            k: row[k] for k in ("path", "sha256", "record_id", "cycle")}, {"input_shapes": task.input_shapes(rank)}
    cursor.prepare_rank_input = load
    s.state_metadata = lambda cycle: {"evidence_kind": "SYNTHETIC_TEST_ONLY_NO_ZeRO3_MASTER_PARTITIONS", "cycle": cycle}
    original = distributed.gather
    def gather(value):
        if isinstance(value, dict) and set(value) == {"rank", "cycle", "versions", "native_global_steps", "input_identity"}:
            # Explicit synthetic peer metadata; never distributed qualification.
            cycle = value["cycle"]
            return [dict(value, rank=rank, input_identity={k: cursor.document["records"][(cycle-1)*16+rank][k]
                    for k in ("path", "sha256", "record_id", "cycle")}) for rank in range(16)]
        return original(value)
    monkeypatch.setattr(distributed, "gather", gather)
    rows = execute_segment(s, stop=3, output=tmp_path/"run", identity={"checkpoint_every": 1, "R32": "chain"},
                           capacity_check=lambda: {"evidence_kind": "SYNTHETIC_TEST_ONLY_CAPACITY"})
    assert [row["cycle"] for row in rows] == [1, 2, 3]
    assert s.engine.global_steps == 6 and cursor.completed == 3
    assert [call[0] for call in s.engine.calls] == ["save", "save", "save"]
    assert (tmp_path/"run/checkpoints/cycle-000003/_SUCCESS").is_file()


def test_common_checkpoint_without_native_owner_retains_legacy_contract(tmp_path):
    state = {"weights": torch.tensor([1.0, 2.0])}
    cp.save_checkpoint(tmp_path/"ckpt", state, identity={"plain": True}, cycle=1, cursor=1, versions={"weights": 1})
    assert "native_checkpoint" not in cp.read_manifest(tmp_path/"ckpt", identity={"plain": True})
    state["weights"].zero_()
    cp.restore_checkpoint(tmp_path/"ckpt", state, identity={"plain": True})
    assert torch.equal(state["weights"], torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize("failure", ["parser", "source_audit"])
def test_preparation_refuses_before_publishing_any_usable_manifest(tmp_path, monkeypatch, failure):
    import distillschedule.training.h3_ar_inputs as inputs
    _, value = stream(tmp_path/"sources", cycles=2)
    rows = [{k: v for k, v in row.items() if k != "metadata"} for row in value["records"]]
    sources = write(tmp_path/"sources.json", rows)
    output = tmp_path/"prepared.json"
    def parse(self, *args, **kwargs):
        if failure == "parser":
            raise ValueError("injected original parser refusal")
        self.pending = self.completed+1
        return {}, None, {}, {}
    monkeypatch.setattr(H3ARInputStream, "prepare_rank_input", parse)  # Explicit orchestration-only parser substitute.
    def guard(): raise ValueError("injected source audit refusal")
    with pytest.raises(ValueError, match="injected"):
        inputs.prepare_stream(sources, output, native=None, torch=torch, cycles=2, seed=17, pre_publication_guard=guard)
    assert not output.exists() and not output.with_name("."+output.name+".preparing").exists()


def _native_sidecar_gloo_worker(rank, rendezvous, root, mode="missing"):
    """Real Gloo/common publisher, SYNTHETIC_TEST_ONLY native engine/CUDA."""
    from datetime import timedelta
    from unittest.mock import patch
    import torch.distributed as dist
    from distillschedule.training import distributed
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2", GROUP_RANK="0", CUDA_VISIBLE_DEVICES="")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://"+rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        root = Path(root)
        if rank == 0: stream(root/"inputs", cycles=2)
        dist.barrier()
        s = session()
        s.rank = rank
        cursor = H3ARInputStream(root/"inputs/stream.json", rank=rank, cycles=2, seed=17)
        s.input_provider = cursor
        native = distributed.checked(lambda: NativeCheckpoint(s), "SYNTHETIC native layout")
        native.bind_writers()
        state = {"native_progress": Progress(s, cursor, native)}
        complete_cycle(s, cursor, 1)
        phase, attempts = {"native_visibility": False, "native_saved": False}, {"count": 0}
        actual_wait, is_file, actual_sha = cp._await_shared_paths, Path.is_file, cp._sha
        def wait(paths, label, **kw):
            phase["native_visibility"] = label == "native checkpoint shared shard visibility"
            phase["native_saved"] |= phase["native_visibility"]
            try: return actual_wait(paths, label, **kw)
            finally: phase["native_visibility"] = False
        def visible(path):
            if mode == "missing" and rank == 0 and phase["native_visibility"] and path.name == "bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt":
                attempts["count"] += 1
                if attempts["count"] <= 3: return False
            return is_file(path)
        def observed_sha(path):
            if mode != "missing" and rank == 0 and phase["native_saved"] and path.name == "bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt":
                attempts["count"] += 1
                if mode == "persistent" or attempts["count"] <= 3: return "0"*64
            return actual_sha(path)
        with patch.object(cp, "_await_shared_paths", wait), patch.object(Path, "is_file", visible), \
                patch.object(cp, "_sha", observed_sha), patch.object(cp, "_STORAGE_VISIBILITY_TIMEOUT", .35 if mode == "persistent" else 1.):
            def save():
                return cp.save_checkpoint(root/"checkpoint", state, identity={"R32": "SYNTHETIC_NATIVE_REAL_GLOO"},
                    cycle=1, cursor=1, versions=s.versions(), native_checkpoint=native)
            if mode == "persistent":
                with pytest.raises(ValueError, match="shared storage visibility timed out"):
                    save()
                assert not (root/"checkpoint/_SUCCESS").exists()
                assert attempts["count"] >= 2 if rank == 0 else attempts["count"] == 0
                write(root/f"rank{rank}.json", dict(evidence_kind="SYNTHETIC_TEST_ONLY_ENGINE_REAL_GLOO",
                    visibility_mode=mode, visibility_attempts=attempts["count"], published=False))
                return
            save()
        header = cp.read_manifest(root/"checkpoint", identity={"R32": "SYNTHETIC_NATIVE_REAL_GLOO"})
        names = header["native_checkpoint"]["files"]
        assert len(names) == 4 and len(header["native_checkpoint"]["owner"]["writers"]) == 2
        assert attempts["count"] >= 4 if rank == 0 else attempts["count"] == 0
        assert cp.read_checkpoint(root/"checkpoint", identity={"R32": "SYNTHETIC_NATIVE_REAL_GLOO"})["cycle"] == 1
        write(root/f"rank{rank}.json", dict(evidence_kind="SYNTHETIC_TEST_ONLY_ENGINE_REAL_GLOO_COMMON_NATIVE_SIDECAR_PUBLICATION",
            native_shards=len(names), visibility_mode=mode, visibility_attempts=attempts["count"], CUDA_initialized=torch.cuda.is_initialized()))
    finally:
        dist.destroy_process_group()


def test_real_gloo_common_publisher_waits_for_all_native_sidecar_shards(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_native_sidecar_gloo_worker, args=(str(tmp_path/"rendezvous"), str(tmp_path)), nprocs=2, join=True)
    assert all(json.loads((tmp_path/f"rank{rank}.json").read_text())["native_shards"] == 4 for rank in range(2))


@pytest.mark.parametrize("mode", ["stale", "persistent"])
def test_real_gloo_native_content_visibility_not_only_existence(tmp_path, mode):
    import torch.multiprocessing as mp
    mp.spawn(_native_sidecar_gloo_worker, args=(str(tmp_path/"rendezvous"), str(tmp_path), mode), nprocs=2, join=True)
    receipts = [json.loads((tmp_path/f"rank{rank}.json").read_text()) for rank in range(2)]
    assert all(row["visibility_mode"] == mode for row in receipts)
    assert receipts[0]["visibility_attempts"] >= 4
    assert (tmp_path/"checkpoint/_SUCCESS").exists() is (mode == "stale")


def test_native_visibility_exact_path_keys_do_not_alias_same_basename(tmp_path, monkeypatch):
    from distillschedule.open_workload import replicated
    paths = [tmp_path/role/"model.pt" for role in ("student", "critic")]
    for index, path in enumerate(paths):
        path.parent.mkdir(); path.write_text(str(index))
    expected = {path: file_sha256(path) for path in paths}
    attempts = {path: 0 for path in paths}
    def observed(path):
        attempts[path] += 1
        return "0"*64 if attempts[path] == 1 else expected[path]
    monkeypatch.setattr(cp.distributed, "rank_world", lambda: (0, 2))
    monkeypatch.setattr(replicated, "metadata_timeout_seconds", lambda: 120.)
    monkeypatch.setattr(cp, "_sha", observed)
    cp._await_shared_paths(paths, "exact native paths", hashes=expected)
    assert attempts == {path: 2 for path in paths}
