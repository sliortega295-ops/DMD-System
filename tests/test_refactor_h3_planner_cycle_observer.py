from __future__ import annotations

import __future__
import ast
from contextlib import nullcontext
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from distillschedule.backends.training.minimax_h3.planner_cycle_observer import (
    H3PlannerCycleObserver,
    planner_two_cycle_enabled,
)


ROOT = Path(__file__).resolve().parents[1]
LOOP = ROOT / "src/distillschedule/backends/training/minimax_h3/trainer_loop.py"


class FakeCuda:
    def __init__(self):
        self.now = 0
        self.complete = False
        self.events = []
        self.synchronizations = 0
        self.nvtx = SimpleNamespace(range_push=lambda _: None, range_pop=lambda: None)

    def Event(self, **kwargs):
        assert kwargs.get("enable_timing") is True
        assert kwargs.get("blocking", False) is False
        cuda = self

        class Event:
            def record(self, stream):
                assert stream == "native-current-stream"
                self.time = cuda.now

            def query(self):
                return cuda.complete

            def elapsed_time(self, end):
                assert cuda.complete, "timing resolved before native final synchronization"
                return end.time - self.time

        event = Event()
        self.events.append(event)
        return event

    def current_stream(self, device=None):
        return "native-current-stream"

    def is_available(self):
        return True

    def reset_peak_memory_stats(self):
        pass

    def memory_allocated(self, device=None):
        return 17

    def synchronize(self, device=None):
        self.synchronizations += 1
        self.complete = True


def trainer_contract(**overrides):
    values = dict(
        max_train_iters=2, save_every_iters=0, auto_resume=False,
        activation_policy="none", activation_checkpoint_segment_size=1,
        activation_offload_enabled=False, gradient_checkpointing=True,
        adaln_grid_enabled=True, adaln_grid_size=1000,
        fa3_replay_cache_storage="cpu_staged", fa3_replay_cache_blocks=tuple(range(50)),
        fa3_replay_cache_max_d2h_inflight=2, fa3_replay_cache_trim_before_backward=True,
        fa3_nograd_num_splits=2, fa3_grad_num_splits=2,
        matched_compute_enabled=True, fake_update_ratio=5,
        gradient_accumulation_iters=1, reorder_critic_rollouts=True,
        shared_model=SimpleNamespace(device="cuda:0", config={"model": {"attention_backend": "_flash_3_hub"}}),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_opt_in_is_explicit_and_default_off():
    assert not planner_two_cycle_enabled({})
    assert not planner_two_cycle_enabled({"H3_PLANNER_TWO_CYCLE": "0"})
    assert planner_two_cycle_enabled({"H3_PLANNER_TWO_CYCLE": "1"})
    with pytest.raises(ValueError, match="requires H3_PLANNER_TWO_CYCLE=1"):
        planner_two_cycle_enabled({"H3_PLANNER_CALIBRATION": "1"})


@pytest.mark.parametrize("mutation", [
    {"max_train_iters": 1}, {"save_every_iters": 1}, {"auto_resume": True},
    {"activation_checkpoint_segment_size": 2},
    {"activation_policy": "checkpoint_boundary_cpu"},
    {"activation_offload_enabled": True}, {"gradient_checkpointing": False},
    {"adaln_grid_size": 999}, {"fa3_replay_cache_storage": "cuda"},
    {"fa3_replay_cache_blocks": (0,)}, {"fa3_replay_cache_max_d2h_inflight": 3},
    {"fa3_replay_cache_trim_before_backward": False},
    {"fa3_nograd_num_splits": 1}, {"fa3_grad_num_splits": 1},
    {"shared_model": SimpleNamespace(config={"model": {"attention_backend": "sdpa"}})},
    {"reorder_critic_rollouts": False}, {"gradient_accumulation_iters": 2},
])
def test_frozen_contract_rejects_changes_before_setup(mutation):
    cuda = FakeCuda()
    observer = H3PlannerCycleObserver(torch=SimpleNamespace(cuda=cuda), rank=0, source_head="a" * 40)
    with pytest.raises(ValueError):
        observer.validate_start(trainer_contract(**mutation), current_iter=0, resume_ckpt_path=None)
    assert not cuda.events


@pytest.mark.parametrize("current_iter,resume", [(1, None), (0, "/checkpoint")])
def test_resume_cannot_enter_two_cycle_timing(current_iter, resume):
    observer = H3PlannerCycleObserver(torch=SimpleNamespace(cuda=FakeCuda()), rank=0, source_head="a" * 40)
    with pytest.raises(ValueError, match="start_iter=0"):
        observer.validate_start(trainer_contract(), current_iter=current_iter, resume_ckpt_path=resume)


def _native_loop_class(torch, calls):
    """Execute the real mixin AST, without importing GPU/LightX2V dependencies."""
    import contextlib
    node = next(n for n in ast.parse(LOOP.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == "H3A100LoopMixin")
    namespace = dict(
        torch=torch, contextlib=contextlib,
        logger=SimpleNamespace(info=lambda *a, **kw: None),
        barrier=lambda: calls.append(("existing_barrier",)),
        is_main_process=lambda: False, get_rank=lambda: 0, get_world_size=lambda: 1,
        reduce_mean=lambda scalar: scalar.item(),
        sync_sequence_parallel_gradients=lambda _: None,
        set_parallel_gradient_sync=lambda *a: None,
        STUDENT_ADAPTER="student", FAKE_ADAPTER="fake",
        STUDENT_ROLE="student", FAKE_ROLE="fake",
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(LOOP), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace["H3A100LoopMixin"]


def _native_fixture(*, observed=True, calibration=False, bad_loss=False):
    from distillschedule.backends.training.minimax_h3.matched_contract import MatchedCycleCensus
    calls, probes = [], []
    cuda = FakeCuda()
    rng = random.Random(20260817)  # Initial fixture state only; never reset per cycle.

    class Scalar:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def __truediv__(self, _):
            return self

        def backward(self):
            cuda.now += 1
            calls.append(("backward",))

        def item(self):
            assert not observed or trainer.planner_cycle_observer.active is None
            cuda.now += 100  # Deliberately expensive, but outside CUDA events.
            return self.value

    torch = SimpleNamespace(cuda=cuda, nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda *a: None)),
        stack=lambda values: SimpleNamespace(mean=lambda: Scalar(sum(v.value for v in values) / len(values))))
    Base = _native_loop_class(torch, calls)

    class FakeTrainer(Base):
        pass

    trainer = FakeTrainer()
    trainer.__dict__.update(vars(trainer_contract()))
    trainer.train_log_every_iters = 1
    trainer.output_train_dir = "unused"
    trainer.trainable_params = ["student"]
    trainer.fake_trainable_params = ["fake"]
    trainer.max_grad_norm = 1.0
    trainer._resolve_resume_a100 = lambda: (None, 0)
    trainer._activation_offload_scope = lambda _: nullcontext()
    trainer._maybe_trim_fa3_replay_cache_before_backward = lambda: None
    trainer._log_residency = lambda _: None
    trainer._log_cuda_memory = lambda _: None
    trainer.log_metrics = lambda *a, **kw: None
    trainer.save_checkpoint_a100 = lambda _: pytest.fail("checkpoint policy changed")
    trainer._encode_conditions = lambda sample: sample["meta"]["id"]
    trainer._latent_shape = lambda sample: (1,)
    trainer._iter_train_samples = lambda: iter({"meta": {"id": i}} for i in range(12))
    trainer.matched_cycle_census = MatchedCycleCensus(expected_world_size=1)
    trainer.fa3_replay_cache_registration = SimpleNamespace(enabled=True, storage="cpu_staged",
        block_indices=tuple(range(50)), max_d2h_inflight=2, trim_before_backward=True)
    trainer.grid_replay_registration = SimpleNamespace(stats=dict(
        checkpoint_wrap_count=0, scoped_execution_count=0, cache_hit_count=0,
        cache_miss_count=0, missing_key_count=0))

    def probe(name):
        assert not observed or trainer.planner_cycle_observer.active is None
        probes.append(name)
        cuda.now += 1000

    def setup(**kwargs):
        probe("setup")
        assert kwargs == {"resume_ckpt_path": None}
        calls.append(("setup",))

    trainer.setup = setup
    for name in ("boundary_offload", "fused_pointwise", "fused_rotary", "fused_qk", "fused_swiglu",
                 "fa3_nograd_split", "fa3_replay_cache", "lora_scale1"):
        setattr(trainer, f"_begin_{name}_cycle", lambda name=name: probe(f"begin/{name}"))
        setattr(trainer, f"_validate_{name}_cycle", lambda i, name=name: probe(f"validate/{i}/{name}"))

    def validate_matched(i):
        probe(f"validate/{i}/matched")
        assert trainer.matched_cycle_census.validate_local() == []

    trainer._validate_matched_cycle = validate_matched

    def runtime_stats():
        probe("runtime_stats")
        return dict(role_transitions=0, role_noops=0, mode_transitions=0, mode_noops=0,
                    physical_role="fake", physical_training_mode=True)

    trainer.shared_model = SimpleNamespace(device="cuda:0", config={"model": {"attention_backend": "_flash_3_hub"}}, runtime_state_stats=runtime_stats,
        role_scope=lambda _: nullcontext(), adaln_scope=lambda *a, **kw: nullcontext(),
        set_transformer_training=lambda _: None, drop_adaln_key=lambda _: None, log_adaln_stats=lambda: None)

    def grid_grad():
        for key, count in (("checkpoint_wrap_count", 50), ("scoped_execution_count", 100), ("cache_hit_count", 100)):
            trainer.grid_replay_registration.stats[key] += count

    def student_loss(shape, conditions, *, current_iter):
        calls.append(("student_loss", shape, conditions, current_iter, rng.random()))
        cuda.now += 10
        for i in range(4):
            trainer.matched_cycle_census.note_forward("student", grad_enabled=i == 3)
        trainer.matched_cycle_census.note_forward("fake", grad_enabled=False)
        trainer.matched_cycle_census.note_forward("teacher", grad_enabled=False)
        grid_grad()
        scalar = Scalar(float("nan") if bad_loss else 0.25)
        return dict(loss=scalar, dmd=scalar, backward_key="student-key")

    def prepare(shape, conditions):
        calls.append(("prepare", shape, conditions, rng.random()))
        cuda.now += 2
        for _ in range(4):
            trainer.matched_cycle_census.note_forward("student", grad_enabled=False)
        return SimpleNamespace(cache_key=conditions)

    def fake_loss(item):
        calls.append(("fake_loss", item.cache_key, rng.random()))
        cuda.now += 3
        trainer.matched_cycle_census.note_forward("fake", grad_enabled=True)
        grid_grad()
        return Scalar(0.5)

    trainer.forward_student_loss = student_loss
    trainer.prepare_fake_update = prepare
    trainer.fake_loss = fake_loss
    actual_updates = {"student": 0, "fake": 0}

    def optimizer(role):
        def step():
            calls.append(("optimizer", role))
            actual_updates[role] += 1
            cuda.now += 1
        return SimpleNamespace(step=step, zero_grad=lambda **kw: None)

    trainer.optimizer = optimizer("student")
    trainer.fake_optimizer = optimizer("fake")
    trainer.lr_scheduler = SimpleNamespace(step=lambda: calls.append(("scheduler", "student")), get_last_lr=lambda: [0.1])
    trainer.fake_lr_scheduler = SimpleNamespace(step=lambda: calls.append(("scheduler", "fake")))

    def memory_probe(*args):
        probe("memory")
        return {"fixture": "CPU_METADATA_ONLY", "time": cuda.now}

    def state_probe(*args, **kwargs):
        assert cuda.complete
        probe("state")
        return {"status": "CPU_FIXTURE_ONLY", "numerical_cross_reference": "NOT_RUN"}

    observer = None
    if observed:
        observer = H3PlannerCycleObserver(torch=torch, rank=0, source_head="a" * 40,
            calibration=calibration, memory_probe=memory_probe, state_probe=state_probe)
        trainer.planner_cycle_observer = observer
    return SimpleNamespace(trainer=trainer, observer=observer, torch=torch, calls=calls,
                           probes=probes, rng=rng, actual_updates=actual_updates)


@pytest.mark.parametrize("calibration", [False, True])
def test_real_native_two_cycle_hooks_preserve_compute_rng_and_exclude_probes(calibration):
    baseline = _native_fixture(observed=False)
    baseline.trainer.train()
    assert baseline.torch.cuda.events == []
    fixture = _native_fixture(calibration=calibration)
    fixture.trainer.train()
    assert fixture.calls == baseline.calls
    assert fixture.rng.getstate() == baseline.rng.getstate()
    assert fixture.actual_updates == {"student": 2, "fake": 10}
    assert len([call for call in fixture.calls if call[0] == "setup"]) == 1
    assert fixture.torch.cuda.synchronizations == 0
    with pytest.raises(RuntimeError, match="final synchronization"):
        fixture.observer.finish_after_sync(fixture.trainer)
    fixture.torch.cuda.synchronize()  # The wrapper's existing final fence.
    receipt = fixture.observer.finish_after_sync(fixture.trainer)
    assert [row["gpu_completion_elapsed_s"] for row in receipt["cycles"]] == pytest.approx([0.047, 0.047])
    assert receipt["final_versions"] == {"generator": 2, "critic": 10, "ema": 0}
    assert receipt["numerical_cross_reference"] == "NOT_RUN"
    assert receipt["formal_timing_eligible"] is False
    assert receipt["extra_explicit_synchronization_calls"] == receipt["extra_collectives"] == 0
    assert [row["census"]["samples"][0]["identity"] for row in receipt["cycles"]] == ['{"id":"0"}', '{"id":"6"}']
    assert all(row["updates"] == {"student": 1, "fake": 5, "ema": 0} for row in receipt["cycles"])
    assert all(row["grid_replay_delta"]["checkpoint_wrap_count"] == 300 for row in receipt["cycles"])
    assert len([p for p in fixture.probes if p.startswith("validate/")]) == 18
    if calibration:
        stages = receipt["stage_calibration"]["rows"]
        expected = ["h3/student_step", "h3/critic_prepare_5xG"] + ["h3/critic_update_F"] * 5
        assert [r["label"] for r in stages] == expected * 2
        assert [r["cycle"] for r in stages] == [1] * 7 + [2] * 7
    else:
        assert receipt["stage_calibration"] is None
        assert len(fixture.torch.cuda.events) == 4
    fixture.observer.close(fixture.trainer)
    with pytest.raises(RuntimeError, match="already resolved"):
        fixture.observer.finish_after_sync(fixture.trainer)


def test_nonfinite_loss_rejects_timing_after_native_validators():
    fixture = _native_fixture(bad_loss=True)
    with pytest.raises(RuntimeError, match="non-finite H3 native cycle loss"):
        fixture.trainer.train()
    assert len([p for p in fixture.probes if p.startswith("validate/")]) == 9
    assert fixture.actual_updates == {"student": 1, "fake": 5}
    assert fixture.observer.active is None
    assert fixture.torch.cuda.synchronizations == 0


def test_final_state_failure_never_yields_a_pass_receipt():
    fixture = _native_fixture()
    fixture.trainer.train()
    fixture.torch.cuda.synchronize()
    def bad_state(*args, **kwargs):
        raise RuntimeError("non-finite H3 state samples")
    fixture.observer.state_probe = bad_state
    with pytest.raises(RuntimeError, match="non-finite H3 state"):
        fixture.observer.finish_after_sync(fixture.trainer)
    assert not fixture.observer.finished


def test_successful_step_calls_not_inferred_from_config():
    fixture = _native_fixture()
    fixture.observer.note_update = lambda _: None
    with pytest.raises(RuntimeError, match="actual optimizer update count mismatch"):
        fixture.trainer.train()
    assert fixture.actual_updates == {"student": 1, "fake": 5}


def test_wrapper_uses_one_final_fence_and_writes_only_after_state_gate(monkeypatch, tmp_path):
    import json
    from distillschedule.backends.training.minimax_h3 import planner_cycle_observer as module
    from distillschedule.backends.training.minimax_h3.integrated_entrypoint import _measure_planner_two_cycle
    fixture = _native_fixture(calibration=True)
    del fixture.trainer.planner_cycle_observer
    original_nvtx = fixture.trainer._nvtx
    monkeypatch.setattr(module, "H3PlannerCycleObserver", lambda **kw: fixture.observer)
    monkeypatch.setenv("EXPECTED_INTEGRATION_HEAD", "a" * 40)
    monkeypatch.setenv("DISTILLSCHEDULE_RUN_ROOT", str(tmp_path))
    _measure_planner_two_cycle(fixture.trainer, type(fixture.trainer).train, torch=fixture.torch, rank=0)
    row = json.loads((tmp_path / "execution/ranks/rank_000.json").read_text())
    assert row["cycles"][1]["gpu_completion_elapsed_s"] == pytest.approx(0.047)
    assert row["state_fingerprints"]["status"] == "CPU_FIXTURE_ONLY"
    assert fixture.torch.cuda.synchronizations == 1
    assert not hasattr(fixture.trainer, "planner_cycle_observer")
    assert fixture.trainer._nvtx == original_nvtx


def test_wrapper_failure_keeps_native_exception_no_json_and_restores_hooks(monkeypatch, tmp_path):
    from distillschedule.backends.training.minimax_h3 import planner_cycle_observer as module
    from distillschedule.backends.training.minimax_h3.integrated_entrypoint import _measure_planner_two_cycle
    fixture = _native_fixture(calibration=True)
    del fixture.trainer.planner_cycle_observer
    original_nvtx = fixture.trainer._nvtx
    monkeypatch.setattr(module, "H3PlannerCycleObserver", lambda **kw: fixture.observer)
    monkeypatch.setenv("EXPECTED_INTEGRATION_HEAD", "a" * 40)
    monkeypatch.setenv("DISTILLSCHEDULE_RUN_ROOT", str(tmp_path))
    def fail_state(*args, **kwargs):
        raise RuntimeError("non-finite H3 state samples")
    fixture.observer.state_probe = fail_state
    with pytest.raises(RuntimeError, match="non-finite H3 state"):
        _measure_planner_two_cycle(fixture.trainer, type(fixture.trainer).train, torch=fixture.torch, rank=0)
    assert not list(tmp_path.rglob("*.json"))
    assert fixture.torch.cuda.synchronizations == 1
    assert fixture.trainer._nvtx == original_nvtx
    assert not hasattr(fixture.trainer, "planner_cycle_observer")


def test_host_fenced_wrapper_has_only_four_fences_and_no_probes_in_host_timer(monkeypatch, tmp_path):
    import hashlib
    import json
    from distillschedule.backends.training.minimax_h3 import planner_cycle_observer as module
    from distillschedule.backends.training.minimax_h3.integrated_entrypoint import _measure_planner_two_cycle
    baseline = _native_fixture(observed=False)
    baseline.trainer.train()
    fixture = _native_fixture()
    fixture.observer.host_fenced_clock = True
    fixture.trainer.config = {"training": {"seed": 20260817}}
    fixture.trainer.matched_expected_world_size = 16
    fixture.torch.distributed = SimpleNamespace(get_world_size=lambda: 16)

    def rng_probe():
        assert fixture.observer.active is None
        fixture.torch.cuda.now += 1000
        digest = hashlib.sha256(repr(fixture.rng.getstate()).encode()).hexdigest()
        return {"torch_cpu": digest, "torch_cuda": digest}

    fixture.observer._rng_fingerprints = rng_probe
    del fixture.trainer.planner_cycle_observer
    monkeypatch.setattr(module, "H3PlannerCycleObserver", lambda **kw: fixture.observer)
    monkeypatch.setattr(module.time, "perf_counter_ns", lambda: fixture.torch.cuda.now * 1_000_000)
    monkeypatch.setenv("EXPECTED_INTEGRATION_HEAD", "a" * 40)
    monkeypatch.setenv("DISTILLSCHEDULE_RUN_ROOT", str(tmp_path))
    monkeypatch.setenv("H3_BENCHMARK_SEED", "20260817")
    _measure_planner_two_cycle(fixture.trainer, type(fixture.trainer).train, torch=fixture.torch, rank=0)
    row = json.loads((tmp_path / "execution/ranks/rank_000.json").read_text())
    assert fixture.torch.cuda.synchronizations == row["native_boundary_fences"] == 4
    assert row["extra_explicit_synchronization_calls"] == 4
    assert [c["native_wall_seconds"] for c in row["cycles"]] == pytest.approx([.047, .047])
    assert fixture.calls == baseline.calls
    assert fixture.rng.getstate() == baseline.rng.getstate()
    assert row["cycles"][0]["rng_after"] == row["cycles"][1]["rng_before"]
    assert row["final_versions"] == dict(generator=2, critic=10, ema=0)
    assert row["formal_timing_eligible"] is False


class _StateTensor:
    def __init__(self, values, *, contiguous=True, reads=None):
        self.values = list(values)
        self.shape, self.dtype, self.device = (len(self.values),), "torch.float32", "cpu"
        self.contiguous, self.reads = contiguous, reads

    def detach(self):
        return self

    def is_contiguous(self):
        return self.contiguous

    def numel(self):
        return len(self.values)

    def view(self, shape):
        assert shape == -1
        return self

    def index_select(self, dimension, indices):
        assert dimension == 0
        return _StateTensor([self.values[i] for i in indices.values], reads=self.reads)

    def float(self):
        return self

    def cpu(self):
        if self.reads is not None:
            self.reads.append(len(self.values))
        return self

    def tolist(self):
        return self.values


def _state_fixture(*, bad=False, contiguous=True):
    reads = []
    params = [_StateTensor([float("nan") if bad else i] * 1000, contiguous=contiguous, reads=reads) for i in range(10)]
    states = {p: {"step": 2, "exp_avg": _StateTensor(range(1000), reads=reads),
                  "exp_avg_sq": _StateTensor(range(1000), reads=reads)} for p in params}
    trainer = SimpleNamespace(trainable_params=params[:5], fake_trainable_params=params[5:],
        optimizer=SimpleNamespace(state=states), fake_optimizer=SimpleNamespace(state=states),
        shared_model=SimpleNamespace(denoiser_module=lambda: SimpleNamespace(
            named_parameters=lambda: [(f"module.layer_{i}.lora", p) for i, p in enumerate(params)])))
    torch = SimpleNamespace(is_tensor=lambda value: isinstance(value, _StateTensor), int64="int64",
                            tensor=lambda values, **kw: _StateTensor(values))
    return trainer, torch, reads


def test_final_state_samples_are_bounded_deterministic_and_do_not_consume_rng():
    from distillschedule.backends.training.minimax_h3.planner_cycle_observer import capture_shard_fingerprints
    trainer, torch, reads = _state_fixture()
    before = random.getstate()
    first = capture_shard_fingerprints(trainer, torch=torch, rank=0, max_tensors=4, samples_per_tensor=3)
    second = capture_shard_fingerprints(trainer, torch=torch, rank=0, max_tensors=4, samples_per_tensor=3)
    assert first == second
    assert random.getstate() == before
    assert first["status"] == "CAPTURED_FINITE_SAMPLES"
    assert first["numerical_cross_reference"] == first["full_state_finite_check"] == "NOT_RUN"
    assert len(first["components"]) == 4
    assert all(c["sampled_tensor_count"] <= 4 for c in first["components"].values())
    assert reads and max(reads) <= 3


@pytest.mark.parametrize("kwargs,reason", [({"bad": True}, "non-finite"), ({"contiguous": False}, "noncontiguous")])
def test_bad_or_copy_requiring_state_is_not_reported_as_finite(kwargs, reason):
    from distillschedule.backends.training.minimax_h3.planner_cycle_observer import capture_shard_fingerprints
    trainer, torch, _ = _state_fixture(**kwargs)
    with pytest.raises(RuntimeError, match=reason):
        capture_shard_fingerprints(trainer, torch=torch, rank=0)
