"""SYNTHETIC_TEST_ONLY: CPU native-boundary/real tensor integration, NOT a W5 tuple.

The imported entrypoint, FA3 module/installer/compact class and TupleCapture are
real. Only the native loop/Grid setup and Prepared dataclass are source-compiled
to avoid importing the unrelated LightX2V/Diffusers model dependencies.
PreparedFakeUpdate is the actual source dataclass. Only CUDA/distributed/provider
boundaries and geometry/storage location are explicitly synthetic. No GPU,
model, native FA3, numerical acceptance, or performance claim follows.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as cpu_torch

from distillschedule.backends.training.minimax_h3 import integrated_entrypoint as entry
from distillschedule.backends.training.minimax_h3 import attention_tuple_capture as capture_module
from distillschedule.backends.training.minimax_h3 import planner_cycle_observer as observer_module
from distillschedule.search.h3_implementation import H3Implementation
from test_refactor_h3_planner_cycle_observer import _native_fixture

SCOPE = "SYNTHETIC_TEST_ONLY"
ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "src/distillschedule/backends/training/minimax_h3"
HEAD = "a" * 40
Q_SHAPE = (1, 3, 2, 2)
LSE_SHAPE = (1, 2, 3)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return {"path": str(path), "sha256": sha(path)}


def source_class(name, file, namespace):
    node = next(n for n in ast.parse(file.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(file), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def native_boundary(monkeypatch, tmp_path):
    assert not cpu_torch.cuda.is_initialized()
    rng = cpu_torch.get_rng_state().clone()
    fixture = _native_fixture()
    trainer, observer, torch = fixture.trainer, fixture.observer, fixture.torch
    observer.host_fenced_clock = True
    del trainer.planner_cycle_observer
    monkeypatch.setattr(observer_module, "H3PlannerCycleObserver", lambda **kwargs: observer)
    impl = H3Implementation()
    config = impl.lower_profile()["runtime"]["model_config"]
    trainer.config = config
    trainer.matched_expected_world_size = 16
    trainer.shared_model._physical_role = "fake"
    trainer.shared_model._physical_training_mode = True
    trainer.shared_model.device = "cuda:0"
    trainer._latent_shape = lambda sample: {"frames": 124, "height": 768, "width": 1344}
    torch.distributed = SimpleNamespace(is_initialized=lambda: True, get_rank=lambda: 0, get_world_size=lambda: 16)
    torch.__version__ = "2.10.0+cu128"  # Synthetic advertised native boundary, actual runtime is CPU 2.14.
    torch.version = SimpleNamespace(cuda="12.8")
    torch.get_rng_state = cpu_torch.get_rng_state
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_rng_state = lambda device: cpu_torch.get_rng_state()
    for name in ("Tensor", "strided", "no_grad", "empty", "save"):
        setattr(torch, name, getattr(cpu_torch, name))
    for name, value in impl.lower_profile()["runtime"]["environment"].items():
        monkeypatch.setenv(name, value)
    for name, value in {"EXPECTED_INTEGRATION_HEAD": HEAD, "DISTILLSCHEDULE_RUN_KIND": "correctness",
                        "DISTILLSCHEDULE_RUN_ROOT": str(tmp_path), "DISTILLSCHEDULE_RUN_ID": "synthetic-capture",
                        "H3_BENCHMARK_SEED": "20260817"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(capture_module, "NATIVE_Q_SHAPE", Q_SHAPE, raising=False)
    monkeypatch.setattr(capture_module, "NATIVE_LSE_SHAPE", LSE_SHAPE, raising=False)
    monkeypatch.setattr(capture_module, "REMOTE_PERSISTENT_ROOT", tmp_path, raising=False)
    monkeypatch.setattr(capture_module, "RAW_BACKWARD_SCHEMA", "SYNTHETIC_TEST_ONLY_RAW22", raising=False)
    original_metadata = capture_module.TorchSnapshotIO.metadata
    io_events = []

    def metadata(self, tensor):
        row = original_metadata(self, tensor)  # Real dtype/shape/stride/storage/pointer metadata.
        assert row["device"] == "cpu" and not cpu_torch.cuda.is_initialized()
        return dict(row, device="cuda:0")  # Explicitly synthetic device boundary only.

    def fence(self, device):
        assert device == "cuda:0" and not cpu_torch.cuda.is_initialized()
        io_events.append("SYNTHETIC_TEST_ONLY_CPU_FENCE")

    monkeypatch.setattr(capture_module.TorchSnapshotIO, "metadata", metadata)
    monkeypatch.setattr(capture_module.TorchSnapshotIO, "fence", fence)
    Prepared = source_class("PreparedFakeUpdate", NATIVE / "trainer_runtime.py", {"dataclasses": dataclasses,
        "Any": object, "torch": cpu_torch, "__name__": __name__})
    old_prepare = trainer.prepare_fake_update
    prepared = []

    def prepare(shape, conditions):
        old_prepare(shape, conditions)  # Preserve original fixture census/RNG/call counters.
        item = Prepared(latent_shape=shape, condition={}, generated=(), noises=(), renoised=(), sigmas=(),
                        cache_key=("score", int(conditions) + 1))
        prepared.append(item)
        return item

    trainer.prepare_fake_update = prepare
    raw_calls, forward_calls, runtime = [], [], SimpleNamespace(tuple_capture=None,
        stats=SimpleNamespace(compact_backward_calls=0), staged_pool=None)
    # Logging and memory staging are native CPU test boundaries, not dependencies
    # to install. The actual FA3 installer below is still executed in full.
    monkeypatch.setitem(sys.modules, "loguru", SimpleNamespace(logger=SimpleNamespace(info=lambda *a, **k: None)))
    native_fa3 = importlib.import_module("distillschedule.backends.training.minimax_h3.fa3_replay_cache")
    monkeypatch.setattr(native_fa3, "_RUNTIME", None)
    prefetches = []
    monkeypatch.setattr(native_fa3, "_PageableStagingPool", lambda *a: SimpleNamespace(
        prefetch_after_backward=lambda block: prefetches.append(block)))

    def forward(q, k, v, scale):
        forward_calls.append((q, k, v, scale))
        return q + k + v, cpu_torch.zeros(LSE_SHAPE, dtype=cpu_torch.float32)

    raw_result = object()

    def raw(*arguments):
        raw_calls.append(arguments)
        assert len(arguments) == 22
        for output in arguments[12:15]:
            output.copy_(arguments[0])
        return raw_result  # Actual source caller ignores softmax_d, never gradients.

    runtime.raw_backward = raw
    monkeypatch.setattr(native_fa3, "_fa3_forward_compact", forward)
    monkeypatch.setattr(native_fa3, "_CACHE_BLOCK_INDEX", SimpleNamespace(get=lambda: 10))
    # The installer accesses the op handle but never invokes it in this fixture.
    forward._opoverload = object()
    namespace = native_fa3.__dict__
    Compact = native_fa3._CompactFA3Autograd
    old_loss = trainer.fake_loss
    applied_items = []

    def fake_loss(item):
        applied_items.append(item)
        assert any(item is actual for actual in prepared)
        scalar = old_loss(item)
        values = [cpu_torch.arange(12, dtype=cpu_torch.float32).to(cpu_torch.bfloat16).reshape(Q_SHAPE)
                  .requires_grad_() for _ in range(3)]
        out = Compact.apply(*values, 128 ** -0.5)
        weight = cpu_torch.arange(12, dtype=cpu_torch.float32).to(cpu_torch.bfloat16).reshape(Q_SHAPE)
        loss = (out * weight).sum()
        old_backward = scalar.backward
        def backward():
            loss.backward()
            old_backward()
        scalar.backward = backward
        return scalar

    trainer.fake_loss = fake_loss
    old_setup = trainer.setup
    captures = []
    native_setup_calls = []
    denoiser = cpu_torch.nn.Module()
    denoiser.transformer_blocks = [object() for _ in range(50)]
    denoiser._gradient_checkpointing_func = lambda function, *a, **k: function(*a)
    trainer.shared_model.denoiser_module = lambda: denoiser
    controller = SimpleNamespace(payload=dict(grid_size=1000, renoise_sigma_min=0.02, renoise_sigma_max=0.98,
                                              video_shift=1.0, audio_shift=1.0))
    trainer.shared_model.adaln_cache = lambda: controller
    trainer.adaln_grid_manifest = "SYNTHETIC_TEST_ONLY"
    trainer.adaln_grid_pin_memory = False
    trainer.dmd_config = {}
    trainer.video_shift = trainer.audio_shift = 1.0
    trainer.fa3_nograd_split_registration = SimpleNamespace(enabled=True, grad_num_splits=2)
    grid_registration = trainer.grid_replay_registration
    def install(transformer, **kwargs):
        native_setup_calls.append(kwargs["tuple_capture"])
        result = native_fa3.install_fa3_replay_cache(transformer, **kwargs,
            kernel_config=SimpleNamespace(kernel_fn=lambda *a, **k: None), raw_forward=forward,
            raw_backward=runtime.raw_backward)
        return result
    setup_node = next(n for n in ast.walk(ast.parse((NATIVE / "grid_adaln.py").read_text()))
                      if isinstance(n, ast.FunctionDef) and n.name == "trainer_setup")
    setup_ns = {"Any": object, "original_trainer_setup": lambda self, **kwargs: old_setup(**kwargs),
        "install_fa3_replay_cache": install,
        "install_grid_checkpoint_replay_scope": lambda *a, **k: grid_registration,
        "logger": SimpleNamespace(info=lambda *a, **k: None)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[setup_node], type_ignores=[])),
                 str(NATIVE / "grid_adaln.py"), "exec"), setup_ns)
    def setup(**kwargs):
        setup_ns["trainer_setup"](trainer, **kwargs)
        if native_fa3._RUNTIME.tuple_capture is not None:
            captures.append(native_fa3._RUNTIME.tuple_capture)
    trainer.setup = setup
    references = {}
    references["native_config"] = write(tmp_path / "native-config.json", config)
    references["input_manifest"] = write(tmp_path / "input-manifest.json", {"scope": SCOPE, "sample_ids": list(range(12))})
    references["owner_receipt"] = write(tmp_path / "owner.json", {"scope": SCOPE, "status": "SYNTHETIC_BOUNDARY_ONLY"})
    references["capacity_receipt"] = write(tmp_path / "capacity.json", {"scope": SCOPE, "status": "SYNTHETIC_BOUNDARY_ONLY"})
    references["source_layout"] = {"path": str(ROOT / "docs/workloads/source_layout.lock.json"),
                                    "sha256": sha(ROOT / "docs/workloads/source_layout.lock.json")}
    provider = write(tmp_path / "provider.json", {"scope": SCOPE, "actual_FA3": "NOT_RUN"})
    source_files = {str(NATIVE / name): sha(NATIVE / name) for name in
                    ("attention_tuple_capture.py", "integrated_entrypoint.py", "trainer_loop.py", "grid_adaln.py", "fa3_replay_cache.py")}
    packet = {"schema_version": "distillschedule.h3_native_attention_capture_opt_in.v1",
        "selector": "h3.w5.c1.fake0.block10.actualtuple.v1", "run_id": "synthetic-capture",
        "candidate_id": impl.id, "profile": False, "formal_timing_eligible": False, "output_relative": "diagnostics/attention_tuple_rank000",
        "target": {"rank": 0, "world_size": 16, "cycle": 1, "role": "fake", "fake_index": 0,
                   "micro_index": 0, "block_index": 10, "versions": {"student": 1, "fake": 0, "ema": 0}},
        "geometry": {"q_shape": list(Q_SHAPE), "lse_shape": list(LSE_SHAPE),
                     "q_dtype": "torch.bfloat16", "lse_dtype": "torch.float32", "layout": "BSHD"},
        "expected_input": {"sample_identity": '{"id":"1"}', "latent_shape": {"frames": 124, "height": 768, "width": 1344},
                           "cache_key": ["score", 2]},
        "references": references, "source_files": source_files,
        "budget": {"max_payload_bytes": 4339077120, "max_replay_span_bytes": 13017231360,
                   "max_tensor_bytes": 541327360, "max_file_bytes": 6442450944, "max_host_working_bytes": 2147483648},
        "provenance": {"source_head": HEAD, "run_kind": "correctness", "provider_schema": "SYNTHETIC_TEST_ONLY_RAW22",
            "provider_files": {provider["path"]: provider["sha256"]},
            "native_config_sha256": references["native_config"]["sha256"],
            "input_manifest_sha256": references["input_manifest"]["sha256"],
            "owner_receipt_sha256": references["owner_receipt"]["sha256"],
            "capacity_receipt_sha256": references["capacity_receipt"]["sha256"],
            "source_layout_sha256": references["source_layout"]["sha256"],
            "runtime_identity": {"python_executable": __import__("sys").executable,
                "python_prefix": __import__("sys").prefix, "torch_version": torch.__version__, "torch_cuda_version": "12.8"}}}
    monkeypatch.setenv("H3_CONFIG", references["native_config"]["path"])
    packet_path = tmp_path / "capture-opt-in.json"
    def seal():
        write(packet_path, packet)
        monkeypatch.setenv("H3_ATTENTION_TUPLE_CAPTURE_CONFIG", str(packet_path))
        monkeypatch.setenv("H3_ATTENTION_TUPLE_CAPTURE_CONFIG_SHA256", sha(packet_path))
    seal()
    result = SimpleNamespace(**vars(fixture), packet=packet, seal=seal, packet_path=packet_path,
        raw_calls=raw_calls, forward_calls=forward_calls, runtime=runtime, io_events=io_events, prepared=prepared,
        applied_items=applied_items, captures=captures, run_root=tmp_path, raw_result=raw_result, namespace=namespace,
        native_fa3=native_fa3, native_setup_calls=native_setup_calls, prefetches=prefetches)
    yield result
    assert not cpu_torch.cuda.is_initialized()
    assert cpu_torch.equal(rng, cpu_torch.get_rng_state())


def run(f):
    entry._measure_planner_two_cycle(f.trainer, type(f.trainer).train, torch=f.torch, rank=f.observer.rank)


def receipt(f):
    return json.loads((f.run_root / "execution/ranks/rank_000.json").read_text())


def test_entrypoint_attaches_exact_prepared_object_and_captures_once_after_original_raw(native_boundary):
    f = native_boundary
    run(f)
    assert len(f.captures) == 1, "Missing actual entrypoint attachment (R253 producer never activated)"
    manifest = capture_module.load_capture(f.run_root / f.packet["output_relative"], max_file_bytes=1 << 20)
    assert manifest["binding"]["sample"]["identity"] == '{"id":"1"}'
    assert manifest["binding"]["cache_key"] == ["score", 2]
    assert manifest["binding"]["versions"] == {"student": 1, "fake": 0, "ema": 0}
    assert f.applied_items[0] is f.prepared[0]
    assert len(f.raw_calls) == len(f.forward_calls) == 10  # Two unchanged cycles x Fake5.
    assert len(f.io_events) == 1 and f.native_fa3._RUNTIME.stats.compact_backward_calls == 10
    assert f.native_setup_calls == f.captures and f.native_setup_calls[0] is f.captures[0]
    assert f.prefetches == [10] * 10 and f.native_fa3._RUNTIME.tuple_capture is None
    assert f.actual_updates == {"student": 2, "fake": 10}
    row = receipt(f)
    assert row["attention_tuple_capture"]["status"] == "CAPTURED_AFTER_NATIVE_SEMANTIC_GATES"
    assert row["attention_tuple_capture"]["manifest_sha256"] == sha(f.run_root / f.packet["output_relative"] / "manifest.json")
    assert row["performance_timing_eligible"] is False and row["formal_timing_eligible"] is False
    assert row["timing_exclusion_reason"] == "ATTENTION_TUPLE_CAPTURE_SYNCHRONIZATION_AND_IO"
    assert row["numerical_cross_reference"] == "NOT_RUN_W5_FAIL_RETAINED"
    assert all(c["performance_timing_eligible"] is False for c in row["cycles"])
    assert not hasattr(f.trainer, "attention_tuple_capture") and not hasattr(f.trainer, "planner_cycle_observer")
    assert not f.captures[0].prepared and f.captures[0].active is None and f.captures[0].trainer is None


def test_default_off_does_not_call_capture_factory_or_import_helper(monkeypatch, native_boundary):
    f = native_boundary
    monkeypatch.delenv("H3_ATTENTION_TUPLE_CAPTURE_CONFIG")
    monkeypatch.delenv("H3_ATTENTION_TUPLE_CAPTURE_CONFIG_SHA256")
    monkeypatch.setattr(capture_module, "attach_native_capture", lambda *a, **k: pytest.fail("default path touched capture helper"), raising=False)
    run(f)
    assert not f.captures and f.io_events == [] and len(f.raw_calls) == 10
    assert "attention_tuple_capture" not in receipt(f)
    assert not (f.run_root / f.packet["output_relative"]).exists()


@pytest.mark.parametrize("mode", ["diagnostic-timing", "calibration", "profile", "native-timing", "timing"])
def test_capture_modes_rejected_before_native_setup(monkeypatch, native_boundary, mode):
    f = native_boundary
    monkeypatch.setenv("DISTILLSCHEDULE_RUN_KIND", mode)
    with pytest.raises(ValueError, match="correctness"):
        run(f)
    assert not f.prepared and not list(f.run_root.glob("execution/ranks/*.json"))
    assert not hasattr(f.trainer, "planner_cycle_observer")


@pytest.mark.parametrize("mutation", [
    ("target", "rank", 1), ("target", "world_size", 8), ("target", "cycle", 2),
    ("target", "role", "student"), ("target", "fake_index", 1), ("target", "micro_index", 1),
    ("target", "block_index", 9), ("target", "versions", {"student": 0, "fake": 0, "ema": 0}),
    ("geometry", "q_shape", [1, 1, 2, 2]), ("geometry", "q_dtype", "torch.float16"),
    ("geometry", "lse_dtype", "torch.bfloat16"), ("geometry", "layout", "BHSD"),
    ("provenance", "source_head", "b" * 40), ("provenance", "native_config_sha256", "b" * 64),
    ("provenance", "input_manifest_sha256", "b" * 64), ("provenance", "runtime_identity", {}),
    ("provenance", "provider_schema", ""), ("provenance", "provider_files", {}),
    (None, "candidate_id", "joint-h3v1-d1-n128"), (None, "selector", "guessed-ordinal"),
    (None, "run_id", "another-run"), (None, "source_files", {}), (None, "expected_input", {}),
])
def test_wrong_or_missing_binding_rejected_before_original_train(native_boundary, mutation):
    f = native_boundary
    section, field, value = mutation
    (f.packet if section is None else f.packet[section])[field] = value
    f.seal()
    with pytest.raises((ValueError, KeyError)):
        run(f)
    assert not f.prepared and not f.captures
    assert not list(f.run_root.glob("execution/ranks/*.json"))
    assert not hasattr(f.trainer, "planner_cycle_observer")


@pytest.mark.parametrize("reference", ["native_config", "input_manifest", "owner_receipt", "capacity_receipt"])
def test_changed_root_admission_evidence_is_not_trusted(native_boundary, reference):
    f = native_boundary
    Path(f.packet["references"][reference]["path"]).write_text("tampered\n")
    with pytest.raises(ValueError, match="checksum"):
        run(f)
    assert not f.prepared


def test_packet_checksum_and_actual_distributed_rank_fail_closed(monkeypatch, native_boundary):
    f = native_boundary
    monkeypatch.setenv("H3_ATTENTION_TUPLE_CAPTURE_CONFIG_SHA256", "b" * 64)
    with pytest.raises(ValueError, match="checksum"):
        run(f)
    f.seal()
    f.torch.distributed.get_rank = lambda: 3
    with pytest.raises(ValueError, match="rank"):
        run(f)
    assert not f.prepared


@pytest.mark.parametrize("field,value", [("sample_identity", '{"id":"other"}'), ("cache_key", ["score", 100]),
                                         ("latent_shape", {"frames": 1})])
def test_actual_prepared_input_metadata_drift_is_rejected(native_boundary, field, value):
    f = native_boundary
    f.packet["expected_input"][field] = value
    f.seal()
    with pytest.raises(ValueError, match="input"):
        run(f)
    assert not list(f.run_root.glob("execution/ranks/*.json"))
    assert not (f.run_root / f.packet["output_relative"] / "_SUCCESS").exists()


def test_complete_tuple_is_required_after_normal_native_gates(native_boundary):
    f = native_boundary
    f.namespace["_CACHE_BLOCK_INDEX"] = SimpleNamespace(get=lambda: 9)
    with pytest.raises(ValueError, match="did not capture"):
        run(f)
    assert f.actual_updates == {"student": 2, "fake": 10} and f.observer.finished
    assert not list(f.run_root.glob("execution/ranks/*.json"))


@pytest.mark.parametrize("failure", ["original_gate", "observer_gate", "raw"])
def test_original_failures_are_not_replaced_by_capture_success(native_boundary, failure):
    f = native_boundary
    error = RuntimeError("SYNTHETIC_TEST_ONLY_original_failure_" + failure)
    if failure == "original_gate":
        def fail(_):
            raise error
        f.trainer._validate_matched_cycle = fail
    elif failure == "observer_gate":
        def fail(*a, **k):
            raise error
        f.observer.state_probe = fail
    else:
        def fail(*a):
            raise error
        f.runtime.raw_backward = fail
    with pytest.raises(RuntimeError) as caught:
        run(f)
    assert caught.value is error
    assert not list(f.run_root.glob("execution/ranks/*.json"))
    assert not hasattr(f.trainer, "attention_tuple_capture") and not hasattr(f.trainer, "planner_cycle_observer")
    if failure != "raw":
        assert (f.run_root / f.packet["output_relative"] / "_SUCCESS").is_file()
        # A complete tuple alone is NOT a successful native/correctness run.
    else:
        assert not (f.run_root / f.packet["output_relative"] / "_SUCCESS").exists()


def test_nonselected_real_rank_is_explicit_not_claimed_captured(native_boundary):
    f = native_boundary
    f.observer.rank = 1
    f.torch.distributed.get_rank = lambda: 1
    run(f)
    row = json.loads((f.run_root / "execution/ranks/rank_001.json").read_text())
    assert row["attention_tuple_capture"]["status"] == "NOT_SELECTED_NO_CAPTURE_IO"
    assert "manifest_sha256" not in row["attention_tuple_capture"]
    assert not (f.run_root / f.packet["output_relative"]).exists() and not f.io_events


def test_main_refuses_capture_without_original_two_cycle_entry(monkeypatch, native_boundary):
    monkeypatch.setenv("H3_PLANNER_TWO_CYCLE", "0")
    with pytest.raises(ValueError, match="two-cycle"):
        entry.main()  # Must reject before importing model/upstream/CUDA entry modules.


@pytest.mark.parametrize("missing", ["H3_ATTENTION_TUPLE_CAPTURE_CONFIG", "H3_ATTENTION_TUPLE_CAPTURE_CONFIG_SHA256"])
def test_half_configured_opt_in_is_not_silently_disabled(monkeypatch, native_boundary, missing):
    f = native_boundary
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match="path and checksum"):
        run(f)
    assert not f.prepared


@pytest.mark.parametrize("bad_field,value", [("profile", True), ("formal_timing_eligible", True)])
def test_packet_cannot_authorize_profile_or_timing(native_boundary, bad_field, value):
    f = native_boundary
    f.packet[bad_field] = value
    f.seal()
    with pytest.raises(ValueError, match="correctness"):
        run(f)
    assert not f.prepared


@pytest.mark.parametrize("boundary", ["world", "uninitialized", "torch", "cuda_version", "device"])
def test_actual_native_boundary_identity_is_not_inferred_from_packet(native_boundary, boundary):
    f = native_boundary
    if boundary == "world":
        f.torch.distributed.get_world_size = lambda: 8
    elif boundary == "uninitialized":
        f.torch.distributed.is_initialized = lambda: False
    elif boundary == "torch":
        f.torch.__version__ = "2.14.0+cpu"
    elif boundary == "cuda_version":
        f.torch.version.cuda = "12.1"
    else:
        f.trainer.shared_model.device = "cuda:7"
    with pytest.raises(ValueError):
        run(f)
    assert not f.prepared


@pytest.mark.parametrize("drift", ["role", "version", "precision", "shape"])
def test_actual_prepared_forward_backward_context_drift_fails_closed(native_boundary, drift):
    f = native_boundary
    if drift == "role":
        f.trainer.shared_model._physical_role = "student"
    elif drift == "version":
        old_note = f.observer.note_update
        def note(role):
            old_note(role)
            if role == "student":
                f.observer.updates["fake"] = 1
        f.observer.note_update = note
    else:
        original = f.native_fa3._fa3_forward_compact
        def forward(*args):
            out, lse = original(*args)
            return (out.to(cpu_torch.float16), lse) if drift == "precision" else (out, lse[..., :2])
        forward._opoverload = original._opoverload
        f.native_fa3._fa3_forward_compact = forward
    with pytest.raises(ValueError):
        run(f)
    assert not (f.run_root / f.packet["output_relative"] / "_SUCCESS").exists()
    assert not list(f.run_root.glob("execution/ranks/*.json"))
    assert f.native_fa3._RUNTIME.tuple_capture is None


def test_partial_io_error_releases_original_prepared_and_runtime_references(monkeypatch, native_boundary):
    f = native_boundary
    original = capture_module.TorchSnapshotIO.save
    count = []
    error = OSError("SYNTHETIC_TEST_ONLY disk failure")
    def save(self, tensor, path):
        count.append(path.name)
        if len(count) == 2:
            raise error
        return original(self, tensor, path)
    monkeypatch.setattr(capture_module.TorchSnapshotIO, "save", save)
    with pytest.raises(OSError) as caught:
        run(f)
    assert caught.value is error and len(f.raw_calls) == 1
    assert (f.run_root / f.packet["output_relative"] / "dout.pt").is_file()
    assert not (f.run_root / f.packet["output_relative"] / "_SUCCESS").exists()
    assert not f.captures[0].prepared and f.captures[0].trainer is None
    assert f.native_fa3._RUNTIME.tuple_capture is None
    assert not list(f.run_root.glob("execution/ranks/*.json"))


def test_tuple_integrity_is_checked_after_original_gates_before_native_receipt(native_boundary):
    f = native_boundary
    def original(trainer):
        type(trainer).train(trainer)
        (f.run_root / f.packet["output_relative"] / "q.pt").write_bytes(b"SYNTHETIC_TEST_ONLY corrupt")
    with pytest.raises(ValueError, match="checksum|size"):
        entry._measure_planner_two_cycle(f.trainer, original, torch=f.torch, rank=0)
    assert f.observer.finished and f.actual_updates == {"student": 2, "fake": 10}
    assert not list(f.run_root.glob("execution/ranks/*.json"))


def test_fresh_process_default_path_does_not_import_capture_helper(tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT / "src"),
               DISTILLSCHEDULE_RUN_ROOT=str(tmp_path), EXPECTED_INTEGRATION_HEAD=HEAD)
    env.pop("H3_ATTENTION_TUPLE_CAPTURE_CONFIG", None)
    env.pop("H3_ATTENTION_TUPLE_CAPTURE_CONFIG_SHA256", None)
    code = '''
import sys
from types import SimpleNamespace
from distillschedule.backends.training.minimax_h3 import integrated_entrypoint as entry
from distillschedule.backends.training.minimax_h3 import planner_cycle_observer as module
name = "distillschedule.backends.training.minimax_h3.attention_tuple_capture"
assert name not in sys.modules
calls = []
class Observer:
    host_fenced_clock = False
    action_trace = None
    def __init__(self, **kwargs): pass
    def finish_after_sync(self, trainer): return {"cycles": [], "scope": "SYNTHETIC_TEST_ONLY"}
    def close(self, trainer): calls.append("original_close")
module.H3PlannerCycleObserver = Observer
trainer = SimpleNamespace()
entry._measure_planner_two_cycle(trainer, lambda t: calls.append("original_train"),
    torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: calls.append("original_fence"))), rank=0)
assert calls == ["original_train", "original_fence", "original_close"]
assert name not in sys.modules
assert not hasattr(trainer, "attention_tuple_capture")
print("SYNTHETIC_TEST_ONLY_DEFAULT_OFF_NO_HELPER_IMPORT_NO_EXTRA_FENCE")
'''
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "SYNTHETIC_TEST_ONLY_DEFAULT_OFF_NO_HELPER_IMPORT" in result.stdout


def test_actual_legacy_timing_cost_and_pair_consumers_refuse_capture_diagnostics(native_boundary):
    from copy import deepcopy
    from test_refactor_h3_search_native import _receipts
    from distillschedule.search.h3_native import summarize_native_two_cycle, compare_native_two_cycle
    f = native_boundary
    run(f)
    captured = receipt(f)
    assert captured["native_runtime_contract_status"] == "PASS_NATIVE_RUNTIME_CONTRACTS"
    assert captured["status"] == "PASS_NATIVE_RUNTIME_CONTRACTS_CAPTURE_DIAGNOSTIC_ONLY"
    assert captured["timing_scope"] == "CAPTURE_INSTRUMENTED_CORRECTNESS_ONLY_NOT_PERFORMANCE"
    rows = _receipts()
    assert summarize_native_two_cycle(rows, expected_head=HEAD, candidate_id=f.packet["candidate_id"])["status"] == "PASS"
    instrumented = deepcopy(rows)
    # A valid all16 synthetic native inventory, changed only by the real hook's
    # status/exclusion markers, must not enter legacy timing/cost/pair readers.
    for row in instrumented:
        for name in ("status", "native_runtime_contract_status", "attention_tuple_capture", "performance_timing_eligible",
                     "timing_exclusion_reason", "timing_scope"):
            row[name] = captured[name]
    with pytest.raises(ValueError, match="status mismatch"):
        summarize_native_two_cycle(instrumented, expected_head=HEAD, candidate_id=f.packet["candidate_id"])
    with pytest.raises(ValueError, match="status mismatch"):
        compare_native_two_cycle(rows, instrumented, expected_head=HEAD)


@pytest.mark.parametrize("metadata", [
    ("target", "cycle", True), ("target", "rank", False), ("target", "block_index", 10.0),
    ("expected_input", "sample_identity", '["not-native-metadata"]'),
    ("expected_input", "sample_identity", '{"id":1}'),
    ("expected_input", "sample_identity", '{"unknown":"not-a-native-field"}'),
    ("expected_input", "cache_key", ["score", -1]),
    ("budget", "max_payload_bytes", 1), ("budget", "max_replay_span_bytes", 1),
    ("budget", "max_host_working_bytes", 1), ("budget", "max_file_bytes", 1),
    ("budget", "max_tensor_bytes", 1), ("budget", "max_file_bytes", 6442450945),
])
def test_invalid_typed_metadata_or_insufficient_budget_fails_before_train(native_boundary, metadata):
    f = native_boundary
    section, name, value = metadata
    f.packet[section][name] = value
    f.seal()
    with pytest.raises(ValueError):
        run(f)
    assert not f.prepared and not f.captures
