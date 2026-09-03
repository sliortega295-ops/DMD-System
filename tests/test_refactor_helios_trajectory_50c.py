from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import dataclasses
from pathlib import Path
from types import SimpleNamespace

from distillschedule.evidence import AppendOnlyTrajectoryWriter, TrajectoryCycleReceipt
from distillschedule.evidence.distributed_state_sketch import (
    sketch_named_tensors,
    write_rank_state_anchor,
)
from runtime.static_world16_trace import _AliasedOperationTrace, StaticWorld16Trace


ROOT = Path(__file__).resolve().parents[1]


def _load_contract_module():
    path = ROOT / "src/distillschedule/runtime/fsdp2_static.py"
    spec = importlib.util.spec_from_file_location("trajectory_fsdp2_static", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_probe_module():
    path = ROOT / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py"
    spec = importlib.util.spec_from_file_location("trajectory_helios_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_50_cycle_contract_advances_versions_without_weakening_full5() -> None:
    module = _load_contract_module()
    assert len(module.contract_for_variant("no-gan").logical_steps) == 5
    assert len(module.contract_for_variant("gan").logical_steps) == 5

    nogan = module.contract_for_trajectory("no-gan", 50)
    assert nogan.logical_steps == tuple(range(10, 260))
    assert nogan.generator_updates == 50
    assert nogan.critic_updates == 250
    assert len(nogan.ema_update_steps) == 250
    assert dict(nogan.final_versions) == {
        "generator": 52,
        "critic": 260,
        "ema": 260,
    }

    gan = module.contract_for_trajectory("gan", 50)
    assert gan.logical_steps == tuple(range(1000, 1250))
    assert gan.ema_update_steps == tuple(range(1000, 1250, 5))
    assert dict(gan.final_versions) == {
        "generator": 250,
        "critic": 1250,
        "ema": 850,
    }


def test_pre_optimizer_observer_is_one_shot_and_sees_clipped_gradients() -> None:
    import torch

    module = _load_contract_module()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    engine = module.FSDP2EngineAdapter(
        model,
        optimizer,
        scheduler,
        physical_step=0,
        gradient_clip=1.0,
    )
    observations: list[tuple[int, bool]] = []

    def observe(value) -> None:
        observations.append(
            (
                value.global_steps,
                all(parameter.grad is not None for parameter in value.module.parameters()),
            )
        )

    engine.set_pre_optimizer_step_observer(observe)
    engine.backward(model(torch.ones(1, 2)).sum())
    engine.step()
    assert observations == [(0, True)]
    engine.backward(model(torch.ones(1, 2)).sum())
    engine.step()
    assert observations == [(0, True)]


def test_trajectory_launcher_is_explicit_and_not_formal_timing() -> None:
    launcher = (
        ROOT / "scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh"
    ).read_text()
    worker = (
        ROOT / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py"
    ).read_text()
    hardening = (
        ROOT / "scripts/distillschedule/probe_static_fsdp2_p0_world16_hardened.py"
    ).read_text()
    controller = (
        ROOT / "scripts/distillschedule/run_static_fsdp2_p0_world16.py"
    ).read_text()
    summarizer = (
        ROOT / "scripts/distillschedule/summarize_trajectory_run.py"
    ).read_text()

    assert '[[ "${MODE}" == trajectory ]]' in launcher
    assert "DISTILLSCHEDULE_TRAJECTORY_CYCLES" in launcher
    assert '--trajectory-cycles "${TRAJECTORY_CYCLES}"' in launcher
    assert 'WORKER+=(--trajectory-mode)' in launcher
    assert "trajectory runs cannot be timing/profile runs" in worker
    assert "FIXED_FIVE_STEP_COHORT_REPLAY" in worker
    assert "Full tensor/gradient dumps on all 300 updates" in hardening
    assert 'choices=("validate", "correctness", "timing", "profile", "trajectory")' in controller
    assert 'value.add_argument("--ssh-port"' in controller
    assert '"--ssh-tty"' in controller
    assert "_summarize_trajectory(args, local)" in controller
    assert "TRAJECTORY_DIAGNOSTIC_NOT_FORMAL_TIMING" in summarizer


class _FakeTrace:
    root = Path("/trace")
    tensors = {}
    trace_hash = "trace-hash"
    manifest = {"operations": [], "tensors": {}}

    def operation(self, key: str):
        return {
            "key": key,
            "operation": "stage3_branch",
            "control": {"global_step": 10, "train_generator": True},
            "tensor_keys": ["step/10/critic/noise"],
        }


def test_trace_alias_rewrites_only_logical_step_and_preserves_source() -> None:
    source = _FakeTrace()
    alias = _AliasedOperationTrace(source, logical_step=15, source_step=10)
    operation = alias.operation("step/15/branch")
    assert operation["key"] == "step/15/branch"
    assert operation["control"] == {"global_step": 15, "train_generator": True}
    assert operation["tensor_keys"] == ["step/15/critic/noise"]
    assert source.operation("step/10/branch")["control"]["global_step"] == 10


def test_single_cycle_trace_returns_original_operation_trace() -> None:
    trace = StaticWorld16Trace.__new__(StaticWorld16Trace)
    trace.cycles = 1
    trace.logical_steps = tuple(range(10, 15))
    trace.source_steps = tuple(range(10, 15))
    source = _FakeTrace()
    trace._loaded = {0: source}
    assert trace._view(0, 10) is source


def test_trajectory_receipt_uses_explicit_trace_source_mapping() -> None:
    module = _load_probe_module()

    class Trace:
        @staticmethod
        def source_step(step: int) -> int:
            return 10 + (step - 10) % 5

    args = SimpleNamespace(
        variant="no-gan",
        run_id="receipt-smoke",
        trajectory_seed=7,
        trajectory_cycles=1,
        trajectory_arm_id="baseline",
    )
    generator = SimpleNamespace(global_steps=3, last_gradient_norm=None)
    critic = SimpleNamespace(global_steps=15)
    ema = SimpleNamespace(optimization_step=15)
    policy = SimpleNamespace(
        id="control",
        mesh_policy="world16-fsdp",
        rollout_batch_size=1,
        pack_real_score_pair=False,
    )
    value = module._trajectory_receipt(
        args=args,
        rank=0,
        world=16,
        cycle_index=1,
        cycle_steps=tuple(range(10, 15)),
        generator_record={
            "surrogate_loss": 1.0,
            "gradient_norm": 2.0,
            "physical_score_forwards": 3,
            "rollout": {"trace_identity_sha256": "g", "call_count": 6},
        },
        rollout_records=[
            {"member_trace_identities": [f"c-{index}"], "call_count": 6}
            for index in range(5)
        ],
        critic_records=[
            {"loss": 1.0, "gradient_norm": 2.0, "score_forwards": 1}
            for _ in range(5)
        ],
        ema_records=[{} for _ in range(5)],
        generator_engine=generator,
        critic_engine=critic,
        ema=ema,
        logical_hash="logical",
        trace_hash="trace",
        policy=policy,
        trace=Trace(),
        state_fingerprints={"state_anchor_sha256": "sha"},
    )
    assert value.finite
    assert len(set(value.rng_identities.values())) == 6
    assert "source_step=10" in value.rng_identities["generator"]


def _receipt(*, logical_step: int) -> TrajectoryCycleReceipt:
    return TrajectoryCycleReceipt(
        run_id="negative",
        workload="helios14b_static_fsdp2",
        variant="no-gan",
        rank=0,
        world_size=1,
        seed=1,
        cycle=1,
        expected_total_cycles=1,
        logical_step=logical_step,
        matched_contract={"trace_cohort": "FIXED_FIVE_STEP_COHORT_REPLAY"},
        implementation={"arm_id": "baseline"},
        sample_identities=[f"sample-{index}" for index in range(6)],
        rng_identities={f"rng-{index}": f"value-{index}" for index in range(6)},
        versions={"generator": 3, "critic": 15, "ema": 15},
        cycle_updates={"generator": 1, "critic": 5, "ema": 5},
        cumulative_updates={"generator": 1, "critic": 5, "ema": 5},
        call_census={"generator": 1},
        losses={"generator": 1.0},
        gradient_norms={"generator": 1.0, **{f"critic_{i}": 1.0 for i in range(5)}},
        tensor_contracts={},
        state_fingerprints={"generator": "sha"},
        finite=True,
    )


def test_trajectory_summarizer_rejects_logical_step_drift(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    writer = AppendOnlyTrajectoryWriter(
        run_root / "trajectory/rank_00.jsonl", run_id="negative", rank=0
    )
    writer.append(_receipt(logical_step=11))
    output = run_root / "summary.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/distillschedule/summarize_trajectory_run.py"),
            "--run-root",
            str(run_root),
            "--run-id",
            "negative",
            "--workload",
            "helios14b_static_fsdp2",
            "--variant",
            "no-gan",
            "--cycles",
            "1",
            "--world-size",
            "1",
            "--expected-final-generator",
            "3",
            "--expected-final-critic",
            "15",
            "--expected-final-ema",
            "15",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    value = json.loads(output.read_text())
    assert value["status"] == "FAIL"
    assert any("logical step" in error for error in value["errors"])


def test_trajectory_summarizer_accepts_verified_state_anchor(tmp_path: Path) -> None:
    import torch

    run_root = tmp_path / "run"
    state_path = run_root / "trajectory/state/rank_00_cycle_001.json"
    state = sketch_named_tensors(
        [("weight", torch.tensor([1.0, 2.0]))], global_rank=0
    )
    references = write_rank_state_anchor(
        state_path,
        run_id="negative",
        workload="helios14b_static_fsdp2",
        variant="no-gan",
        cycle=1,
        rank=0,
        world_size=1,
        states={
            "generator_parameter": state,
            "critic_parameter": state,
            "generator_adam": state,
            "critic_adam": state,
            "generator_ema": state,
            "generator_gradient": state,
            **{f"critic_{index}_gradient": state for index in range(5)},
        },
    )
    writer = AppendOnlyTrajectoryWriter(
        run_root / "trajectory/rank_00.jsonl", run_id="negative", rank=0
    )
    writer.append(
        dataclasses.replace(
            _receipt(logical_step=10),
            state_fingerprints=references,
        )
    )
    output = run_root / "summary.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/distillschedule/summarize_trajectory_run.py"),
            "--run-root",
            str(run_root),
            "--run-id",
            "negative",
            "--workload",
            "helios14b_static_fsdp2",
            "--variant",
            "no-gan",
            "--cycles",
            "1",
            "--world-size",
            "1",
            "--expected-final-generator",
            "3",
            "--expected-final-critic",
            "15",
            "--expected-final-ema",
            "15",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text())["status"] == "PASS"
