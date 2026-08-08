from __future__ import annotations

import json
from pathlib import Path

import pytest

from distillschedule.core.contracts import ContractViolation
from distillschedule.runtime.launch_config import (
    inspect_accelerate_config,
    render_accelerate_config,
    require_routea_zero1_contract,
)
from distillschedule.runtime.topology import TopologyRecord, validate_topology_records

ROOT = Path(__file__).resolve().parents[1]


def test_rendered_accelerate_config_is_node_qualified_and_zero1(tmp_path: Path) -> None:
    template = ROOT / "scripts/accelerate_configs/multi_node_zero1_cpu_optimizer_offload.yaml"
    deepspeed = ROOT / "scripts/accelerate_configs/zero1_cpu_optimizer_offload.json"
    output = tmp_path / "accelerate.node1.yaml"
    contract = render_accelerate_config(
        template_path=template,
        deepspeed_config_path=deepspeed,
        output_path=output,
        num_machines=2,
        num_processes=16,
        machine_rank=1,
    )
    assert contract.machine_rank == 1
    assert contract.num_machines == 2
    assert contract.num_processes == 16
    assert contract.multinode_launcher == "standard"
    assert contract.deepspeed.zero_stage == 1
    assert contract.deepspeed.optimizer_offload == "cpu"
    assert contract.deepspeed.parameter_offload == "none"
    assert str(deepspeed.resolve()) in output.read_text()
    assert inspect_accelerate_config(output).to_dict() == contract.to_dict()


def test_routea_launcher_rejects_zero_stage_drift(tmp_path: Path) -> None:
    path = tmp_path / "zero2.json"
    path.write_text(
        json.dumps(
            {
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {"device": "cpu"},
                }
            }
        )
    )
    with pytest.raises(ContractViolation, match="requires ZeRO-1"):
        require_routea_zero1_contract(path)


def _world16_records() -> list[TopologyRecord]:
    records = []
    for rank in range(16):
        node_rank = rank // 8
        local_rank = rank % 8
        records.append(
            TopologyRecord(
                rank=rank,
                local_rank=local_rank,
                node_rank=node_rank,
                world_size=16,
                local_world_size=8,
                hostname=f"node-{node_rank}",
                cuda_device_index=local_rank,
                cuda_device_count=8,
            )
        )
    return records


def test_world16_topology_accepts_two_hosts_and_sixteen_unique_ranks() -> None:
    summary = validate_topology_records(
        _world16_records(),
        expected_nodes=2,
        expected_local_world_size=8,
        expected_world_size=16,
    )
    assert summary.ranks_by_node[0] == tuple(range(8))
    assert summary.ranks_by_node[1] == tuple(range(8, 16))


def test_world16_topology_rejects_two_independent_world8_groups() -> None:
    records = _world16_records()
    records[8] = TopologyRecord(
        rank=0,
        local_rank=0,
        node_rank=1,
        world_size=8,
        local_world_size=8,
        hostname="node-1",
        cuda_device_index=0,
        cuda_device_count=8,
    )
    with pytest.raises(ContractViolation, match="global ranks"):
        validate_topology_records(
            records,
            expected_nodes=2,
            expected_local_world_size=8,
            expected_world_size=16,
        )


def test_world16_topology_rejects_duplicate_local_rank() -> None:
    records = _world16_records()
    records[7] = TopologyRecord(
        rank=7,
        local_rank=6,
        node_rank=0,
        world_size=16,
        local_world_size=8,
        hostname="node-0",
        cuda_device_index=6,
        cuda_device_count=8,
    )
    with pytest.raises(ContractViolation, match="local ranks mismatch"):
        validate_topology_records(
            records,
            expected_nodes=2,
            expected_local_world_size=8,
            expected_world_size=16,
        )


def _load_summarizer_module():
    import importlib.util

    path = ROOT / "scripts/distillschedule/summarize_routea_world16_typed.py"
    spec = importlib.util.spec_from_file_location("routea_summarizer", path)
    assert spec is not None and spec.loader is not None
    import sys

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_jsonl_reader_ignores_only_unterminated_partial_tail(tmp_path: Path) -> None:
    module = _load_summarizer_module()
    path = tmp_path / "rank_0.jsonl"
    path.write_text('{"phase":"ok"}\n{"phase":', encoding="utf-8")
    result = module.read_jsonl_tail_safe(path)
    assert len(result.events) == 1
    assert result.ignored_partial_tail is True


def test_jsonl_reader_rejects_malformed_complete_record(tmp_path: Path) -> None:
    module = _load_summarizer_module()
    path = tmp_path / "rank_0.jsonl"
    path.write_text('{"phase":"ok"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(module.EvidenceError, match="malformed JSONL"):
        module.read_jsonl_tail_safe(path)


def test_summarizer_closes_a_complete_world16_run(tmp_path: Path) -> None:
    module = _load_summarizer_module()
    run = tmp_path / "run"
    identity_values = {
        "dmd_system_head": "d" * 40,
        "dmd_system_plan_sha256": "1" * 64,
        "resolved_plan_digest": "2" * 64,
        "helios_head": "h" * 40,
        "helios_entrypoint_sha256": "3" * 64,
        "python_path": "/env/bin/python",
        "python_sha256": "4" * 64,
        "accelerate_path": "/env/bin/accelerate",
        "accelerate_sha256": "5" * 64,
        "deepspeed_path": "/env/bin/deepspeed",
        "deepspeed_sha256": "6" * 64,
        "accelerate_config_sha256": "7" * 64,
        "deepspeed_config_sha256": "8" * 64,
    }
    for node_rank in (0, 1):
        launcher = run / "launcher" / f"node_{node_rank}"
        launcher.mkdir(parents=True)
        lines = [
            f"node_rank={node_rank}",
            f"host=node-{node_rank}",
            "memory_events_before=oom 0;oom_kill 0;",
            "memory_events_after=oom 0;oom_kill 0;",
            "final_return_code=0",
        ] + [f"{key}={value}" for key, value in identity_values.items()]
        (launcher / "launch_manifest.txt").write_text("\n".join(lines) + "\n")
        (launcher / "console.log").write_text("complete\n")
        (launcher / "resource.csv").write_text(
            "unix_time,cgroup_bytes,host_used_bytes,gpu_used_mib,gpu_util_percent\n"
            "0,1,100,0,0\n1,1,200,0,0\n"
        )
    topology_records = [record.to_dict() for record in _world16_records()]
    (run / "topology_probe.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "summary": {"world_size": 16, "nodes": 2, "local_world_size": 8},
                "records": topology_records,
            }
        )
    )
    phase = run / "phase_trace"
    phase.mkdir()
    for rank in range(16):
        events = [
            {
                "phase": "parameter_versions_initialized",
                "generator_version": 2,
                "critic_version": 10,
            },
            {
                "phase": "operation_trace_input_validated",
                "trace_hash": f"trace-{rank}",
                "source_authority": f"rank-{rank}",
            },
        ]
        for step in range(10, 15):
            events.append(
                {"phase": "training_step_start", "global_step": step, "time_unix": float(step)}
            )
            events.append({"phase": "critic_optimizer_step"})
            events.append({"phase": "ema_step"})
            if step == 10:
                events.append({"phase": "generator_optimizer_step"})
            events.append(
                {
                    "phase": "training_step_end",
                    "global_step": step,
                    "time_unix": float(step) + 1.0,
                }
            )
        events.extend(
            [
                {"phase": "activation_offload_final", "stats": {"live_cpu_bytes": 0}},
                {
                    "phase": "frozen_block_streamer_final",
                    "stats": {"resident_bytes": 0, "materialized_bytes": 10, "materialize_count": 1},
                },
                {
                    "phase": "frozen_block_streamer_final",
                    "stats": {"resident_bytes": 0, "materialized_bytes": 20, "materialize_count": 2},
                },
                {"phase": "training_complete_skip_final_save"},
            ]
        )
        (phase / f"rank_{rank}.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events)
        )
    checkpoint = run / "checkpoint-15"
    checkpoint.mkdir()
    (checkpoint / "state.bin").write_bytes(b"state")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"timing_summary": {"global_training_window_seconds": 5.0}})
    )
    output = tmp_path / "result.json"
    result = module.summarize(
        run_id="synthetic",
        root=tmp_path,
        run_dir=run,
        output=output,
        baseline_result=baseline,
    )
    assert result["status"] == "PASS"
    assert result["functional_status"] == "PASS"
    assert result["timing"]["status"] == "PASS"
    assert result["evidence_errors"] == []
    assert output.is_file()
