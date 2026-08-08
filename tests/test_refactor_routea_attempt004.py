from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
from omegaconf import OmegaConf

from distillschedule.core.config import load_plan
from distillschedule.core.contracts import ContractViolation
from distillschedule.runtime.routea_evidence import (
    atomic_json,
    build_node_contract,
    finalize_node_contracts,
    inspect_rendered_helios_config,
    inspect_trace_bundle,
    object_digest,
    validate_final_contract,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/distillschedule/plans/helios_routea_full5_world16_typed.json"


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses and other import-time helpers resolve annotations through
    # sys.modules.  Mirror normal import semantics before executing a script
    # loaded directly from its filesystem path.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


SUMMARIZER = load_script(
    "routea_summarizer",
    ROOT / "scripts/distillschedule/summarize_routea_world16_typed.py",
)
CONTROLLER = load_script(
    "routea_controller",
    ROOT / "scripts/distillschedule/run_routea_world16.py",
)
SUITE = load_script(
    "no_gan_suite",
    ROOT / "scripts/distillschedule/run_no_gan_suite.py",
)
OFFLOAD = load_script(
    "g_offload_search",
    ROOT / "scripts/distillschedule/run_g_offload_search.py",
)
PAIRED = load_script(
    "paired_formal",
    ROOT / "scripts/distillschedule/run_paired_formal.py",
)


def write_trace_bundle(root: Path, world_size: int = 16) -> Path:
    ranks = []
    for rank in range(world_size):
        rank_root = root / f"rank_{rank}"
        rank_root.mkdir(parents=True)
        trace = {
            "schema_version": "distillschedule.operation_trace.v1",
            "source": {
                "source_authority": f"frozen-source-rank-{rank}",
                "sample_id": rank,
            },
            "contract": {"rank": rank},
            "operations": [
                {"operation": "rollout", "seed": rank},
                {"operation": "critic_update", "step": 10 + rank},
            ],
        }
        trace["trace_hash"] = object_digest(trace)
        (rank_root / "trace.json").write_text(
            json.dumps(trace, indent=2, sort_keys=True) + "\n"
        )
        ranks.append(
            {
                "rank": rank,
                "path": f"rank_{rank}",
                "trace_hash": trace["trace_hash"],
            }
        )
    bundle = {
        "schema_version": "distillschedule.operation_trace_bundle.v1",
        "ranks": ranks,
    }
    bundle["bundle_hash"] = object_digest(bundle)
    (root / "bundle.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n"
    )
    return root


def rendered_config(path: Path, *, output_dir: str, batch: int = 1) -> Path:
    payload = {
        "output_dir": output_dir,
        "data_config": {
            "single_height": 384,
            "single_width": 640,
            "gan_data_root": ["/data/rank16"],
        },
        "model_config": {
            "transformer_model_name_or_path": "/models/generator",
            "real_score_model_name_or_path": "/models/score",
        },
        "training_config": {
            "mixed_precision": "bf16",
            "max_train_steps": 15,
            "train_batch_size": batch,
            "gradient_accumulation_steps": 1,
            "checkpointing_steps": 15,
            "gradient_checkpointing": True,
            "dfake_gen_update_ratio": 5,
            "latent_window_size": [9],
            "use_ema": True,
            "is_train_dmd": True,
            "is_dmd_vae_decode": False,
            "is_use_gan": False,
            "is_use_reward_model": False,
            "ema_start_step": 750,
            "resume_from_checkpoint": "/checkpoint/checkpoint-10",
            "dmd_generator_deepspeed_config": str(
                ROOT / "scripts/accelerate_configs/zero1_cpu_optimizer_offload.json"
            ),
            "dmd_critic_deepspeed_config": str(
                ROOT / "scripts/accelerate_configs/zero1_cpu_optimizer_offload.json"
            ),
        },
        "validation_config": {
            "validation_height": 384,
            "validation_width": 640,
            "validation_latent_window_size": [9],
        },
    }
    OmegaConf.save(OmegaConf.create(payload), path)
    return path


def accelerate_config(path: Path, node_rank: int) -> Path:
    path.write_text(
        "\n".join(
            [
                "compute_environment: LOCAL_MACHINE",
                "distributed_type: DEEPSPEED",
                "deepspeed_config:",
                f"  deepspeed_config_file: {ROOT / 'scripts/accelerate_configs/zero1_cpu_optimizer_offload.json'}",
                "  deepspeed_multinode_launcher: standard",
                "gradient_accumulation_steps: 1",
                f"machine_rank: {node_rank}",
                "main_training_function: main",
                "mixed_precision: bf16",
                "num_machines: 2",
                "num_processes: 16",
                "rdzv_backend: static",
                "same_network: true",
                "use_cpu: false",
                "",
            ]
        )
    )
    return path


def set_plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in (
        "HELIOS_RUNTIME_ROOT",
        "HELIOS_DATA_ROOT",
        "HELIOS_DISTILLED_ROOT",
        "HELIOS_BASE_ROOT",
        "DISTILLSCHEDULE_PRECAST_GENERATOR_PATH",
        "DISTILLSCHEDULE_PRECAST_SCORE_PATH",
    ):
        directory = tmp_path / name.lower()
        directory.mkdir(exist_ok=True)
        monkeypatch.setenv(name, str(directory))
    monkeypatch.setenv("DISTILLSCHEDULE_PYTHON", "/runtime/python")
    monkeypatch.setenv("DISTILLSCHEDULE_ACCELERATE", "/runtime/accelerate")
    monkeypatch.setenv("DISTILLSCHEDULE_OUTPUT_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("DISTILLSCHEDULE_RUN_ID", "attempt004-test")
    monkeypatch.setenv("DISTILLSCHEDULE_FROZEN_BLOCK_MMAP_CACHE", str(tmp_path / "frozen"))
    monkeypatch.setenv("DISTILLSCHEDULE_IMMUTABLE_MODEL_MMAP_CACHE", str(tmp_path / "immutable"))
    monkeypatch.setenv("DISTILLSCHEDULE_TRACE_BUNDLE", str(tmp_path / "trace"))
    monkeypatch.setenv("DISTILLSCHEDULE_CHECKPOINT", str(tmp_path / "checkpoint"))
    monkeypatch.setenv("DISTILLSCHEDULE_ACCELERATE_CONFIG", str(tmp_path / "accelerate.yaml"))
    monkeypatch.setenv("DISTILLSCHEDULE_MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("DISTILLSCHEDULE_MASTER_PORT", "29677")


def compiled_preflight(path: Path, plan_digest: str, node_rank: int) -> Path:
    payload = {
        "plan_id": "helios-routea-full5-world16-typed-revalidation-v1",
        "plan_digest": plan_digest,
        "invocation": {
            "command": [
                "/runtime/accelerate",
                "launch",
                "--num_machines",
                "2",
                "--num_processes",
                "16",
                "--machine_rank",
                str(node_rank),
                "--mixed_precision",
                "bf16",
                "--gradient_accumulation_steps",
                "1",
                "--config_file",
                "/tmp/accelerate.yaml",
                "train_helios.py",
            ]
        },
    }
    path.write_text(json.dumps(payload))
    return path


def test_trace_bundle_allows_rank_specific_hashes_but_freezes_mapping(tmp_path: Path) -> None:
    trace_root = write_trace_bundle(tmp_path / "trace")
    contract = inspect_trace_bundle(trace_root)
    hashes = [entry["trace_hash"] for entry in contract["ranks"]]
    assert len(set(hashes)) == 16
    assert contract["rank_count"] == 16
    assert [entry["rank"] for entry in contract["ranks"]] == list(range(16))
    assert all(len(entry["operation_fingerprint"]) == 64 for entry in contract["ranks"])


def test_normalized_config_ignores_output_path_but_rejects_batch_drift(tmp_path: Path) -> None:
    first = inspect_rendered_helios_config(
        rendered_config(tmp_path / "node0.yaml", output_dir="/run/a")
    )
    second = inspect_rendered_helios_config(
        rendered_config(tmp_path / "node1.yaml", output_dir="/run/b")
    )
    assert first["normalized_digest"] == second["normalized_digest"]
    with pytest.raises(ContractViolation, match="expected 1"):
        inspect_rendered_helios_config(
            rendered_config(tmp_path / "bad.yaml", output_dir="/run/c", batch=2)
        )


def test_node_contracts_require_same_normalized_training_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_plan_env(monkeypatch, tmp_path)
    trace = write_trace_bundle(tmp_path / "trace")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "state.json").write_text("{}")
    contracts = []
    for rank in (0, 1):
        monkeypatch.setenv("DISTILLSCHEDULE_NODE_RANK", str(rank))
        accel = accelerate_config(tmp_path / f"accelerate-{rank}.yaml", rank)
        monkeypatch.setenv("DISTILLSCHEDULE_ACCELERATE_CONFIG", str(accel))
        # The raw resolved digest is node-qualified because the launcher path
        # is part of the compiled invocation.  Production compiles each node
        # after rendering its own Accelerate config, then normalizes the shared
        # training contract before the two node contracts are compared.
        loaded = load_plan(PLAN)
        config = rendered_config(
            tmp_path / f"rendered-{rank}.yaml", output_dir=f"/run/node-{rank}"
        )
        compiled = compiled_preflight(
            tmp_path / f"compiled-{rank}.json", loaded.digest, rank
        )
        payload = build_node_contract(
            node_rank=rank,
            plan_path=PLAN,
            repo_root=ROOT,
            compiled_preflight_path=compiled,
            rendered_config_path=config,
            accelerate_config_path=accel,
            trace_bundle_path=trace,
            checkpoint_path=checkpoint,
            assets={"data": tmp_path / "helios_data_root"},
        )
        path = tmp_path / f"node-{rank}.json"
        atomic_json(path, payload)
        contracts.append(path)
    final = finalize_node_contracts(contracts)
    validate_final_contract(final)
    node_payloads = [json.loads(path.read_text()) for path in contracts]
    assert len(
        {payload["node"]["resolved_plan_digest"] for payload in node_payloads}
    ) == 2
    assert len({payload["shared_digest"] for payload in node_payloads}) == 1
    assert final["status"] == "PASS"
    assert final["shared"]["batch_contract"]["global_batch"] == 16
    assert final["shared"]["memory_acceptance"]["host_reserve_gate"] is False


def version_events(*, include_ema_predicate: bool = True, include_final: bool = True):
    events = [
        {
            "phase": "parameter_versions_initialized",
            "generator_version": 2,
            "critic_version": 10,
            "ema_version": 10,
        }
    ]
    for step in range(10, 15):
        if step == 10:
            events.append(
                {
                    "phase": "generator_optimizer_step",
                    "global_step": step,
                    "generator_version": 3,
                }
            )
        events.append(
            {
                "phase": "critic_optimizer_step",
                "global_step": step,
                "critic_version": step + 1,
            }
        )
        ema = {
            "phase": "ema_step",
            "global_step": step,
            "ema_version": step + 1,
        }
        if include_ema_predicate:
            ema["ema_update_enabled"] = True
        events.append(ema)
    if include_final:
        events.append(
            {
                "phase": "parameter_versions_final",
                "global_step": 14,
                "generator_version": 3,
                "critic_version": 15,
                "ema_version": 15,
            }
        )
    return events


def test_version_trajectory_requires_observed_ema_predicate_and_final_record() -> None:
    exact = SUMMARIZER.validate_version_trajectory(version_events(), rank=0)
    assert exact["status"] == "OBSERVED_EXACT"
    missing_predicate = SUMMARIZER.validate_version_trajectory(
        version_events(include_ema_predicate=False), rank=0
    )
    assert missing_predicate["status"] == "INCOMPLETE_OR_MISMATCH"
    assert any("predicate" in error for error in missing_predicate["errors"])
    missing_final = SUMMARIZER.validate_version_trajectory(
        version_events(include_final=False), rank=0
    )
    assert any("final observed" in error for error in missing_final["errors"])


def test_strict_launcher_manifest_rejects_attempt003_mixed_csv(tmp_path: Path) -> None:
    manifest = tmp_path / "launch_manifest.txt"
    manifest.write_text(
        "schema_version=old\nrun_id=attempt003\n0, GPU-uuid, NVIDIA A100, 40960 MiB\n"
    )
    with pytest.raises(SUMMARIZER.EvidenceError, match="legacy/raw CSV"):
        SUMMARIZER.parse_strict_kv(manifest)


def test_high_host_pressure_without_oom_is_not_normative_failure() -> None:
    assert SUMMARIZER.CUDA_OOM.search("ordinary allocation pressure") is None
    before = SUMMARIZER.parse_memory_events("oom 0;oom_kill 0;high 10;")
    after = SUMMARIZER.parse_memory_events("oom 0;oom_kill 0;high 1000;")
    assert after["oom"] - before["oom"] == 0
    assert after["oom_kill"] - before["oom_kill"] == 0


def test_suite_dag_marks_target_search_as_run_required() -> None:
    suite = SUITE.expand_env(
        SUITE.load_yaml(
            ROOT / "configs/distillschedule/suites/helios_no_gan_world16.yaml"
        )
    )
    order, stages = SUITE.stage_map(suite)
    assert order.index("n1_routea_attempt004") < order.index(
        "n2_target_g_offload_search"
    )
    assert stages["n2_target_g_offload_search"]["disposition"] == "RUN_REQUIRED"
    assert stages["r4_exact_b0_capacity_diagnostic"]["disposition"] == "REUSED_BOUNDED_DIAGNOSTIC"
    assert "n2_target_g_offload_search" in SUITE.select_stages(
        order, only=[], from_stage="n2_target_g_offload_search", stop_after=None
    )


def test_suite_remote_pair_qualifies_preflight_output_by_node(tmp_path: Path) -> None:
    suite = SUITE.expand_env(
        SUITE.load_yaml(
            ROOT / "configs/distillschedule/suites/helios_no_gan_world16.yaml"
        )
    )
    _order, stages = SUITE.stage_map(suite)
    context = {
        "python": "/python",
        "repo_root": "/repo",
        "suite_run_id": "suite",
        "stage_run_id": "stage",
        "master_addr": "10.0.0.1",
        "master_port": "29677",
    }
    command = SUITE.format_command(stages["n0_preflight"]["command"], context)
    pair = SUITE.RemotePair(
        ssh="ssh",
        hosts=("node0", "node1"),
        root=tmp_path,
        site_env="site.env",
        suite_run_id="suite",
        stage_id="n0_preflight",
        stage_run_id="stage",
        command=command,
        log_dir=tmp_path / "logs",
        dry_run=True,
    )
    commands = pair.commands()
    assert "node_0" in commands[0][-1]
    assert "node_1" in commands[1][-1]


def test_suite_routea_command_uses_explicit_compute_rendezvous() -> None:
    suite = SUITE.expand_env(
        SUITE.load_yaml(
            ROOT / "configs/distillschedule/suites/helios_no_gan_world16.yaml"
        )
    )
    _order, stages = SUITE.stage_map(suite)
    command = SUITE.format_command(
        stages["n1_routea_attempt004"]["command"],
        {
            "python": "/python",
            "repo_root": "/repo",
            "hosts_csv": "ssh-node0,ssh-node1",
            "stage_run_id": "attempt",
            "site_env": "/site.env",
            "master_addr": "172.24.129.137",
            "master_port": "29677",
        },
    )
    assert command[command.index("--master-addr") + 1] == "172.24.129.137"
    assert command[command.index("--master-port") + 1] == "29677"


def test_baai_smoke_isolates_bytecode_on_shared_filesystem() -> None:
    script = (
        ROOT / "scripts/distillschedule/baai_server_smoke.sh"
    ).read_text()
    assert 'export PYTHONPYCACHEPREFIX="${OUT}/pycache"' in script
    assert script.index("export PYTHONPYCACHEPREFIX") < script.index(
        '"${PYTHON}" -m compileall'
    )
    assert 'cat "${OUT}/node_contract_summary.json" >&2' in script


def test_routea_plan_explicitly_freezes_accumulation() -> None:
    plan = json.loads(
        (
            ROOT
            / "configs/distillschedule/plans/helios_routea_full5_world16_typed.json"
        ).read_text()
    )
    command = plan["legacy"]["command"]
    index = command.index("--gradient_accumulation_steps")
    assert command[index + 1] == "1"


def test_routea_controller_helpers_are_executable() -> None:
    for relative in (
        "scripts/distillschedule/remote_stage_wrapper.sh",
        "scripts/distillschedule/cancel_remote_stage.sh",
        "scripts/distillschedule/launch_routea_world16_typed_node.sh",
    ):
        assert (ROOT / relative).stat().st_mode & 0o111, relative


def test_routea_worker_uses_package_metadata_for_runtime_versions() -> None:
    script = (
        ROOT / "scripts/distillschedule/launch_routea_world16_typed_node.sh"
    ).read_text()
    assert "package_version('accelerate')" in script
    assert "package_version('deepspeed')" in script
    assert "[str(accelerate),'--version']" not in script


def test_routea_controller_summarizes_remote_root_on_compute_host() -> None:
    args = CONTROLLER.parser().parse_args(
        [
            "--hosts",
            "node0,node1",
            "--run-id",
            "attempt004",
            "--repo-root",
            "/local/repo",
            "--remote-root",
            "/remote/repo",
            "--site-env",
            "/shared/site.env",
            "--ssh",
            "ssh-wrapper",
        ]
    )
    command, output = CONTROLLER._remote_summary_command(
        args=args,
        identity={"remote_root": "/remote/repo"},
    )
    assert command[:2] == ["ssh-wrapper", "node0"]
    assert "source /shared/site.env" in command[-1]
    assert "/remote/repo/scripts/distillschedule/summarize_routea_world16_typed.py" in command[-1]
    assert output == Path(
        "/remote/repo/artifacts/manifests/attempt004_result.json"
    )


def test_dry_run_stage_is_never_reused_as_real_evidence() -> None:
    dry = {
        "status": "PASS",
        "dry_run": True,
        "stage_contract_digest": "same",
    }
    assert not SUITE.reusable_stage_result(dry, stage_contract_digest="same")
    assert SUITE.dependency_accepted("DRY_RUN", dry_run=True)
    assert not SUITE.dependency_accepted("DRY_RUN", dry_run=False)


def test_offload_search_treats_only_capacity_outcomes_as_bounds() -> None:
    assert OFFLOAD.is_capacity_failure("OOM")
    assert OFFLOAD.is_capacity_failure("CAPACITY_REJECTED")
    assert not OFFLOAD.is_capacity_failure("FAIL")
    assert not OFFLOAD.is_capacity_failure("EVIDENCE_INVALID")


def test_offload_search_recognizes_zero_block_minimum() -> None:
    observations = {40: {"status": "PASS"}, 0: {"status": "PASS"}}
    assert OFFLOAD.minimum_is_exact(
        passing_size=0,
        failing_size=None,
        observations=observations,
        nonmonotonic=False,
    )


def test_formal_timing_reports_dispersion() -> None:
    summary = PAIRED.distribution_summary([10.0, 12.0, 14.0])
    assert summary["count"] == 3
    assert summary["median"] == 12.0
    assert summary["range"] == 4.0
    assert summary["sample_stdev"] == 2.0
    assert summary["coefficient_of_variation"] == pytest.approx(1.0 / 6.0)
    assert summary["median_absolute_deviation"] == 2.0
