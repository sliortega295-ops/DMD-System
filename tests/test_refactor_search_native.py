"""CPU contract tests; fixture measurements are never GPU evidence."""
from __future__ import annotations
import ast
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from distillschedule.search.contracts import semantic_contract
from distillschedule.search.registry import WAN, LIVEWAN, candidate_profile, get_candidate
from distillschedule.search.execution import NativeTrialRunner, resolve_candidate
from distillschedule.search.native import handle_request
from distillschedule.workloads.best_known.evidence import summary_command
from distillschedule.workloads.best_known.dispatch import ssh_command
from distillschedule.workloads.best_known.registry import fingerprint


def _native_comparison_fixture():
    return {"schema_version": "distillschedule.trajectory_coarse_comparison.v1",
            "comparison_kind": "semantic", "expected_cycles": 2, "expected_world_size": 16,
            "complete": True, "status": "TRAJECTORY_MATCHED_50C_SINGLE_SEED",
            "contract": {"status": "PASS", "mismatches": []}}


@pytest.mark.parametrize("status", ["TRAJECTORY_MATCHED_50C_SINGLE_SEED", "TRAJECTORY_REVIEW_50C_COMPLETED"])
def test_native_comparison_uses_real_schema_and_two_cycle_inventory(status):
    from distillschedule.search.native import validate_native_comparison
    value = _native_comparison_fixture()
    value["status"] = status
    validate_native_comparison(value)
    for key, bad in (("expected_cycles", 50), ("expected_world_size", 8),
                     ("comparison_kind", "approximation"), ("complete", False),
                     ("schema_version", "invented"), ("status", "PASS"),
                     ("status", "CONTRACT_MISMATCH_50C_COMPLETED")):
        changed = {**value, key: bad}
        with pytest.raises(ValueError):
            validate_native_comparison(changed)


def test_wan_search_maps_to_real_two_cycle_workers():
    for key in ("fsdp-reference", "resident-tree", "resident-chunk2048", "resident-lora-epilogue"):
        p = candidate_profile(get_candidate(WAN, key))
        assert p["runtime"]["native_run_modes"] == {"correctness": "trajectory", "diagnostic-timing": "two-cycle-timing"}
        assert p["runtime"]["search_two_cycle_ready"]
        assert p["contract"]["final_versions"] == {"generator": 4, "critic": 20, "ema": 20}
        assert p["runtime"]["environment"]["DISTILLSCHEDULE_TWO_CYCLE_SEED"] == "20260903"
        assert p["runtime"]["environment"]["DISTILLSCHEDULE_SEARCH_CANDIDATE_ID"] == key
    assert not get_candidate(WAN, "resident-flat512").executable


def test_native_summary_dispatch_does_not_fall_back_to_one_cycle():
    r = resolve_candidate(WAN, "resident-tree")
    r.update(nodes=[{"repo_root": str(ROOT), "python": sys.executable, "environment": {}}],
             asset_config={"run_root": "/remote/persistent/run"}, run_id="UNIT_TEST")
    r["native_mode"] = "trajectory"
    c = summary_command(r)
    assert c[1].endswith("summarize_wan13b_trajectory_world16.py")
    assert c[-2:] == ["--cycles", "2"]
    r["native_mode"] = "two-cycle-timing"
    assert summary_command(r)[1].endswith("summarize_wan13b_two_cycle_timing_world16.py")


def test_remote_dispatch_uses_allocated_transport_without_reconfiguring_ssh():
    c = ssh_command({"host": "allocated-node", "ssh_port": 2222, "address_family": "ipv4"})
    assert c[-1] == "allocated-node" and "-4" in c
    assert c[c.index("-p") + 1] == "2222"
    assert "-oControlMaster=no" in c


def test_remote_rpc_rejects_wrong_revision_before_writing():
    with patch("distillschedule.search.native.git_head", return_value="a" * 40):
        with pytest.raises(ValueError, match="source changed"):
            handle_request({"source_head": "b" * 40, "action": "resolve"})


def test_native_worker_dynamic_import_dependencies_exist():
    # The previous integration AST check covered import statements but missed
    # ROOT / file imports used by the Wan/Helios native adapter chain.
    files = ["probe_wan13b_two_cycle_timing_world16.py", "probe_wan13b_trajectory_world16.py",
             "probe_wan13b_fair_world16.py", "probe_wan13b_upper_bound_world16.py",
             "probe_static_fsdp2_p0_world16.py", "probe_static_fsdp2_p0_world16_hardened.py"]
    for name in files:
        tree = ast.parse((ROOT / "scripts/distillschedule" / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and isinstance(node.left, ast.Name) and node.left.id == "ROOT" and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str) and node.right.value.endswith(".py"):
                assert (ROOT / node.right.value).is_file(), (name, node.right.value)
    from distillschedule.runtime.nsys_range import StaticFSDP2Full5NsightRange
    assert StaticFSDP2Full5NsightRange(variant="no-gan", enabled=False, rank=0, local_rank=0).enabled is False


def test_cpu_api_gate_catches_drift_without_executing_model_functions(tmp_path):
    import importlib.util
    gate_path = ROOT / "scripts/distillschedule/check_search_native_api.py"
    spec = importlib.util.spec_from_file_location("search_native_api_gate", gate_path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    source = tmp_path / "unit_api.py"
    source.write_text("def helper(value):\n    raise AssertionError('must not run')\ndef worker():\n    return helper(1, missing_option=True)\n")
    spec = importlib.util.spec_from_file_location("unit_api", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checked, errors = gate.check_module(module)
    assert checked >= 1 and len(errors) == 1
    assert "missing_option" in errors[0]["error"]


def test_generic_block_sharding_keeps_helios40_default_and_wan30_opt_in():
    import inspect
    from distillschedule.runtime.fsdp2_p0 import apply_blockwise_fsdp2_policy
    signature = inspect.signature(apply_blockwise_fsdp2_policy)
    assert signature.parameters["expected_block_count"].default == 40
    bound = signature.bind(None, mesh_policy="world16-fsdp", expected_block_count=30)
    assert bound.arguments["expected_block_count"] == 30
    source = inspect.getsource(apply_blockwise_fsdp2_policy)
    assert 'f"{expected_block_count}_BLOCKS_BOTTOM_UP_THEN_ROOT"' in source
    assert 'len(blocks) != expected_block_count' in source


def test_preparation_is_serial_and_once_per_selected_candidate(tmp_path):
    assets = {"assets": {"test": "UNIT_TEST_NOT_ASSETS"}, "nodes": []}
    identity = {"source_head": "a" * 40, "asset_fingerprint": fingerprint(assets["assets"])}
    r = NativeTrialRunner(workload=WAN, identity=identity, assets=assets,
                          output=tmp_path, run_prefix="test", bootstrap_peer="resident-tree")
    calls = []
    def trajectory(c):
        calls.append(c.id)
        return {"id": c.id}
    def rpc(action, **payload):
        assert action == "prepare_pair"
        (tmp_path / "preparation").mkdir(exist_ok=True)
        return {key: {"source_head": "a" * 40, "candidate_id": key, "assets": assets} for key in
                (payload["reference"]["id"], payload["candidate"]["id"])}
    with patch.object(r, "_trajectory", side_effect=trajectory), patch.object(r, "_rpc", side_effect=rpc):
        r._prepare(get_candidate(WAN, "fsdp-reference"))
        r._prepare(get_candidate(WAN, "resident-tree"))
        assert calls == ["fsdp-reference", "resident-tree"]
        r._prepare(get_candidate(WAN, "resident-chunk2048"))
        assert calls[-2:] == ["fsdp-reference", "resident-chunk2048"]


def test_resource_observer_is_not_installed_on_canonical_paths():
    source = (ROOT / "src/distillschedule/workloads/best_known/dispatch.py").read_text()
    assert 'if "search_candidate" in receipt:\n        from distillschedule.search.resources import ResourceObserver' in source
    observer = (ROOT / "src/distillschedule/search/resources.py").read_text()
    tree = ast.parse(observer)
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) and "torch" in ast.unparse(n) for n in ast.walk(tree))
    assert "SAMPLED_DRIVER_AND_NODE_CGROUP_NOT_CONTINUOUS_EXTREMA" in observer


@pytest.mark.parametrize("workload", [WAN, LIVEWAN])
def test_normalizer_requires_native_gates_and_preserves_review(tmp_path, workload):
    """Schema fixtures only. This test never starts a worker or claims GPU PASS."""
    from distillschedule.search.measurement import normalize_native
    from distillschedule.search.native import binding
    from distillschedule.search.registry import search_identity
    c = get_candidate(workload, "fsdp-reference")
    identity = search_identity(workload, source_head="a" * 40, assets={"fixture": True},
                               hardware={"world_size": 16, "nodes": 2, "gpu": "A100-SXM4-40GB"})
    receipt = {"search_candidate": c.to_dict(), "profile_id": workload, "native_mode": "two-cycle-timing",
               "integration_head": identity["source_head"], "asset_fingerprint": identity["asset_fingerprint"],
               "workload_semantics_hash": identity["semantic_fingerprint"], "run_id": "UNIT_TEST_NOT_GPU"}
    versions = {"generator": 4, "critic": 20, "ema": 20} if workload == WAN else {"generator": 2, "critic": 10, "ema": 0}
    summary = {"profiled": False, "profiler_status": "NOT_RUN", "cycles_in_same_process": 2,
               "final_versions": versions, "cycle_1": {"world_max_seconds": 100, "world_completion_endpoint_seconds": 100},
               "cycle_2": {"claimable": True, "world_max_seconds": 90, "seconds": 90}}
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    put(tmp_path / "bound-summary.json", summary)
    cv = _native_comparison_fixture()
    put(tmp_path / "compare.json", cv)
    preparation = {"source_head": "a" * 40, "candidate_id": c.id,
        "comparison": binding(tmp_path / "compare.json"), "correctness": binding(tmp_path / "bound-summary.json")}
    for node in (0, 1):
        sample = {"cgroup_events": {"oom": 0, "oom_kill": 0}, "cgroup_current_bytes": 100,
                  "gpus": [{"device": i, "free_bytes": 30} for i in range(8)]}
        put(tmp_path / f"search-resources.node{node}.json", {"status": "PASS", "node_rank": node,
            "run_id": receipt["run_id"], "samples": [sample, sample]})
        path = tmp_path / f"launcher/node_{node}"
        path.mkdir(parents=True)
        (path / "launch_manifest.txt").write_text("UNIT_TEST_NOT_GPU\n")
        (path / "teardown_status.txt").write_text("status=PASS\ngpu_process_count=0\nowner_run_id=UNIT_TEST_NOT_GPU\n")
    for rank in range(16):
        raw = {"rank": rank, "status": "PASS", "final_versions": versions,
               "finite_status": [True, True], "cycle_census": [{}, {}],
               "two_cycle_timing": {"per_cycle_contract": [{}, {}]},
               "memory": {"max_allocated_bytes": 10, "max_reserved_bytes": 12},
               "peak_memory": {"max_allocated_bytes": 10, "max_reserved_bytes": 12}}
        path = f"execution/ranks/rank_{rank:02d}.json" if workload == WAN else f"receipts/rank_{rank:03d}.json"
        put(tmp_path / path, raw)
    with patch("distillschedule.search.measurement.checked_summary", return_value=(tmp_path, summary)):
        result = normalize_native(receipt, preparation, identity)
        assert result["numerical_status"] == "REVIEW" and result["native_numerical_status"] == cv["status"]
        assert result["cycle_wall_seconds"] == {"1": 100, "2": 90}
        assert len(result["ranks"]) == 16
        cv["status"] = "FAIL"
        put(tmp_path / "compare.json", cv)
        preparation["comparison"] = binding(tmp_path / "compare.json")
        with pytest.raises(ValueError, match="numerical gate failed"):
            normalize_native(receipt, preparation, identity)


def test_saved_winner_requires_new_run_id_before_remote_access(tmp_path):
    from distillschedule.search.execution import rebind_plan
    r = resolve_candidate(WAN, "resident-tree")
    with pytest.raises(ValueError, match="fresh explicit"):
        rebind_plan(r, run_id=r["run_id"], coordinator_output=tmp_path / "should-not-exist")
    assert not (tmp_path / "should-not-exist").exists()
