"""Reuse prior actual-draw evidence, never promote a sampled macro receipt.

This admits diagnostic performance only. Unchanged native mathematics/helpers,
assets, depth/tile, and all sixteen original action receipts must be supplied.
The new metadata-only run still checks actual calls/samples/updates/RNG itself.
"""
import json
import hashlib
import subprocess
from pathlib import Path

from distillschedule.workloads.best_known.registry import file_sha256
from .contracts import require
from .h3_implementation import H3Implementation


def core_paths(root):
    base = root / "src/distillschedule/backends/training/minimax_h3"
    ignored = {"cycle_actions.py", "planner_cycle_observer.py"}
    return sorted(str(p.relative_to(root)) for p in base.rglob("*.py") if p.name not in ignored) + [
        "src/distillschedule/runtime/cycle_plan.py", "src/distillschedule/runtime/role_runner.py",
        "src/distillschedule/runtime/version_sync.py"]


def validate_action_admission(data, receipt, root):
    from .h3_native import summarize_native_two_cycle
    proof = data.get("h3_action_proof", {})
    require(proof.get("schema_version") == "distillschedule.h3_prior_action_proof.v1",
            "H3 v2 timing/calibration requires prior native action proof, not legacy macro correctness")
    require(proof.get("asset_fingerprint") == receipt["asset_fingerprint"], "H3 action proof assets changed")
    core = proof.get("native_core_sha256", {})
    require(set(core) == set(core_paths(root)), "H3 native source proof coverage changed")
    require(all(file_sha256(root/p) == sha for p,sha in core.items()), "H3 native core differs from action proof")
    for path, sha in core.items():
        old = subprocess.check_output(['git','-C',str(root),'show',f"{proof['source_head']}:{path}"])
        require(hashlib.sha256(old).hexdigest()==sha,"H3 source bridge differs from the actually tested Git object")
    candidate = H3Implementation.parse(receipt["search_candidate"]["id"])
    source_impl = H3Implementation.parse(proof["candidate_id"])
    require((candidate.prefetch_depth,candidate.lora_epilogue_block_n)==
            (source_impl.prefetch_depth,source_impl.lora_epilogue_block_n),
            "H3 kernel/cache policy requires its own action proof")
    references = proof.get("ranks", [])
    require(len(references)==16 and len({r['path'] for r in references})==16, "H3 action proof requires sixteen sources")
    rows = []
    for ref in references:
        path=Path(ref['path'])
        require(path.is_file() and file_sha256(path)==ref['sha256'], "H3 action proof rank hash changed")
        row=json.loads(path.read_text())
        require(row.get('calibration_only') is False, "calibration cannot replace action correctness")
        require(all(c.get('action_execution',{}).get('tensor_value_reads') ==
                    'CORRECTNESS_ONLY_SYNCHRONIZING_FULL_DRAW_HASHES' for c in row['cycles']),
                "H3 action proof must contain actual full stochastic draw hashes")
        rows.append(row)
    summary=summarize_native_two_cycle(rows,expected_head=proof['source_head'],candidate_id=proof['candidate_id'])
    require(summary['status']=='PASS', "H3 prior native action proof failed")
    return dict(status="PASS_PRIOR_NATIVE_ACTION_PROOF", source_head=proof['source_head'],
        actual_draw_values_current_run="NOT_RECHECKED", numerical_cross_reference="REVIEW_NOT_STRICT_PASS",
        source_bridge="NATIVE_CORE_BYTE_IDENTICAL; OBSERVER_AND_CALIBRATION_CHANGES_REVIEWED_SEPARATELY")
