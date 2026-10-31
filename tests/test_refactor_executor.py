from dataclasses import replace
from pathlib import Path
import pytest
from distillschedule.core.config import load_plan
from distillschedule.core.contracts import ContractViolation
from distillschedule.core.plan import ExecutionPlan
from distillschedule.runtime import DistillExecutor

ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path, monkeypatch, nodes=1):
    monkeypatch.setenv('DISTILLSCHEDULE_OUTPUT_DIR', str(tmp_path / 'run'))
    loaded = load_plan(ROOT / 'tests/fixtures/dmd_full_cycle.json')
    if nodes > 1:
        raw = loaded.plan.to_dict()
        raw['hardware'].update(nodes=nodes, world_size=nodes)
        raw['rank_groups'][0]['ranks'] = list(range(nodes))
        loaded = replace(loaded, plan=ExecutionPlan.from_mapping(raw))
    return loaded


def test_dry_run_writes_resolved_and_compiled_artifacts(tmp_path, monkeypatch):
    result = DistillExecutor().run(fixture(tmp_path, monkeypatch), repo_root=ROOT, dry_run=True)
    assert result.dry_run and result.return_code is None
    assert (result.output_dir / 'resolved_plan.json').is_file()
    assert (result.output_dir / 'compiled_run.json').is_file()
    assert result.command == ()


def test_multi_node_dry_run_writes_node_qualified_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv('DISTILLSCHEDULE_NODE_RANK', '1')
    result = DistillExecutor().run(fixture(tmp_path, monkeypatch, nodes=2), repo_root=ROOT, dry_run=True)
    assert (result.output_dir / 'resolved_plan.node1.json').is_file()
    assert (result.output_dir / 'compiled_run.node1.json').is_file()
    assert not (result.output_dir / 'resolved_plan.json').exists()


@pytest.mark.parametrize('rank', [None, '2', 'invalid'])
def test_multi_node_dry_run_rejects_missing_or_invalid_rank(tmp_path, monkeypatch, rank):
    monkeypatch.delenv('DISTILLSCHEDULE_NODE_RANK', raising=False)
    if rank is not None:
        monkeypatch.setenv('DISTILLSCHEDULE_NODE_RANK', rank)
    with pytest.raises(ContractViolation, match='DISTILLSCHEDULE_NODE_RANK'):
        DistillExecutor().run(fixture(tmp_path, monkeypatch, nodes=2), repo_root=ROOT, dry_run=True)
