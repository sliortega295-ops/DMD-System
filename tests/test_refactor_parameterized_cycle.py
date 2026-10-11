"""Only CPU action execution is established for the new parameterized protocol."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from distillschedule.runtime.parameterized_cycle import build_parameterized_cycle_graph,parameterized_window_order
from distillschedule.runtime.cycle_plan import build_cycle_graph,rollout_window_order
from distillschedule.search.plan_ir import validate_versions
from test_refactor_prepared_cycle import INITIAL,runner


@pytest.mark.parametrize('n',[1,2,3,5,8])
@pytest.mark.parametrize('ema',['none','first','all'])
def test_n_update_program_runs_common_prepared_runner(n,ema):
    positions=() if ema=='none' else (0,) if ema=='first' else tuple(range(n))
    g=build_parameterized_cycle_graph(source_steps=tuple(range(n)),initial_versions=INITIAL,ema_after=positions)
    validate_versions(g)
    for window in sorted({1,n}):
        order=parameterized_window_order(g,max_live_rollouts=window)
        events=[];r=runner(events);p=r.prepare(g,order=order,retain_outputs=tuple(f'c{i}' for i in range(n)))
        p.run()
        assert len(events)==1+2*n+len(positions)
        assert p.receipt()['peak_live_inputs']==window
        assert r.versions.snapshot().values==dict(generator=3,critic=10+n,ema=10+len(positions))
        assert p.receipt()['executed_operations']==list(order)


def test_legacy_full5_graph_is_not_relaxed_and_v2_preserves_math():
    a=build_cycle_graph(source_steps=(10,11,12,13,14),initial_versions=INITIAL,ema_after=(0,1,2,3,4))
    b=build_parameterized_cycle_graph(source_steps=(10,11,12,13,14),initial_versions=INITIAL,ema_after=(0,1,2,3,4))
    assert a.operations==b.operations and a.dependencies==b.dependencies
    for w in range(1,6):
        assert rollout_window_order(a,max_live_rollouts=w)==parameterized_window_order(b,max_live_rollouts=w)
    with pytest.raises(ValueError):build_cycle_graph(source_steps=(0,),initial_versions=INITIAL,ema_after=())


@pytest.mark.parametrize('steps',[(),(0,0),(-1,),(True,),tuple(range(65))])
def test_invalid_count_does_not_silently_change_training(steps):
    with pytest.raises(ValueError):build_parameterized_cycle_graph(source_steps=steps,initial_versions=INITIAL)


def test_unsupported_precommit_protocol_is_not_guessed():
    with pytest.raises(ValueError,match='protocol'):
        build_parameterized_cycle_graph(source_steps=(0,),initial_versions=INITIAL,protocol='livewan')
    g=build_parameterized_cycle_graph(source_steps=(0,),initial_versions=INITIAL)
    with pytest.raises(ValueError):parameterized_window_order(g,max_live_rollouts=5)


def test_public_cycle_frontdoor_selects_and_executes_cpu_actions_under_memory_limit(tmp_path):
    from distillschedule.search.plan_ir import graph_from_dict
    root=Path(__file__).parents[1];out=tmp_path/'plan.json'
    cmd=[sys.executable,str(root/'scripts/distillctl.py'),'workloads','search','--strategy','cycle-frontier','--problem',
         str(root/'configs/distillschedule/planner_examples/cycle_g1f3_cpu.json'),'--output',str(out)]
    result=subprocess.run(cmd,capture_output=True,text=True,cwd=root)
    assert result.returncode==0,result.stderr
    doc=json.loads(out.read_text());g=graph_from_dict(doc['program'])
    assert doc['evidence_scope']=='CPU_FIXTURE_ONLY' and doc['gpu_execution']=='NOT_RUN'
    r=runner([])
    from distillschedule.runtime.version_sync import VersionLedger
    r.versions=VersionLedger(g.metadata['initial_versions'])
    prepared=r.prepare(g,order=tuple(doc['order']))
    prepared.run()
    assert prepared.receipt()['peak_live_inputs']==1
    assert prepared.receipt()['final_versions']==g.metadata['final_versions']
    assert subprocess.run(cmd,capture_output=True,text=True,cwd=root).returncode!=0  # immutable output


def test_cpu_contract_cannot_claim_native_costs_or_unimplemented_shape_support():
    from distillschedule.search.cycle_cli import plan_cycle
    root=Path(__file__).parents[1]
    doc=json.loads((root/'configs/distillschedule/planner_examples/cycle_g1f3_cpu.json').read_text())
    with pytest.raises(ValueError):plan_cycle({**doc,'scope':'MEASURED'})
    with pytest.raises(ValueError):plan_cycle({**doc,'resolution':[720,1280]})
    with pytest.raises(ValueError):plan_cycle({**doc,'action_costs':{'g':doc['action_costs']['g']}})
