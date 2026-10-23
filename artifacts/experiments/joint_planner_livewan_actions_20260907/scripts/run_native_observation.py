"""Run the single missing action/transport observation, not an old matrix."""
import json
from pathlib import Path
from distillschedule.search.execution import run_plan
E=Path(__file__).resolve().parents[1]
plan=json.loads((E/'configs/native-observation-plan.json').read_text())
assert plan['mode']=='action-observation' and plan['native_mode']=='trajectory'
assert plan['integration_head']=='78e501bc1727013d94e356f697051b4d11cba4be'
assert all(n['environment']['DISTILLSCHEDULE_LIVEWAN_OPERATION_CALIBRATION']=='1' for n in plan['nodes'])
run_plan(plan,coordinator_output=E/'native-observation-dispatch')
