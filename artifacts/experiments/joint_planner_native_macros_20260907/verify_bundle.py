"""CPU-only integrity and bounded-search replay. Does not launch GPU or SSH."""
import copy
import hashlib
import json
from pathlib import Path

from distillschedule.search.cost_model import CostModel
from distillschedule.search.model_search import search_plans
from distillschedule.search.plan_ir import PlanningProblem

root = Path(__file__).resolve().parent
hashes = json.loads((root / 'artifact_sha256.json').read_text())
for name, expected in hashes.items():
    path = (root / name).resolve()
    assert path.is_relative_to(root) and path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name

value = copy.deepcopy(json.loads((root / 'livewan/costs.json').read_text()))
for row in value['records']:
    original = Path(row['evidence']['source_receipt'])
    row['evidence']['source_receipt'] = str(root / 'livewan/receipts' / original.parent.name / original.name)
costs = CostModel.from_dict(value)
costs.verify_receipts()
# Replay the source-bound, already generated historical IR. Fresh lowering on
# a different current HEAD must (and does) reject these old calibration costs.
problem = PlanningProblem.from_dict(json.loads((root / 'livewan/problem.json').read_text()))
result = search_plans(problem, costs, seed=0, max_proposals=1000, timeout_seconds=60, exhaustive=True)
original = json.loads((root / 'livewan/search.json').read_text())
assert problem.digest == original['problem_sha256']
assert result['status'] == 'PASS_CPU_SEARCH'
for key in ('assignment', 'predicted_cycle1_ns', 'predicted_cycle2_ns'):
    assert result['predicted_best'][key] == original['predicted_best'][key], key
h3 = json.loads((root / 'h3/outcome.json').read_text())
assert h3['status'] == 'BLOCKED_NUMERICAL_GATE'
assert h3['cross_comparison']['status'] == h3['self_comparison']['status'] == 'FAIL'
assert h3['calibration'] == h3['native_timing_validation'] == 'NOT_RUN'
print(json.dumps({'status': 'PASS_PUBLISHED_CPU_REPLAY', 'files_verified': len(hashes),
                  'livewan_operation_costs': len(costs.records), 'h3': 'BLOCKED_NUMERICAL_GATE',
                  'new_gpu_execution': 'NOT_RUN', 'strict_winner': None}))
