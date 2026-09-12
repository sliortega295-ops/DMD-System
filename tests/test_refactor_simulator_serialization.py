"""Regression from world16 control-plane admission; CPU fixture only."""
from dataclasses import replace
import json

from test_refactor_model_planning import fixture
from distillschedule.search.plan_ir import PlanningProblem
from distillschedule.search.model_search import search_plans, validate_saved_prediction
from distillschedule.search.simulator import simulate


def test_world16_sorted_json_roundtrip_preserves_prediction():
    problem, costs, assignment = fixture()
    problem = replace(problem, memory_limits={rank: 100 for rank in range(16)})
    restored = PlanningProblem.from_dict(json.loads(json.dumps(problem.to_dict(), sort_keys=True)))
    assert restored.digest == problem.digest
    assert simulate(problem, assignment, costs) == simulate(restored, assignment, costs)
    result = search_plans(problem, costs, exhaustive=True, max_proposals=8)
    validate_saved_prediction(json.loads(json.dumps(result, sort_keys=True)), restored, costs)
