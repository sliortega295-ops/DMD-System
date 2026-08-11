import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "distillschedule" / "evaluate_routeb_candidate_a_numerical_contract.py"
    spec = importlib.util.spec_from_file_location("routeb_numeric", path)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def _state(value=0.5):
    return {
        "master": {"relative_l2": value, "max_abs": value, "exact_fraction": 1.0 if value == 0 else 0.5},
        "exp_avg": {"relative_l2": value, "max_abs": value, "exact_fraction": 1.0 if value == 0 else 0.5},
        "exp_avg_sq": {"relative_l2": value, "max_abs": value, "exact_fraction": 1.0 if value == 0 else 0.5},
    }


def test_numerical_contract_passes_at_frozen_limits():
    module = _module()
    contract = {
        "status": "ACTIVE",
        "frozen_before_pipeline": True,
        "transition_manifest_sha256": "x",
        "frozen_limits": {
            "loss_max_abs": 1.0,
            "critic": {key: {"relative_l2": 1.0, "max_abs": 1.0} for key in ("master", "exp_avg", "exp_avg_sq")},
            "generator_all_state_exact": True,
        },
    }
    semantic = {"status": "PASS", "reference_run": "s", "candidate_run": "p", "numerical_diagnostics": [{"max_loss_abs": 0.5}]}
    state = {"critic": _state(0.5), "generator": _state(0.0)}
    assert module.evaluate(contract, semantic, state)["status"] == "PASS"


def test_numerical_contract_fails_closed_on_exact_or_state_drift():
    module = _module()
    contract = {
        "status": "ACTIVE",
        "frozen_before_pipeline": True,
        "transition_manifest_sha256": "x",
        "frozen_limits": {
            "loss_max_abs": 1.0,
            "critic": {key: {"relative_l2": 0.1, "max_abs": 0.1} for key in ("master", "exp_avg", "exp_avg_sq")},
            "generator_all_state_exact": True,
        },
    }
    semantic = {"status": "FAIL", "exact_contract_errors": ["x"], "numerical_diagnostics": [{"max_loss_abs": 2.0}]}
    state = {"critic": _state(0.5), "generator": _state(0.5)}
    result = module.evaluate(contract, semantic, state)
    assert result["status"] == "FAIL"
    assert len(result["errors"]) >= 3
