"""Synthetic-only credential boundaries; no authentication or GPU execution."""
from copy import deepcopy
import importlib
import json
from types import SimpleNamespace

import pytest


@pytest.fixture(params=["livewan_reference_assets", "h3_grid_training"])
def owner(request):
    return importlib.import_module("distillschedule.training." + request.param)


def explicit_environment(owner, extra):
    if owner.__name__.endswith("h3_grid_training"):
        value = {"H3_ADALN_GRID_MANIFEST": "/synthetic/grid.json",
                 "H3_FLASH_ATTN3_LOCAL_PATH": "/synthetic/fa3", **extra}
        config = {"training": {"a100": {"adaln_grid": {"manifest": "/synthetic/grid.json"}}}}
        return value, lambda: owner._environment(value, config)
    return extra, lambda: owner._environment(extra)


def test_snapshot_serialization_excludes_credentials_and_unknown_hub_keys(owner):
    credentials = {name: "SYNTHETIC_CREDENTIAL_NOT_REAL" for name in (
        "HF_TOKEN", "HF_HUB_TOKEN", "HF_STORED_TOKENS_PATH", "HF_CLIENT_SECRET",
        "TRANSFORMERS_TOKEN", "TRANSFORMERS_API_KEY", "NCCL_AUTH_TOKEN", "CUDA_PRIVATE_KEY")}
    controls = {"HF_HOME": "/synthetic/cache", "HF_HUB_CACHE": "/synthetic/cache/hub",
                "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                "TRANSFORMERS_CACHE": "/synthetic/transformers", "TRANSFORMERS_OFFLINE": "1",
                "OMP_NUM_THREADS": "1", "TORCH_DISTRIBUTED_DEBUG": "DETAIL"}
    snapshot = owner.environment_snapshot({**controls, **credentials,
        "HF_UNKNOWN_SETTING": "not-a-declared-control", "TRANSFORMERS_UNKNOWN_SETTING": "unknown"})
    assert snapshot == dict(sorted(controls.items()))
    assert list(snapshot) == sorted(snapshot)
    serialized = json.dumps({"runtime": {"environment": snapshot}})
    assert "SYNTHETIC_CREDENTIAL_NOT_REAL" not in serialized
    assert not set(credentials).intersection(snapshot)


@pytest.mark.parametrize("key", ["HF_TOKEN", "HF_HUB_TOKEN", "HF_STORED_TOKENS_PATH", "HF_CLIENT_SECRET",
    "TRANSFORMERS_TOKEN", "TRANSFORMERS_API_KEY", "NCCL_AUTH_TOKEN", "CUDA_PRIVATE_KEY", "AWS_SECRET_ACCESS_KEY"])
def test_explicit_environment_rejects_credentials_without_echoing_values(owner, key):
    value, admit = explicit_environment(owner, {key: "SYNTHETIC_CREDENTIAL_NOT_REAL"})
    before = deepcopy(value)
    with pytest.raises(ValueError, match="credential") as caught:
        admit()
    assert "SYNTHETIC_CREDENTIAL_NOT_REAL" not in str(caught.value)
    assert value == before


def test_token_rotation_does_not_change_checkpoint_identity(owner):
    def identity(token, *, offline="1"):
        environment = owner.environment_snapshot({"HF_TOKEN": token, "HF_HUB_OFFLINE": offline})
        if owner.__name__.endswith("h3_grid_training"):
            assets = {"environment": environment, "native_config": {"sha256": "a" * 64}}
            return owner.training_identity("b" * 40, assets, {"inventory_sha256": "c" * 64}, 3)
        from distillschedule.training.livewan_training import training_identity
        config = SimpleNamespace(trainer={"cycles": 3, "seed": 71, "checkpoint_every": 1})
        return training_identity("b" * 40, {"runtime": {"environment": environment}},
                                 {"input_sha256": "c" * 64}, config, {})
    baseline = identity("SYNTHETIC_TOKEN_BEFORE")
    assert identity("SYNTHETIC_TOKEN_AFTER") == baseline
    assert identity("SYNTHETIC_TOKEN_AFTER", offline="0") != baseline


@pytest.mark.parametrize("key", ["LIVEWAN_TRAJECTORY_CYCLES", "DMD_OPTIMIZE", "DISTILLSCHEDULE_PLAN",
    "NSYS_PROFILE", "NCU_PROFILE", "TORCH_COMPILE", "TORCH_LOGS", "LOCAL_KERNELS", "LD_PRELOAD",
    "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"])
def test_livewan_forbidden_controls_remain_visible_and_rejected(key):
    from distillschedule.training.livewan_reference_assets import environment_snapshot, _environment
    value = environment_snapshot({key: "1", "HF_TOKEN": "SYNTHETIC_CREDENTIAL_NOT_REAL"})
    assert value[key] == "1"
    with pytest.raises(ValueError, match="optimization/profiling/allocator overrides"):
        _environment(value)


@pytest.mark.parametrize("key", ["USE_TF", "USE_FLAX"])
def test_livewan_optional_framework_import_controls_are_bound(key):
    from distillschedule.training.livewan_reference_assets import environment_snapshot, _environment
    value = environment_snapshot({key: "0", "HF_TOKEN": "SYNTHETIC_CREDENTIAL_NOT_REAL"})
    assert value == {key: "0"}
    _environment(value)
    assert environment_snapshot({key: "1"}) != value


@pytest.mark.parametrize("key", ["H3_PLANNER_TWO_CYCLE", "DISTILLSCHEDULE_PLAN", "LOCAL_KERNELS"])
def test_h3_forbidden_controls_remain_visible_and_rejected(key):
    from distillschedule.training import h3_grid_training as owner
    value = owner.environment_snapshot({key: "1", "HF_TOKEN": "SYNTHETIC_CREDENTIAL_NOT_REAL"})
    assert value[key] == "1"
    _, admit = explicit_environment(owner, value)
    with pytest.raises(ValueError, match="benchmark/planner/optimized environment overrides"):
        admit()


def test_h3_required_grid_and_kernel_controls_still_admit_only_matching_paths():
    from distillschedule.training import h3_grid_training as owner
    value, admit = explicit_environment(owner, {"HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"})
    before = deepcopy(value)
    admit()
    assert value == before
    value["H3_ADALN_GRID_MANIFEST"] = "/synthetic/different.json"
    with pytest.raises(ValueError, match="matching Grid table"):
        admit()
