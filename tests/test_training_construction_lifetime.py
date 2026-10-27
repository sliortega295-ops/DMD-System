"""CPU_TEST_ONLY: real provider hooks and caller-owned Gloo construction lifetime.

These are lifecycle regressions, not GPU math, native-model or quality evidence.
"""
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist

from distillschedule.open_workload.torch_backend import signature
from distillschedule.training import session
from distillschedule.training.config import TrainingConfig
from test_training_control_admission import _fixture
from test_training_policy_continuation import _run_workers


class _HookProvider:
    """Own actual module/optimizer hooks without replacing any numerical call."""

    def __init__(self, components, *, close_error=None):
        self.components = components
        self.model = components.state["student"]
        self.before = signature(components.state)
        self.close_calls = self.forward_calls = 0
        self.updates = {name: 0 for name in ("student_optimizer", "fake_optimizer")}
        self.close_error, self.original_close = close_error, components.close
        self.handles = [self.model.register_forward_pre_hook(self._forward)]
        for name in self.updates:
            def updated(*_args, name=name):
                self.updates[name] += 1
            self.handles.append(components.state[name].register_step_post_hook(updated))
        components.close = self.close

    def _forward(self, *_args):
        self.forward_calls += 1

    def close(self):
        self.close_calls += 1
        for handle in self.handles:
            handle.remove()
        if self.original_close is not None:
            self.original_close()
        if self.close_error is not None:
            raise self.close_error

    def probe(self):
        with torch.no_grad():
            self.model(torch.zeros(1, 4, 2, 4, 4), torch.zeros(1, 4, 1, 1, 1))

    def report(self):
        forwards = self.forward_calls
        unchanged = signature(self.components.state) == self.before
        self.probe()
        return {"close_calls": self.close_calls, "forwards": forwards,
                "post_close_hook_calls": self.forward_calls - forwards,
                "module_hooks": len(self.model._forward_pre_hooks),
                "optimizer_hooks": [len(self.components.state[name]._optimizer_step_post_hooks)
                                    for name in self.updates],
                "updates": self.updates, "state_unchanged": unchanged}


def _track_factory(patch, config, providers, *, invalid=False, close_error=None):
    original = session.load_callable

    def load(reference):
        factory, source = original(reference)
        if reference != config.model["factory"]:
            return factory, source

        def build(parameters):
            components = factory(parameters)
            providers.append(_HookProvider(components, close_error=close_error))
            if invalid:
                components.prediction = "CPU_TEST_ONLY: invalid returned prediction"
            return components
        return build, source

    patch.setattr(session, "load_callable", load)


@pytest.fixture
def local_config(tmp_path):
    config, raw = _fixture(tmp_path / "media")
    raw["trainer"].pop("distributed")
    raw["data"].pop("sharding")
    raw["trainer"].update(cycles=1, output_dir=str(tmp_path / "never-training"))
    return TrainingConfig.from_dict(raw, source=config.source)


def _assert_released(provider):
    assert provider["close_calls"] == 1, provider
    assert provider["module_hooks"] == 0 and provider["optimizer_hooks"] == [0, 0], provider
    assert provider["post_close_hook_calls"] == 0, provider


@pytest.mark.parametrize("include_validation", [False, True])
def test_returned_validation_failure_releases_real_provider_once(local_config, monkeypatch, include_validation):
    providers = []
    _track_factory(monkeypatch, local_config, providers, invalid=True)
    with pytest.raises(ValueError, match="prediction must be an explicit PredictionSpec"):
        session.construct(local_config, include_validation=include_validation)
    assert len(providers) == 1
    observed = providers[0].report()
    _assert_released(observed)
    assert observed["state_unchanged"] and not any(observed["updates"].values())
    assert not Path(local_config.trainer["output_dir"]).exists()


@pytest.mark.parametrize("close", [False, 17, "not-callable"])
def test_noncallable_close_preserves_original_validation_error(local_config, monkeypatch, close):
    original = session.load_callable

    def load(reference):
        factory, source = original(reference)

        def build(parameters):
            components = factory(parameters)
            components.close = close
            return components
        return build, source

    monkeypatch.setattr(session, "load_callable", load)
    with pytest.raises(ValueError, match="model close must be callable"):
        session._construct_model(local_config)


@pytest.mark.parametrize("stage", ["validation", "after_factory", "managed"])
def test_cleanup_callback_exception_policy_is_unchanged(local_config, monkeypatch, stage):
    providers, close_error = [], OSError("CPU_TEST_ONLY: close callback failed")
    _track_factory(monkeypatch, local_config, providers, invalid=stage == "validation", close_error=close_error)
    config = local_config
    if stage == "after_factory":
        raw = config.to_dict()
        raw["algorithm"]["factory"] = "distillschedule.examples.tiny_av:missing_algorithm"
        config = TrainingConfig.from_dict(raw, source=config.source)
    with pytest.raises(OSError) as caught:
        result = session.construct(config)
        with result[0].managed():
            raise RuntimeError("CPU_TEST_ONLY: owned scope failed")
    assert caught.value is close_error
    assert caught.value.__context__ is not None
    _assert_released(providers[0].report())


def _peer_finished(root, rank, name):
    (root / f"{name}-{rank}").touch()
    deadline = time.monotonic() + 15
    while not (root / f"{name}-{1-rank}").exists():
        if time.monotonic() >= deadline:
            raise AssertionError("peer did not leave the public invocation within 15 seconds")
        time.sleep(.02)


def _probe_group():
    initialized = dist.is_initialized()
    probe, error = torch.tensor(dist.get_rank() + 1), None
    try:
        dist.all_reduce(probe)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {"initialized": initialized, "error": error, "sum": probe.item()}


def _construction_worker(rank, rendezvous, output, source, raw, command, fault):
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank),
                      LOCAL_WORLD_SIZE="2", GROUP_RANK="0")
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=10))
    root, providers = Path(output), []
    raw = deepcopy(raw)
    raw["trainer"].update(cycles=1, output_dir=str(root / "never-training"))
    if rank == 1 and fault == "before_model":
        raw["data"]["prepared"] = str(root / "missing-prepared.json")
    if rank == 1 and fault == "after_factory":
        raw["algorithm"]["factory"] = "distillschedule.examples.tiny_av:missing_algorithm"
    config = TrainingConfig.from_dict(raw, source=source)
    try:
        with pytest.MonkeyPatch.context() as patch:
            _track_factory(patch, config, providers, invalid=rank == 1 and fault == "validation")
            error = result = None
            started = time.monotonic()
            try:
                if command == "train":
                    result = session.train(config)
                else:
                    result = getattr(session, command)(config, root / "missing-checkpoint", root / "never-output")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - started
        observed = [provider.report() for provider in providers]
        # A separate filesystem rendezvous prevents the test's recovery probe
        # from accidentally matching a peer still in a production collective.
        _peer_finished(root, rank, "invocation-finished")
        group = _probe_group()
        (root / f"rank-{rank}.json").write_text(json.dumps({
            "status": "CPU_TEST_ONLY", "error": error, "elapsed": elapsed, "result": result,
            "providers": observed, "group": group,
            "training_output_exists": Path(config.trainer["output_dir"]).exists(),
            "output_exists": (root / "never-output").exists()}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("command", ["train", "evaluate", "export"])
@pytest.mark.parametrize("fault", ["before_model", "validation", "after_factory"])
def test_asymmetric_construction_failure_releases_all_returned_providers(tmp_path, command, fault):
    config, raw = _fixture(tmp_path / "media")
    rows = _run_workers(tmp_path / "workers", _construction_worker, str(config.source), raw, command, fault)
    label = "training construction" if command == "train" else "output construction"
    message = {"before_model": "missing-prepared.json", "validation": "prediction must be an explicit PredictionSpec",
               "after_factory": "adapter is not callable"}[fault]
    assert rows[0]["error"] == rows[1]["error"], rows
    for rank, row in enumerate(rows):
        assert row["error"] and label in row["error"] and message in row["error"], rows
        assert row["result"] is None and row["elapsed"] < 8, row
        assert row["group"] == {"initialized": True, "error": None, "sum": 3}, row
        assert not row["training_output_exists"] and not row["output_exists"], row
        assert len(row["providers"]) == (0 if rank == 1 and fault == "before_model" else 1), row
        for provider in row["providers"]:
            _assert_released(provider)
            assert provider["state_unchanged"] and provider["forwards"] == 0, provider
            assert not any(provider["updates"].values()), provider


def _success_worker(rank, rendezvous, output, source, raw):
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank),
                      LOCAL_WORLD_SIZE="2", GROUP_RANK="0")
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=10))
    root, providers, restore_live, export_live = Path(output), [], [], []
    raw = deepcopy(raw)
    raw["trainer"].update(cycles=1, checkpoint_every=1, output_dir=str(root / "training"))
    config = TrainingConfig.from_dict(raw, source=source)
    try:
        with pytest.MonkeyPatch.context() as patch:
            _track_factory(patch, config, providers)
            original_restore = session._restore_for_output

            def restore(*args, **kwargs):
                assert providers[-1].close_calls == 0
                restored = original_restore(*args, **kwargs)
                restore_live.append(providers[-1].close_calls == 0)
                return restored

            patch.setattr(session, "_restore_for_output", restore)
            from distillschedule.training import inference
            original_export = inference.publish_export

            def publish(*args, **kwargs):
                assert providers[-1].close_calls == 0
                exported = original_export(*args, **kwargs)
                export_live.append(providers[-1].close_calls == 0)
                return exported

            patch.setattr(inference, "publish_export", publish)
            trained = session.train(config)
            session.evaluate(config, trained["checkpoint"], root / "evaluation")
            session.export(config, trained["checkpoint"], root / "weights")
            if rank == 0:
                # Public construct must stay usable without a peer collective.
                components, *_ = session.construct(config)
                assert providers[-1].close_calls == 0
                with components.managed():
                    providers[-1].probe()
                    assert providers[-1].forward_calls == 1 and providers[-1].close_calls == 0
                (root / "direct-construction-finished").touch()
            else:
                deadline = time.monotonic() + 15
                while not (root / "direct-construction-finished").exists():
                    if time.monotonic() >= deadline:
                        raise AssertionError("public construct unexpectedly required its peer")
                    time.sleep(.02)
        _peer_finished(root, rank, "success-finished")
        group = _probe_group()
        (root / f"rank-{rank}.json").write_text(json.dumps({
            "status": "CPU_TEST_ONLY", "providers": [provider.report() for provider in providers],
            "group": group, "restore_live": restore_live, "export_live": export_live,
            "trained": trained["status"], "cycles": trained["completed_cycles"]}))
    finally:
        dist.destroy_process_group()


def test_success_transfers_live_provider_to_training_and_output_managed_scopes(tmp_path):
    config, raw = _fixture(tmp_path / "media")
    rows = _run_workers(tmp_path / "workers", _success_worker, str(config.source), raw)
    for rank, row in enumerate(rows):
        assert row["trained"] == "COMPLETE" and row["cycles"] == 1, row
        assert row["restore_live"] == [True, True] and row["export_live"] == [True], row
        assert row["group"] == {"initialized": True, "error": None, "sum": 3}, row
        assert len(row["providers"]) == (4 if rank == 0 else 3), row
        for provider in row["providers"]:
            _assert_released(provider)
        trained, evaluated, exported = row["providers"][:3]
        assert trained["forwards"] > 0 and trained["updates"] == {"student_optimizer": 1, "fake_optimizer": 2}
        assert evaluated["forwards"] > 0 and not any(evaluated["updates"].values())
        assert not any(exported["updates"].values())
    for relative in ("training/summary.json", "training/checkpoints/cycle-000001/_SUCCESS",
                     "evaluation/_SUCCESS", "weights/_SUCCESS"):
        assert (tmp_path / "workers" / relative).is_file()
