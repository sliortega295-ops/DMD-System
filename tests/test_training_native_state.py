"""Native engine checkpoint state on real CPU modules/Adam, without model stubs."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter


def _engine():
    torch.manual_seed(7)
    module = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 2))
    module[0].requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.7)
    return FSDP2EngineAdapter(module, optimizer, scheduler, physical_step=0, global_batch=2)


def _update(engine):
    engine.backward(engine(torch.ones(2, 2)).square().mean())
    engine.step()


def test_native_engine_continuation_retains_parameter_objects_moments_and_scheduler(tmp_path):
    from distillschedule.training.native_state import engine_state
    from distillschedule.training.checkpoint import save_checkpoint, restore_checkpoint, pack_state
    from test_training_end_to_end import equal
    control = _engine()
    _update(control)
    state = engine_state({"generator": control})
    checkpoint = save_checkpoint(tmp_path / "native", state, identity={"native": "CPU_TEST_ONLY"},
                                 cycle=1, cursor=1, versions={"generator": 1})
    packed = pack_state(state)
    # Frozen base is reconstructed from validated asset identity, not multiplied
    # into every rank's training-state checkpoint.
    assert set(packed["generator.weights"]["value"]["tensors"]) == {"1.weight", "1.bias"}
    restored = _engine()
    parameter_ids = [id(p) for p in restored.module.parameters()]
    restore_checkpoint(checkpoint, engine_state({"generator": restored}), identity={"native": "CPU_TEST_ONLY"})
    assert parameter_ids == [id(p) for p in restored.module.parameters()]
    assert restored.global_steps == 1 and restored.global_samples == 2
    assert restored._step_calls == 1
    _update(control); _update(restored)
    assert equal(control.module.state_dict(), restored.module.state_dict())
    assert equal(control.optimizer.state_dict(), restored.optimizer.state_dict())
    assert equal(control.lr_scheduler.state_dict(), restored.lr_scheduler.state_dict())


def test_native_shadow_and_invalid_optimizer_are_validated_before_mutation(tmp_path):
    from distillschedule.training.native_state import engine_state, EMAState
    from distillschedule.training.checkpoint import pack_state, unpack_state
    from test_training_end_to_end import equal
    engine = _engine(); _update(engine)
    names, parameters = zip(*[(name, p) for name, p in engine.module.named_parameters() if p.requires_grad])
    ema = SimpleNamespace(_names=list(names), _parameters=list(parameters),
                          _shadows=[p.detach().clone() for p in parameters],
                          optimization_step=1, decay=.999, min_decay=0., update_after_step=0,
                          cur_decay_value=0., device="cpu")
    state = engine_state({"generator": engine}, ema=ema)
    before = deepcopy(engine.module.state_dict())
    packed = pack_state(state)
    packed["generator.optimizer"]["value"]["parameters"][0].reverse()
    with pytest.raises(ValueError, match="ownership"):
        unpack_state(packed, state)
    assert equal(before, engine.module.state_dict())
    snapshot = deepcopy(EMAState(ema).state_dict())
    with torch.no_grad():
        ema._shadows[0].zero_()
    EMAState(ema).load_state_dict(snapshot)
    assert torch.equal(ema._shadows[0], snapshot["shadows"][0])


def test_native_optimizer_rejects_raw_moment_cast_before_any_live_mutation():
    from distillschedule.training.native_state import engine_state
    from distillschedule.training.checkpoint import pack_state, unpack_state, rng_state
    from test_training_end_to_end import equal
    engine = _engine(); _update(engine)
    state = engine_state({"generator": engine})
    before = pack_state(state)
    random_before = rng_state()
    parameter_ids = [id(p) for p in engine.module.parameters()]
    malformed = deepcopy(before)
    # If admission is delayed until after AdamW.load_state_dict, this source
    # dtype is silently cast back to FP32 and the malformed shard is accepted.
    saved = malformed["generator.optimizer"]["value"]["optimizer"]
    first = next(iter(saved["state"]))
    saved["state"][first]["exp_avg"] = saved["state"][first]["exp_avg"].double()
    with pytest.raises(ValueError, match="shape/dtype"):
        unpack_state(malformed, state)
    assert equal(before, pack_state(state))
    assert equal(random_before, rng_state())
    assert parameter_ids == [id(p) for p in engine.module.parameters()]


def test_native_optimizer_raw_validation_does_not_impose_fixed_w4_policy():
    from distillschedule.training.native_state import OptimizerState
    engine = _engine(); _update(engine)
    saved = deepcopy(OptimizerState(engine).state_dict())
    # Scheduled native engines restore the saved group policy. W4's stricter
    # fixed AdamW options/scalar-step representation do not belong here.
    saved["optimizer"]["param_groups"][0]["lr"] = .0003
    for slots in saved["optimizer"]["state"].values():
        slots["step"] = slots["step"].double()
    OptimizerState(engine).load_state_dict(saved)
    assert engine.optimizer.param_groups[0]["lr"] == .0003
    assert all(slots["step"].dtype == torch.float64 for slots in engine.optimizer.state.values())


def test_actual_resident_clocks_and_unnamed_ema_owner_are_supported():
    from distillschedule.runtime.wan13b_upper_bound import ReplicatedTrainableAllReduceEngine
    from distillschedule.training.native_state import engine_state
    original = _engine()
    engine = ReplicatedTrainableAllReduceEngine(original.module, original.optimizer,
                                                original.lr_scheduler, physical_step=0)
    parameters = [p for p in engine.module.parameters() if p.requires_grad]
    # FreshTrainableEMA's actual field surface has neither _names nor device.
    ema = SimpleNamespace(_parameters=parameters, _shadows=[p.detach().clone() for p in parameters],
        optimization_step=0, decay=.999, min_decay=0., update_after_step=0, cur_decay_value=None,
        implementation="reference")
    state = engine_state({"generator": engine}, ema=ema)
    assert state["generator.clocks"].state_dict()["policy"]["global_batch"] == 16
    assert state["ema"].state_dict()["names"] == ["1.weight", "1.bias"]
    state["generator.clocks"].validate_checkpoint_boundary()
    parameters[0].grad = torch.ones_like(parameters[0])
    with pytest.raises(ValueError, match="unconsumed gradients"):
        state["generator.clocks"].validate_checkpoint_boundary()


def test_original_dispatcher_checkpoint_resume_matches_uninterrupted_cpu_engine_state(tmp_path, monkeypatch):
    """Actual native dispatcher + CPU Adam; scalar helpers are not Wan quality."""
    from distillschedule.strategies.wan_bridge import OriginalWanSession
    from distillschedule.strategies.wan_task import OriginalWanTask
    from distillschedule.strategies.wan_strategy import WanStrategy, strategy_environment
    from distillschedule.strategies.wan_stream import task_at, bind_cohort, restore_stream_cursor
    from distillschedule.strategies.wan_worker import execute_cycles
    from distillschedule.training.native_state import engine_state
    from distillschedule.training.checkpoint import save_checkpoint, restore_checkpoint, pack_state
    from test_training_end_to_end import equal
    import distillschedule.strategies.wan_trace as traces

    task = OriginalWanTask(fake_updates=1, first_step=2, lora_rank=64)
    strategy = WanStrategy('resident', 512, False, 1)
    stream = {'stream_sha256': 'CPU_HELPER_CHECKPOINT_TEST_ONLY', 'cohorts': [
        {'task': task_at(task, i).to_dict(), 'trace_bundle': str(i), 'trace_hash': str(i)} for i in range(3)]}
    monkeypatch.setattr(traces, 'load_original_trace', lambda t, *a, **k: SimpleNamespace(
        variant='no-gan', logical_steps=t.source_steps))

    def build():
        g, c = _engine(), _engine()
        names, parameters = zip(*[(name, p) for name, p in g.module.named_parameters() if p.requires_grad])
        ema = SimpleNamespace(_names=list(names), _parameters=list(parameters),
            _shadows=[p.detach().clone() for p in parameters], optimization_step=0,
            decay=.9, min_decay=0., update_after_step=0, cur_decay_value=None, device='cpu')
        def gupdate(**kwargs):
            _update(g); return {}
        def rollout(**kwargs):
            step = kwargs['steps'][0]
            kwargs['entry_outputs'][step] = SimpleNamespace(tensors={})
            return {step: g(torch.ones(1, 2)).detach() + torch.randn(1, 2)}, [{'batch_size': 1}], {}
        def critic(**kwargs):
            c.backward((c(torch.ones(1, 2)) - kwargs['x0']).square().mean()); c.step(); return {}
        def estep():
            from distillschedule.runtime.fsdp2_static import ema_update_reference
            for shadow, parameter in zip(ema._shadows, ema._parameters):
                ema_update_reference(shadow, parameter.detach(), .9)
            ema.optimization_step += 1; ema.cur_decay_value = .9
        ema.step = estep
        p0 = SimpleNamespace(_run_nogan_generator_update_p0=gupdate, _precompute_critic_rollouts=rollout,
                             _run_nogan_critic_precomputed=critic)
        session = OriginalWanSession(task=task, strategy=strategy,
            fair=SimpleNamespace(base=SimpleNamespace(_build_engines=lambda **k: (g, c, ema, {}), p0=p0)),
            bootstrap=SimpleNamespace(manifest={'lora_rank': 64}), model_root='CPU_TEST_ONLY', device='cpu', rank=0,
            trace=SimpleNamespace(variant='no-gan', logical_steps=task.source_steps), negative=None, diagnostics=None)
        return session, engine_state({'generator': g, 'critic': c}, ema=ema)

    stub_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda d: None))
    stub_dist = SimpleNamespace(barrier=lambda: None)
    identity = {'fixture': 'CPU_NATIVE_DISPATCH_NOT_PRETRAINED_WAN'}
    before = lambda cycle, current: bind_cohort(current, task, stream, cycle-1, ('g','r0','c0','e0'))
    with strategy_environment(strategy):
        control, state = build()
        def save(cycle, current, record):
            if cycle == 1:
                save_checkpoint(tmp_path / 'checkpoint', state, identity=identity, cycle=1, cursor=1,
                                versions=current.versions(), policy={'stream': stream['stream_sha256']})
        execute_cycles(control, cycles=3, mode='run', torch=stub_torch, distributed=stub_dist,
                       before_cycle=before, after_cycle=save)
        expected = pack_state(state)
        resumed, state = build()
        restored = restore_checkpoint(tmp_path / 'checkpoint', state, identity=identity)
        assert resumed.versions() == restored['versions']
        restore_stream_cursor(resumed, task, stream, completed_cycles=1)
        execute_cycles(resumed, cycles=3, start_cycle=1, mode='run', torch=stub_torch,
                       distributed=stub_dist, before_cycle=before)
        assert equal(pack_state(state), expected)
