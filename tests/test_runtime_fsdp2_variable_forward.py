"""Control/cleanup tests; real FSDP/PEFT integration has a separate CPU lane."""
from types import SimpleNamespace as NS

import pytest

from distillschedule.runtime.fsdp2_variable_forward import fsdp2_variable_forward_phase


class Unit:
    def __init__(self, events, name):
        self.events, self.name = events, name
        self.group = NS(is_unsharded=False, _reshard_after_forward=True, reshard_after_backward=True)
        self.units = [self]

    def modules(self):
        return iter(self.units)

    def _get_fsdp_state(self):
        return NS(_fsdp_param_group=self.group)

    def set_reshard_after_forward(self, value, *, recurse):
        self.events.append((self.name, 'forward_policy', value, recurse))
        self.group._reshard_after_forward = value

    def set_reshard_after_backward(self, value, *, recurse):
        self.events.append((self.name, 'backward_policy', value, recurse))
        self.group.reshard_after_backward = value

    def unshard(self, *, async_op):
        self.events.append((self.name, 'unshard', async_op))
        self.group.is_unsharded = True

    def reshard(self):
        self.events.append((self.name, 'reshard'))
        self.group.is_unsharded = False


class Engine:
    def __init__(self):
        self.events = []
        self.module = Unit(self.events, 'root')
        self.module.units.append(BlockUnit(self.events, 'block'))
        self.optimizer = NS(param_groups=[dict(params=[object()])])
        self.global_steps = 0

    def __call__(self, value):
        self.events.append(('forward', value))
        return value

    def backward(self, loss):
        self.events.append(('backward', loss))

    def step(self):
        assert not any(u.group.is_unsharded for u in self.module.units)
        self.events.append(('original_clip_adam_scheduler_zero_grad',))
        self.global_steps += 1


def scope(engine, **kwargs):
    return fsdp2_variable_forward_phase(engine, enabled=True, expected_units=2, module_type=Unit,
                                       checkpoint_block_type=BlockUnit, **kwargs)


class BlockUnit(Unit):
    pass


def test_materializes_before_calls_releases_before_original_step_and_restores_policy():
    engine = Engine()
    with scope(engine, with_grad=True) as (view, receipt):
        assert view.module is engine.module and view.global_steps == 0
        assert receipt['unshard_count'] == 2
        view(1); view(2); view.backward('loss'); view.step()
        assert view.global_steps == 1
        assert not any(u.group.is_unsharded for u in engine.module.units)
        with pytest.raises(ValueError, match='released'):
            view(3)
    assert receipt['backward_count'] == receipt['commit_count'] == 1
    assert receipt['reshard_count'] == 2 and receipt['restored'] is True
    assert engine.events[:6] == [('root', 'forward_policy', False, False),
        ('root', 'backward_policy', False, False), ('block', 'forward_policy', False, False),
        ('block', 'backward_policy', False, False), ('root', 'unshard', False), ('block', 'unshard', False)]
    assert engine.events[-1] == ('original_clip_adam_scheduler_zero_grad',)
    assert ('root', 'forward_policy', True, False) in engine.events
    assert ('block', 'backward_policy', True, False) in engine.events


def test_no_grad_phase_has_no_update_and_releases_at_exit():
    engine = Engine()
    with scope(engine, with_grad=False) as (view, receipt):
        for value in range(7):
            view(value)
        with pytest.raises(ValueError, match='no-grad'):
            view.backward('loss')
        with pytest.raises(ValueError, match='no-grad'):
            view.step()
    assert receipt['commit_count'] == receipt['backward_count'] == 0
    assert receipt['restored'] and engine.global_steps == 0


@pytest.mark.parametrize('where', ['forward', 'backward', 'materialize'])
def test_failure_restores_every_unit_without_committing(where):
    engine = Engine()
    if where == 'materialize':
        def fail(**kwargs):
            raise RuntimeError('original failure')
        engine.module.units[1].unshard = fail
    with pytest.raises(RuntimeError, match='original failure'):
        with scope(engine, with_grad=True) as (view, receipt):
            view(1)
            if where == 'backward':
                def fail(loss):
                    raise RuntimeError('original failure')
                engine.backward = fail
                view.backward('loss')
            raise RuntimeError('original failure')
    assert all(not u.group.is_unsharded for u in engine.module.units)
    assert not engine.global_steps
    assert ('root', 'backward_policy', True, False) in engine.events


def test_missing_backward_or_commit_cannot_be_reported_success():
    engine = Engine()
    with pytest.raises(ValueError, match='one backward'):
        with scope(engine, with_grad=True) as (view, receipt):
            view.step()
    assert receipt['restored']
    with pytest.raises(ValueError, match='commit'):
        with scope(engine, with_grad=True) as (view, receipt):
            view.backward('loss')
    assert receipt['restored'] and engine.global_steps == 0


def test_changed_ownership_nested_scope_and_wrong_unit_count_are_rejected():
    engine = Engine()
    with pytest.raises(ValueError, match='ownership'):
        with scope(engine, with_grad=True) as (view, receipt):
            view.backward('loss')
            engine.optimizer.param_groups[0]['params'] = [object()]
            view.step()
    assert receipt['restored'] and engine.global_steps == 0
    with scope(engine, with_grad=False):
        with pytest.raises(ValueError, match='unsharded'):
            with scope(engine, with_grad=False):
                pytest.fail('nested entry')
    with pytest.raises(ValueError, match='unit count'):
        with fsdp2_variable_forward_phase(engine, enabled=True, with_grad=False,
                                         expected_units=41, module_type=Unit):
            pytest.fail('incorrect topology')


def test_off_path_returns_original_engine_without_touching_modules():
    engine = NS()
    with fsdp2_variable_forward_phase(engine, enabled=False, with_grad=True, expected_units=41) as (view, receipt):
        assert view is engine and receipt == dict(enabled=False, restored=True)


def test_grad_phase_requires_whole_checkpoint_blocks_not_just_matching_unit_count():
    engine = Engine()
    with pytest.raises(ValueError, match='checkpoint block'):
        with fsdp2_variable_forward_phase(engine, enabled=True, with_grad=True, expected_units=2, module_type=Unit):
            pytest.fail('missing checkpoint boundary')
    # Same unit count, but an interior primitive was sharded instead of BlockUnit.
    engine.module.units[1] = Unit(engine.events, 'interior-linear')
    with pytest.raises(ValueError, match='checkpoint block'):
        with scope(engine, with_grad=True):
            pytest.fail('wrong checkpoint boundary')
    assert engine.events == []  # Reject before policy changes or collectives.


def test_non_fsdp_root_is_rejected_even_when_child_count_matches():
    engine = Engine()
    children = list(engine.module.units)
    engine.module = NS(modules=lambda: iter(children))
    with pytest.raises(ValueError, match='root owner'):
        with scope(engine, with_grad=True):
            pytest.fail('non-FSDP root')
