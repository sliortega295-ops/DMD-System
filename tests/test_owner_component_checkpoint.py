"""Per-component checkpoint policies must prove their actual execution."""
from copy import deepcopy

import pytest

from distillschedule.open_workload.owner_execution import inventory, evidence
from test_owner_execution_capability import NativeOwner


class ComponentOwner(NativeOwner):
    def execution_options(self):
        return dict(schema_version='distillschedule.owner_execution.v1',
            baseline=dict(generator_checkpointing=True, fake_checkpointing=True),
            options=[dict(generator_checkpointing=True, fake_checkpointing=True),
                     dict(generator_checkpointing=True, fake_checkpointing=False)],
            effect=dict(kind='native_component_checkpoint',
                        enabled_keys=dict(generator='generator_checkpointing', fake='fake_checkpointing')),
            identity={'source': 'COMPONENT_CPU_CONTRACT_TEST_ONLY'})


def selected():
    return {'owner_execution': {'owner': dict(generator_checkpointing=True, fake_checkpointing=False)}}


def rows():
    return [{'owner_execution': {'owner': dict(option=selected()['owner_execution']['owner'],
        effect=ComponentOwner().execution_options()['effect'], components=dict(
            generator=dict(checkpoint_requests=2, recompute_calls=2, forward_calls=1,
                           retained_calls=0, disabled_verified=False),
            fake=dict(checkpoint_requests=0, recompute_calls=0, forward_calls=10,
                      retained_calls=10, disabled_verified=True)))}}]


def test_per_component_inventory_and_real_effect_gate():
    value = inventory({'owner': ComponentOwner()})
    assert value['owner']['descriptor']['effect']['enabled_keys'] == dict(
        generator='generator_checkpointing', fake='fake_checkpointing')
    assert evidence(selected(), rows())['active']


@pytest.mark.parametrize('fault', ['unused_fake', 'fake_recomputed', 'generator_not_replayed',
                                  'missing_component', 'wrong_option', 'unknown_key'])
def test_per_component_effect_rejects_missing_or_wrong_runtime(fault):
    value = rows()
    actual = value[0]['owner_execution']['owner']
    if fault == 'unused_fake':
        actual['components']['fake'].update(forward_calls=0, retained_calls=0)
    elif fault == 'fake_recomputed':
        actual['components']['fake']['recompute_calls'] = 1
    elif fault == 'generator_not_replayed':
        actual['components']['generator']['recompute_calls'] = 0
    elif fault == 'missing_component':
        del actual['components']['fake']
    elif fault == 'wrong_option':
        actual['option']['fake_checkpointing'] = True
    else:
        actual['effect']['enabled_keys']['fake'] = 'not_declared'
    assert not evidence(selected(), value)['active']


@pytest.mark.parametrize('component', [None, [], 3, True])
@pytest.mark.parametrize('whole_map', [True, False])
def test_per_component_effect_rejects_malformed_runtime(component, whole_map):
    value = rows()
    actual = value[0]['owner_execution']['owner']
    if whole_map:
        actual['components'] = component
    else:
        actual['components']['fake'] = component
    assert not evidence(selected(), value)['active']


@pytest.mark.parametrize('mapping', [{}, {'fake': 'missing'}, {'fake': False},
                                    {'fake': 'fake_checkpointing', 'other': 'fake_checkpointing'}])
def test_per_component_inventory_rejects_invalid_bindings(mapping):
    class Bad(ComponentOwner):
        def execution_options(self):
            value = super().execution_options()
            value['effect']['enabled_keys'] = deepcopy(mapping)
            return value
    with pytest.raises(ValueError, match='effect'):
        inventory({'owner': Bad()})
