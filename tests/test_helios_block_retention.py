"""CPU identity/lowering and canonical rejection tests, no GPU evidence."""
from dataclasses import replace
import pytest
from distillschedule.search.helios_implementation import HeliosImplementation,current_implementation,MODE_ENV
from distillschedule.search.helios_model_adapter import safe_incumbent


def test_optin_identity_environment_and_lowering_roundtrip():
    old=safe_incumbent();new=replace(old,retained_critic_blocks=(38,39))
    assert new.id==old.id+'-rc38_39'
    assert HeliosImplementation.parse(new.id)==new
    assert HeliosImplementation.from_assignment(new.assignment)==new
    env=new.environment();env[MODE_ENV]='native'
    assert current_implementation(env)==new
    assert new.lower_profile()['contract']==old.lower_profile()['contract']
    assert new.lower_profile()['runtime']['planner_implementation']==new.assignment
    assert new.assignment['retained_critic_blocks']=='38,39'


def test_default_ids_environment_do_not_gain_hidden_retention():
    old=safe_incumbent()
    assert 'retained_critic_blocks' not in old.assignment and '-rc' not in old.id
    assert not any('RETAIN_CRITIC_BLOCKS' in k for k in old.environment())


@pytest.mark.parametrize('blocks',[(40,),(-1,),(True,),(39,38),(38,38),(36,37,38)])
def test_bad_blocks_rejected(blocks):
    with pytest.raises(Exception):replace(safe_incumbent(),retained_critic_blocks=blocks)


def test_no_opaque_calibration_or_gan_retention():
    x=replace(safe_incumbent(),retained_critic_blocks=(39,))
    env=x.environment();env[MODE_ENV]='calibration'
    with pytest.raises(Exception,match='uncalibrated'):current_implementation(env)
    with pytest.raises(Exception):HeliosImplementation('gan','world16-fsdp',2048,5,True,retained_critic_blocks=(39,))
