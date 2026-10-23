"""Explicit larger state window; no default changes or calibration bypass."""
import pytest
from distillschedule.strategies.wan_strategy import WanStrategy
from distillschedule.search.wan_candidate_bundle import INCUMBENT

def test_existing_default_and_retain4_serialization_preserved_and_new_distinct():
 a=WanStrategy('resident',2048,True,5,INCUMBENT)
 assert 'retention_policy'not in a.to_dict()and'retained_critic_blocks'not in a.to_dict()
 b=WanStrategy('resident',2048,True,5,INCUMBENT,retained_critic_blocks=(26,27,28,29))
 assert'retention_policy'not in b.to_dict()
 c=WanStrategy('resident',2048,True,5,INCUMBENT,retained_critic_blocks=tuple(range(22,30)),retention_policy='tail8.v1')
 assert WanStrategy.from_dict(c.to_dict())==c and c.id!=a.id and c.id!=b.id
 assert c.backend==a.backend and c.rollout_window==a.rollout_window
 for key in a.environment():
  if key!='DISTILLSCHEDULE_WAN13B_RETAIN_CRITIC_BLOCKS':assert a.environment()[key]==c.environment()[key]

@pytest.mark.parametrize('kwargs',[
 dict(retained_critic_blocks=tuple(range(22,30))),
 dict(retained_critic_blocks=tuple(range(23,30)),retention_policy='tail8.v1'),
 dict(retained_critic_blocks=(),retention_policy='tail8.v1'),
 dict(retained_critic_blocks=tuple(range(22,30)),retention_policy='tail9.v1'),
])
def test_new_policy_does_not_weaken_old_rejects_or_accept_arbitrary_cache(kwargs):
 with pytest.raises(ValueError):WanStrategy('resident',2048,True,5,INCUMBENT,**kwargs)
