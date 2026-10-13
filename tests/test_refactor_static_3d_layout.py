import pytest
from distillschedule.runtime.static_3d_layout import Static3DLayout


def test_all_rank_coordinates_and_group_locality():
    layout=Static3DLayout()
    for rank in range(16):
        assert layout.physical_rank(*layout.coordinate(rank))==rank
    groups=layout.groups()
    assert groups['tp']==[[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
    assert all(len({r//8 for r in group})==1 for group in groups['tp'])
    assert all(len({r//8 for r in group})==2 for group in groups['pp'])
    assert all(len({r//8 for r in group})==1 for group in groups['dp'])


def test_logical_global_batch_is_sixteen_not_dp_two():
    layout=Static3DLayout()
    identities=[layout.logical_sample_rank(d,m) for d in range(2) for m in range(8)]
    assert identities==list(range(16))
    assert layout.receipt()['global_batch']==16
    assert layout.receipt()['timing']=='NOT_RUN_NO_NATIVE_EXECUTOR_IMPLIED'


@pytest.mark.parametrize('blocks,expected',[(30,((0,15),(15,30))),
    (40,((0,20),(20,40))),(50,((0,25),(25,50)))])
def test_balanced_stage_ranges(blocks,expected):
    assert Static3DLayout().block_ranges(blocks)==expected


@pytest.mark.parametrize('kwargs',[dict(tp=2),dict(dp=8),dict(pp=1),dict(microbatches=1)])
def test_no_silent_layout_or_batch_fallback(kwargs):
    with pytest.raises(ValueError):Static3DLayout(**kwargs)
