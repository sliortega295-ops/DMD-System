from scripts.distillschedule.analyze_w16_pool_nsys import (
    classify_kernel,
    merge_interval_nanoseconds,
)


def test_merge_interval_nanoseconds_merges_overlaps_and_touching_ranges():
    assert merge_interval_nanoseconds([(1, 3), (2, 5), (5, 7), (10, 12)]) == 8


def test_merge_interval_nanoseconds_ignores_empty_ranges():
    assert merge_interval_nanoseconds([(3, 3), (5, 4), (8, 10)]) == 2


def test_classify_kernel():
    assert classify_kernel("ncclDevKernel_AllReduce") == "nccl"
    assert classify_kernel("flash_fwd_kernel") == "compute"
