import copy

import pytest

from distillschedule.search.allocation import ExecutionAllocation, PhysicalDevice


def allocation(world=8, arch="sm90", *, roles=("generator", "critic", "teacher")):
    devices = tuple(PhysicalDevice(rank, rank // 8, rank % 8, f"GPU-test-{rank}",
                                   arch, (80 if arch == "sm90" else 40) << 30)
                    for rank in range(world))
    slots = tuple(tuple(range(rank, 16, world)) for rank in range(world))
    return ExecutionAllocation(devices, 16, {role: slots for role in roles})


def test_world8_retains_sixteen_independent_samples_and_one_global_mean():
    contract = allocation()
    assert contract.world_size == 8 and contract.node_count == 1
    assert contract.role_slots["generator"][3] == (3, 11)
    assert contract.loss_scale_before_rank_mean("generator") == 0.5
    # Sum two independent losses locally, then rank-mean: every sample has1/16.
    assert contract.loss_scale_before_rank_mean("generator") / 8 == 1 / 16
    assert ExecutionAllocation.from_dict(contract.to_dict()) == contract


def test_world16_keeps_existing_batch_without_hardware_identity_aliasing():
    a100 = allocation(16, "sm80")
    h100 = allocation()
    assert a100.logical_global_batch == h100.logical_global_batch == 16
    assert a100.node_count == 2 and a100.loss_scale_before_rank_mean("critic") == 1
    assert a100.digest != h100.digest
    assert set(a100.memory_limits) == set(range(16))


def test_role_pools_and_uneven_slots_do_not_change_mean_weighting():
    raw = allocation().to_dict()
    raw["role_slots"] = {
        "generator": [list(range(rank, 16, 4)) if rank < 4 else [] for rank in range(8)],
        "critic": [[] if rank < 4 else list(range(rank - 4, 16, 4)) for rank in range(8)],
    }
    contract = ExecutionAllocation.from_dict(raw)
    assert contract.role_ranks("generator") == (0, 1, 2, 3)
    assert contract.role_ranks("critic") == (4, 5, 6, 7)
    assert contract.loss_scale_before_rank_mean("critic") == 0.25
    raw["role_slots"]["critic"][4].remove(0)
    raw["role_slots"]["critic"][5].append(0)
    uneven = ExecutionAllocation.from_dict(raw)
    assert uneven.loss_scale_before_rank_mean("critic") == 0.25


@pytest.mark.parametrize("edit", ["drop_sample", "duplicate_sample", "wrong_batch", "duplicate_gpu"])
def test_bad_mapping_rejected_before_any_runtime(edit):
    raw = allocation().to_dict()
    if edit == "drop_sample":
        raw["role_slots"]["generator"][0].pop()
    elif edit == "duplicate_sample":
        raw["role_slots"]["generator"][0][1] = 1
    elif edit == "wrong_batch":
        raw["logical_global_batch"] = 8
    else:
        raw["devices"][1]["uuid"] = raw["devices"][0]["uuid"]
    with pytest.raises(ValueError):
        ExecutionAllocation.from_dict(raw)


def test_receipts_check_physical_and_logical_coverage_separately():
    contract = allocation()
    receipts = [{"rank": rank, "allocation_sha256": contract.digest,
                 "logical_slots": {role: list(rows[rank])
                                   for role, rows in contract.role_slots.items()}}
                for rank in range(8)]
    contract.validate_slot_receipts(receipts)
    invalid = copy.deepcopy(receipts)
    invalid[7]["logical_slots"]["generator"] = [7, 7]
    with pytest.raises(ValueError, match="logical samples"):
        contract.validate_slot_receipts(invalid)
    with pytest.raises(ValueError, match="physical worker"):
        contract.validate_slot_receipts(receipts[:7])
    contract.validate_worker(rank=0, world_size=8, node=0, local_rank=0,
                             device_uuid="GPU-test-0")
    with pytest.raises(ValueError, match="placement"):
        contract.validate_worker(rank=0, world_size=16, node=0, local_rank=0,
                                 device_uuid="GPU-test-0")


def test_slot_map_cannot_be_mutated_after_validation():
    contract = allocation()
    with pytest.raises(TypeError):
        contract.role_slots["generator"] = tuple(() for _ in range(8))
    exported = contract.to_dict()
    exported["role_slots"]["generator"][0].clear()
    assert contract.role_slots["generator"][0] == (0, 8)


@pytest.mark.parametrize("field,value", [("world_size", 8.0), ("node", False),
                                        ("local_rank", False), ("rank", False)])
def test_worker_numeric_aliases_are_not_valid_rank_identity(field, value):
    kwargs = dict(rank=0, world_size=8, node=0, local_rank=0, device_uuid="GPU-test-0")
    kwargs[field] = value
    with pytest.raises(ValueError):
        allocation().validate_worker(**kwargs)


@pytest.mark.parametrize("field", ["rank", "slot"])
def test_receipt_boolean_alias_cannot_replace_sample_or_worker_zero(field):
    contract = allocation()
    rows = [{"rank": rank, "allocation_sha256": contract.digest,
             "logical_slots": {role: list(slots[rank])
                               for role, slots in contract.role_slots.items()}}
            for rank in range(8)]
    if field == "rank":
        rows[0]["rank"] = False
    else:
        rows[0]["logical_slots"]["generator"][0] = False
    with pytest.raises(ValueError):
        contract.validate_slot_receipts(rows)
