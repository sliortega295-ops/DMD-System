from runtime.host_admission import GIB, evaluate_host_admission, strict_numa_policies


def test_strict_numa_policy_is_rejected_even_with_node_headroom() -> None:
    value = evaluate_host_admission(
        projected_peak_bytes=160 * GIB,
        reserve_bytes=64 * GIB,
        physical_total_bytes=1024 * GIB,
        physical_available_bytes=900 * GIB,
        cgroup_max_bytes=760 * GIB,
        cgroup_current_bytes=1 * GIB,
        numa_policies=("bind:3",),
    )
    assert not value.admitted
    assert value.strict_numa_policies == ("bind:3",)
    assert "strict NUMA membind" in value.rejection_reasons[0]


def test_unbound_official_probe_is_admitted_with_reserve() -> None:
    value = evaluate_host_admission(
        projected_peak_bytes=160 * GIB,
        reserve_bytes=64 * GIB,
        physical_total_bytes=1024 * GIB,
        physical_available_bytes=900 * GIB,
        cgroup_max_bytes=760 * GIB,
        cgroup_current_bytes=1 * GIB,
    )
    assert value.admitted
    assert value.effective_cap_bytes == 760 * GIB
    assert value.required_bytes == 224 * GIB


def test_peak_plus_reserve_over_cgroup_cap_is_rejected() -> None:
    value = evaluate_host_admission(
        projected_peak_bytes=720 * GIB,
        reserve_bytes=64 * GIB,
        physical_total_bytes=1024 * GIB,
        physical_available_bytes=900 * GIB,
        cgroup_max_bytes=760 * GIB,
        cgroup_current_bytes=1 * GIB,
    )
    assert not value.admitted
    assert "exceeds effective cap" in value.rejection_reasons[0]


def test_numa_policy_parser_only_flags_strict_bind() -> None:
    lines = [
        "1000 default file=/bin/python N0=1",
        "2000 interleave:0-7 anon=1 N2=1",
        "3000 bind:3 heap anon=1 N3=1",
        "4000 bind:3 anon=1 N3=1",
    ]
    assert strict_numa_policies(lines) == ("bind:3",)


def test_external_physical_pressure_is_rejected() -> None:
    value = evaluate_host_admission(
        projected_peak_bytes=160 * GIB,
        reserve_bytes=64 * GIB,
        physical_total_bytes=1024 * GIB,
        physical_available_bytes=100 * GIB,
        cgroup_max_bytes=760 * GIB,
        cgroup_current_bytes=20 * GIB,
    )
    assert not value.admitted
    assert value.additional_bytes_needed == 140 * GIB
    assert value.physical_required_bytes == 204 * GIB
    assert any("physical available" in reason for reason in value.rejection_reasons)


def test_minimum_share_closes_measured_a_local_x8_deficit() -> None:
    from runtime.host_admission import SharedBlockCandidate, plan_minimum_shared_blocks

    candidates = [
        SharedBlockCandidate("generator", index, 702_788_608, 16)
        for index in range(40)
    ] + [
        SharedBlockCandidate("score", index, 702_788_608, 7)
        for index in range(40)
    ]
    plan = plan_minimum_shared_blocks(
        base_peak_bytes=757_807_587_328,
        effective_cap_bytes=816_043_786_240,
        reserve_bytes=64 * GIB,
        rank_count=8,
        candidates=candidates,
    )
    assert plan.admitted
    assert plan.deficit_bytes == 10_483_277_824
    assert len(plan.selected) == 3
    assert [(item.component, item.block_index) for item in plan.selected] == [
        ("score", 0),
        ("score", 1),
        ("score", 2),
    ]
    assert plan.projected_savings_bytes == 14_758_560_768
    assert plan.projected_peak_bytes == 743_049_026_560
    assert plan.projected_headroom_bytes == 4_275_282_944


def test_minimum_share_rejects_peak_before_ownership_policy_is_active() -> None:
    from runtime.host_admission import SharedBlockCandidate, plan_minimum_shared_blocks

    plan = plan_minimum_shared_blocks(
        base_peak_bytes=80 * GIB,
        unaffected_peak_bytes=95 * GIB,
        effective_cap_bytes=100 * GIB,
        reserve_bytes=10 * GIB,
        rank_count=8,
        candidates=[SharedBlockCandidate("score", index, 2 * GIB) for index in range(8)],
    )
    assert not plan.admitted
    assert plan.selected == ()
    assert plan.projected_peak_bytes == 95 * GIB
    assert plan.projected_affected_peak_bytes == 80 * GIB
    assert "ownership-unaffected lifecycle phase" in plan.rejection_reason


def test_minimum_share_rejects_insufficient_candidates() -> None:
    from runtime.host_admission import SharedBlockCandidate, plan_minimum_shared_blocks

    plan = plan_minimum_shared_blocks(
        base_peak_bytes=100 * GIB,
        effective_cap_bytes=100 * GIB,
        reserve_bytes=10 * GIB,
        rank_count=2,
        candidates=[SharedBlockCandidate("score", 0, GIB)],
    )
    assert not plan.admitted
    assert plan.projected_headroom_bytes < 0
