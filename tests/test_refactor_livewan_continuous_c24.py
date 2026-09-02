from __future__ import annotations

from pathlib import Path

from distillschedule.adapters.livewan.adapter import LiveWanSFDMDAdapter
from distillschedule.core.config import load_plan


ROOT = Path(__file__).resolve().parents[1]
C21 = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c21_continuous_b16.json"
C24 = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c24_continuous_b16_score_modulation.json"


def test_c24_changes_only_exact_score_modulation_metadata(monkeypatch) -> None:
    for key in (
        "LIVEWAN_STUDENT_CHECKPOINT",
        "LIVEWAN_CRITIC_CHECKPOINT",
        "LIVEWAN_TEACHER_CHECKPOINT",
        "LIVEWAN_OPERATION_TRACE",
        "LIVEWAN_SYSTEMS_BOOTSTRAP",
        "DISTILLSCHEDULE_OUTPUT_DIR",
    ):
        monkeypatch.setenv(key, f"/tmp/{key.lower()}")

    reference = load_plan(C21).plan.to_dict()
    candidate_plan = load_plan(C24).plan
    LiveWanSFDMDAdapter().validate(candidate_plan)
    candidate = candidate_plan.to_dict()
    metadata = dict(candidate["metadata"])
    assert metadata.pop("score_modulation_kernel") == {
        "enabled": True,
        "name": "score_bf16_modulation_fusion_v1",
        "scope": "score_ranks_only_critic_and_teacher",
        "source_modulate_sha256": (
            "f244152360d40d6745e28bed4add081604312d04fe2f342842a32432f4bd20ad"
        ),
        "math_change": "fuse_exact_bf16_to_fp32_modulation_pointwise_chain",
        "normalized_shift_scale_dtype": "bfloat16",
        "output_dtype": "float32",
        "per_frame": False,
        "triton_block_size": 256,
    }
    assert metadata == reference["metadata"]
    for key in ("workload", "hardware", "rank_groups", "lifecycle", "training_backends"):
        assert candidate[key] == reference[key]


def test_c24_is_registered_in_controller_launcher_and_transport_probe() -> None:
    controller = (
        ROOT / "scripts/distillschedule/run_livewan_crossscale_world16.py"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT / "scripts/distillschedule/launch_livewan_crossscale_world16_node.sh"
    ).read_text(encoding="utf-8")
    probe = (
        ROOT / "scripts/distillschedule/probe_livewan_crossscale_transport.py"
    ).read_text(encoding="utf-8")
    variant = '"c24-continuous-b16-score-modulation"'
    assert variant in controller
    assert probe.count(variant) == 11
    assert "livewan_wan13b_14b_sf_dmd_c24_continuous_b16_score_modulation.json" in launcher


def test_c24_reuses_the_source_pinned_exact_modulation_kernel() -> None:
    kernel = (
        ROOT / "src/distillschedule/runtime/livewan_modulation_kernel.py"
    ).read_text(encoding="utf-8")
    assert "EXPECTED_MODULATE_SHA256" in kernel
    assert "f244152360d40d6745e28bed4add081604312d04fe2f342842a32432f4bd20ad" in kernel
    assert "normalized.dtype != torch.bfloat16" in kernel
    assert "shift.requires_grad" in kernel
    assert "scale.requires_grad" in kernel
    assert "output = torch.empty_like(normalized, dtype=torch.float32)" in kernel
    assert "return (gradient * (1 + scale)).to(torch.bfloat16), None, None" in kernel
