from dataclasses import replace

import pytest

from distillschedule.search.sharing import RoleState, enumerate_sharing_partitions, sharing_compatibility


def role(name):
    return RoleState(name, "a" * 64, "b" * 64, "bfloat16", "c" * 64,
                     True, (name + ":loraA", name + ":loraB"), name, True)


def test_shared_base_with_disjoint_named_lora_and_teacher_has_five_choices():
    teacher = replace(role("teacher"), trainable_storage_ids=(), adapter_scope="disabled")
    choices = enumerate_sharing_partitions([role("student"), role("fake"), teacher])
    assert len(choices) == 5
    assert (("fake", "student", "teacher"),) in choices
    assert (("fake",), ("student",), ("teacher",)) in choices


def test_full_trainable_student_cannot_share_with_fake_or_teacher():
    student = replace(role("student"), base_is_frozen=False)
    fake = role("fake")
    assert sharing_compatibility(student, fake)["reasons"] == ["MUTABLE_BASE"]
    assert enumerate_sharing_partitions([student, fake]) == [(("fake",), ("student",))]


@pytest.mark.parametrize("field,value", [("frozen_base_sha256", "e" * 64),
    ("architecture_sha256", "e" * 64), ("placement_fingerprint", "e" * 64),
    ("base_dtype", "float32"), ("scoped_checkpoint_replay", False),
    ("trainable_storage_ids", ("student:loraA",)), ("adapter_scope", "student")])
def test_never_share_from_only_model_name_or_equal_size(field, value):
    assert sharing_compatibility(role("student"), replace(role("fake"), **{field: value}))["status"] == "REJECTED"


def test_missing_sha_and_duplicate_storage_ids_fail():
    with pytest.raises(ValueError):
        replace(role("student"), frozen_base_sha256="same-model")
    with pytest.raises(ValueError):
        replace(role("student"), trainable_storage_ids=("x", "x"))
