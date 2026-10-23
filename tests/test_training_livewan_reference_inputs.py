"""Safe distinct CPU1 cohorts; no native world16 schema relaxation or pickle."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from test_training_livewan_public_cpu import files, bounded, json_file, equal, config_for


def recode(feeds, change):
    from distillschedule.data.livewan_reference_inputs import _canonical
    count = int(feeds["control_length"].item())
    control = json.loads(bytes(feeds["control"][:count].tolist()))
    change(control)
    code = _canonical(control)
    feeds["control"].zero_()
    feeds["control"][:len(code)] = torch.tensor(list(code), dtype=torch.uint8)
    feeds["control_length"][0] = len(code)
    return feeds


def test_each_prepared_sample_is_whole_five_record_cpu_cycle_with_original_streams(files):
    from distillschedule.data import PreparedDataset
    from distillschedule.data.livewan_reference_inputs import decode_cycle, descriptor, validate_source_cycle, STREAMS
    from distillschedule.data.livewan_inputs import _derived_seed, CONTRACT
    from distillschedule.training.livewan_public import GEOMETRY, CPUReferenceRequest
    dataset = PreparedDataset(files["prepared"])
    assert len(dataset) == 2 and dataset.validate(sources=True)["status"] == "PASS"
    assert CONTRACT["resolution"] == [368, 640] and CONTRACT["world_size"] == CONTRACT["global_batch"] == 16
    assert GEOMETRY["resolution"] == [32, 32] and GEOMETRY["world_size"] == GEOMETRY["global_batch"] == 1
    ids, hashes, controls, worlds = [], [], [], []
    for cycle in range(2):
        feeds = dataset.sample(cycle)
        row = decode_cycle(feeds, cycle=cycle)
        validate_source_cycle(row)
        assert descriptor(row) == dataset.manifest["records"][cycle]["metadata"]
        controls.append(row["control_sha256"])
        assert len(row["entries"]) == 5 and row["negative"].shape == (1, 512, 4096)
        for entry, raw in zip(row["control"]["entries"], row["entries"]):
            operation = entry["operation"]
            ids.append(entry["id"])
            hashes.append(entry["content_sha256"])
            assert set(operation["seeds"]) == set(STREAMS)
            assert operation["seeds"] == {name: _derived_seed(row["control"]["seed"], operation["operation_key"], name) for name in STREAMS}
            assert len(set(operation["seeds"].values())) == 6
            worlds.append(operation["world_frames"])
            assert raw["latents"].shape == (16, 21, 4, 4) and raw["context"].shape == (512, 4096)
            assert CPUReferenceRequest().new_blocks(operation["world_frames"]) in (4, 5, 6)
    assert len(set(ids)) == len(set(hashes)) == 10 and len(set(controls)) == 2
    assert set(worlds) == {3, 6, 9}


@pytest.mark.parametrize("change", ["native_geometry", "native_precision", "format", "missing_negative", "partial_count", "cycle_bool", "seed_bool", "duplicate_id", "duplicate_content"])
def test_invalid_source_cohort_rejected_without_publication(files, tmp_path, change):
    from distillschedule.data.livewan_reference_inputs import prepare_inputs
    value = json.loads(Path(files["source"]).read_bytes())
    if change == "native_geometry":
        value["geometry"]["resolution"] = [368, 640]
    elif change == "native_precision":
        value["precision"] = "BF16"
    elif change == "format":
        value["format"] = "original_livewan_teacher_cohorts.v1"
    elif change == "missing_negative":
        value.pop("negative")
    elif change == "partial_count":
        value["records"].pop()
    elif change == "cycle_bool":
        value["cycles"] = True
    elif change == "seed_bool":
        value["seed"] = True
    elif change == "duplicate_id":
        value["records"][1]["id"] = value["records"][0]["id"]
    elif change == "duplicate_content":
        from safetensors.torch import load_file, save_file
        from distillschedule.training.livewan_public import pin_file
        raw = load_file(value["records"][0]["source"]["path"])
        path = tmp_path / "copy.safetensors"
        save_file(raw, str(path), metadata=dict(evidence_kind=value["evidence_kind"], container_note="distinct bytes same source content"))
        value["records"][1]["source"] = pin_file(path)
    source = json_file(tmp_path / "bad.json", value)
    before = __import__("distillschedule.training.checkpoint", fromlist=["rng_state"]).rng_state()
    with pytest.raises((ValueError, RuntimeError)):
        prepare_inputs(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert equal(before, __import__("distillschedule.training.checkpoint", fromlist=["rng_state"]).rng_state())


@pytest.mark.parametrize("change", ["missing_history", "missing_text", "missing_negative", "latent_shape", "text_dtype", "nonfinite", "control_seed", "negative_content", "row_order", "padding", "length_dtype"])
def test_invalid_prepared_tensor_envelope_rejects_no_conversion_or_relabel(files, change):
    from distillschedule.data import PreparedDataset
    from distillschedule.data.livewan_reference_inputs import decode_cycle
    row = PreparedDataset(files["prepared"]).sample(0)
    if change == "missing_history":
        row["entries"][0].pop("latents")
    elif change == "missing_text":
        row["entries"][0].pop("context")
    elif change == "missing_negative":
        row.pop("negative")
    elif change == "latent_shape":
        row["entries"][0]["latents"] = row["entries"][0]["latents"][:, :20]
    elif change == "text_dtype":
        row["entries"][0]["context"] = row["entries"][0]["context"].bfloat16()
    elif change == "nonfinite":
        row["entries"][0]["latents"][0, 0, 0, 0] = float("nan")
    elif change == "control_seed":
        recode(row, lambda value: value["entries"][0]["operation"]["seeds"].update(rollout=8))
    elif change == "negative_content":
        row["negative"].add_(1.)
    elif change == "row_order":
        row["entries"].reverse()
    elif change == "padding":
        row["control"][-1] = 1
    elif change == "length_dtype":
        row["control_length"] = row["control_length"].int()
    with pytest.raises((ValueError, RuntimeError)):
        decode_cycle(row, cycle=0)


@pytest.mark.parametrize("dtype_shape", ["bf16", "native_shape", "nan", "missing_text"])
def test_safe_raw_archives_require_exact_fp32_uncropped_history_and_raw_text(files, tmp_path, dtype_shape):
    from safetensors.torch import load_file, save_file
    from distillschedule.training.livewan_public import pin_file
    from distillschedule.data.livewan_reference_inputs import prepare_inputs
    value = json.loads(Path(files["source"]).read_bytes())
    raw = load_file(value["records"][0]["source"]["path"])
    if dtype_shape == "bf16":
        raw["context"] = raw["context"].bfloat16()
    elif dtype_shape == "native_shape":
        raw["latents"] = torch.zeros(16, 21, 46, 80)
    elif dtype_shape == "nan":
        raw["latents"][0, 0, 0, 0] = float("nan")
    else:
        raw.pop("context")
    path = tmp_path / "invalid.safetensors"
    save_file(raw, str(path), metadata=dict(evidence_kind=value["evidence_kind"]))
    value["records"][0]["source"] = pin_file(path)
    source = json_file(tmp_path / "bad.json", value)
    with pytest.raises((ValueError, RuntimeError)):
        prepare_inputs(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_pickle_payload_is_never_ingested_and_native_namespaces_remain_absent(files, tmp_path):
    import os
    import sys
    from distillschedule.training.livewan_public import pin_file
    from distillschedule.data.livewan_reference_inputs import prepare_inputs
    marker = tmp_path / "UNSAFE_CODE_RAN"
    class Unsafe:
        def __reduce__(self):
            return os.system, ("touch " + str(marker),)
    path = tmp_path / "unsafe.pt"
    torch.save(Unsafe(), path)
    value = json.loads(Path(files["source"]).read_bytes())
    value["records"][0]["source"] = pin_file(path)
    source = json_file(tmp_path / "bad.json", value)
    namespaces = {k: v for k, v in sys.modules.items() if k == "wan" or k.startswith("wan.") or k == "wanstreamer" or k.startswith("wanstreamer.")}
    with pytest.raises(ValueError, match="safetensors"):
        prepare_inputs(source, tmp_path / "out")
    assert not marker.exists() and not (tmp_path / "out").exists()
    assert namespaces == {k: v for k, v in sys.modules.items() if k == "wan" or k.startswith("wan.") or k == "wanstreamer" or k.startswith("wanstreamer.")}


@pytest.mark.parametrize("change", ["dtype", "content", "control"])
def test_worker_input_view_readmits_decoded_negative_and_control_before_teacher(files, change):
    from distillschedule.reference import livewan_wan21 as provider
    from distillschedule.training.livewan_public import LiveWanCPUContext
    from distillschedule.data import PreparedDataset
    from distillschedule.data.livewan_reference_inputs import decode_cycle, CPUCycleInputs
    row = decode_cycle(PreparedDataset(files["prepared"]).sample(0), cycle=0)
    context = LiveWanCPUContext(api=provider.load_cpu(backend="torch-sdpa-cpu", precision="float32"), source=provider.source_identity())
    if change == "dtype":
        row["negative"] = row["negative"].bfloat16()
    elif change == "content":
        row["negative"].add_(1.)
    else:
        row["control"]["seed"] += 1
    with pytest.raises(ValueError, match="negative|control"):
        CPUCycleInputs(row, reference_context=context)
