"""Original flat/split cached parser equivalence and strict tensor-only publication."""
import json

import pytest
import torch

from test_training_h3_cached_public_cpu import make_case, pin, json_file
from test_h3_cached_original_cpu import original, equal_tree


@pytest.mark.parametrize("packed", [False, True])
def test_original_cache_publication_parser_equivalence_and_rng(tmp_path, packed):
    from distillschedule.data.h3_cached_inputs import prepare_inputs, decode_row
    from distillschedule.data import PreparedDataset
    from distillschedule.training.checkpoint import rng_state
    from distillschedule.data.tensors import tensor_tree
    _, sources = make_case(tmp_path, packed=packed)
    before = rng_state()
    path = prepare_inputs(sources, tmp_path / "prepared")
    equal_tree(before, rng_state())
    dataset = PreparedDataset(path)
    assert dataset.validate(sources=True)["status"] == "PASS"
    for index, record in enumerate(json.loads(sources.read_text())["records"]):
        raw = torch.load(record["cache"]["path"], weights_only=True)
        expected = original().dmd2.prepare_batch(raw)
        tree = dataset.sample(index)
        tensor_tree(tree)
        row = decode_row(tree, absolute_step=index, geometry=dataset.manifest["metadata"]["geometry"])
        equal_tree(expected, original().dmd2.prepare_batch(row["raw"]))
        assert "seq_len" not in row["raw"].get("packed", {})


@pytest.mark.parametrize("bad", ["mask", "dirty", "fp16", "missing_previous", "nonfinite", "unknown", "copied", "packed_control"])
def test_cache_refusals_before_publication(tmp_path, bad):
    from distillschedule.data.h3_cached_inputs import prepare_inputs
    _, sources = make_case(tmp_path, packed=bad in ("dirty", "packed_control"))
    manifest = json.loads(sources.read_text())
    path = tmp_path / "cache-0.pth"
    raw = torch.load(path, weights_only=True)
    shared = raw[0] if isinstance(raw, tuple) else raw
    if bad == "mask": shared["padding_mask"] = torch.ones(1)
    if bad == "dirty": shared["imgvid_cond_noise_aug"] = 0.5
    if bad == "fp16": shared["input_latents"] = shared["input_latents"].half()
    if bad == "missing_previous": shared.pop("ref_video_latents")
    if bad == "nonfinite": shared["audio_input_latents"].fill_(float("nan"))
    if bad == "unknown": shared["unadmitted_control"] = torch.ones(1)
    if bad == "packed_control": raw[1]["packed"]["control_rows"] = torch.ones(1)
    torch.save(raw, path)
    manifest["records"][0]["cache"] = pin(path)
    if bad == "copied": manifest["records"][1]["cache"] = pin(path)
    json_file(sources, manifest)
    with pytest.raises(ValueError):
        prepare_inputs(sources, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


def test_cli_public_preparation(tmp_path, capsys):
    from distillschedule.data.cli import main
    _, sources = make_case(tmp_path)
    assert main(["prepare-h3-cached-reference", "--sources", str(sources), "--output", str(tmp_path / "prepared")]) == 0
    assert json.loads(capsys.readouterr().out)["samples"] == 4
