"""H3 codec manifest refusals before importing original source or allocating."""
from copy import deepcopy

import pytest


def manifest(root):
    from distillschedule.training import h3_av_codec as codec
    pin=dict(path=str(root / "weights.safetensors"),bytes=16,sha256="a"*64)
    return dict(schema_version=codec.SCHEMA,precision=codec.PRECISION,storage_dtype="float32",evidence_kind=codec.FIXTURE,
        source=dict(root=str(root),files=deepcopy(codec.SOURCE_FILES),provenance=dict(origin="provided H3 snapshot",
            license="operator supplied source terms",terms_accepted=True)),
        video_decoder={**codec.VIDEO_NATIVE,"num_layers":1,"heads":1,"dim_head":16},
        geometry=dict(video_latent_shape=[1,24,7,2,2],audio_latent_shape=[2,32,2],
            video_shape=[22,32,32,3],audio_shape=[1600,2],fps=24,sample_rate=32000),
        weights=dict(video=[deepcopy(pin)],audio=[deepcopy(pin)]),
        budget=dict(max_parameters=70_000_000,max_loaded_bytes=2**31,max_activation_elements=100_000_000,
                    max_attention_elements=10_000_000,host_reserve_bytes=2**20))


@pytest.mark.parametrize("fault", ["schema","precision","evidence","source","terms","width","video_shape",
    "audio_shape","channels","fps","rate","budget","unknown","pin"])
def test_h3_codec_invalid_contract_refuses_before_source_import(tmp_path, monkeypatch, fault):
    from distillschedule.training import h3_av_codec as codec
    p=manifest(tmp_path)
    if fault=="schema":p["schema_version"]="unknown"
    elif fault=="precision":p["precision"]="bf16"
    elif fault=="evidence":p["evidence_kind"]="PRETRAINED_VERIFIED"
    elif fault=="source":p["source"]["files"][next(iter(p["source"]["files"]))]="0"*64
    elif fault=="terms":p["source"]["provenance"]["terms_accepted"]=False
    elif fault=="width":p["video_decoder"]["dim_head"]=8
    elif fault=="video_shape":p["geometry"]["video_shape"][-1]=1
    elif fault=="audio_shape":p["geometry"]["audio_shape"][-1]=1
    elif fault=="channels":p["geometry"]["audio_latent_shape"][0]=1
    elif fault=="fps":p["geometry"]["fps"]=0
    elif fault=="rate":p["geometry"]["sample_rate"]=16000
    elif fault=="budget":p["budget"]["max_parameters"]=True
    elif fault=="unknown":p["silent_resize"]=True
    else:p["weights"]["video"][0]["sha256"]="wrong"
    monkeypatch.setattr(codec,"source_api",lambda *a:pytest.fail("invalid contract imported codec source"))
    with pytest.raises(ValueError):codec.validate_manifest(p)


def test_real_weights_do_not_admit_tiny_decoder_as_original_native(tmp_path):
    from distillschedule.training import h3_av_codec as codec
    p=manifest(tmp_path);p["evidence_kind"]=codec.LOCAL
    with pytest.raises(ValueError,match="original native"):
        codec.validate_manifest(p)


def test_host_admission_is_the_same_shared_headroom_function():
    from distillschedule.training import host_memory,helios_native_inference
    assert helios_native_inference._host_admit is host_memory.admit_host_memory


@pytest.mark.parametrize("fault", ["missing", "unknown", "shape", "dtype", "duplicate", "nonfinite", "hash"])
def test_safe_weight_loader_rejects_invalid_state_without_materializing_model(tmp_path, fault):
    import torch
    from safetensors.torch import save_file
    from distillschedule.training import h3_av_codec as codec
    from distillschedule.workloads.best_known.registry import file_sha256
    model = torch.nn.Linear(2, 3, device="meta")
    original = {k: (tuple(v.shape), v.dtype) for k, v in model.state_dict().items()}
    values = {"weight": torch.ones(3, 2), "bias": torch.zeros(3)}
    if fault == "missing": del values["bias"]
    elif fault == "unknown": values["surprise"] = torch.zeros(1)
    elif fault == "shape": values["weight"] = torch.ones(3, 3)
    elif fault == "dtype": values["weight"] = values["weight"].half()
    elif fault == "nonfinite": values["weight"][0, 0] = float("nan")
    path = tmp_path / "state.safetensors"
    save_file(values, path)
    pin = dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha256(path))
    if fault == "hash": pin["sha256"] = "0" * 64
    pins = [pin, deepcopy(pin)] if fault == "duplicate" else [pin]
    with pytest.raises(ValueError): codec._load_weights(model, pins, original, "float32")
    assert all(t.is_meta for t in model.state_dict().values())


def test_explicit_bf16_storage_loads_only_selected_decoder_keys_exactly(tmp_path):
    import torch
    from safetensors.torch import save_file
    from distillschedule.training import h3_av_codec as codec
    from distillschedule.workloads.best_known.registry import file_sha256
    model = torch.nn.Linear(2, 3, device="meta")
    original = {k: (tuple(v.shape), v.dtype) for k, v in model.state_dict().items()}
    original["unused_encoder"] = ((4,), torch.float32)
    values = {"weight": torch.arange(6).reshape(3, 2).bfloat16(),
              "bias": torch.arange(3).bfloat16(), "unused_encoder": torch.zeros(4).bfloat16()}
    path = tmp_path / "full.safetensors"; save_file(values, path)
    pin = dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha256(path))
    codec._load_weights(model, [pin], original, "bfloat16_to_float32")
    assert set(model.state_dict()) == {"weight", "bias"}
    assert all(torch.equal(t, values[k].float()) and t.dtype == torch.float32
               and t.device.type == "cpu" for k, t in model.state_dict().items())
    assert not model.training and all(not p.requires_grad for p in model.parameters())
