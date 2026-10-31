"""Real public data/decode/writer chain; test plugin is not an H3 model."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from distillschedule.workloads.best_known.registry import file_sha256


def decoding_case(root, *, bad_decoder=False):
    from distillschedule.data import import_preencoded
    root.mkdir(parents=True, exist_ok=True)
    torch.save({"latents": torch.linspace(-1, 1, 4).reshape(1, 4)}, root / "input.pt")
    (root / "inputs.json").write_text(json.dumps(dict(schema_version="distillschedule.data.preencoded.v1",
        records=[dict(id="new-decoder-input", features=dict(path="input.pt", sha256=file_sha256(root / "input.pt")))],
        metadata=dict(scope="SYNTHETIC_DECODER_CONSUMER_TEST_NOT_H3"))))
    import_preencoded(root / "inputs.json", root / "prepared")
    source = root / "codec.py"
    source.write_text("from pathlib import Path\nimport torch\nfrom distillschedule.training.adapters import ModelComponents\ndef factory(p):\n    Path(p['entered']).touch()\n    torch.rand(1)\n    def decode(feeds, options):\n        assert options == {} and feeds['latents'].shape == (1,4)\n        if p['fail']:raise ValueError('injected decode failure')\n        pixels=torch.zeros(2,2,2,3,dtype=torch.uint8)\n        samples=feeds['latents'].t().repeat(1,2)\n        return dict(video=pixels,audio=dict(samples=samples,sample_rate=32000))\n    return ModelComponents(state={'codec_identity':'TEST_ONLY'},decode=decode,\n        close=lambda:Path(p['closed']).touch(),metadata={'purpose':'TEST_ONLY_CODEC'}).validate()\n")
    conf = dict(schema_version="distillschedule.decode_prepared.v1",
        decoder=dict(factory="codec.py:factory", parameters=dict(entered=str(root / "entered"),
            closed=str(root / "closed"), fail=bad_decoder)),
        data=dict(prepared="prepared/manifest.json", sample_index=0), output_kind="audio_video")
    path = root / "decode.json"; path.write_text(json.dumps(conf))
    return path


def test_public_decode_uses_real_prepared_input_and_writes_png_wav(tmp_path, capsys):
    from distillschedule.cli import main
    from distillschedule.training.checkpoint import rng_state
    from test_training_end_to_end import equal
    path = decoding_case(tmp_path)
    before = rng_state(device="cpu")
    assert main(["decode", str(path), "--output", str(tmp_path / "output")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "DECODED" and result["sample_id"] == "new-decoder-input"
    assert result["output_kind"] == "audio_video" and (tmp_path / "output/audio.wav").is_file()
    assert len(list((tmp_path / "output/frames").glob("*.png"))) == 2
    assert (tmp_path / "output/_SUCCESS").read_text().strip() == file_sha256(tmp_path / "output/generation.json")
    assert (tmp_path / "entered").exists() and (tmp_path / "closed").exists()
    assert equal(before, rng_state(device="cpu"))


def test_failed_decode_closes_provider_and_restores_rng_without_success(tmp_path):
    from distillschedule.training.decoding import decode_prepared
    from distillschedule.training.checkpoint import rng_state
    from test_training_end_to_end import equal
    path = decoding_case(tmp_path, bad_decoder=True)
    before = rng_state(device="cpu")
    with pytest.raises(ValueError, match="injected decode failure"):
        decode_prepared(path, tmp_path / "failed")
    assert (tmp_path / "closed").exists() and not (tmp_path / "failed/_SUCCESS").exists()
    assert equal(before, rng_state(device="cpu"))


@pytest.mark.parametrize("fault", ["data_hash", "duplicate_json", "kind", "sample", "output_exists", "no_decoder"])
def test_decode_refuses_invalid_configuration_and_inputs(tmp_path, fault):
    from distillschedule.training.decoding import decode_prepared
    path = decoding_case(tmp_path);raw = json.loads(path.read_text());output=tmp_path / "output"
    if fault == "data_hash":
        next((tmp_path / "prepared").glob("**/*.pt")).write_bytes(b"changed")
    elif fault == "kind":raw["output_kind"]="latents"
    elif fault == "sample":raw["data"]["sample_index"]=True
    elif fault == "output_exists":output.mkdir()
    elif fault == "no_decoder":
        source=tmp_path / "codec.py";source.write_text(source.read_text().replace("decode=decode,", "decode=None,"))
    if fault == "duplicate_json":path.write_text('{"schema_version":1,"schema_version":2}')
    else:path.write_text(json.dumps(raw))
    with pytest.raises((ValueError, FileExistsError)):
        decode_prepared(path, output)
    assert not (output / "_SUCCESS").exists()
    if fault in ("data_hash", "duplicate_json", "kind", "sample", "output_exists"):
        assert not (tmp_path / "entered").exists()
    if fault == "no_decoder":assert (tmp_path / "closed").exists()


def test_decode_help_stays_torch_and_model_free():
    script="from distillschedule.cli import main; import sys;\ntry:main(['decode','--help'])\nexcept SystemExit as e:assert e.code==0\nassert 'torch' not in sys.modules"
    result=subprocess.run([sys.executable,"-B","-c",script],text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
