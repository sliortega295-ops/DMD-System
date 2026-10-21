from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import wave

import pytest
import torch

from distillschedule.data import PreparedDataset, import_preencoded, prepare
from distillschedule.data.decoding import decode_record


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def raw_fixture(root: Path, *, count=2):
    encoder = root / "encoder.py"
    encoder.write_text(
        "import torch\n"
        "def encode(record, parameters):\n"
        "    return {'latents': record['video'].permute(3,0,1,2).to(torch.float64) * parameters['scale'],\n"
        "            'condition': {'audio': record['audio']['samples'],\n"
        "                          'tokens': [torch.tensor(list(record['text'].encode()), dtype=torch.int64)]}}\n",
        encoding="utf-8",
    )
    records = []
    for index in range(count):
        frames = root / f"frames-{index}"
        frames.mkdir()
        # PPM is an actual decoded-media fixture, not a stand-in latent tensor.
        for frame in range(2):
            (frames / f"{frame:02d}.ppm").write_bytes(
                b"P6\n4 4\n255\n" + bytes([20 + index * 30, 40 + frame * 20, 60]) * 16
            )
        audio = root / f"audio-{index}.wav"
        with wave.open(str(audio), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(8000)
            output.writeframes(bytes([0, 0, 0, 64, 0, 128, 255, 127]))
        caption = root / f"text-{index}.txt"
        caption.write_text(f"sample {index}\n", encoding="utf-8")
        records.append({"id": f"record-{index}", "video": frames.name,
                        "audio": audio.name, "text": {"path": caption.name}})
    path = write_json(root / "raw.json", {
        "schema_version": "distillschedule.data.raw.v1",
        "encoder": {"factory": "encoder.py:encode", "parameters": {"scale": 0.5}},
        "records": records,
    })
    return path


def test_actual_raw_media_to_hashed_native_features(tmp_path):
    raw = raw_fixture(tmp_path)
    manifest = prepare(raw, tmp_path / "prepared")
    dataset = PreparedDataset(manifest)
    assert len(dataset) == 2
    assert [dataset.sample_id(i) for i in range(2)] == ["record-0", "record-1"]
    first, second = dataset.sample(0), dataset.sample(1)
    assert first["latents"].shape == (3, 2, 4, 4)
    assert first["latents"].dtype == torch.float64
    assert first["latents"][0, 0, 0, 0] == 10
    assert second["latents"][0, 0, 0, 0] == 25
    assert first["condition"]["audio"].shape == (4, 1)
    assert torch.equal(first["condition"]["tokens"][0], torch.tensor(list(b"sample 0\n")))
    assert len(dataset.identity) == 64
    assert dataset.identity == PreparedDataset(manifest).identity
    assert dataset.validate()["samples"] == 2
    with pytest.raises(IndexError):
        dataset.sample(2)
    with pytest.raises(IndexError):
        dataset.sample(-1)


def test_real_video_decoder_without_resize_or_temporal_recycling(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("actual video decoding needs existing ffmpeg and ffprobe")
    raw_fixture(tmp_path, count=1)
    video = tmp_path / "clip.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-framerate", "2", "-i",
                    str(tmp_path / "frames-0/%02d.ppm"), "-c:v", "ffv1", str(video)],
                   check=True, capture_output=True)
    decoded = decode_record({"id": "video", "video": str(video)})
    assert decoded["video"].shape == (2, 4, 4, 3)
    assert decoded["video"].dtype == torch.uint8
    assert decoded["video"][0, 0, 0].tolist() == [20, 40, 60]


def test_video_decoder_rejects_unhashed_external_segment_dependencies(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("actual video decoding needs existing ffmpeg and ffprobe")
    raw_fixture(tmp_path, count=1)
    video = tmp_path / "clip.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-framerate", "2", "-i",
                    str(tmp_path / "frames-0/%02d.ppm"), "-c:v", "ffv1", str(video)],
                   check=True, capture_output=True)
    playlist = tmp_path / "unbound.ffconcat"
    playlist.write_text("ffconcat version 1.0\nfile 'clip.mkv'\n")
    with pytest.raises(ValueError, match="cannot decode video"):
        decode_record({"id": "external", "video": str(playlist)})


@pytest.mark.parametrize("change,match", [
    (lambda value: value["records"].append(dict(value["records"][0])), "duplicate"),
    (lambda value: value["records"][0].update(video="missing.mp4"), "does not exist"),
    (lambda value: value["records"][0].update(text={"path": "missing.txt"}), "does not exist"),
    (lambda value: value["encoder"].update(parameters={"bad": float("nan")}), "JSON"),
])
def test_bad_raw_manifests_fail_before_publication(tmp_path, change, match):
    raw = raw_fixture(tmp_path)
    value = json.loads(raw.read_text())
    change(value)
    write_json(raw, value)
    with pytest.raises(ValueError, match=match):
        prepare(raw, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_failed_encoder_does_not_publish_partial_dataset(tmp_path):
    raw = raw_fixture(tmp_path)
    (tmp_path / "encoder.py").write_text(
        "import torch\n"
        "def encode(record, parameters):\n"
        "    if record['id'] == 'record-1': raise ValueError('encoder failed')\n"
        "    return {'latents': torch.ones(1)}\n"
    )
    with pytest.raises(ValueError, match="encoder failed"):
        prepare(raw, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()
    assert not list(tmp_path.glob(".prepared.*"))


def test_completed_data_is_never_overwritten(tmp_path):
    raw = raw_fixture(tmp_path)
    manifest = prepare(raw, tmp_path / "prepared")
    original = manifest.read_bytes()
    with pytest.raises(FileExistsError):
        prepare(raw, tmp_path / "prepared")
    assert manifest.read_bytes() == original


def test_feature_drift_is_rejected_at_open_and_each_sample(tmp_path):
    raw = raw_fixture(tmp_path)
    manifest = prepare(raw, tmp_path / "prepared")
    dataset = PreparedDataset(manifest)
    feature = manifest.parent / dataset.manifest["records"][0]["features"]["path"]
    feature.write_bytes(feature.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        dataset.sample(0)
    with pytest.raises(ValueError, match="hash"):
        PreparedDataset(manifest)


@pytest.mark.parametrize("target", ["text-0.txt", "encoder.py"])
def test_explicit_source_validation_detects_raw_and_encoder_drift(tmp_path, target):
    raw = raw_fixture(tmp_path)
    manifest = prepare(raw, tmp_path / "prepared")
    dataset = PreparedDataset(manifest)
    (tmp_path / target).write_text("changed")
    assert dataset.validate()["samples"] == 2  # prepared data is portable without raw media
    with pytest.raises(ValueError, match="source.*(changed|hash)"):
        dataset.validate(sources=True)


def test_encoder_cannot_change_source_silently_during_preparation(tmp_path):
    raw = raw_fixture(tmp_path)
    (tmp_path / "encoder.py").write_text(
        "import torch\nfrom pathlib import Path\n"
        "def encode(record, parameters):\n"
        "    Path(__file__).with_name('text-0.txt').write_text('drift')\n"
        "    return {'latents': torch.ones(1)}\n"
    )
    with pytest.raises(ValueError, match="source.*(changed|hash)"):
        prepare(raw, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def preencoded_fixture(root, value=None, **feature_options):
    feature = root / "native.pt"
    torch.save(value if value is not None else {"video_latents": torch.ones((2, 3, 5), dtype=torch.bfloat16),
                                              "conditioning": (torch.arange(7),)}, feature)
    digest = hashlib.sha256(feature.read_bytes()).hexdigest()
    return write_json(root / "preencoded.json", {
        "schema_version": "distillschedule.data.preencoded.v1",
        "metadata": {"native_format": "test-native-v1"},
        "records": [{"id": "native-0", "features": {"path": feature.name, "sha256": digest,
                                                      **feature_options}}],
    })


def test_import_preserves_arbitrary_native_shapes_dtypes_and_tree(tmp_path):
    source = preencoded_fixture(tmp_path)
    manifest = import_preencoded(source, tmp_path / "prepared")
    dataset = PreparedDataset(manifest)
    value = dataset.sample(0)
    assert value["video_latents"].shape == (2, 3, 5)
    assert value["video_latents"].dtype == torch.bfloat16
    assert isinstance(value["conditioning"], tuple)
    assert torch.equal(value["conditioning"][0], torch.arange(7))
    assert dataset.validate()["samples"] == 1


@pytest.mark.parametrize("mutation,match", [
    (lambda value: value["records"][0]["features"].update(sha256="not-a-hash"), "SHA256"),
    (lambda value: value["records"][0]["features"].update(sha256="0" * 64), "hash"),
    (lambda value: value["records"].append(dict(value["records"][0])), "duplicate"),
])
def test_preencoded_import_rejects_invalid_hashes_and_ids(tmp_path, mutation, match):
    source = preencoded_fixture(tmp_path)
    value = json.loads(source.read_text())
    mutation(value)
    write_json(source, value)
    with pytest.raises(ValueError, match=match):
        import_preencoded(source, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_import_rejects_non_tensor_payload_with_safe_weights_only_loader(tmp_path):
    source = preencoded_fixture(tmp_path, {"latent": Path("not-an-allowlisted-tensor")})
    with pytest.raises(ValueError, match="safe|tensor"):
        import_preencoded(source, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_malformed_and_false_tensor_metadata_rejected(tmp_path):
    raw = raw_fixture(tmp_path)
    manifest = prepare(raw, tmp_path / "prepared")
    value = json.loads(manifest.read_text())
    tensors = value["records"][0]["features"]["tensors"]
    tensors["items"]["latents"]["shape"][0] = -1
    write_json(manifest, value)
    with pytest.raises(ValueError, match="shape"):
        PreparedDataset(manifest)
    tensors["items"]["latents"]["shape"][0] = 99
    write_json(manifest, value)
    with pytest.raises(ValueError, match="metadata"):
        PreparedDataset(manifest).validate()


def test_prepared_features_cannot_reference_outside_dataset(tmp_path):
    raw = raw_fixture(tmp_path)
    manifest = prepare(raw, tmp_path / "prepared")
    value = json.loads(manifest.read_text())
    value["records"][0]["features"]["path"] = "../native.pt"
    write_json(manifest, value)
    with pytest.raises(ValueError, match="inside|relative"):
        PreparedDataset(manifest)


@pytest.mark.parametrize("mutation,match", [
    (lambda value: value["provenance"]["encoder"].update(parameters_sha256="0" * 64), "parameters"),
    (lambda value: value["provenance"]["sources"][0]["media"]["video"].update(sha256="invalid"), "SHA256"),
    (lambda value: value["provenance"]["sources"].pop(), "source"),
])
def test_prepared_provenance_cannot_omit_or_falsify_required_identity(tmp_path, mutation, match):
    raw = raw_fixture(tmp_path)
    manifest = prepare(raw, tmp_path / "prepared")
    value = json.loads(manifest.read_text())
    mutation(value)
    write_json(manifest, value)
    with pytest.raises(ValueError, match=match):
        PreparedDataset(manifest)


def test_manifest_bytes_bind_identity_and_open_reader(tmp_path):
    raw = raw_fixture(tmp_path)
    manifest = prepare(raw, tmp_path / "prepared")
    dataset = PreparedDataset(manifest)
    original_identity = dataset.identity
    value = json.loads(manifest.read_text())
    value["metadata"]["native_format"] = "another explicit model identity"
    write_json(manifest, value)
    with pytest.raises(ValueError, match="manifest hash"):
        dataset.sample(0)
    assert PreparedDataset(manifest).identity != original_identity


@pytest.mark.parametrize("width,data,expected", [
    (1, bytes([0, 128, 255]), [-1.0, 0.0, 127 / 128]),
    (2, b"\x00\x80\x00\x00\xff\x7f", [-1.0, 0.0, 32767 / 32768]),
    (3, b"\x00\x00\x80\x00\x00\x00\xff\xff\x7f", [-1.0, 0.0, 8388607 / 8388608]),
    (4, b"\x00\x00\x00\x80\x00\x00\x00\x00\xff\xff\xff\x7f", [-1.0, 0.0, 1.0]),
])
def test_pcm_audio_widths_preserve_rate_and_samples(tmp_path, width, data, expected):
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(width)
        output.setframerate(22050)
        output.writeframes(data)
    decoded = decode_record({"id": "pcm", "audio": str(audio)})["audio"]
    assert decoded["sample_rate"] == 22050
    assert decoded["samples"].dtype == torch.float32
    assert decoded["samples"].shape == (3, 1)
    assert torch.allclose(decoded["samples"][:, 0], torch.tensor(expected))


def test_varying_frame_sizes_are_rejected_not_resized(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "0.ppm").write_bytes(b"P6\n2 2\n255\n" + bytes([0, 0, 0]) * 4)
    (frames / "1.ppm").write_bytes(b"P6\n4 4\n255\n" + bytes([0, 0, 0]) * 16)
    with pytest.raises(ValueError, match="no implicit resize"):
        decode_record({"id": "bad-frames", "video": str(frames)})
