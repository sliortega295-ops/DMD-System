"""Actual current-G native LiveWan generation; random complete-class CPU fixtures."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from test_training_livewan_public_cpu import bounded, equal, json_file


def fresh_inputs(root, *, world_frames=6, raw=None):
    from safetensors.torch import save_file
    from distillschedule.data.livewan_inference_inputs import SCHEMA, FORMAT, producer_identity, prepare_inputs
    from distillschedule.training.livewan_public import SCOPE, PRECISION, GEOMETRY, BACKEND, FIXTURE, pin_file
    root.mkdir(parents=True)
    if raw is None:
        gen = torch.Generator().manual_seed(5411)
        raw = dict(world=torch.randn(1, 16, world_frames, 4, 4, generator=gen),
                   context=torch.randn(1, 512, 4096, generator=gen))
    path = root / "NEW-world-and-positive.safetensors"
    save_file(raw, str(path), metadata=dict(evidence_kind=FIXTURE))
    source = json_file(root / "fresh.json", dict(schema_version=SCHEMA, format=FORMAT, scope=SCOPE,
        precision=PRECISION, geometry=GEOMETRY, backend=BACKEND, producer=producer_identity(), evidence_kind=FIXTURE,
        records=[dict(id="NEW-supplied-clean-world-positive-not-training-row", source=pin_file(path), world_frames=world_frames)]))
    return prepare_inputs(source, root / "prepared")


def inference_config(root, manifest, prepared, *, seed=729):
    from distillschedule.training.config import TrainingConfig
    return TrainingConfig.from_dict(dict(schema_version="distillschedule.training.v1",
        model=dict(factory=manifest["model"]["factory"], parameters=manifest["model"]["parameters"]),
        algorithm=dict(factory="distillschedule.training.livewan_inference:refuse_training"),
        data=dict(prepared=str(prepared)), execution=dict(adapter="program", mode="reference"),
        trainer=dict(cycles=1, seed=seed, device="cpu", output_dir=str(root / "not-training"))),
        source=root / "inference.json")


def native_control(student, row, *, seed):
    from distillschedule.reference import livewan_wan21 as provider
    from distillschedule.training import session
    from distillschedule.training.livewan_inference import STREAMER
    api = provider.load_cpu(backend="torch-sdpa-cpu", precision="float32")
    session._seed(seed)
    cfg = {k: v for k, v in STREAMER.items() if k not in {"rope_gap", "latent_norm"}}
    with torch.no_grad():
        stream = api.stream.FewStepStreamer(student, device="cpu", dtype=torch.float32, **cfg)
        stream.set_text(student.text_embedding(row["context"]))
        stream.set_world(row["world"])
        stream.rope_gap = 0
        values = [stream.generate_block() for _ in range((21-row["world"].shape[2])//3)]
    return torch.cat([row["world"], *values], dim=2)


@pytest.fixture(scope="module")
def portable(tmp_path_factory):
    from distillschedule.data import PreparedDataset
    from distillschedule.training import session
    from distillschedule.training.checkpoint import rng_state
    from distillschedule.training.inference import restore_modules
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.livewan_public import content_digest
    from test_training_livewan_public_cpu import fixture_files, config_for
    root = tmp_path_factory.mktemp("livewan-current-G-inference")
    old = root / "training-payloads"
    files = fixture_files(old, count=2)
    config = config_for(files, old / "run", cycles=1)
    result = session.train(config)
    components, data, _, _, _ = session._load_for_output(config, result["checkpoint"])
    owner = components.state["native_cycle"]
    before, rng = owner.state_dict(), rng_state()
    ids = ({n: id(p) for n, p in owner._named.items()}, {n: m.training for n, m in owner._modules.items()},
           {n: p.requires_grad for n, p in owner._named.items()})
    forbidden = {fn.__code__ for fn in (torch.nn.Module.eval, torch.nn.Module.train, torch.nn.Module.requires_grad_,
                                      torch.nn.Module.to, owner.context.api.model.WanModel.forward)}
    calls = []
    def observe(frame, event, arg):
        if event == "call" and frame.f_code in forbidden:
            calls.append(frame.f_code.co_name)
    try:
        sys.setprofile(observe)
        projection = components.inference_projection(components, {"seed": 773})
    finally:
        sys.setprofile(None)
    assert not calls
    assert projection.components.state == {"student": owner.models["student"]}
    assert projection.components.generate is projection.components.decode is None
    assert equal(before, owner.state_dict()) and equal(rng, rng_state())
    assert ids == ({n: id(p) for n, p in owner._named.items()}, {n: m.training for n, m in owner._modules.items()},
                   {n: p.requires_grad for n, p in owner._named.items()})
    manifest = session.export(config, result["checkpoint"], root / "export")
    weights = torch.load(root / "export/weights.pt", weights_only=True)
    assert set(weights) == {"student"}
    assert equal(weights["student"], dict(owner.models["student"].state_dict()))
    assert manifest["model"]["parameters"] == dict(projection.parameters)
    assert {"optional_codec", "pinned_source_importer"} <= manifest["model"]["parameters"]["source"]["helpers"].keys()
    serialized = json.dumps(manifest["model"]["parameters"])
    assert str(old) not in serialized and "asset_manifest" not in serialized
    fresh = fresh_inputs(root / "NEW-inputs")
    row = PreparedDataset(fresh).sample(0)
    cfg = inference_config(root, manifest, fresh)
    factory, _ = load_callable(manifest["model"]["factory"])
    inference = factory({**manifest["model"]["parameters"], "device": "cpu"})
    with pytest.raises(ValueError, match="strict exported weights"):
        inference.generate(inference.state, row, {})
    restore_modules(inference, root / "export", manifest)
    assert equal(inference.state["student"].state_dict(), owner.models["student"].state_dict())
    expected = native_control(owner.models["student"], row, seed=729)
    generated = session.generate_from_export(cfg, root / "export", root / "expected", prepared=fresh, output_kind="latents")
    values = torch.load(generated["artifact"], weights_only=True)
    assert torch.equal(expected, values["generated"]["latents"])
    assert generated["output_kind"] == "latents" and generated["video"]["status"] == "NOT_AVAILABLE"
    assert not (root / "expected/frames").exists()
    # Projection/export must not change the NEXT actual training cycle either.
    other, _, _, _, _ = session._load_for_output(config, result["checkpoint"])
    from distillschedule.data.livewan_reference_inputs import decode_cycle
    next_row = decode_cycle(data.sample(1), cycle=1)
    assert equal(owner.run(next_row, cycle=1), other.state["native_cycle"].run(decode_cycle(data.sample(1), cycle=1), cycle=1))
    assert equal(owner.state_dict(), other.state["native_cycle"].state_dict())
    json_file(root / "inference.json", cfg.to_dict())
    json_file(root / "proof.json", dict(status="PASS_CURRENT_G_NATIVE_FULL_STEP_EXACT_NO_QUALITY_CLAIM",
        cycle=1, generator=1, critic=5, ema=0, projection_read_only=True, next_cycle_exact=True,
        current_student_weights_exact=True, native_control_exact=True,
        latent_content=content_digest(values["generated"]), training_process_pid=os.getpid(),
        quality_validation="NOT_RUN", native_world16="NOT_RUN", c51_restart="NOT_IMPLEMENTED", claimable_speedup=False))
    shutil.rmtree(old)
    assert not old.exists()
    return root, manifest, fresh


def test_public_inference_factory_is_explicit_and_not_training_owner():
    from distillschedule.training import livewan_inference
    assert livewan_inference.FACTORY == "distillschedule.training.livewan_inference:build_inference_model"


def test_optional_codec_has_its_own_original_wan21_source_pin():
    from distillschedule.training import wan21_codec
    assert wan21_codec.SOURCE_FILES["wan/modules/vae.py"] == "4912ffd7a9fbb17ceef9a17d4a9c4b8aee36ec37c32594f54e57693b29909498"


def test_current_g_exact_fresh_process_after_all_training_assets_and_owners_removed(portable):
    root, manifest, fresh = portable
    script = r'''
import json,os,sys
from pathlib import Path
import torch
from distillschedule.training import session,livewan_public
from distillschedule.training.config import load_training_config
root=Path(sys.argv[1]);assert not (root/'training-payloads').exists()
def forbidden(*args,**kwargs):raise AssertionError('training owner/Teacher/C/Adam constructed during exported inference')
livewan_public.LiveWanCPUOwner.__init__=forbidden
livewan_public.build_model=forbidden
torch.optim.AdamW.__init__=forbidden
cfg=load_training_config(root/'inference.json')
result=session.generate_from_export(cfg,root/'export',root/'fresh-process',prepared=cfg.data['prepared'],output_kind='latents')
expected=torch.load(root/'expected/generation.pt',weights_only=True)['generated']['latents']
actual=torch.load(result['artifact'],weights_only=True)['generated']['latents']
assert torch.equal(actual,expected)
assert not torch.cuda.is_initialized()
assert not any(n=='wan' or n.startswith('wan.') or n=='wanstreamer' or n.startswith('wanstreamer.') for n in sys.modules)
(root/'fresh-process-proof.json').write_text(json.dumps(dict(status='PASS_FRESH_OS_CURRENT_G_EXACT_TRAINING_REMOVED',pid=os.getpid(),seed=result['inference_seed'],shape=list(actual.shape))))
'''
    result = subprocess.run([sys.executable, "-B", "-c", script, str(root)], cwd=Path.cwd(),
                            env=dict(os.environ), text=True, capture_output=True, timeout=150)
    assert result.returncode == 0, result.stdout+result.stderr
    assert json.loads((root / "fresh-process-proof.json").read_text())["pid"] != os.getpid()
    evidence = os.environ.get("DMD_W4_INFERENCE_EVIDENCE")
    if evidence:
        target = Path(evidence)
        target.mkdir(parents=True, exist_ok=True)
        for name in ("proof.json", "fresh-process-proof.json"):
            shutil.copy2(root / name, target / name)


def test_native_call_census_fresh_cache_rng_continuation_and_current_conditions(portable, tmp_path):
    from distillschedule.training import session
    from distillschedule.reference import livewan_wan21 as provider
    from distillschedule.training.inference import restore_modules
    from distillschedule.training.livewan_inference import build_inference_model
    from distillschedule.data import PreparedDataset
    root, manifest, fresh = portable
    cfg = inference_config(root, manifest, fresh)
    model = build_inference_model(manifest["model"]["parameters"])
    restore_modules(model, root / "export", manifest)
    api = provider.load_cpu(backend="torch-sdpa-cpu", precision="float32")
    row = PreparedDataset(fresh).sample(0)
    events, caches = [], []
    def observe(frame, event, arg):
        if event == "call" and frame.f_code is api.stream.FewStepStreamer._fwd.__code__:
            events.append(frame.f_locals["t"])
        if event == "return" and frame.f_code is api.stream.FewStepStreamer.__init__.__code__:
            caches.append(frame.f_locals["self"].cache)
    session._seed(729)
    try:
        sys.setprofile(observe)
        actual = model.generate(model.state, row, {})["latents"]
    finally:
        sys.setprofile(None)
    assert events == [0.0]+[1.0, .75, .5, .25, 0.0]*5
    session._seed(729)
    again = model.generate(model.state, row, {})["latents"]
    assert torch.equal(actual, again)
    assert torch.equal(actual, native_control(model.state["student"], row, seed=729))
    session._seed(730)
    changed_seed = model.generate(model.state, row, {})["latents"]
    assert torch.equal(changed_seed[:, :, :6], actual[:, :, :6])
    assert not torch.equal(changed_seed[:, :, 6:], actual[:, :, 6:])
    for key in ("world", "context"):
        raw = {k: row[k].clone() for k in ("world", "context")}
        raw[key].add_(.125)
        new = fresh_inputs(tmp_path / key, raw=raw)
        session._seed(729)
        changed = model.generate(model.state, PreparedDataset(new).sample(0), {})["latents"]
        assert not torch.equal(changed[:, :, 6:], actual[:, :, 6:])
    assert caches[0].num_tokens == 21*4


@pytest.mark.parametrize("change", ["source", "codec_source", "importer_source", "runtime", "dtype", "shape", "capacity", "sampler", "architecture", "training_path"])
def test_inference_factory_rejects_unbound_or_changed_contract_before_use(portable, change):
    from distillschedule.training.livewan_inference import build_inference_model
    _, manifest, _ = portable
    p = deepcopy(manifest["model"]["parameters"])
    if change == "source": p["source"]["helpers"]["provider"] = "0"*64
    elif change == "codec_source": p["source"]["helpers"]["optional_codec"] = "0"*64
    elif change == "importer_source": p["source"]["helpers"]["pinned_source_importer"] = "0"*64
    elif change == "runtime": p["runtime"]["torch"]["version"] = "unqualified"
    elif change == "dtype": p["precision"] = "BF16"
    elif change == "shape": p["geometry"]["latent_shape"][-1] = 80
    elif change == "capacity": p["budget"]["max_parameters"] = 1
    elif change == "sampler": p["streamer"]["sampler"] = "euler"
    elif change == "architecture": p["architecture"]["dim"] = 1_000_000
    else: p["asset_manifest"] = "/nonexistent/training-assets.json"
    with pytest.raises(ValueError): build_inference_model(p)


def test_no_decoder_fails_without_publication_and_cli_reaches_native_latents(portable, tmp_path):
    from distillschedule.training import session
    root, manifest, fresh = portable
    cfg = inference_config(root, manifest, fresh)
    with pytest.raises(ValueError, match="--output-kind latents"):
        session.generate_from_export(cfg, root / "export", tmp_path / "no-video", prepared=fresh, output_kind="video")
    assert not (tmp_path / "no-video").exists()
    args = [sys.executable, "-B", "scripts/distillctl.py", "generate", str(root / "inference.json"),
            "--weights", str(root / "export"), "--prepared", str(fresh), "--output", str(tmp_path / "cli-latents"),
            "--output-kind", "latents"]
    result = subprocess.run(args, cwd=Path.cwd(), env=dict(os.environ), text=True, capture_output=True, timeout=150)
    assert result.returncode == 0, result.stdout+result.stderr
    actual = torch.load(tmp_path / "cli-latents/generation.pt", weights_only=True)["generated"]
    assert equal(actual, torch.load(root / "expected/generation.pt", weights_only=True)["generated"])
    assert not (tmp_path / "cli-latents/frames").exists()


def test_cold_or_poisoned_owner_cannot_be_projected(tmp_path):
    from test_training_livewan_public_cpu import fixture_files
    from distillschedule.training.livewan_public import build_model
    files = fixture_files(tmp_path / "assets", count=1)
    components = build_model(dict(asset_manifest=str(files["assets"])))
    with pytest.raises(ValueError, match="complete original G1/C5"):
        components.inference_projection(components, {"seed": 5})
    components.state["native_cycle"].poisoned = True
    with pytest.raises(ValueError, match="poisoned"):
        components.inference_projection(components, {"seed": 5})


@pytest.mark.parametrize("world_frames", [3, 6, 9])
def test_fresh_inference_rows_require_supplied_complete_world_and_positive_only(tmp_path, world_frames):
    from distillschedule.data import PreparedDataset
    from distillschedule.data.livewan_inference_inputs import decode_row
    from distillschedule.training.livewan_public import FIXTURE
    fresh = fresh_inputs(tmp_path / "fresh", world_frames=world_frames)
    feeds = PreparedDataset(fresh).sample(0)
    row = decode_row(feeds, evidence_kind=FIXTURE)
    assert row["world"].shape == (1, 16, world_frames, 4, 4)
    for key in ("world", "context"):
        changed = deepcopy(feeds)
        changed[key] = changed[key].bfloat16()
        with pytest.raises(ValueError, match="FP32"):
            decode_row(changed, evidence_kind=FIXTURE)
    changed = deepcopy(feeds)
    changed["world"].add_(1)
    with pytest.raises(ValueError, match="content"):
        decode_row(changed, evidence_kind=FIXTURE)


@pytest.fixture(scope="module")
def codec_files(tmp_path_factory):
    from safetensors.torch import save_file
    from distillschedule.training import wan21_codec as codec
    from distillschedule.training.livewan_public import pin_file
    upstream = os.environ.get("DMD_WAN21_TEST_UPSTREAM")
    if not upstream:
        pytest.skip("set DMD_WAN21_TEST_UPSTREAM to the separately hash-pinned ORIGINAL Wan2.1 source")
    root = tmp_path_factory.mktemp("original-wan21-random-codec")
    source = root / "source"
    for name in codec.SOURCE_FILES:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(upstream) / name, target)
    loader, api = codec._source(codec.source_spec(source))
    architecture = dict(codec.ARCHITECTURE, dim=4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(303)
        native = api.WanVAE_(**architecture)
    weights = root / "RANDOM-NOT-PRETRAINED.safetensors"
    save_file({k: v.detach().contiguous() for k, v in native.state_dict().items()}, str(weights), metadata=dict(evidence_kind=codec.FIXTURE))
    manifest = dict(schema_version=codec.SCHEMA, scope=codec.SCOPE, precision=codec.PRECISION,
        source=codec.source_spec(source), runtime=codec.runtime_identity(), architecture=architecture,
        normalization=codec.NORMALIZATION, decode_policy=codec.POLICY, geometry=codec.GEOMETRY,
        evidence_kind=codec.FIXTURE, weight_provenance=dict(status="RANDOM_NOT_PRETRAINED",
            origin="bounded original WanVAE_ complete-class CPU test fixture", license="test-generated random values; no pretrained weight license asserted"),
        weights=[pin_file(weights)], budget=dict(max_parameters=2_000_000, max_loaded_bytes=64_000_000,
            max_activation_elements=100_000, max_attention_elements=100_000))
    path = json_file(root / "decoder.json", manifest)
    return root, path, native, loader, api


def test_actual_original_codec_21_to81_uint8_exact_native_decode_once_no_raw_clamp(codec_files):
    from distillschedule.training import wan21_codec as codec
    from distillschedule.training.livewan_inference import latent_schema
    from distillschedule.training.livewan_public import pin_file, content_digest
    from distillschedule.training.checkpoint import rng_state
    root, manifest, native, loader, api = codec_files
    gen = torch.Generator().manual_seed(945)
    z = 6*torch.randn(1, 16, 21, 4, 4, generator=gen)
    raw = content_digest(z)
    generated = dict(latents=z, latent_schema=latent_schema())
    before = rng_state()
    actual = codec.decode_video(generated, pin_file(manifest))
    assert equal(before, rng_state()) and content_digest(z) == raw
    mean = torch.tensor(codec.NORMALIZATION["mean"], dtype=torch.float32)
    std = torch.tensor(codec.NORMALIZATION["std"], dtype=torch.float32)
    native.eval().requires_grad_(False)
    with torch.no_grad():
        direct = native.decode(z.clamp(-4, 4), [mean, 1/std]).clamp(-1, 1)
    expected = ((direct[0].permute(1, 2, 3, 0)+1)/2*255).clamp(0, 255).to(torch.uint8)
    assert torch.equal(actual["video"], expected)
    assert actual["video"].shape == (81, 32, 32, 3) and actual["video"].dtype == torch.uint8
    assert actual["decoder"]["evidence_kind"] == codec.FIXTURE
    assert actual["decoder"]["source"]["revision"] == codec.REVISION
    assert actual["decoder"]["decode_policy"]["raw_latents"] == "UNCHANGED"
    assert all(v is None for v in native._feat_map)
    assert not any(name == "wan" or name.startswith("wan.") for name in sys.modules)
    evidence = os.environ.get("DMD_W4_INFERENCE_EVIDENCE")
    if evidence:
        json_file(Path(evidence) / "random-codec-proof.json", dict(status="PASS_ACTUAL_ORIGINAL_WAN21_NATIVE_DECODE_EXACT_RANDOM_NOT_PRETRAINED",
            output_shape=list(actual["video"].shape), native_pixels_exact=True, raw_latents_unchanged=True,
            normalization_applied_once=True, decoder=actual["decoder"], pixel_content=content_digest(actual["video"])))


@pytest.mark.parametrize("change", ["source", "license", "runtime", "config", "normalization", "policy", "capacity", "corrupt", "dtype", "shape", "nan"])
def test_codec_rejects_untrusted_changed_config_weights_and_capacity(codec_files, tmp_path, change):
    from safetensors.torch import load_file, save_file
    from distillschedule.training import wan21_codec as codec
    from distillschedule.training.livewan_public import pin_file
    root, path, _, _, _ = codec_files
    value = json.loads(path.read_bytes())
    if change == "source": value["source"]["revision"] = "0"*40
    elif change == "license":
        copy = tmp_path / "source"
        shutil.copytree(root / "source", copy)
        (copy / "LICENSE.txt").write_text("wrong license")
        value["source"] = codec.source_spec(copy)
    elif change == "runtime": value["runtime"]["torch"]["version"] = "unqualified"
    elif change == "config": value["architecture"]["temperal_downsample"] = [True, True, False]
    elif change == "normalization": value["normalization"]["mean"][0] += .1
    elif change == "policy": value["decode_policy"]["latent_copy_clamp"] = [-1, 1]
    elif change == "capacity": value["budget"]["max_parameters"] = 1
    else:
        weight = tmp_path / "invalid.safetensors"
        shutil.copy2(value["weights"][0]["path"], weight)
        if change == "corrupt":
            value["weights"] = [pin_file(weight)]
            with weight.open("ab") as stream: stream.write(b"corrupt")
        else:
            raw = load_file(str(weight))
            key = next(iter(raw))
            if change == "dtype": raw[key] = raw[key].bfloat16()
            elif change == "shape": raw.pop(key)
            else: raw[key].view(-1)[0] = float("nan")
            save_file(raw, str(weight), metadata=dict(evidence_kind=codec.FIXTURE))
            value["weights"] = [pin_file(weight)]
    manifest = json_file(tmp_path / "bad-codec.json", value)
    with pytest.raises((ValueError, RuntimeError)):
        codec.admit_manifest(pin_file(manifest))


def test_cli_actual_original_optional_video_parameter_and_no_fallback(portable, codec_files, tmp_path):
    from distillschedule.training.livewan_public import pin_file
    root, manifest, fresh = portable
    _, decoder, _, _, _ = codec_files
    parameters = json_file(tmp_path / "parameters.json", dict(decoder_manifest=pin_file(decoder)))
    args = [sys.executable, "-B", "scripts/distillctl.py", "generate", str(root / "inference.json"),
            "--weights", str(root / "export"), "--prepared", str(fresh), "--parameters", str(parameters),
            "--output-kind", "video", "--output", str(tmp_path / "cli-video")]
    result = subprocess.run(args, cwd=Path.cwd(), env=dict(os.environ), text=True, capture_output=True, timeout=150)
    assert result.returncode == 0, result.stdout+result.stderr
    receipt = json.loads((tmp_path / "cli-video/generation.json").read_bytes())
    assert receipt["video"]["shape"] == [81, 32, 32, 3]
    assert len(list((tmp_path / "cli-video/frames").glob("*.png"))) == 81
    values = torch.load(tmp_path / "cli-video/generation.pt", weights_only=True)
    assert values["decoded"]["decoder"]["evidence_kind"] == "SYNTHETIC_RANDOM_ORIGINAL_WAN21_CODEC_NOT_PRETRAINED"
    assert equal(values["generated"], torch.load(root / "expected/generation.pt", weights_only=True)["generated"])
    # A declared broken codec must fail even if latent output was requested.
    parameters.write_text(json.dumps(dict(decoder_manifest={**pin_file(decoder), "sha256": "0"*64})))
    args[-1] = str(tmp_path / "broken-no-fallback")
    args[args.index("--output-kind")+1] = "latents"
    failed = subprocess.run(args, cwd=Path.cwd(), env=dict(os.environ), text=True, capture_output=True, timeout=150)
    assert failed.returncode != 0 and not (tmp_path / "broken-no-fallback").exists()
