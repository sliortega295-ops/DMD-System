"""CURRENT unmerged original Student, cached AV LATENT DIAGNOSTIC only."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from test_training_h3_cached_public_cpu import make_case, request as training_request, json_file, pin
from test_h3_cached_original_cpu import original, equal_tree, sample
from distillschedule.data.h3_cached_inputs import prepare_inputs, decode_row
from distillschedule.training import session
from distillschedule.training.config import TrainingConfig
from distillschedule.workloads.best_known.registry import file_sha256, repository_root


def trained(root, cycles=2):
    assets, sources = make_case(root, count=4)
    prepared = prepare_inputs(sources, root / "prepared")
    config = training_request(root / "run", assets, prepared, cycles)
    result = session.train(config)
    return config, result


def new_inputs(root, *, raw=None, packed=False):
    from distillschedule.data.h3_cached_inputs import SCHEMA
    from distillschedule.reference import diffsynth_h3_cached as provider
    from distillschedule.training.h3_cached_assets import FIXTURE, PRECISION
    root.mkdir(parents=True)
    if raw is None:
        original().training.seed_all(2981)
        raw = sample(71)
    if packed:
        parsed, _ = original().dmd2.prepare_batch(raw)
        raw = (dict(input_latents=raw["input_latents"], audio_input_latents=raw["audio_input_latents"],
                    ref_visual_anchor=parsed["ref_visual_anchor"], ref_audio_anchor=parsed["ref_audio_anchor"],
                    imgvid_cond_noise_aug=1.0, audio_cond_noise_aug=1.0),
               dict(prompt_embeds=raw["prompt_embeds"], packed=parsed["packed"]), {})
    torch.save(raw, root / "new-cache.pth")
    parsed, _ = original().dmd2.prepare_batch(raw)
    geometry = dict(video_shape=list(parsed["input_latents"].shape), audio_shape=list(parsed["audio_input_latents"].shape),
        prompt_shape=list(parsed["prompt_embeds"].shape), reference_video_shape=[1, 24, 3, 4, 4],
        reference_audio_shape=[2, 32, 5])
    path = root / "new-cache.pth"
    sources = json_file(root / "inputs.json", dict(schema_version=SCHEMA, evidence_kind=FIXTURE,
        source=provider.source_identity(), precision=PRECISION, geometry=geometry,
        records=[dict(id="NEW-genuine-cached-AV-condition", cache=pin(path))]))
    return prepare_inputs(sources, root / "prepared")


def inference_config(root, manifest, prepared):
    return TrainingConfig.from_dict(dict(schema_version="distillschedule.training.v1",
        model=dict(factory=manifest["model"]["factory"], parameters=manifest["model"]["parameters"]),
        algorithm=dict(factory="distillschedule.training.h3_cached_inference:refuse_training"),
        data=dict(prepared=str(prepared)), execution=dict(adapter="program", mode="reference"),
        trainer=dict(cycles=1, seed=729, device="cpu", output_dir=str(root / "not-training"))),
        source=root / "inference.json")


def process(script, root):
    result = subprocess.run([sys.executable, "-B", "-c", script, str(root)], cwd=repository_root(),
        env=dict(os.environ), text=True, capture_output=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module", params=[False, True], ids=["flat", "packed"])
def portable(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("h3-current-student-" + ("packed" if request.param else "flat"))
    old = root / "training-payloads"
    assets, sources = make_case(old, count=4, packed=request.param)
    prepared = prepare_inputs(sources, old / "prepared")
    config = training_request(old / "partial", assets, prepared, 2)
    partial = session.train(config)
    config = training_request(old / "resumed", assets, prepared, 3, resume=partial["checkpoint"])
    json_file(root / "resume.json", config.to_dict())
    fresh = new_inputs(root / "NEW-inputs", packed=request.param)
    exported = process(r'''
import hashlib,json,os,sys
from copy import deepcopy
from pathlib import Path
import torch
from distillschedule.training import session
from distillschedule.training.config import load_training_config
from distillschedule.training.inference import restore_modules
from distillschedule.training.plugins import load_callable
from distillschedule.data import PreparedDataset
from distillschedule.reference import diffsynth_h3_cached as provider
from distillschedule.workloads.best_known.registry import file_sha256
from test_training_h3_cached_inference_cpu import inference_config
root=Path(sys.argv[1]);config=load_training_config(root/'resume.json')
resumed=session.train(config)
assert resumed['resume_cycle']==2 and resumed['completed_cycles']==3
load=session._load_for_output;captured=[]
def capture(*args,**kwargs):
    result=load(*args,**kwargs);captured.append(result[0]);return result
session._load_for_output=capture
manifest=session.export(config,resumed['checkpoint'],root/'export')
owner=captured[0].state['native_update'];assert owner._last_metrics['gan_active']
current={n:t for n,t in owner.adapter.state_dict().items() if '.adapters.fake_score.' not in n}
weights=torch.load(root/'export/weights.pt',weights_only=True)
assert set(weights)=={'student'} and weights['student'].keys()==current.keys()
assert all(torch.equal(weights['student'][n],t) for n,t in current.items())
assert any('adapters.student.B.weight' in n and torch.count_nonzero(t) for n,t in current.items())
assert not any('fake_score' in n or 'head' in n for n in current)
assert manifest['identity']['training_provenance']==resumed['identity']
assert manifest['export']['format']=='torch_state_dicts'
assert manifest['model']['parameters']['source']['source16']==provider.source_identity()
factory,_=load_callable(manifest['model']['factory'])
inference=factory({**manifest['model']['parameters'],'device':'cpu'})
restore_modules(inference,root/'export',manifest)
restored_student=inference.state['student'].state_dict()
assert all(torch.equal(restored_student[n],t) for n,t in current.items())
cfg=inference_config(root,manifest,root/'NEW-inputs/prepared/manifest.json')
(root/'inference.json').write_text(json.dumps(cfg.to_dict()))
actual=session.generate_from_export(cfg,root/'export',root/'expected',prepared=cfg.data['prepared'],output_kind='latents')
api=provider.load_cpu(backend='torch-dense-cpu')
raw=PreparedDataset(cfg.data['prepared']).sample(0)['raw'];inputs,negative=api.dmd2.prepare_batch(raw,1.0)
assert negative is None
direct=deepcopy(owner.adapter);session._seed(729)
values,_=api.dmd2.rollout(direct,(inputs['input_latents'],inputs['audio_input_latents']),inputs,api.dmd2.make_schedule(owner.config))
generated=torch.load(actual['artifact'],weights_only=True)['generated']
assert torch.equal(values[0],generated['video_latents']) and torch.equal(values[1],generated['audio_latents'])
hashes={n:hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest() for n,t in current.items()}
(root/'student-hashes.json').write_text(json.dumps(hashes,sort_keys=True))
(root/'export-proof.json').write_text(json.dumps(dict(status='PASS_CURRENT_FULL_BASE_UNMERGED_STUDENT_NOT_QUALITY',
    pid=os.getpid(),fresh_process_resume_cycle=2,completed_cycles=3,gan_start=2,protocol='GAN_START_2_TEST_FIXTURE',
    gan_active=True,state_keys=len(hashes),full_weights_sha256=file_sha256(root/'export/weights.pt'),
    source16=provider.source_identity(),current_weights_exact=True,direct_original_eight_step_exact=True,
    fake_discriminator_optimizer_rng_weights=False,quality_validation='NOT_RUN',claimable_speedup=False)))
print('FRESH_RESUME_GAN2_CURRENT_STUDENT_FULL_WEIGHTS_AND_DIRECT_ORIGINAL_EXACT_PASS')
''', root)
    shutil.rmtree(old)
    (root / "resume.json").unlink()
    assert not old.exists()
    assert json.loads((root / "export-proof.json").read_text())["pid"] != os.getpid()
    evidence = os.environ.get("H3_CACHED_INFERENCE_EVIDENCE")
    if evidence:
        destination = Path(evidence) / ("packed" if request.param else "flat")
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "export-proof.json", destination / "export-proof.json")
        (destination / "export-process.stdout").write_text(exported.stdout)
    return root, json.loads((root / "export/export.json").read_text()), fresh, request.param


def test_projection_is_student_only_nonmutating_and_next_update_exact(tmp_path):
    from distillschedule.training.checkpoint import rng_state
    config, result = trained(tmp_path)
    components, data, _, _, _ = session._load_for_output(config, result["checkpoint"])
    owner = components.state["native_update"]
    assert callable(components.inference_projection), "W6 has no current-Student inference projection"
    before, rng = owner.state_dict(), rng_state()
    ids = ({n: id(p) for n, p in owner._named_parameters().items()},
           {n: id(m) for n, m in owner._modules().items()})
    parameter_ids = set(ids[0].values())
    forbidden = {fn.__code__ for fn in (torch.nn.Module.eval, torch.nn.Module.train, torch.nn.Module.requires_grad_,
        torch.nn.Module.to, original().dmd2.H3Adapter.select, original().dmd2.H3Adapter.forward,
        original().model.MiniMaxH3DiT.forward)}

    def observe(action):
        def profile(frame, event, arg):
            if event == "call" and frame.f_code in forbidden:
                module = frame.f_locals.get("self")
                assert not parameter_ids.intersection(id(p) for p in module.parameters()), "mutating/forward API on shared view"
        previous = sys.getprofile()
        try:
            sys.setprofile(profile)
            return action()
        finally:
            sys.setprofile(previous)

    projection = observe(lambda: components.inference_projection(components, {"seed": 777}))
    assert projection.components.generate is None
    student = projection.components.state["student"]
    assert type(student) is original().dmd2.H3Adapter and student is not owner.adapter
    expected = {n: t for n, t in owner.adapter.state_dict().items() if ".adapters.fake_score." not in n}
    equal_tree(expected, student.state_dict())
    assert not any("fake_score" in n for n in student.state_dict())
    for name, module in student.named_modules():
        live = owner.adapter.get_submodule(name)
        assert type(module) is type(live)
        assert module is not live and module._modules is not live._modules
        assert module._parameters is not live._parameters and module._buffers is not live._buffers
    live_parameters = dict(owner.adapter.named_parameters())
    for name, parameter in student.named_parameters():
        assert parameter is live_parameters[name]
    assert all(layer is student.dit.get_submodule(name) for name, layer in student.layers)
    from distillschedule.training.inference import publish_export
    from distillschedule.training.plugins import load_callable
    _, source = load_callable(projection.factory)
    observe(lambda: publish_export(projection.components, tmp_path / "projection-export",
        model=dict(factory=projection.factory, source=source, parameters=dict(projection.parameters),
                   metadata=dict(projection.components.metadata)),
        identity=dict(model=source, model_metadata=dict(projection.components.metadata)), cycle=2))
    equal_tree(before, owner.state_dict()); equal_tree(rng, rng_state())
    assert ids == ({n: id(p) for n, p in owner._named_parameters().items()},
                   {n: id(m) for n, m in owner._modules().items()})
    # A projection/export may not change the next actual original update.
    other, _, _, _, _ = session._load_for_output(config, result["checkpoint"])
    row = decode_row(data.sample(2), absolute_step=2, geometry=owner.assets["geometry"])
    equal_tree(owner.update(row, cycle=2), other.state["native_update"].update(row, cycle=2))
    equal_tree(owner.state_dict(), other.state["native_update"].state_dict())


def test_untrained_owner_cannot_be_projected_as_current_restored_student(tmp_path):
    from distillschedule.training.h3_cached_public import build_model
    assets, _ = make_case(tmp_path, count=1)
    components = build_model(dict(asset_manifest=str(assets), device="cpu", seed=777))
    with pytest.raises(ValueError, match="checkpoint requires complete"):
        components.inference_projection(components, {"seed": 777})


def test_common_export_actual_current_student_and_not_optimizer(tmp_path):
    config, result = trained(tmp_path, cycles=1)
    manifest = session.export(config, result["checkpoint"], tmp_path / "export")
    assert manifest["model"]["factory"] == "distillschedule.training.h3_cached_inference:build_inference_model"
    weights = torch.load(tmp_path / "export/weights.pt", weights_only=True)
    assert set(weights) == {"student"}
    assert any("adapters.student.B.weight" in n and torch.count_nonzero(t) for n, t in weights["student"].items())
    assert not any(any(word in n for word in ("fake_score", "head", "optimizer", "rng", "cursor"))
                   for n in weights["student"])


def test_inference_factory_is_separate_and_torch_free_at_discovery():
    result = subprocess.run([sys.executable, "-B", "-c", """
import sys
import distillschedule.training.h3_cached_inference as inference
from distillschedule.training.plugins import load_callable
load_callable(inference.FACTORY)
assert 'torch' not in sys.modules
assert 'distillschedule.training.h3_cached_owner' not in sys.modules
assert 'distillschedule.training.h3_cached_public' not in sys.modules
from distillschedule.training.cli import main
try: main(['--help'])
except SystemExit as exc: assert exc.code == 0
assert 'torch' not in sys.modules
"""], cwd=repository_root(), env=dict(os.environ), text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_fresh_process_only_export_and_NEW_conditions_no_C_Adam_old_reads(portable):
    root, manifest, _, packed = portable
    generated = process(r'''
import hashlib,json,os,sys
from pathlib import Path
import torch
from distillschedule.training.config import load_training_config
from distillschedule.training import session
root=Path(sys.argv[1]);forbidden=str(root/'training-payloads')
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        assert forbidden not in str(args[0]), 'old training payload read'
sys.addaudithook(audit)
class NoTraining:
    def find_spec(self,fullname,path=None,target=None):
        assert fullname not in ('distillschedule.training.h3_cached_public','distillschedule.training.h3_cached_owner'),fullname
sys.meta_path.insert(0,NoTraining())
def no_optimizer(*args,**kwargs): raise AssertionError('optimizer constructed during inference')
torch.optim.Adam.__init__=torch.optim.AdamW.__init__=no_optimizer
from distillschedule.reference import diffsynth_h3_cached as provider
api=provider.load_cpu(backend='torch-dense-cpu');models=[];forwards=[];rollouts=[]
constructor=api.model.MiniMaxH3DiT.__init__.__code__
rollout=api.dmd2.rollout.__wrapped__.__code__
def profile(frame,event,arg):
    if event=='call' and frame.f_code is api.dmd2.VelocityDiscriminator.__init__.__code__:
        raise AssertionError('discriminator constructed during inference')
    if event=='return' and frame.f_code is constructor: models.append(frame.f_locals['self'])
    if event=='call' and frame.f_code is api.model.MiniMaxH3DiT.forward.__code__: forwards.append(True)
    if event=='return' and frame.f_code is rollout:
        assert len(frame.f_locals['schedule'][0])==9 and frame.f_locals['schedule'][0][-1]==0
        rollouts.append(True)
config=load_training_config(root/'inference.json');load=session.load_callable
def only_model(reference):
    assert reference!=config.algorithm['factory'],'training algorithm loaded'
    return load(reference)
session.load_callable=only_model
sys.setprofile(profile)
try:
    result=session.generate_from_export(config,root/'export',root/'fresh',prepared=config.data['prepared'],output_kind='latents')
finally: sys.setprofile(None)
assert len(models)==2 and forwards==[True]*8 and rollouts==[True],(len(models),len(forwards),len(rollouts))
assert set(sys.modules).isdisjoint({'distillschedule.training.h3_cached_public','distillschedule.training.h3_cached_owner'})
assert not any('fake_score' in n for n in models[-1].state_dict())
hashes={'dit.'+n:hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest() for n,t in models[-1].state_dict().items()}
assert hashes==json.loads((root/'student-hashes.json').read_text())
actual=torch.load(result['artifact'],weights_only=True)['generated']
expected=torch.load(root/'expected/generation.pt',weights_only=True)['generated']
for name,shape,layout in [('video_latents',[1,24,2,4,4],'BCTHW'),('audio_latents',[2,32,3],'SCT')]:
    assert torch.equal(actual[name],expected[name]) and list(actual[name].shape)==shape
    assert actual[name].dtype==torch.float32 and actual[name].isfinite().all()
    assert result['latents']['tensors'][name]['layout']==layout
assert not torch.cuda.is_initialized() and not (root/'fresh/frames').exists()
assert result['quality_validation']=='NOT_RUN' and result['video']['status']=='NOT_AVAILABLE'
assert result['latents']['interpretation']=='LATENT_DIAGNOSTIC_NOT_MEDIA_QUALITY'
(root/'fresh-proof.json').write_text(json.dumps(dict(status='PASS_NEW_OS_PROCESS_ORIGINAL_CACHED_AV_LATENT_DIAGNOSTIC',
    pid=os.getpid(),actual_original_dit_calls=8,original_schedule_boundaries=9,
    constructor_counts=dict(meta=1,cpu=1,discriminator=0,optimizer=0),strict_current_weights_exact=True,
    same_seed_exact=True,training_payloads_removed=True,forbidden_old_payload_reads=0,
    training_algorithm_loaded=False,cuda_initialized=False,quality_validation='NOT_RUN',claimable_speedup=False,
    tensors={n:dict(shape=list(actual[n].shape),dtype=str(actual[n].dtype),
        sha256=hashlib.sha256(actual[n].contiguous().numpy().tobytes()).hexdigest())
        for n in ('video_latents','audio_latents')})))
print('FRESH_ORIGINAL_8_STEP_AV_NO_C_ADAM_OLD_READS_CUDA_PASS')
''', root)
    assert "FRESH_ORIGINAL_8_STEP_AV" in generated.stdout
    evidence = os.environ.get("H3_CACHED_INFERENCE_EVIDENCE")
    if evidence:
        destination = Path(evidence) / ("packed" if packed else "flat")
        shutil.copy2(root / "fresh-proof.json", destination / "fresh-proof.json")
        (destination / "inference-process.stdout").write_text(generated.stdout)
    text = json.dumps({k: manifest["model"][k] for k in ("parameters", "metadata")})
    assert str(root) not in text and "/home/" not in text and "/tmp/" not in text


def test_current_AV_is_only_geometry_and_real_prompt_previous_AV_are_conditions(portable, tmp_path):
    root, manifest, prepared, _ = portable
    from distillschedule.data import PreparedDataset
    # Rebuild a genuine flat raw cache from the original canonical row.
    raw = PreparedDataset(prepared).sample(0)["raw"]
    baseline = torch.load(root / "expected/generation.pt", weights_only=True)["generated"]
    for field in ("templates", "prompt_embeds", "ref_visual_anchor", "ref_audio_anchor"):
        changed = deepcopy(raw)
        if field == "templates":
            changed["input_latents"].fill_(37); changed["audio_input_latents"].fill_(-19)
        else:
            changed[field] = changed[field] + 3
        inputs = new_inputs(tmp_path / field, raw=changed)
        config = inference_config(tmp_path, manifest, inputs)
        result = session.generate_from_export(config, root / "export", tmp_path / (field + "-output"),
                                              prepared=inputs, output_kind="latents")
        actual = torch.load(result["artifact"], weights_only=True)["generated"]
        if field == "templates":
            equal_tree(baseline, actual)
        else:
            assert any(not torch.equal(baseline[n], actual[n]) for n in ("video_latents", "audio_latents")), field


def test_video_default_and_inference_factory_refuse_training(portable, tmp_path):
    root, manifest, prepared, _ = portable
    config = inference_config(tmp_path, manifest, prepared)
    with pytest.raises(ValueError, match="actual decoder"):
        session.generate_from_export(config, root / "export", tmp_path / "video", prepared=prepared)
    assert not (tmp_path / "video").exists()
    with pytest.raises(ValueError, match="inference-only"):
        session.train(config)
    assert not Path(config.trainer["output_dir"]).exists()
    with pytest.raises(ValueError, match="inference-only"):
        session.evaluate(config, root / "missing", tmp_path / "evaluation")
    assert not (tmp_path / "evaluation").exists()


@pytest.mark.parametrize("fault", ["source", "runtime", "lora", "rank_type", "schedule", "schedule_type", "architecture",
                                  "layers", "rope", "geometry", "budget", "input_budget", "precision", "device",
                                  "topology", "unknown"])
def test_factory_contract_rejects_before_real_allocation(portable, monkeypatch, fault):
    from distillschedule.training.h3_cached_inference import build_inference_model
    _, manifest, _, _ = portable
    p = deepcopy(manifest["model"]["parameters"])
    if fault == "source": p["source"]["source16"]["manifest_sha256"] = "0" * 64
    elif fault == "runtime": p["runtime"]["torch"]["version"] = "changed"
    elif fault == "lora": p["lora"]["lora_alpha"] = 32
    elif fault == "rank_type": p["lora"]["lora_rank"] = 64.0
    elif fault == "schedule": p["schedule"]["num_inference_steps"] = 7
    elif fault == "schedule_type": p["schedule"]["num_inference_steps"] = 8.0
    elif fault == "architecture": p["architecture"]["text_dim"] = 2
    elif fault == "layers": p["architecture"]["num_layers"] = 10**9
    elif fault == "rope": p["architecture"]["rope_inv_freq_len"] = 100
    elif fault == "geometry": p["geometry"]["video_shape"][-1] = 3
    elif fault == "budget": p["budget"]["max_parameters"] = 1
    elif fault == "input_budget": p["budget"]["max_input_elements"] = 1
    elif fault == "precision": p["precision"] = "BF16"
    elif fault == "device": p["device"] = "cuda"
    elif fault == "topology": monkeypatch.setenv("WORLD_SIZE", "2")
    else: p["training_asset_manifest"] = "/forbidden"
    calls, constructor = [], original().model.MiniMaxH3DiT.__init__.__code__
    def profile(frame, event, arg):
        if event == "return" and frame.f_code is constructor:
            calls.append(frame.f_locals["self"])
    previous = sys.getprofile()
    rng_before = torch.get_rng_state().clone()
    try:
        sys.setprofile(profile)
        with pytest.raises(ValueError): build_inference_model(p)
    finally: sys.setprofile(previous)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert all(next(model.parameters()).device.type == "meta" for model in calls)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("fault", ["missing", "extra", "dtype", "shape", "nonfinite", "unhashed_modified"])
def test_common_restore_stages_all_weights_before_mutating_live_state(portable, tmp_path, fault):
    from distillschedule.training.h3_cached_inference import build_inference_model
    from distillschedule.training.inference import read_export, restore_modules
    root, manifest, _, _ = portable
    destination = tmp_path / "export"; shutil.copytree(root / "export", destination)
    weights = torch.load(destination / "weights.pt", weights_only=True)
    key = list(weights["student"])[-1]
    if fault == "missing": weights["student"].pop(key)
    elif fault == "extra": weights["student"]["dit.extra"] = torch.ones(1)
    elif fault == "dtype": weights["student"][key] = weights["student"][key].double()
    elif fault == "shape": weights["student"][key] = torch.zeros(1)
    elif fault == "nonfinite": weights["student"][key].fill_(float("nan"))
    else: weights["student"][key].add_(1)
    torch.save(weights, destination / "weights.pt")
    if fault != "unhashed_modified":
        manifest = deepcopy(manifest); manifest["files"]["weights.pt"] = file_sha256(destination / "weights.pt")
        json_file(destination / "export.json", manifest)
        (destination / "_SUCCESS").write_text(file_sha256(destination / "export.json") + "\n")
    components = build_inference_model(manifest["model"]["parameters"])
    student = components.state["student"]
    before = deepcopy(student.state_dict())
    with pytest.raises(ValueError, match="layout|nonfinite|hash"):
        _, admitted = read_export(destination)
        restore_modules(components, destination, admitted)
    equal_tree(before, student.state_dict())
    assert not student._inference_weights_restored


@pytest.mark.parametrize("fault", ["mask", "dirty_anchor", "geometry", "dtype", "nonfinite", "unknown"])
def test_actual_cache_admission_rejects_without_forward_or_output(portable, tmp_path, fault):
    from distillschedule.data import PreparedDataset
    from distillschedule.data.h3_cached_inputs import row_identity, _id_tensor
    from distillschedule.data import import_preencoded
    from distillschedule.training.h3_cached_assets import content_digest
    root, manifest, prepared, _ = portable
    row = PreparedDataset(prepared).sample(0)
    if fault == "mask": row["raw"]["attention_mask"] = torch.ones(1)
    elif fault == "dirty_anchor": row["raw"]["imgvid_cond_noise_aug"] = torch.tensor(0.9)
    elif fault == "geometry": row["raw"]["input_latents"] = torch.ones(1, 24, 3, 4, 4)
    elif fault == "dtype": row["raw"]["prompt_embeds"] = row["raw"]["prompt_embeds"].double()
    elif fault == "nonfinite": row["raw"]["ref_audio_anchor"].fill_(float("inf"))
    else: row["raw"]["unknown_condition"] = torch.ones(1)
    # Rehash where possible: reject semantics, not merely a stale row hash.
    if fault != "nonfinite":
        content = content_digest(row["raw"])
        source = bytes(row["source_sha256"].tolist()).hex()
        row["content_sha256"] = _id_tensor(content)
        row["identity"] = _id_tensor(row_identity(source, content, 0))
    torch.save(row, tmp_path / "row.pt")
    sources = json_file(tmp_path / "bad.json", dict(schema_version="distillschedule.data.preencoded.v1",
        records=[dict(id="bad", features=dict(path="row.pt", sha256=file_sha256(tmp_path / "row.pt")))]))
    inputs = import_preencoded(sources, tmp_path / "prepared")
    config = inference_config(tmp_path, manifest, inputs)
    calls, forward = [], original().model.MiniMaxH3DiT.forward.__code__
    def profile(frame, event, arg):
        if event == "call" and frame.f_code is forward: calls.append(True)
    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        with pytest.raises(ValueError):
            session.generate_from_export(config, root / "export", tmp_path / "output", prepared=inputs, output_kind="latents")
    finally: sys.setprofile(previous)
    assert not calls and not (tmp_path / "output").exists()


def test_factory_guard_CPU_default_dtype_and_public_CLI(portable, tmp_path, capsys, monkeypatch):
    from distillschedule.training.h3_cached_inference import build_inference_model
    from distillschedule.data import PreparedDataset
    from distillschedule.training.cli import main
    root, manifest, prepared, _ = portable
    def no_reseed(*args, **kwargs): raise AssertionError("inference factory reseeded")
    with monkeypatch.context() as patch, torch.device("meta"):
        patch.setattr(torch, "manual_seed", no_reseed)
        components = build_inference_model(manifest["model"]["parameters"])
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 and not p.requires_grad
               for p in components.state["student"].parameters())
    assert components.export is None and components.load_export is None
    with pytest.raises(ValueError, match="before any forward"):
        components.generate(components.state, PreparedDataset(prepared).sample(0), {})
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError, match="FP32"): build_inference_model(manifest["model"]["parameters"])
    finally: torch.set_default_dtype(dtype)
    config = inference_config(tmp_path, manifest, prepared)
    path = json_file(tmp_path / "inference.json", config.to_dict())
    assert main(["generate", str(path), "--weights", str(root / "export"), "--prepared", str(prepared),
                 "--output-kind", "latents", "--output", str(tmp_path / "cli")]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["output_kind"] == "latents" and receipt["quality_validation"] == "NOT_RUN"
    assert receipt["latents"]["tensors"]["audio_latents"]["layout"] == "SCT"
    assert not torch.cuda.is_initialized()
