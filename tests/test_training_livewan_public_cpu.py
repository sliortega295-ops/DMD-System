"""Complete original Self-Forcing CPU1 fixture; not native world16 or quality."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import random
import resource
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def bounded():
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized() and not torch.distributed.is_initialized()
    start = time.monotonic()
    old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("180s CPU probe limit")))
    signal.alarm(180)
    try:
        yield
        assert time.monotonic()-start < 180
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 < 2*1024**3
        assert not torch.cuda.is_initialized() and not torch.distributed.is_initialized()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def json_file(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return Path(path)


def fixture_files(root, *, count=2):
    from safetensors.torch import save_file
    from distillschedule.reference import livewan_wan21 as provider
    from distillschedule.training.livewan_public import (ASSETS, SCOPE, GEOMETRY, PRECISION, BACKEND,
        FIXTURE, runtime_identity, pin_file)
    from distillschedule.data.livewan_reference_inputs import SCHEMA, FORMAT, producer_identity, prepare_inputs
    from test_livewan_reference_cpu import fixture_model, tiny_config
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    role_specs = {}
    for index, role in enumerate(("student", "teacher", "critic")):
        model = fixture_model(951+index, teacher=role == "teacher")
        config = tiny_config(role == "teacher")
        config.update(patch_size=[1, 2, 2], window_size=[-1, -1], qk_norm=True, cross_attn_norm=True, eps=1e-6)
        cfg = json_file(root / (role + ".json"), config)
        weight = root / (role + ".safetensors")
        save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, str(weight), metadata={"evidence_kind": FIXTURE})
        role_specs[role] = dict(config=pin_file(cfg), weights=[pin_file(weight)])
    gen = torch.Generator().manual_seed(811)
    negative = root / "negative.safetensors"
    save_file(dict(negative=torch.randn(1, 512, 4096, generator=gen)), str(negative), metadata={"evidence_kind": FIXTURE})
    assets = dict(schema_version=ASSETS, scope=SCOPE, precision=PRECISION, backend=BACKEND, evidence_kind=FIXTURE,
        source=provider.source_identity(), runtime=runtime_identity(), geometry=GEOMETRY, lora_seed=121,
        negative=pin_file(negative), budget=dict(max_parameters=2_000_000, max_loaded_bytes=32_000_000,
        max_cycles=count, max_attention_elements=500_000), **role_specs)
    asset_path = json_file(root / "assets.json", assets)
    records = []
    for index in range(5*count):
        record = root / f"teacher-{index:03d}.safetensors"
        raw = dict(latents=torch.randn(16, 21, 4, 4, generator=gen), context=torch.randn(512, 4096, generator=gen))
        save_file(raw, str(record), metadata={"evidence_kind": FIXTURE})
        records.append(dict(id=f"synthetic-original-full-class-source-{index:03d}", source=pin_file(record)))
    source_path = json_file(root / "inputs.json", dict(schema_version=SCHEMA, format=FORMAT, scope=SCOPE,
        precision=PRECISION, geometry=GEOMETRY, evidence_kind=FIXTURE, producer=producer_identity(),
        seed=171, cycles=count, negative=pin_file(negative), records=records))
    before = __import__("distillschedule.training.checkpoint", fromlist=["rng_state"]).rng_state()
    prepared = prepare_inputs(source_path, root / "prepared")
    after = __import__("distillschedule.training.checkpoint", fromlist=["rng_state"]).rng_state()
    assert equal(before, after)
    return dict(assets=asset_path, source=source_path, prepared=prepared)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    return fixture_files(tmp_path_factory.mktemp("livewan-public") / "local")


def config_for(files, output, *, cycles=2, resume=None, **overrides):
    from distillschedule.training.config import TrainingConfig
    value = dict(schema_version="distillschedule.training.v1",
        model=dict(factory="distillschedule.training.livewan_public:build_model", parameters=dict(asset_manifest=str(files["assets"]))),
        algorithm=dict(factory="distillschedule.training.livewan_public:build_algorithm"),
        data=dict(prepared=str(files["prepared"]), sampling=dict(kind="finite", shuffle=False)),
        execution=dict(adapter="program", mode="reference"),
        trainer=dict(cycles=cycles, seed=773, device="cpu", checkpoint_every=1, output_dir=str(output)))
    if resume is not None:
        value["trainer"]["resume_from"] = str(resume)
    for key, fields in overrides.items():
        value[key].update(fields)
    return TrainingConfig.from_dict(value, source=Path(output).parent / "training.json")


def run_capturing(config):
    """Observe actual frame events, with no patched math, hooks, or fake models."""
    from distillschedule.training import session
    from distillschedule.runtime import livewan_crossscale_worker as worker
    from distillschedule.reference import livewan_wan21 as provider
    api = provider.load_cpu(backend="torch-sdpa-cpu", precision="float32")
    owners, receipts, calls, gradients = [], [], [], []
    construct_code, run_code = session.construct.__code__, __import__("distillschedule.training.livewan_public", fromlist=["LiveWanCPUOwner"]).LiveWanCPUOwner.run.__code__
    body_codes = {worker._run_baseline.__code__: "complete_baseline", api.blockcausal.block_forward.__code__: "block_forward",
        api.stream.FewStepStreamer.generate_block.__code__: "generate_block", api.stream.FewStepStreamer.recompute_step.__code__: "recompute_step",
        api.stream.FewStepStreamer._commit.__code__: "cache_commit", torch.optim.AdamW.step.__code__: "adam_step"}
    clip_code = torch.nn.utils.clip_grad_norm_.__wrapped__.__code__
    def profile(frame, event, arg):
        if event == "return" and frame.f_code is construct_code and isinstance(arg, tuple):
            owners.append(arg[0].state["native_cycle"])
        if event == "return" and frame.f_code is run_code and isinstance(arg, dict):
            receipts.append(deepcopy(arg))
        if event == "call" and frame.f_code in body_codes:
            kind = body_codes[frame.f_code]
            model = frame.f_locals.get("model")
            role = next((k for k, v in owners[0].models.items() if model is v), None) if owners and model is not None else None
            calls.append(dict(kind=kind, role=role, grad=torch.is_grad_enabled(),
                              checkpoint=frame.f_locals.get("grad_checkpoint"), t=frame.f_locals.get("t") if type(frame.f_locals.get("t")) is float else None))
        if event == "call" and frame.f_code is clip_code and owners:
            owner = owners[0]
            parameters = {id(p) for p in frame.f_locals["parameters"]}
            gradients.append({n: float(p.grad.detach().norm()) for n, p in owner._named.items()
                              if p.grad is not None and id(p) in parameters})
    try:
        sys.setprofile(profile)
        result = session.train(config)
    finally:
        sys.setprofile(None)
    assert len(owners) == 1
    return result, owners[0], receipts, dict(calls=calls, gradients=gradients)


def test_actual_original_cpu1_two_fresh_complete_cycles(files, tmp_path, record_property):
    from distillschedule.training.livewan_public import content_digest
    from distillschedule.data import PreparedDataset
    from distillschedule.training import session
    config = config_for(files, tmp_path / "two")
    components, dataset, algorithm, _ = session.construct(config)
    workload = session.build_cycle(config, components, dataset, algorithm, cycle=0, cursor=0)
    assert len(workload.program.calls) == 1
    call = workload.program.calls[0]
    assert call.grad == "external_grad" and call.replay_safe is False and call.reads == call.writes == ("native_cycle",)
    assert components.generate is components.decode is None
    assert callable(components.inference_projection)
    initial = components.state["native_cycle"]
    frozen = deepcopy(initial._frozen)
    before = {role: content_digest(model.state_dict()) for role, model in initial.models.items()}
    result, owner, receipts, observed = run_capturing(config)
    assert result["status"] == "COMPLETE" and result["cursor"] == 2 and result["final_versions"] == {"native_cycle": 2}
    assert owner.cycle == 2 and owner.versions == dict(generator=2, critic=10, ema=0)
    assert not owner.poisoned and owner._frozen == frozen
    assert content_digest(owner.models["teacher"].state_dict()) == before["teacher"]
    assert content_digest(owner.models["student"].state_dict()) != before["student"]
    assert content_digest(owner.models["critic"].state_dict()) != before["critic"]
    assert len(receipts) == 2 and len(observed["gradients"]) == 12
    all_ids = []
    expected_nograd = 0
    expected_blocks = 0
    for cycle, receipt in enumerate(receipts):
        assert receipt["cycle"] == cycle+1 and receipt["versions"] == owner.expected(cycle+1)
        assert len(receipt["inputs"]) == 5 and len(receipt["losses"]) == 8
        assert [r["kind"] for r in receipt["losses"]] == ["generator"]*3 + ["critic"]*5
        assert all(r["finite"] for r in receipt["losses"])
        assert len(receipt["optimizer_norms"]) == 6 and all(r["clip_input_norm"] > 0 for r in receipt["optimizer_norms"])
        assert receipt["logical_score_calls"]["teacher/nograd/plain"] == receipt["logical_score_calls"]["critic/nograd/plain"] == 1
        assert receipt["logical_score_calls"]["critic/grad/checkpoint"] == 5
        assert all(r["buckets"] == 0 and r["gradient_tensors"] == 12 for r in receipt["gradient_sync"])
        for row in receipt["inputs"]:
            blocks = (21-row["world_frames"])//3
            expected_blocks += blocks
            expected_nograd += 1+5*blocks
            all_ids.append(row["record_id"])
    # Three recomputes are interleaved before the first Critic loss, not a
    # replacement three-loss loop or one-row pretending to complete five rows.
    assert len(set(all_ids)) == 10
    calls = observed["calls"]
    assert sum(c["kind"] == "complete_baseline" for c in calls) == 2
    assert sum(c["kind"] == "generate_block" for c in calls) == expected_blocks
    assert sum(c["kind"] == "cache_commit" for c in calls) == 10+expected_blocks
    assert sum(c["kind"] == "recompute_step" for c in calls) == 6
    assert sum(c["kind"] == "block_forward" and c["role"] == "student" and not c["grad"] for c in calls) == expected_nograd
    assert sum(c["kind"] == "block_forward" and c["role"] == "student" and c["grad"] and c["checkpoint"] for c in calls) == 6
    assert sum(c["kind"] == "block_forward" and c["role"] == "teacher" for c in calls) == 2
    assert sum(c["kind"] == "block_forward" and c["role"] == "critic" and not c["grad"] for c in calls) == 2
    assert sum(c["kind"] == "block_forward" and c["role"] == "critic" and c["grad"] and c["checkpoint"] for c in calls) == 10
    ggrads = [row for row in observed["gradients"] if any(k.startswith("student.") for k in row)]
    assert len(ggrads) == 2
    assert all(any(v > 0 and k.startswith(prefix) for k, v in row.items()) for row in ggrads for prefix in (
        "student.patch_embedding", "student.time_embedding", "student.blocks.0.self_attn", "student.blocks.0.ffn", "student.head.head"))
    assert all(not any(k.startswith("student.text_embedding.") for k in row) for row in ggrads)
    assert all(not any(k.startswith("teacher.") for k in row) for row in observed["gradients"])
    cgrads = [row for row in observed["gradients"] if any(k.startswith("critic.") for k in row)]
    assert len(cgrads) == 10 and len(cgrads[-1]) == 12 and all(value > 0 for value in cgrads[-1].values())
    assert all(name.endswith((".a", ".b")) for row in cgrads for name in row)
    assert any(name.endswith(".a") and value == 0. for name, value in cgrads[0].items()), "original cold-zero B initialization was replaced"
    for role in owner.optimizers:
        slots = owner._optimizer_value(role)["state"]
        assert slots and any(bool(s["exp_avg"].count_nonzero()) for s in slots.values())
        assert all(s["step"].item() == (2 if role == "generator" else 10) for s in slots.values())
    record_property("real_cycle_receipts", json.dumps(receipts, sort_keys=True))
    record_property("real_call_census", json.dumps(calls, sort_keys=True))
    evidence = os.environ.get("DISTILLSCHEDULE_W4_PUBLIC_EVIDENCE")
    if evidence:
        root = Path(evidence)
        json_file(root / "artifacts/two-cycle-proof.json", dict(result=result, receipts=receipts,
            observations=observed, current_student_sha256=content_digest(owner.models["student"].state_dict()),
            teacher_unchanged=True, critic_base_unchanged=True, cuda_initialized=torch.cuda.is_initialized(),
            distributed_initialized=torch.distributed.is_initialized(), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        from distillschedule.training.livewan_public import runtime_identity, pin_file
        from distillschedule.training.checkpoint import read_checkpoint
        from distillschedule.reference import livewan_wan21 as provider
        payload = read_checkpoint(result["checkpoint"], identity=result["identity"])
        torch.save(payload, root / "artifacts/complete-two-cycle-safe-state.pt")
        json_file(root / "artifacts/runtime-identity.json", dict(executable=sys.executable, python=sys.version,
            runtime=runtime_identity(), cuda_initialized=torch.cuda.is_initialized(), distributed_initialized=torch.distributed.is_initialized(),
            model_classes={role: type(model).__module__ + "." + type(model).__qualname__ for role, model in owner.models.items()},
            model_parameter_counts={role: sum(p.numel() for p in model.parameters()) for role, model in owner.models.items()},
            provider_source=provider.source_identity(), provider_api_origins={key: str(Path(module.__file__).resolve())
                for key, module in vars(owner.context.api).items()}, evidence_kind=owner.assets["evidence_kind"],
            worker=pin_file(__import__("distillschedule.runtime.livewan_crossscale_worker", fromlist=["_run_baseline"]).__file__),
            environment={key: os.environ.get(key) for key in ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PIP_NO_INDEX", "PYTHONDONTWRITEBYTECODE")}))
        json_file(root / "artifacts/supplied-input-asset-identities.json", dict(
            assets=json.loads(Path(files["assets"]).read_bytes()), source=json.loads(Path(files["source"]).read_bytes()),
            prepared=dataset.manifest, prepared_identity=dataset.identity,
            safe_checkpoint=pin_file(root / "artifacts/complete-two-cycle-safe-state.pt"),
            raw_fixture_payloads="REPRODUCIBLE_SYNTHETIC_TEST_SCRATCH_NOT_COPIED_AS_PRODUCTION_DATA"))


def test_fresh_os_process_exact_uninterrupted2_vs1_resume1(files, tmp_path):
    from distillschedule.training.checkpoint import read_checkpoint
    from distillschedule.training import session
    full, owner, receipts, _ = run_capturing(config_for(files, tmp_path / "full"))
    expected = read_checkpoint(full["checkpoint"], identity=full["identity"])
    part = session.train(config_for(files, tmp_path / "part", cycles=1))
    first = read_checkpoint(Path(full["checkpoint"]).parent / "cycle-000001", identity=full["identity"])
    partial = read_checkpoint(part["checkpoint"], identity=part["identity"])
    assert equal(first["state"], partial["state"]) and equal(first["rng"], partial["rng"])
    torch.save(dict(state=expected["state"], rng=expected["rng"], receipts=receipts[1:]), tmp_path / "expected.pt")
    config = config_for(files, tmp_path / "resumed", resume=part["checkpoint"])
    json_file(tmp_path / "resume.json", config.to_dict())
    script = r'''
import sys,json,resource,os,time
from pathlib import Path
import torch
from test_training_livewan_public_cpu import run_capturing,equal
from distillschedule.training.config import load_training_config
from distillschedule.training.checkpoint import read_checkpoint,rng_state
from distillschedule.training.livewan_public import content_digest
root=Path(sys.argv[1]); torch.set_num_threads(1); start=time.monotonic()
before_pid=os.getpid()
result,owner,receipts,observed=run_capturing(load_training_config(root/'resume.json'))
expected=torch.load(root/'expected.pt',weights_only=True)
actual=read_checkpoint(result['checkpoint'],identity=result['identity'])
assert equal(expected['state'],actual['state']), 'all full G/C/Teacher, named Adam/options/step, source, clocks or inner RNG differ'
assert equal(expected['rng'],actual['rng']) and equal(actual['rng'],rng_state()), 'Torch/Python/NumPy outer RNG differ'
assert equal(expected['receipts'],receipts), 'actual original next loss/source/control/census differs'
assert result['resume_cycle']==1 and result['cursor']==2 and result['final_versions']=={'native_cycle':2}
assert owner.versions==dict(generator=2,critic=10,ema=0)
assert not torch.cuda.is_initialized() and not torch.distributed.is_initialized()
assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 < 2*1024**3 and time.monotonic()-start <180
print(json.dumps(dict(status='PASS_FRESH_OS_EXACT_2_VS_1_PLUS_RESUME1',pid=before_pid,
   current_student_sha256=content_digest(owner.models['student'].state_dict()),
   complete_state_sha256=content_digest(actual['state']),inner_outer_rng_exact=True,
   source_rows_not_replayed=True,versions=owner.versions,losses_exact=True,adam_options_slots_exact=True,
   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,cuda_initialized=False)))
'''
    process = subprocess.run([sys.executable, "-B", "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], env=dict(os.environ), text=True, capture_output=True, timeout=180)
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(process.stdout.splitlines()[-1])
    assert report["status"] == "PASS_FRESH_OS_EXACT_2_VS_1_PLUS_RESUME1" and report["pid"] != os.getpid()
    evidence = os.environ.get("DISTILLSCHEDULE_W4_PUBLIC_EVIDENCE")
    if evidence:
        root = Path(evidence)
        (root / "logs/restart-process.stdout").write_text(process.stdout)
        (root / "logs/restart-process.stderr").write_text(process.stderr)
        json_file(root / "artifacts/restart-proof.json", dict(**report, parent_pid=os.getpid(),
            full=full, partial=part, resume_configuration=config.to_dict(),
            comparison="torch.equal all complete named owners/optimizer slots/options/source/control/clocks/loss/census/inner+outer RNG"))


def complete_boundary(files, root):
    from distillschedule.training import session
    from distillschedule.data.livewan_reference_inputs import decode_cycle
    components, data, _, _ = session.construct(config_for(files, root))
    owner = components.state["native_cycle"]
    owner.run(decode_cycle(data.sample(0), cycle=0), cycle=0)
    return owner, data, deepcopy(owner.state_dict())


@pytest.fixture(scope="module")
def complete(files, tmp_path_factory):
    owner, dataset, boundary = complete_boundary(files, tmp_path_factory.mktemp("completed") / "not-created")
    return dataset, boundary


def owner_from_boundary(files, complete, output):
    from distillschedule.training import session
    components = session.construct(config_for(files, output))[0]
    owner = components.state["native_cycle"]
    owner.load_state_dict(complete[1])
    return owner, complete[0]


@pytest.mark.parametrize("failure", ["after_generator", "teacher", "critic", "read"])
def test_partial_native_failure_poisons_all_owners_and_restores_previous_boundary(files, complete, tmp_path, failure):
    from distillschedule.training.checkpoint import rng_state, pack_state, unpack_state, save_checkpoint
    from distillschedule.training.tensor_state import live_meshes
    from distillschedule.training.livewan_public import content_digest
    from distillschedule.data.livewan_reference_inputs import decode_cycle, CPUCycleInputs
    from distillschedule.runtime import livewan_crossscale_worker as worker
    owner, data = owner_from_boundary(files, complete, tmp_path / "not-created")
    saved = deepcopy(owner.state_dict())
    packed = pack_state(dict(native_cycle=owner))
    rng = deepcopy(rng_state())
    ids = deepcopy(owner._parameter_ids)
    bindings = dict(owner._optimizer_ids)
    targets = dict(after_generator=worker._mean_replicated_gradients.__code__, teacher=owner.context.api.dmd.cfg_velocity.__code__,
        critic=worker.critic_backward_for_sample.__code__, read=CPUCycleInputs.load.__code__)
    fired = []
    def fail(frame, event, arg):
        if frame.f_code is not targets[failure] or fired:
            return
        trigger = ((failure == "after_generator" and event == "call")
            or (failure in ("teacher", "critic") and event == "return" and arg is not None)
            or (failure == "read" and event == "call" and frame.f_locals["self"].read_count == 2))
        if trigger:
            fired.append(dict(failure=failure, versions=deepcopy(owner.versions)))
            raise RuntimeError("injected actual " + failure + " failure")
    try:
        sys.setprofile(fail)
        with pytest.raises(RuntimeError, match="injected actual"):
            owner.run(decode_cycle(data.sample(1), cycle=1), cycle=1)
    finally:
        sys.setprofile(None)
    assert len(fired) == 1 and owner.poisoned and not owner._busy
    assert equal(rng, rng_state()), "failed original update leaked its private RNG into the caller"
    if failure == "after_generator":
        assert owner.versions == dict(generator=2, critic=5, ema=0)
        assert content_digest(owner.models["student"].state_dict()) != content_digest(saved["modules"]["student"])
    partial = dict(clocks=deepcopy(owner.versions), private_rng=deepcopy(owner._rng),
        models={role: content_digest(m.state_dict()) for role, m in owner.models.items()},
        gradients={n: content_digest(p.grad) if p.grad is not None else None for n, p in owner._named.items()})
    diagnostic = owner.state_dict()
    assert diagnostic["checkpoint_status"] == "POISONED_NOT_RESUMABLE"
    assert live_meshes(dict(native_cycle=owner)) == []
    assert partial["models"] == {role: content_digest(m.state_dict()) for role, m in owner.models.items()}
    assert partial["gradients"] == {n: content_digest(p.grad) if p.grad is not None else None for n, p in owner._named.items()}
    assert partial["clocks"] == owner.versions and equal(partial["private_rng"], owner._rng) and equal(rng, rng_state())
    poison_path = tmp_path / "poisoned-checkpoint"
    with pytest.raises(ValueError, match="poisoned"):
        save_checkpoint(poison_path, dict(native_cycle=owner), identity=dict(test=failure),
            cycle=1, cursor=1, versions=dict(native_cycle=1))
    assert not poison_path.exists() and not (poison_path / "_SUCCESS").exists()
    with pytest.raises(ValueError, match="poisoned"):
        owner.validate_state_dict(diagnostic)
    with pytest.raises(ValueError, match="poisoned"):
        owner.load_state_dict(diagnostic)
    with pytest.raises(ValueError, match="poisoned"):
        owner.run(decode_cycle(data.sample(1), cycle=1), cycle=1)
    with pytest.raises(ValueError):
        owner.validate_checkpoint_boundary()
    assert equal(saved, complete[1]), "partial work mutated the preceding safe checkpoint"
    unpack_state(packed, dict(native_cycle=owner), progress=dict(cycle=1, cursor=1, versions=dict(native_cycle=1)))
    assert not owner.poisoned and equal(saved, owner.state_dict())
    assert ids == owner._parameter_ids and bindings == owner._optimizer_ids and equal(rng, rng_state())
    # Only recovery, never retry of the partial update: reconstruct scratch and
    # consume the genuinely next five-entry source cohort once.
    owner.run(decode_cycle(data.sample(1), cycle=1), cycle=1)
    assert owner.versions == dict(generator=2, critic=10, ema=0)
    assert len({e["id"] for row in owner.consumed for e in row["entries"]}) == 10


def corrupt_checkpoint(saved, change):
    value = deepcopy(saved)
    g = value["optimizers"]["generator"]
    c = value["optimizers"]["critic"]
    if change == "missing_role":
        value["modules"].pop("teacher")
    elif change == "precision":
        value["identity"]["precision"] = "bf16"
    elif change == "source":
        value["identity"]["assets"]["source"]["manifest_sha256"] = "0"*64
    elif change == "geometry":
        value["identity"]["geometry"]["resolution"] = [368, 640]
    elif change == "dtype":
        value["modules"]["student"]["head.head.weight"] = value["modules"]["student"]["head.head.weight"].bfloat16()
    elif change == "shape":
        value["modules"]["student"]["head.head.weight"] = value["modules"]["student"]["head.head.weight"][:1]
    elif change == "layout":
        value["modules"]["student"]["head.head.weight"] = value["modules"]["student"]["head.head.weight"].to_sparse()
    elif change == "frozen_teacher":
        value["modules"]["teacher"]["head.head.weight"].add_(.1)
    elif change == "frozen_critic":
        name = next(n for n in value["modules"]["critic"] if ".base.weight" in n)
        value["modules"]["critic"][name].add_(.1)
    elif change == "detached_text":
        value["modules"]["student"]["text_embedding.0.weight"].add_(.1)
    elif change == "adam_order":
        g["groups"][0]["params"].reverse()
    elif change == "adam_options":
        c["groups"][0]["betas"] = (.9, .99)
    elif change == "adam_defaults":
        g["defaults"]["eps"] = .1
    elif change == "adam_missing":
        c["state"].pop(next(iter(c["state"])))
    elif change == "adam_step":
        next(iter(c["state"].values()))["step"].fill_(4)
    elif change == "adam_dtype":
        slot = next(iter(g["state"].values()))
        slot["exp_avg"] = slot["exp_avg"].double()
    elif change == "adam_nan":
        next(iter(g["state"].values()))["exp_avg"].fill_(float("nan"))
    elif change == "adam_negative_variance":
        next(iter(c["state"].values()))["exp_avg_sq"].fill_(-1.)
    elif change == "detached_text_slot":
        g["state"]["student.text_embedding.0.weight"] = deepcopy(next(iter(g["state"].values())))
    elif change == "wrong_clock":
        value["versions"]["critic"] = 4
    elif change == "bool_clock":
        value["cycle"] = True
    elif change == "incomplete_boundary":
        value["cycle"] = 0
        value["versions"] = dict(generator=0, critic=0, ema=0)
    elif change == "source_chain":
        value["consumed"][0]["chain"] = "0"*64
    elif change == "source_control":
        value["consumed"][0]["control_sha256"] = "0"*64
    elif change == "source_replay":
        value["consumed"] += deepcopy(value["consumed"])
    elif change == "rng_owner":
        value["rng"].pop("python")
    elif change == "rng_torch":
        value["rng"]["torch"] = value["rng"]["torch"][:10]
    elif change == "rng_numpy":
        value["rng"]["numpy"][1].pop()
    elif change == "loss_receipt":
        value["last_metrics"]["losses"][0]["mean"] = float("nan")
    elif change == "call_census":
        value["last_metrics"]["logical_score_calls"]["critic"] = 5
    elif change == "recompute_count":
        value["last_metrics"]["losses"].pop(0)
    elif change == "receipt_rng":
        value["last_metrics"]["rng_sha256"]["torch"] = "0"*64
    elif change == "operation_seed":
        value["last_metrics"]["inputs"][0]["seeds"]["rollout"] = 1
    else:
        raise AssertionError(change)
    return value


@pytest.mark.parametrize("change", ["missing_role", "precision", "source", "geometry", "dtype", "shape", "layout",
    "frozen_teacher", "frozen_critic", "detached_text", "adam_order", "adam_options", "adam_defaults", "adam_missing",
    "adam_step", "adam_dtype", "adam_nan", "adam_negative_variance", "detached_text_slot", "wrong_clock", "bool_clock",
    "incomplete_boundary", "source_chain", "source_control", "source_replay", "rng_owner", "rng_torch", "rng_numpy", "loss_receipt",
    "call_census", "recompute_count", "receipt_rng", "operation_seed"])
def test_corrupt_checkpoint_all_owner_staging_rejects_before_any_mutation(files, complete, tmp_path, change):
    from distillschedule.training.livewan_public import content_digest
    from distillschedule.training.checkpoint import rng_state
    owner, _ = owner_from_boundary(files, complete, tmp_path / "not-created")
    saved = deepcopy(owner.state_dict())
    before = dict(modules={role: content_digest(model.state_dict()) for role, model in owner.models.items()},
                  optimizers={role: content_digest(owner._optimizer_value(role)) for role in owner.optimizers})
    rng = deepcopy(rng_state())
    value = corrupt_checkpoint(saved, change)
    with pytest.raises((ValueError, RuntimeError, TypeError, KeyError)):
        owner.load_state_dict(value)
    assert owner.poisoned and equal(rng, rng_state())
    assert before["modules"] == {role: content_digest(model.state_dict()) for role, model in owner.models.items()}
    assert before["optimizers"] == {role: content_digest(owner._optimizer_value(role)) for role in owner.optimizers}
    assert owner.cycle == 1 and owner.versions == dict(generator=1, critic=5, ema=0)
    owner.load_state_dict(saved)
    assert equal(saved, owner.state_dict())


@pytest.mark.parametrize("point", ["student", "teacher", "critic", "adam_staging"])
def test_actual_restore_copy_failure_poison_requires_previous_complete_recovery(files, complete, tmp_path, point):
    from distillschedule.data.livewan_reference_inputs import decode_cycle
    from distillschedule.training.livewan_public import content_digest
    from distillschedule.training.checkpoint import rng_state, pack_state, unpack_state
    owner, data = owner_from_boundary(files, complete, tmp_path / "not-created")
    saved = deepcopy(owner.state_dict())
    packed = pack_state(dict(native_cycle=owner))
    owner.run(decode_cycle(data.sample(1), cycle=1), cycle=1)
    current = {role: content_digest(model.state_dict()) for role, model in owner.models.items()}
    rng, ids = deepcopy(rng_state()), deepcopy(owner._parameter_ids)
    module_code = torch.nn.Module.load_state_dict.__code__
    optimizer_code = torch.optim.AdamW.load_state_dict.__wrapped__.__code__
    fired, adam_calls = [], []
    def fail(frame, event, arg):
        if event != "return" or fired:
            return
        if point == "adam_staging" and frame.f_code is optimizer_code:
            adam_calls.append(1)
            if len(adam_calls) == 2:
                fired.append(point)
                raise RuntimeError("injected actual Adam restore staging failure")
        elif point != "adam_staging" and frame.f_code is module_code and frame.f_locals["self"] is owner.models[point]:
            fired.append(point)
            raise RuntimeError("injected actual model restore copy failure")
    try:
        sys.setprofile(fail)
        with pytest.raises(RuntimeError, match="injected actual"):
            unpack_state(packed, dict(native_cycle=owner), progress=dict(cycle=1, cursor=1, versions=dict(native_cycle=1)))
    finally:
        sys.setprofile(None)
    assert fired and owner.poisoned and equal(rng, rng_state())
    if point == "adam_staging":
        assert current == {role: content_digest(model.state_dict()) for role, model in owner.models.items()}
    with pytest.raises(ValueError, match="poisoned"):
        owner.validate_checkpoint_boundary()
    unpack_state(packed, dict(native_cycle=owner), progress=dict(cycle=1, cursor=1, versions=dict(native_cycle=1)))
    assert equal(saved, owner.state_dict()) and ids == owner._parameter_ids and equal(rng, rng_state())


def test_outer_progress_rejection_poisons_without_mutation(files, complete, tmp_path):
    from distillschedule.training.livewan_public import content_digest
    from distillschedule.training.checkpoint import rng_state
    owner, _ = owner_from_boundary(files, complete, tmp_path / "not-created")
    saved = deepcopy(owner.state_dict())
    before = {role: content_digest(m.state_dict()) for role, m in owner.models.items()}
    rng = deepcopy(rng_state())
    with pytest.raises(ValueError, match="inner/outer"):
        owner.validate_checkpoint_progress(saved, dict(cycle=1, cursor=2, versions=dict(native_cycle=1)))
    assert owner.poisoned and before == {role: content_digest(m.state_dict()) for role, m in owner.models.items()}
    assert equal(rng, rng_state())
    owner.load_state_dict(saved)
    assert equal(saved, owner.state_dict())


def test_common_safe_state_api_stages_every_owner_before_copy_and_preserves_objects(files, complete, tmp_path):
    from distillschedule.training.checkpoint import pack_state, unpack_state, rng_state
    from distillschedule.training.livewan_public import content_digest
    owner, _ = owner_from_boundary(files, complete, tmp_path / "not-created")
    saved = deepcopy(owner.state_dict())
    packed = pack_state(dict(native_cycle=owner))
    assert packed["native_cycle"]["kind"] == "stateful"
    bad = deepcopy(packed)
    spec = bad["native_cycle"]["value"]["optimizers"]["critic"]["state"]
    next(iter(spec.values()))["step"].fill_(4.)
    before = {role: content_digest(model.state_dict()) for role, model in owner.models.items()}
    rng, ids, optimizer_ids = deepcopy(rng_state()), deepcopy(owner._parameter_ids), dict(owner._optimizer_ids)
    progress = dict(cycle=1, cursor=1, versions=dict(native_cycle=1))
    with pytest.raises(ValueError, match="clock"):
        unpack_state(bad, dict(native_cycle=owner), progress=progress)
    assert owner.poisoned and before == {role: content_digest(m.state_dict()) for role, m in owner.models.items()}
    assert equal(rng, rng_state())
    unpack_state(packed, dict(native_cycle=owner), progress=progress)
    assert not owner.poisoned and equal(saved, owner.state_dict())
    assert ids == owner._parameter_ids and optimizer_ids == owner._optimizer_ids and equal(rng, rng_state())


def test_actual_changed_source_read_after_g_commit_poisons_without_cache_fallback(files, complete, tmp_path):
    from distillschedule.data.livewan_reference_inputs import decode_cycle, CPUCycleInputs
    owner, data = owner_from_boundary(files, complete, tmp_path / "not-created")
    saved = deepcopy(owner.state_dict())
    row = decode_cycle(data.sample(1), cycle=1)
    path = Path(row["control"]["entries"][2]["source"]["path"])
    original = path.read_bytes()
    changed = []
    def change(frame, event, arg):
        if event == "call" and frame.f_code is CPUCycleInputs.load.__code__ and frame.f_locals["self"].read_count == 2 and not changed:
            altered = bytearray(original)
            altered[-1] ^= 1
            path.write_bytes(altered)
            changed.append(1)
    try:
        sys.setprofile(change)
        with pytest.raises(ValueError, match="source bytes changed"):
            owner.run(row, cycle=1)
    finally:
        sys.setprofile(None)
        path.write_bytes(original)
    assert changed and owner.poisoned and owner.versions == dict(generator=2, critic=7, ema=0)
    assert owner.state_dict()["checkpoint_status"] == "POISONED_NOT_RESUMABLE"
    owner.load_state_dict(saved)
    assert equal(saved, owner.state_dict())


def test_native_optional_context_keeps_legacy_duck_callers_and_bf16_flags(monkeypatch):
    """Dispatch characterization only, NOT actual full-model numerical proof."""
    from contextlib import nullcontext
    from types import SimpleNamespace
    from distillschedule.runtime import livewan_crossscale_worker as worker
    calls = []
    def cfg(model, z, t, rope, positive, negative, lengths, dtype, guidance, scale):
        calls.append(("teacher", z.dtype, dtype, guidance, scale))
        return torch.ones_like(z)
    def velocity(model, z, t, rope, context, lengths, dtype, scale):
        calls.append(("fake", z.dtype, dtype, scale))
        return torch.ones_like(z)
    parameter = torch.nn.Parameter(torch.ones(()))
    def critic(model, x0, t, rope, context, lengths, dtype, scale, *, grad_checkpoint, noise):
        calls.append(("critic", x0.dtype, dtype, scale, grad_checkpoint))
        return parameter * x0.float().mean()
    native = SimpleNamespace(dmd=SimpleNamespace(cfg_velocity=cfg, bidirectional_velocity=velocity, critic_loss=critic))
    monkeypatch.setitem(sys.modules, "wanstreamer", native)
    monkeypatch.setitem(sys.modules, "wanstreamer.lora", SimpleNamespace(lora_enabled=lambda *_: nullcontext()))
    class Projection:
        def text_embedding(self, value):
            calls.append(("projection", value.dtype))
            return value
    bundle = SimpleNamespace(teacher=Projection(), critic=Projection(), critic_gradient_reducer=None)
    raw = torch.ones(1, 512, 1)
    projected = worker.prepare_score_sample_context(torch, bundle=bundle, raw_context=raw, include_teacher=True)
    assert projected.reference_context is None
    score = SimpleNamespace(rope=None, negative_teacher_context=raw.bfloat16())
    sample = SimpleNamespace(teacher=raw.bfloat16(), critic=raw.bfloat16())
    flow = SimpleNamespace(z_t=torch.ones(1, 16, 21, 1, 1), t=torch.tensor([.6]))
    kwargs = dict(score_context=score, flow=flow, context_lens=torch.tensor([512]))
    assert worker._dmd_teacher_velocity(torch, bundle=bundle, sample_context=sample, **kwargs).dtype == torch.float32
    assert worker._dmd_fake_velocity(torch, critic_model=bundle.critic, critic_context=raw.bfloat16(), **kwargs).dtype == torch.float32
    row = SimpleNamespace(seeds=dict(critic_t=1, critic_noise=2))
    result = worker.critic_backward_for_sample(torch, bundle=bundle, score_context=score, row=row,
        x0=flow.z_t, sample_context=sample, context_lens=torch.tensor([512]), sample_weight=1.)
    assert result.dtype == torch.float32 and parameter.grad is not None
    assert calls == [("projection", torch.bfloat16), ("projection", torch.bfloat16),
        ("teacher", torch.bfloat16, torch.bfloat16, 5., 1000.),
        ("fake", torch.bfloat16, torch.bfloat16, 1000.),
        ("critic", torch.bfloat16, torch.bfloat16, 1000., True)]


def test_unadmitted_context_world_size_and_autocast_reject_before_math(files, tmp_path):
    from types import SimpleNamespace
    from distillschedule.runtime import livewan_crossscale_worker as worker
    from distillschedule.training.livewan_public import build_model
    with pytest.raises(RuntimeError, match="unadmitted"):
        worker._build_streamers(torch, student=None, count=1, device="cpu", reference_context=SimpleNamespace())
    components = __import__("distillschedule.training.session", fromlist=["construct"]).construct(config_for(files, tmp_path / "not-created"))[0]
    context = components.state["native_cycle"].context
    with pytest.raises(RuntimeError, match="world1"):
        worker._mean_replicated_gradients(torch, [], group=None, group_size=16, bucket_mib=128, reference_context=context)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="no-autocast"):
            build_model(dict(asset_manifest=str(files["assets"]), device="cpu"))


def test_preconstruction_patched_native_cycle_is_not_admitted(files, monkeypatch):
    from distillschedule.training.livewan_public import build_model
    from distillschedule.runtime import livewan_crossscale_worker as worker
    monkeypatch.setattr(worker, "_run_baseline", lambda **_: None)
    with pytest.raises(ValueError, match="source|binding"):
        build_model(dict(asset_manifest=str(files["assets"]), device="cpu"))


@pytest.mark.parametrize("change", ["budget", "head", "architecture", "runtime", "precision", "geometry", "missing_role"])
def test_assets_fail_closed_before_real_model_construction(files, tmp_path, change):
    from distillschedule.training.livewan_public import build_model, pin_file
    from distillschedule.reference import livewan_wan21 as provider
    from safetensors.torch import load_file, save_file
    value = json.loads(Path(files["assets"]).read_bytes())
    if change == "budget":
        value["budget"]["max_parameters"] = 1
    elif change == "head":
        raw = load_file(value["student"]["weights"][0]["path"])
        raw["head.head.weight"].zero_()
        path = tmp_path / "cold-zero.safetensors"
        save_file(raw, str(path), metadata=dict(evidence_kind=value["evidence_kind"]))
        value["student"]["weights"] = [pin_file(path)]
    elif change == "architecture":
        cfg = json.loads(Path(value["student"]["config"]["path"]).read_bytes())
        cfg["dim"] = 100_000_000
        path = json_file(tmp_path / "architecture.json", cfg)
        value["student"]["config"] = pin_file(path)
    elif change == "runtime":
        value["runtime"]["torch"]["version"] = "bogus"
    elif change == "precision":
        value["precision"] = "bfloat16"
    elif change == "geometry":
        value["geometry"]["latent_shape"][-1] = 80
    elif change == "missing_role":
        value.pop("teacher")
    assets = json_file(tmp_path / "bad-assets.json", value)
    api = provider.load_cpu(backend="torch-sdpa-cpu", precision="float32")
    init = api.model.WanModel.__init__.__wrapped__.__code__
    constructed = []
    def profile(frame, event, arg):
        if event == "return" and frame.f_code is init:
            model = frame.f_locals["self"]
            if model.patch_embedding.weight.device.type != "meta":
                constructed.append(1)
    try:
        sys.setprofile(profile)
        with pytest.raises((ValueError, RuntimeError)):
            build_model(dict(asset_manifest=str(assets), device="cpu"))
    finally:
        sys.setprofile(None)
    # A cold-zero head is an actual supplied-value admission after safe load;
    # every other malformed control/header fails before real construction.
    assert constructed == ([1] if change == "head" else [])


def test_original_livewan_public_atomic_factory_and_explicit_context_exist():
    from distillschedule.training.livewan_public import build_model, build_algorithm, LiveWanCPUContext
    from distillschedule.data.livewan_reference_inputs import prepare_inputs
    from distillschedule.runtime import livewan_crossscale_worker as worker
    import inspect
    assert all(callable(fn) for fn in (build_model, build_algorithm, prepare_inputs))
    assert LiveWanCPUContext.__module__ == "distillschedule.training.livewan_public"
    assert "reference_context" in inspect.signature(worker.prepare_rollout).parameters
    assert "reference_context" in inspect.signature(worker._mean_replicated_gradients).parameters
