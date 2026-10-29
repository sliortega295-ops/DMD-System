"""C51 continuation protocol evidence; NOT a CUDA/model numerical gate.

Native-shaped archives and original TeacherLatents/LoRA are real where the
explicit licensed source fixture is present. Tiny native-math / CUDA / Gloo
stand-ins below test only orchestration, not Wan outputs, CUDA or full world16.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test_training_livewan_state import original_source, original_lora, owner, equal


def test_explicit_stream_preset_is_not_the_legacy_two_cycle_binding():
    from distillschedule.training.native_presets import describe_preset
    value = describe_preset("W4", "self_forcing.c51_stream.v1")
    assert value["continuous_training"] is True
    assert value["cycles"]["max"] == 100
    assert value["numerical_status"] == "REVIEW"
    assert describe_preset("W4", "self_forcing.c51.v1")["continuous_training"] is False


def test_full_c51_lifecycle_module_exists_without_importing_native_models():
    from distillschedule.training.livewan_c51_training import descriptor
    value = descriptor()
    assert value["global_batch"] == 16
    assert value["updates"] == {"generator": 1, "critic": 5, "ema": 0}
    assert value["native_math"] == "_run_pipeline_8x8_continuous/_run_pipeline_8x8"


@pytest.fixture(scope="module")
def stream(tmp_path_factory, original_source):
    import torch
    from distillschedule.data.livewan_inputs import prepare_c51_inputs
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("real-original-c51-inputs")
    data = root/"teacher"
    data.mkdir()
    prompts = root/"prompts.pt"
    torch.save(dict(pos=torch.tensor([.25, .5], dtype=torch.float16).view(2, 1, 1).expand(2, 512, 4096),
                    neg=torch.tensor(-.25, dtype=torch.float16).expand(1, 512, 4096),
                    prompts=["CPU native-shape A", "CPU native-shape B"], neg_prompt="CPU fixture"), prompts)
    for i in range(320):
        torch.save(dict(latents=torch.tensor(i/512, dtype=torch.float16).expand(16, 21, 46, 80),
                        prompt_idx=i % 2, seed=i, size="640*368", frames=81, steps=40, cfg=5.), data/f"sample-{i:04d}.pt")
    manifest = root/"stream.json"
    value = prepare_c51_inputs(data, prompts, manifest, source_root=original_source, cycles=4, seed=31415,
                              evidence_kind="CPU_NATIVE_SHAPE_FIXTURE_NOT_PRETRAINED")
    yield manifest, value
    torch.set_num_threads(old)


_LOADERS = {}


def pool(stream, rank=0):
    import torch
    from distillschedule.data.livewan_inputs import C51_KIND
    from distillschedule.runtime.livewan_crossscale_worker import InputLoader, _pipeline_8x8_slot_pairs
    from distillschedule.training.livewan_c51_training import make_plan
    from distillschedule.training.livewan_c51_state import PoolInputs
    plan = make_plan(dict(native=dict(models={r: dict(root="/explicit/"+r) for r in ("student", "critic", "teacher")})), "/unused")
    path = str(stream[0])
    if path not in _LOADERS:
        _LOADERS[path] = InputLoader(torch=torch, input_kind=C51_KIND, prepared=stream[0])
    loader = _LOADERS[path]
    return PoolInputs(loader, rank=rank, pairs=_pipeline_8x8_slot_pairs(plan))


def consume(cursor, cycle):
    from distillschedule.training.livewan_c51_state import physical_versions
    for step in range(5):
        for slot in cursor.slots:
            row = cursor.inputs.row(5*cycle+step, slot)
            if cursor.rank < 8:
                latent, context, pi = cursor.load(row)
                assert tuple(latent.shape) == (16, 21, 46, 80) and pi == row.prompt_idx
            else:
                value = cursor.context_from_trace(row, device="cpu")
                assert tuple(value.shape) == (1, 512, 4096)
    rows = list(cursor.pending)
    cursor.complete(cycle+1, physical_versions(cursor.rank, cycle+1))
    return rows


def test_real_original_dual_wave_consumption_and_public_c51_preparation(stream):
    from distillschedule.data.livewan_inputs import C51_SCHEMA
    g, score = pool(stream, 0), pool(stream, 8)
    assert stream[1]["schema_version"] == C51_SCHEMA
    assert stream[1]["evidence_kind"] == "CPU_NATIVE_SHAPE_FIXTURE_NOT_PRETRAINED"
    assert type(g.inputs.dataset).__name__ == "TeacherLatents"
    assert consume(g, 0) == consume(score, 0)
    assert consume(g, 1) == consume(score, 1)
    assert g.state_dict()["cursor"] == score.state_dict()["cursor"] == 10
    assert len({r["record_id"] for r in g.expected(0)+g.expected(1)}) == 20
    assert [r["world_frames"] for r in g.expected(0)] != [r["world_frames"] for r in g.expected(1)]


@pytest.mark.parametrize("bad", ["missing", "duplicate", "wrong_slot", "future"])
def test_half_wave_duplicate_wrong_slot_and_future_cannot_commit(stream, bad):
    from distillschedule.training.livewan_c51_state import physical_versions
    cursor = pool(stream)
    before = cursor.state_dict()
    row = cursor.inputs.row(0, cursor.slots[0])
    if bad == "missing":
        with pytest.raises(ValueError, match="dual-wave"):
            cursor.complete(1, physical_versions(0, 1))
    elif bad == "duplicate":
        cursor.load(row)
        with pytest.raises(ValueError, match="duplicate"):
            cursor.load(row)
        with pytest.raises(ValueError, match="half-wave"):
            cursor.state_dict()
        assert cursor.cycle == 0
        return
    else:
        changed = cursor.inputs.row(5 if bad == "future" else 0, cursor.slots[0] if bad == "future" else cursor.slots[1])
        with pytest.raises(ValueError, match="lane sample"):
            cursor.load(changed)
    assert cursor.state_dict() == before


@pytest.mark.parametrize("key,value", [("rank", 1), ("input_sha256", "0"*64), ("cursor", 1), ("pairs", []), ("history", ["0"*64])])
def test_pool_restore_rejects_identity_or_cursor_change_before_mutation(stream, key, value):
    cursor = pool(stream)
    consume(cursor, 0)
    saved = cursor.state_dict()
    bad = deepcopy(saved); bad[key] = value
    with pytest.raises(ValueError):
        cursor.load_state_dict(bad)
    assert cursor.state_dict() == saved


def test_public_options_require_new_stream_and_reject_old_fallback(tmp_path):
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training.native import _options, describe_native
    raw = dict(schema_version="distillschedule.training.v1", model=dict(workload="W4"),
        algorithm=dict(name="original_self_forcing_c51"),
        data=dict(input_kind="original_livewan_c51_teacher_stream.v1", prepared="/explicit/input.json"),
        execution=dict(adapter="native", mode="manual", task="/explicit/task.json", assets="/explicit/assets.json",
                       capacity_policy="/explicit/capacity.json", preset="self_forcing.c51_stream.v1"),
        trainer=dict(cycles=100, seed=31415, checkpoint_every=50, output_dir="/explicit/output"))
    conf = TrainingConfig.from_dict(raw, source=tmp_path/"training.json")
    assert _options(conf, "W4") == "manual"
    assert describe_native(conf)["capabilities"]["checkpoint"]["supported"] is True
    raw["execution"]["preset"] = "self_forcing.c51.v1"
    with pytest.raises(ValueError, match="exactly 2"):
        _options(TrainingConfig.from_dict(raw, source=conf.source), "W4")
    raw["execution"]["preset"] = "self_forcing.c51_stream.v1"
    raw["data"]["input_kind"] = "original_livewan_teacher_cohorts.v1"
    with pytest.raises(ValueError):
        _options(TrainingConfig.from_dict(raw, source=conf.source), "W4")


def test_plan_keeps_frozen_full_bundle_mechanisms_and_legacy_cycle_refusals():
    from distillschedule.core.plan import ExecutionPlan
    from distillschedule.runtime.livewan_c51_bundle import source_template, validate_plan, TEMPLATE_SHA256
    from distillschedule.workloads.best_known.registry import fingerprint
    from distillschedule.training.livewan_c51_training import make_plan
    assert fingerprint(source_template()) == TEMPLATE_SHA256
    plan = make_plan(dict(native=dict(models={r: dict(root="/explicit/"+r) for r in ("student", "critic", "teacher")})), "/unused")
    assert validate_plan(plan, mode="pipeline")
    bad = deepcopy(plan); bad["metadata"]["continuous_pipeline"]["cycles"] = 100
    with pytest.raises(RuntimeError, match="two continuous cycles"):
        validate_plan(bad, mode="pipeline")
    bad = deepcopy(plan); bad["metadata"]["critic_step1_overlap"] = True
    with pytest.raises(RuntimeError):
        validate_plan(bad, mode="pipeline")


class CPUEvent:
    def __init__(self, *args, **kwargs): self.complete = False
    def record(self, *args): self.complete = True
    def synchronize(self): assert self.complete


class CPUCuda:
    Event = CPUEvent
    class Stream:
        def __init__(self, **kwargs): pass
        def wait_stream(self, stream): pass
    @staticmethod
    def current_stream(*args): return None
    @staticmethod
    def set_device(*args): pass
    @staticmethod
    def stream(*args):
        from contextlib import nullcontext
        return nullcontext()


class CPUTensor:
    """Only device/pinning/copy events substituted; storage/value are real Torch."""
    def __init__(self, value, device):
        self.value, self.device = value, SimpleNamespace(type=device)
    @property
    def shape(self): return self.value.shape
    @property
    def dtype(self): return self.value.dtype
    def element_size(self): return self.value.element_size()
    def detach(self): return self
    def contiguous(self): return self
    def copy_(self, source, **kwargs):
        self.value.copy_(source.value)
        return self
    def is_pinned(self): return self.device.type == "cpu"


class CPUTorch:
    cuda = CPUCuda
    @staticmethod
    def empty(shape, *, dtype, device="cpu", **kwargs):
        import torch
        return CPUTensor(torch.empty(shape, dtype=dtype), str(device))


class CPULink:
    """Narrow in-process Gloo substitute, no invented completion counters."""
    def __init__(self):
        from queue import Queue
        self.queue = Queue()
    def isend(self, value, **kwargs):
        from threading import Event
        ready = Event()
        self.queue.put((kwargs["tag"], value.value.clone(), ready))
        return SimpleNamespace(wait=lambda: ready.wait(10) or (_ for _ in ()).throw(RuntimeError("send timeout")))
    def irecv(self, target, **kwargs):
        def wait():
            tag, value, ready = self.queue.get(timeout=10)
            assert tag == kwargs["tag"]
            target.value.copy_(value)
            ready.set()
        return SimpleNamespace(wait=wait)


def test_actual_bounded_transport_copy_send_receive_nonterminal_drain_and_backpressure():
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from distillschedule.training.livewan_c51_state import WindowedRunaheadTransport
    link = CPULink()
    kw = dict(torch=CPUTorch, dist=link, group=object(), device="cuda", shape=(3,), dtype=torch.float32,
              cycles=100, cross_cycle_lookahead=True, waves_per_step=2)
    g = WindowedRunaheadTransport(rank=0, peer=8, is_generator=True, **kw)
    score = WindowedRunaheadTransport(rank=8, peer=0, is_generator=False, **kw)
    def produce():
        for step in range(1, 5):
            for wave in (0, 1):
                x = CPUTensor(torch.full((3,), step*10+wave, dtype=torch.float32), "cuda")
                g.enqueue_send(step=step, wave=wave, tensor=x)
                g.wait_staged(step=step, wave=wave)
        g.drain_through(1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(produce)
        for step in range(1, 5):
            for wave in (0, 1):
                assert torch.equal(score.receive_to_device(step=step, wave=wave).value,
                                   torch.full((3,), step*10+wave, dtype=torch.float32))
        score.drain_through(1)
        future.result(timeout=10)
    assert len(g._slots) == len(score._slots) == 2
    assert g.receipt()["backpressure_waits"] == 8
    assert not g.buffers and not score.buffers and not g.send_futures
    with pytest.raises(ValueError, match="duplicate"):
        score.receive_to_device(step=1, wave=0)
    with pytest.raises(ValueError, match="incomplete waves"):
        g.drain_through(2)
    g.executor.shutdown(wait=True)


def test_joint_header_requires_all_ranks_both_real_waves_and_physical_clocks(stream):
    from distillschedule.training.livewan_c51_training import validate_joint
    from distillschedule.training.livewan_c51_state import physical_versions
    cursor = pool(stream)
    rows = []
    for rank in range(16):
        lane = pool(stream, rank)
        rows.append(dict(rank=rank, cycle=1, versions=physical_versions(rank, 1), slots=list(lane.slots),
                         rows=lane.expected(0), quiescent=True))
    assert validate_joint(rows, cycle=1, pairs=cursor.pairs)["versions"] == dict(generator=1, critic=5, ema=0)
    for mutation in (lambda a: a.pop(), lambda a: a[8]["rows"].pop(),
                     lambda a: a[1]["versions"].update(critic=5), lambda a: a[0].update(quiescent=False)):
        bad = deepcopy(rows); mutation(bad)
        with pytest.raises(ValueError): validate_joint(bad, cycle=1, pairs=cursor.pairs)


def test_new_schema_does_not_raise_legacy_capacity_or_shape_limits(original_source):
    from distillschedule.data.livewan_inputs import _inventory, C51_KIND
    with pytest.raises(ValueError, match="1..64"):
        _inventory("/missing", "/missing", source_root=original_source, cycles=100, seed=0,
                   evidence_kind="CPU_NATIVE_SHAPE_FIXTURE_NOT_PRETRAINED")
    with pytest.raises(ValueError, match="missing teacher record"):
        _inventory("/missing", "/missing", source_root=original_source, cycles=100, seed=0,
                   evidence_kind="CPU_NATIVE_SHAPE_FIXTURE_NOT_PRETRAINED", input_kind=C51_KIND)
    with pytest.raises(ValueError, match="1..100"):
        _inventory("/missing", "/missing", source_root=original_source, cycles=101, seed=0,
                   evidence_kind="CPU_NATIVE_SHAPE_FIXTURE_NOT_PRETRAINED", input_kind=C51_KIND)


def cpu_owner_chain(manifest, source, output, *, role, resume=None, stop=4):
    """Real original LoRA/Adam/input/checkpoint, deliberately NOT native DMD."""
    import importlib.util
    import random
    import numpy as np
    import torch
    from distillschedule.training.checkpoint import pack_state, save_checkpoint, restore_checkpoint, cpu_tree, rng_state
    from distillschedule.training.livewan_c51_state import owner_state, Progress, physical_versions
    from test_training_livewan_state import update
    torch.set_num_threads(1)
    torch.manual_seed(2026); random.seed(2026); np.random.seed(2026)
    spec = importlib.util.spec_from_file_location("real_original_lora_c51_cpu", Path(source)/"wanstreamer/lora.py")
    lora = importlib.util.module_from_spec(spec); spec.loader.exec_module(lora)
    current = owner(lora, role)
    rank = 0 if role == "generator" else 8
    cursor = pool((Path(manifest), None), rank)
    actual = physical_versions(rank, 0)
    progress = Progress(cursor, actual, restore_cycle=2 if resume else None)
    state = owner_state(current.bundle, source_root=source, progress=progress)
    identity = dict(scope="REAL_CPU_ORIGINAL_OWNER_AND_DATA_NOT_WORLD16_NOT_DMD_MATH", input=cursor.inputs.input_sha256,
                    role=role, horizon=4)
    if resume:
        restore_checkpoint(resume, state, identity=identity)
        progress.restore_cycle = None
    outputs = []
    for cycle in range(cursor.cycle, stop):
        for _ in range(1 if rank == 0 else 5):
            outputs.append(update(current))
            actual[role] += 1
        consume(cursor, cycle)
        progress.validate_checkpoint_boundary()
        if cycle == 1:
            save_checkpoint(Path(output)/"midpoint", state, identity=identity, cycle=2, cursor=10,
                            versions=dict(generator=2, critic=10, ema=0), policy=dict(scope="CPU_ONLY"))
    result = dict(state=pack_state(state), rng=cpu_tree(rng_state()), outputs=cpu_tree(outputs),
                  parameter_ids=[id(p) for p in current.parameters],
                  evidence_scope=identity["scope"])
    Path(output).mkdir(parents=True, exist_ok=True)
    torch.save(result, Path(output)/"result.pt")
    return result


@pytest.mark.parametrize("role", ["generator", "critic"])
def test_actual_native_owner_named_adam_rng_cursor_checkpoint_new_process_resume(stream, original_source, tmp_path, role):
    import torch
    full = cpu_owner_chain(stream[0], original_source, tmp_path/"full", role=role)
    cpu_owner_chain(stream[0], original_source, tmp_path/"first", role=role, stop=2)
    command = [sys.executable, "-c", (
        "import sys; from test_training_livewan_c51_continuation import cpu_owner_chain; "
        "cpu_owner_chain(sys.argv[1],sys.argv[2],sys.argv[3],role=sys.argv[4],resume=sys.argv[5])"),
        str(stream[0]), str(original_source), str(tmp_path/"restored"), role, str(tmp_path/"first/midpoint")]
    result = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
    restored = torch.load(tmp_path/"restored/result.pt", weights_only=True)
    assert equal(full["state"], restored["state"])
    assert equal(full["rng"], restored["rng"])
    assert equal(full["outputs"][2 if role == "generator" else 10:], restored["outputs"])
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("damage", ["step", "moment_shape", "parameter_name", "stride", "cursor"])
def test_saved_owner_layout_adam_and_progress_refuse_before_mutating_any_owner(stream, original_source, original_lora, damage):
    import torch
    from distillschedule.training.checkpoint import pack_state, unpack_state, cpu_tree
    from distillschedule.training.livewan_c51_state import owner_state, Progress, physical_versions
    cursor = pool(stream)
    current = owner(original_lora, "generator")
    actual = physical_versions(0, 0)
    progress = Progress(cursor, actual)
    state = owner_state(current.bundle, source_root=original_source, progress=progress)
    saved = cpu_tree(pack_state(state)); before = cpu_tree(pack_state(state))
    weights = saved["generator.weights"]["value"]
    optim = saved["generator.optimizer"]["value"]
    if damage == "step": next(iter(optim["optimizer"]["state"].values()))["step"].fill_(1)
    elif damage == "moment_shape": next(iter(optim["optimizer"]["state"].values()))["exp_avg"] = torch.zeros(1)
    elif damage == "parameter_name": optim["parameters"][0][0] += "changed"
    elif damage == "stride":
        key = next(k for k, v in weights["tensors"].items() if v.ndim == 2)
        weights["tensors"][key] = weights["tensors"][key].T.contiguous().T
    else: saved["pool.progress"]["value"]["data"]["cursor"] = 1
    with pytest.raises(ValueError): unpack_state(saved, state)
    assert equal(cpu_tree(pack_state(state)), before)


def detached_text_generator(stream, original_source, original_lora, *, restore_cycle=None):
    """Real materialized AdamW with native-style detached text, not Wan math."""
    import torch
    from distillschedule.runtime.livewan_crossscale_worker import (
        _bind_trajectory_optimizer_parameter_names, materialize_adam_state,
    )
    from distillschedule.training.livewan_c51_state import owner_state, Progress, physical_versions
    current = owner(original_lora, "generator")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(72)
        current.model.add_module("text_embedding", torch.nn.Linear(3, 3))
    current.model.eval()
    current.parameters = list(current.model.parameters())
    current.optimizer.param_groups[0]["params"] = current.parameters
    current.bundle.generator_optimizer_parameter_names = _bind_trajectory_optimizer_parameter_names(
        named_parameters=list(current.model.named_parameters()), optimizer=current.optimizer,
        expected_parameter_ids={id(p) for p in current.parameters})
    current.bundle.metadata["student"]["optimizer"] = materialize_adam_state(torch, current.optimizer)
    current.progress = Progress(pool(stream), physical_versions(0, 0), restore_cycle=restore_cycle)
    current.state = owner_state(current.bundle, source_root=original_source, progress=current.progress)
    return current


def advance_detached_text_generator(current):
    import torch
    cycle = current.progress.cursor.cycle
    current.optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        context = current.model.text_embedding(torch.arange(6, dtype=torch.float32).view(2, 3)/10+cycle)
    (current.model.projection(context)-.5).square().mean().backward()
    assert all(p.grad is None for p in current.model.text_embedding.parameters())
    current.optimizer.step()
    current.optimizer.zero_grad(set_to_none=True)
    current.progress.actual["generator"] += 1
    consume(current.progress.cursor, cycle)


def test_detached_text_generator_native_adam_positive_cycle_checkpoint_restore(
        stream, original_source, original_lora, tmp_path):
    import torch
    from distillschedule.training.checkpoint import pack_state, save_checkpoint, restore_checkpoint, cpu_tree
    current = detached_text_generator(stream, original_source, original_lora)
    initial = {n: p.detach().clone() for n, p in current.model.named_parameters()}
    advance_detached_text_generator(current)
    advance_detached_text_generator(current)
    for name, parameter in current.model.named_parameters():
        slots = current.optimizer.state[parameter]
        detached = name.startswith("text_embedding.")
        assert slots["step"].item() == (0 if detached else 2)
        assert torch.equal(parameter, initial[name]) is detached
        for key in ("exp_avg", "exp_avg_sq"):
            assert (torch.count_nonzero(slots[key]).item() == 0) is detached
    saved = cpu_tree(pack_state(current.state))
    identity = dict(scope="CPU_DETACHED_TEXT_NATIVE_ADAM_NOT_WAN_OR_WORLD16")
    path = save_checkpoint(tmp_path/"detached-text", current.state, identity=identity,
                           cycle=2, cursor=10, versions=dict(generator=2, critic=10, ema=0))

    restored = detached_text_generator(stream, original_source, original_lora, restore_cycle=2)
    with torch.no_grad():
        for parameter in restored.parameters:
            parameter.add_(3.)
    # Live serialization must use cycle zero, not the expected restore cycle.
    assert cpu_tree(pack_state(restored.state))["pool.progress"]["value"]["data"]["cycle"] == 0
    parameter_ids = [id(p) for p in restored.parameters]
    optimizer_id = id(restored.optimizer)
    restore_checkpoint(path, restored.state, identity=identity)
    restored.progress.restore_cycle = None
    assert equal(cpu_tree(pack_state(restored.state)), saved)
    assert [id(p) for p in restored.model.parameters()] == parameter_ids
    assert [[id(p) for p in group["params"]] for group in restored.optimizer.param_groups] == [parameter_ids]
    assert id(restored.bundle.generator_optimizer) == optimizer_id
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("slot", ["step", "exp_avg", "exp_avg_sq"])
def test_detached_text_adam_corruption_rejected_before_any_owner_mutation(
        stream, original_source, original_lora, slot):
    import torch
    from distillschedule.training.checkpoint import pack_state, unpack_state, cpu_tree
    current = detached_text_generator(stream, original_source, original_lora)
    advance_detached_text_generator(current)
    advance_detached_text_generator(current)
    saved = cpu_tree(pack_state(current.state))
    value = saved["generator.optimizer"]["value"]
    ids_by_name = dict(zip(value["parameters"][0], value["optimizer"]["param_groups"][0]["params"], strict=True))
    value["optimizer"]["state"][ids_by_name["text_embedding.weight"]][slot].fill_(1.)
    restored = detached_text_generator(stream, original_source, original_lora, restore_cycle=2)
    with torch.no_grad():
        restored.model.projection.weight.add_(3.)
    before = cpu_tree(pack_state(restored.state))
    parameter_ids = [id(p) for p in restored.parameters]
    message = "step/version mismatch: text_embedding" if slot == "step" else "detached text_embedding AdamW moment"
    with pytest.raises(ValueError, match=message):
        unpack_state(saved, restored.state, progress=dict(cycle=2, cursor=10,
                     versions=dict(generator=2, critic=10, ema=0)))
    assert equal(cpu_tree(pack_state(restored.state)), before)
    assert [id(p) for p in restored.model.parameters()] == parameter_ids
    assert not torch.cuda.is_initialized()


def cpu_continuation_chain(manifest, source, output, *, resume=None, stop=4):
    """Actual public dispatch -> continuous loop -> publication -> safe restore.

    Native-math body, CUDA synchronization, Gloo transport, peer-rank metadata
    and CUDA mechanism validation are explicit stand-ins. Source data, original
    owner Parameters/Adam/RNG, loop/lookahead/drain policy and checkpoint I/O are
    real. This is NOT all-rank or full-C51 numeric qualification.
    """
    import importlib.util
    import random
    import numpy as np
    import torch
    from distillschedule.runtime import livewan_crossscale_worker as worker, livewan_c51_bundle as bundle_api
    from distillschedule.training import native, livewan_c51_training as training, distributed
    from distillschedule.training.checkpoint import pack_state, cpu_tree, rng_state, read_manifest
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training.livewan_c51_state import physical_versions
    from test_training_livewan_state import update
    torch.set_num_threads(1)
    random.seed(2718); np.random.seed(2718); torch.manual_seed(2718)
    spec = importlib.util.spec_from_file_location("c51_real_lora_runtime_protocol", Path(source)/"wanstreamer/lora.py")
    lora = importlib.util.module_from_spec(spec); spec.loader.exec_module(lora)
    current = owner(lora, "generator")
    raw = dict(schema_version="distillschedule.training.v1", model=dict(workload="W4"), algorithm=dict(name="original_self_forcing_c51"),
        data=dict(input_kind="original_livewan_c51_teacher_stream.v1", prepared=str(manifest)),
        execution=dict(adapter="native", mode="manual", preset="self_forcing.c51_stream.v1", task="/explicit/task", assets="/explicit/assets",
                       capacity_policy="/explicit/capacity"),
        trainer=dict(cycles=4, checkpoint_every=2, stop_after_cycle=stop, seed=31415, output_dir=str(output),
                     **(dict(resume_from=str(resume)) if resume else {})))
    config = TrainingConfig.from_dict(raw, source=Path(output)/"training.json")
    cursor = pool((manifest, None))
    identity = dict(schema_version=training.SCHEMA, scope="CPU_CONTINUOUS_LIFECYCLE_WITH_DECLARED_CUDA_GLOO_MATH_STANDINS", horizon=4,
                    input_sha256=cursor.inputs.input_sha256)
    header = read_manifest(resume, identity=identity) if resume else None
    capacity_checks = []
    cont = training.Continuation(config=config, admitted=dict(assets=dict(native=dict(sources=dict(livewan=dict(root=str(source)))))),
        identity=identity, output=output, bootstrap=None, loader=cursor.loader, header=header,
        capacity_check=lambda: capacity_checks.append(True))
    actual = physical_versions(0, 0)
    loader = cont.bind(torch=torch, rank=0, bundle=current.bundle, versions=actual, pairs=cursor.pairs)
    evidence, inventory = worker.RunEvidence(), worker.CallInventory()
    timeline = []
    class Transport:
        def __init__(self): self.sent = set(); self.drained = cont.start_cycle
        def enqueue_send(self, *, step, wave, tensor):
            assert (step, wave) not in self.sent
            self.sent.add((step, wave)); timeline.append(("send", step, wave))
        def wait_staged(self, *, step, wave): assert (step, wave) in self.sent
        def drain_through(self, cycle):
            assert not any(step >= 5*cycle for step, _ in self.sent)
            self.drained = cycle; timeline.append(("drain", cycle))
        def finish(self): self.drain_through(stop)
        def receipt(self): return dict(scope="CPU_TRANSPORT_STANDIN", drained=self.drained)
    transport = Transport()
    shadow = SimpleNamespace(events=[], pending_source_work=None, pending_source_version=None,
                             finish=lambda: None, receipt=lambda: dict(scope="CPU_SHADOW_STANDIN"))
    def prepare(torch, *, row, loader, **kwargs):
        latent, _, _ = loader.load(row)
        inventory.counts["student/nograd/plain"] = inventory.counts.get("student/nograd/plain", 0)+1+5*row.new_blocks
        return SimpleNamespace(row=row, x0=latent.reshape(-1)[:3].clone())
    def body(**kwargs):
        cycle = kwargs["cycle_index"]
        if cycle:
            shadow.events.append(dict(version=5*cycle, status="PASS"))
        current.optimizer.zero_grad(set_to_none=True)  # original G helper owns this at each cycle entry
        update(current)
        actual["generator"] += 1
        inventory.counts["student/grad/checkpoint"] = inventory.counts.get("student/grad/checkpoint", 0)+6
        for step in range(5):
            for wave, slot in enumerate(cursor.slots):
                if step == wave == 0 and kwargs["precomputed_first"] is not None:
                    continue
                row = cursor.inputs.row(5*cycle+step, slot)
                value = prepare(torch, row=row, loader=loader)
                worker._record_output(torch, evidence, row=row, x0=value.x0, run_kind="continuation")
                if step:
                    transport.enqueue_send(step=5*cycle+step, wave=wave, tensor=value.x0)
        evidence.loss_fingerprints.extend(dict(kind="generator", finite=True, mean=1., norm=1., max_abs=1.) for _ in range(6))
        evidence.optimizer_diagnostics.append(dict(clip_input_norm=1.))
    original_gather = distributed.gather
    def gather(value):
        if type(value) is dict and set(value) == {"rank", "cycle", "versions", "slots", "rows", "quiescent"}:
            # Only the peer rank metadata plane is substituted. No PASS from
            # this fixture may be used as world16 consumption evidence.
            values = []
            for rank in range(16):
                view = object.__new__(type(loader)); view.__dict__.update(loader.__dict__)
                view.rank, view.slots = rank, cursor.pairs[rank % 8]
                values.append(dict(rank=rank, cycle=value["cycle"], versions=physical_versions(rank, value["cycle"]),
                    slots=list(view.slots), rows=view.expected(value["cycle"]-1), quiescent=True))
            return values
        return original_gather(value)
    cont.mechanism_receipt = lambda: dict(scope="EXPLICIT_CPU_NATIVE_MECHANISM_VALIDATOR_STANDIN")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "synchronize", lambda *a: None)
        patch.setattr(worker, "prepare_rollout", prepare)
        patch.setattr(worker, "_run_pipeline_8x8", body)
        patch.setattr(distributed, "gather", gather)
        patch.setattr(bundle_api, "runtime_receipt_errors", lambda *a, **k: [])
        patch.setattr(bundle_api, "continuation_receipt_errors", lambda *a, **k: [])
        def dispatch(conf, *, admit):
            assert conf is config and native._options(conf, "W4") == "manual"
            worker._run_pipeline_8x8_continuous(torch=torch, rank=0, trace=cursor.inputs, loader=loader,
                bundle=current.bundle, streamers=[SimpleNamespace(), SimpleNamespace()], score_context=None,
                run_kind="continuation", evidence=evidence, versions=actual, device="cpu",
                runahead_host_transport=transport, critic_shadow_hot_sync=shadow, slot_pairs_by_lane=cursor.pairs,
                full_context_lens_elision=True, critic_step1_split_overlap=False, critic_b1_attribution=False,
                inventory=inventory, continuation=cont)
            return 0
        patch.setattr(training, "run_training", dispatch)  # public dispatch only; no CUDA environment pretence
        assert native.run_native(config) == 0
    result = dict(state=pack_state(cont.state), rng=cpu_tree(rng_state()), timeline=timeline,
                  completed=loader.cycle, scope=identity["scope"], capacity_checks=len(capacity_checks))
    Path(output).mkdir(parents=True, exist_ok=True)
    torch.save(result, Path(output)/"result.pt")
    return result


def test_actual_continuous_loop_boundary_no_future_calls_and_fresh_process_restore(stream, original_source, tmp_path, record_property):
    import torch
    full = cpu_continuation_chain(stream[0], original_source, tmp_path/"full")
    first = cpu_continuation_chain(stream[0], original_source, tmp_path/"first", stop=2)
    assert first["completed"] == 2 and (tmp_path/"first/checkpoints/cycle-000002/_SUCCESS").is_file()
    assert not any(row[0] == "send" and row[1] >= 10 for row in first["timeline"])
    assert first["timeline"][-1] == ("drain", 2)
    run = subprocess.run([sys.executable, "-c", (
        "import sys; from test_training_livewan_c51_continuation import cpu_continuation_chain; "
        "cpu_continuation_chain(sys.argv[1],sys.argv[2],sys.argv[3],resume=sys.argv[4])"),
        str(stream[0]), str(original_source), str(tmp_path/"restored"), str(tmp_path/"first/checkpoints/cycle-000002")],
        capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout+run.stderr
    resumed = torch.load(tmp_path/"restored/result.pt", weights_only=True)
    assert equal(full["state"], resumed["state"])
    assert equal(full["rng"], resumed["rng"])
    # One per cycle, one with the live snapshot per checkpoint, and one before
    # the final segment receipt: full=4+2+1; each half=2+1+1.
    assert full["capacity_checks"] == 7 and first["capacity_checks"] == resumed["capacity_checks"] == 4
    assert resumed["timeline"][0] == ("send", 10, 0)  # exactly the first real resume row, no padding
    record_property("scope", "REAL_CPU_DATA_OWNER_LOOP_CHECKPOINT; EXPLICIT_CUDA_GLOO_PEERS_NATIVE_MATH_STANDINS; NOT_GPU_NUMERICAL")


def test_horizon100_actual_continuous_loop_checkpoint50_does_not_consume_cycle51(tmp_path, monkeypatch, record_property):
    """100-cycle scheduling fixture only; NOT 8000 real Teacher archives/math."""
    import torch
    from distillschedule.runtime import livewan_crossscale_worker as worker
    from distillschedule.training.checkpoint import save_checkpoint, restore_checkpoint
    pairs = tuple((i, i+8) for i in range(8))
    consumed, sends, completed = [], [], []
    state = {"physical": dict(generator=0, critic=0, ema=0)}
    identity = dict(scope="SYNTHETIC_OPERATION_METADATA_SCHEDULER_ONLY_NOT_NATIVE_MATH", horizon=100)
    def row(step, slot):
        return SimpleNamespace(operation_key=f"cycle={step//5}/step={step%5}/slot={slot}", logical_step=step,
                               logical_slot=slot, new_blocks=4, world_frames=9, rope_gap=0, seeds={})
    trace = SimpleNamespace(cycles=100, global_batch=16, row=row, source_reuse_across_cycles=False)
    def prepare(torch, *, row, **kw):
        consumed.append((row.logical_step, row.logical_slot))
        return SimpleNamespace(row=row, x0=torch.zeros(1))
    class Host:
        def wait_staged(self, *, step, wave): assert (step, wave) in sends
        def enqueue_send(self, *, step, wave, tensor): sends.append((step, wave))
        def finish(self): pass
        def receipt(self): return dict(scope="CPU_METADATA_TRANSPORT_ONLY")
    shadow = SimpleNamespace(begin_source=lambda **kw: None, finish=lambda: None,
                             receipt=lambda: dict(scope="CPU_SHADOW_METADATA_ONLY"))
    def body(**kw):
        cycle = kw["cycle_index"]
        for step in range(5):
            for wave, slot in enumerate(pairs[0]):
                if step == wave == 0 and kw["precomputed_first"] is not None:
                    continue
                value = prepare(torch, row=row(5*cycle+step, slot))
                if step: kw["runahead_host_transport"].enqueue_send(step=5*cycle+step, wave=wave, tensor=value.x0)
        state["physical"]["generator"] += 1
    class Boundary:
        def __init__(self, start, end):
            self.start_cycle, self.end_cycle = start, end
            self.cycles_executed = end-start
            self.identity = dict(schema_version="CPU_METADATA_SCHEDULING_ONLY_NOT_C51_QUALIFICATION")
        def prime(self, **kw):
            value = prepare(torch, row=row(5*self.start_cycle, pairs[0][0]))
            kw["transport"].enqueue_send(step=5*self.start_cycle, wave=0, tensor=value.x0)
            return value
        def cycle_complete(self, *, cycle, **kw):
            assert consumed[-10:] == [(5*(cycle-1)+s, slot) for s in range(5) for slot in pairs[0]]
            completed.append(cycle)
            if cycle == self.end_cycle:
                assert not any(step >= 5*cycle for step, _ in consumed)
                assert not any(step >= 5*cycle for step, _ in sends)
                save_checkpoint(tmp_path/f"scheduler-cycle-{cycle}", state, identity=identity, cycle=cycle,
                    cursor=5*cycle, versions=dict(state["physical"]), policy=dict(scope="SCHEDULER_ONLY_NOT_C51_ROLE_STATE"))
    monkeypatch.setattr(worker, "prepare_rollout", prepare)
    monkeypatch.setattr(worker, "_run_pipeline_8x8", body)
    def run(start, end):
        worker._run_pipeline_8x8_continuous(torch=torch, rank=0, trace=trace, loader=None,
            bundle=SimpleNamespace(student=None), streamers=[None, None], score_context=None,
            run_kind="continuation", evidence=worker.RunEvidence(), versions=state["physical"], device="cpu",
            runahead_host_transport=Host(), critic_shadow_hot_sync=shadow, slot_pairs_by_lane=pairs,
            full_context_lens_elision=True, critic_step1_split_overlap=False, critic_b1_attribution=False,
            inventory=worker.CallInventory(), continuation=Boundary(start, end))
    run(0, 50)
    assert (tmp_path/"scheduler-cycle-50/_SUCCESS").is_file()
    assert len(consumed) == 500 and max(step for step, _ in consumed) == 249
    state = {"physical": dict(generator=0, critic=0, ema=0)}
    restore_checkpoint(tmp_path/"scheduler-cycle-50", state, identity=identity)
    run(50, 100)
    assert completed == list(range(1, 101))
    assert consumed == [(s, slot) for s in range(500) for slot in pairs[0]]
    assert len(set(consumed)) == 1000
    assert state["physical"] == dict(generator=100, critic=0, ema=0)
    record_property("scope", identity["scope"])


def test_source_overlay_chain_preserves_frozen_rows_and_rejects_malformed_chains(tmp_path):
    import shutil
    from distillschedule.evidence.source_layout import reviewed_imports
    from distillschedule.workloads.best_known.registry import repository_root, file_sha256
    root = repository_root()
    docs = tmp_path/"docs/workloads"; docs.mkdir(parents=True)
    shutil.copyfile(root/"docs/workloads/source_import_map.json", docs/"source_import_map.json")
    lock = json.loads((root/"docs/workloads/source_layout.lock.json").read_text())
    lock.pop("integration_overlays", None)
    path = docs/"source_layout.lock.json"
    path.write_text(json.dumps(lock))
    before = reviewed_imports(tmp_path)
    path.write_text(json.dumps({**lock, "integration_overlays": []}))
    assert reviewed_imports(tmp_path) == before
    target = "src/distillschedule/runtime/livewan_crossscale_worker.py"
    previous = next(r["integrated_sha256"] for r in before if r["target"] == target)
    overlay = dict(schema_version="distillschedule.source_integration_overlay.v1", target=target,
        previous_sha256=previous, integrated_sha256=file_sha256(root/target),
        source_commit="90cb5d3b888ab128049fd5397420356c448aff2c", change="R31 explicit C51 continuation over the original worker")
    path.write_text(json.dumps({**lock, "integration_overlays": [overlay]}))
    after = reviewed_imports(tmp_path)
    assert next(r["integrated_sha256"] for r in after if r["target"] == target) == file_sha256(root/target)
    assert json.loads(path.read_text())["files"] == lock["files"]
    for bad in ({**overlay, "target": "unknown.py"}, {**overlay, "source_commit": "0"*40},
                {**overlay, "previous_sha256": "0"*64}, {**overlay, "integrated_sha256": previous},
                {**overlay, "change": ""}):
        path.write_text(json.dumps({**lock, "integration_overlays": [bad]}))
        with pytest.raises(ValueError): reviewed_imports(tmp_path)
    path.write_text(json.dumps({**lock, "integration_overlays": [overlay, overlay]}))
    with pytest.raises(ValueError): reviewed_imports(tmp_path)


def test_missing_world16_checkpoint_rank_refused_before_owner_construction(tmp_path, monkeypatch):
    import torch
    from distillschedule.training import distributed
    from distillschedule.training.checkpoint import SCHEMA, read_manifest
    from distillschedule.training.livewan_c51_training import TOPOLOGY
    from distillschedule.workloads.best_known.registry import file_sha256
    identity = dict(scope="CORRUPT_WORLD16_HEADER_NEGATIVE_FIXTURE_NOT_EXECUTION")
    manifest = dict(schema_version=SCHEMA, kind="RESUMABLE_COMPLETE_CYCLE", identity=identity,
        cycle=50, cursor=250, versions=dict(generator=50, critic=250, ema=0), policy={},
        topology=TOPOLOGY, torch=torch.__version__, files={f"rank-{r:05d}.pt": "0"*64 for r in range(16)})
    for rank in range(15): (tmp_path/f"rank-{rank:05d}.pt").touch()
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    (tmp_path/"_SUCCESS").write_text(file_sha256(tmp_path/"manifest.json"))
    monkeypatch.setattr(distributed, "rank_world", lambda: (0, 16))
    monkeypatch.setattr(distributed, "topology", lambda: TOPOLOGY)
    with pytest.raises(ValueError, match="shard missing"):
        read_manifest(tmp_path, identity=identity)


def test_unsupported_capacity_policy_and_untyped_continuation_fail_closed(tmp_path):
    from distillschedule.training.livewan_c51_training import capacity_policy, make_plan
    from distillschedule.runtime.livewan_crossscale_worker import run_worker
    with pytest.raises(ValueError, match="C51 capacity"):
        capacity_policy(dict(schema_version="old_or_disabled_policy"))
    plan = make_plan(dict(native=dict(models={r: dict(root="/explicit/"+r) for r in ("student", "critic", "teacher")})), "/unused")
    path = tmp_path/"plan.json"; path.write_text(json.dumps(plan))
    with pytest.raises(RuntimeError, match="typed native owner"):
        run_worker(plan_path=path, mode="pipeline", run_kind="continuation", repo_root=tmp_path, continuation=object())


def test_bounded_transport_cannot_reuse_host_slot_before_h2d_completion():
    import torch
    from distillschedule.training.livewan_c51_state import WindowedRunaheadTransport
    link = CPULink()
    score = WindowedRunaheadTransport(torch=CPUTorch, dist=link, group=object(), device="cuda", shape=(1,),
        dtype=torch.float32, cycles=2, cross_cycle_lookahead=True, waves_per_step=2,
        rank=8, peer=0, is_generator=False)
    link.isend(CPUTensor(torch.ones(1), "cpu"), tag=27102)
    score.receive_to_device(step=1, wave=0)
    score._h2d[(1, 0)].complete = False  # CUDA event stand-in: consumer copy is still pending.
    with pytest.raises(AssertionError): score.receive_to_device(step=2, wave=0)
    assert score._owners == {0: (1, 0)} and (2, 0) not in score.received_keys


def test_c51_overlay_uses_exact_real_gitlink_patches_not_clean_unpatched_wan(original_source, tmp_path):
    import shutil
    from distillschedule.training.livewan_reference_assets import source_identity
    from distillschedule.training.livewan_c51_assets import overlay_inventory
    value = os.environ.get("DMD_WAN21_TEST_UPSTREAM")
    if not value:
        pytest.skip("original C51 source-overlay test requires DMD_WAN21_TEST_UPSTREAM")
    native = dict(sources=dict(livewan=source_identity(original_source, project="livewan"),
                               wan21=source_identity(Path(value), project="wan21")))
    root = tmp_path/"verified-original-overlay"
    for name in native["sources"]["wan21"]["files"]:
        target = root/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(value)/name, target)
    with pytest.raises(ValueError, match="overlay source missing/changed"):
        overlay_inventory(native, root)  # unpatched upstream is NOT C51
    shutil.copyfile(original_source/"wan21_patches/modules/attention.py", root/"wan/modules/attention.py")
    shutil.copyfile(original_source/"wan21_patches/configs/__init__.py", root/"wan/configs/__init__.py")
    result = overlay_inventory(native, root)
    assert result["root"] == str(root) and not (root/".git").exists()
    assert result["files"]["wan/modules/attention.py"] != native["sources"]["wan21"]["files"]["wan/modules/attention.py"]
    (root/"wan/unadmitted.py").write_text("raise RuntimeError('must never import')")
    with pytest.raises(ValueError, match="Python inventory changed"):
        overlay_inventory(native, root)


def continuation_runtime_fixture(inputs, pairs, *, rank, start, end):
    """SYNTHETIC_TEST_ONLY: actual row controls, never observed CUDA counters.

    Keep every original full-bundle source/shape/operator validator enabled.
    Only the observations are synthetic; row ordering and census come from the
    actual admitted native stream, including a nonzero resumed segment.
    """
    from test_refactor_livewan_c51_planner import _runtime_fixture
    from distillschedule.training.livewan_c51_training import SCHEMA
    from distillschedule.training.livewan_c51_state import physical_versions
    value = _runtime_fixture(rank)
    value["fixture_scope"] = "SYNTHETIC_TEST_ONLY_CPU_CONTINUATION_RECEIPT_NOT_GPU"
    cycles = end-start
    rows = [inputs.row(5*c+s, slot) for c in range(start, end) for s in range(5) for slot in pairs[rank % 8]]
    if rank < 8:
        inventory = {"student/nograd/plain": sum(1+5*r.new_blocks for r in rows),
                     "student/grad/checkpoint": 6*cycles}
        for key, factor, tail in (("generator_rms_epilogue", 90, 30), ("generator_modulation", 61, 0),
                                  ("generator_gate", 60, 0)):
            receipt = value[key]
            receipt.update(scope_entries=len(rows), scope_exits=len(rows),
                           scope_forward_calls=[(1+5*r.new_blocks)*factor+tail for r in rows])
            receipt["calls"] = sum(receipt["scope_forward_calls"])
        value["generator_rms_epilogue"]["shape_histogram"] = {"1x2760x1536": value["generator_rms_epilogue"]["calls"]}
        value["generator_rollout_cast_cache"].update(scope_entries=len(rows), scope_exits=len(rows))
        value["output_fingerprints"] = [dict(operation_key=r.operation_key, finite=True) for r in rows]
    else:
        inventory = {"teacher/nograd/plain": 2*cycles, "critic/grad/checkpoint": 5*cycles}
        for key in ("checkpoint_calls", "original_fa_calls", "replay_fa_cache_reads", "expected_checkpoint_calls"):
            value["critic_attention_checkpoint"][key] = 150*cycles
        value["score_attention_backend"].update(invocation_count=5*cycles, backward_invocation_count=5*cycles)
        value["score_rmsnorm_kernel"].update(forward_calls=5*cycles, backward_calls=5*cycles)
    value["call_inventory"] = value["physical_call_inventory"] = inventory
    if rank >= 8:
        score_runtime_census(value)
    value["continuation"] = dict(schema_version=SCHEMA, start_cycle=start, completed_cycles=end,
        requested_stop_cycle=end, target_cycles=inputs.cycles, input_sha256=inputs.input_sha256,
        executed_cycles=cycles, source_reuse_across_cycles=False, numerical_status="REVIEW",
        physical_final_versions=physical_versions(rank, end))
    return value


def score_runtime_census(value):
    """Derive synthetic operator counts from the declared physical inventory."""
    inventory = value["physical_call_inventory"]
    teacher, critic = inventory["teacher/nograd/plain"], inventory["critic/grad/checkpoint"]
    value["score_residual_gate"].update(forward_calls=80*teacher+120*critic, backward_calls=60*critic)
    norms = 121*teacher+181*critic
    value["score_layernorm"].update(total_norm_calls=norms, selected_entries=norms,
                                  forward_calls=norms, backward_calls=critic)


@pytest.mark.parametrize("rank", [0, 8], ids=["Generator", "Score"])
@pytest.mark.parametrize("start", [0, 2], ids=["initial", "resumed"])
def test_real_continuation_receipt_composes_full_bundle_and_fresh_rows(stream, rank, start, record_property):
    from distillschedule.runtime.livewan_c51_bundle import runtime_receipt_errors, continuation_receipt_errors
    cursor = pool(stream, rank)
    value = continuation_runtime_fixture(cursor.inputs, cursor.pairs, rank=rank, start=start, end=start+2)
    assert runtime_receipt_errors(value, rank=rank, cycles=2) == []
    assert continuation_receipt_errors(value, rank=rank, inputs=cursor.inputs, pairs=cursor.pairs) == []
    for key, bad_value in (("input_sha256", "0"*64), ("start_cycle", True), ("executed_cycles", 1),
                           ("completed_cycles", start+1), ("requested_stop_cycle", start+1),
                           ("target_cycles", 100), ("source_reuse_across_cycles", True),
                           ("physical_final_versions", {})):
        broken = deepcopy(value); broken["continuation"][key] = bad_value
        assert continuation_receipt_errors(broken, rank=rank, inputs=cursor.inputs, pairs=cursor.pairs), key
    for key, bad_value in (("installed_plan_sha256", "invalid"), ("actual_latent_shapes", [[1, 16, 18, 46, 80]]),
                           ("planner_c51_bundle", {})):
        broken = deepcopy(value); broken[key] = bad_value
        assert continuation_receipt_errors(broken, rank=rank, inputs=cursor.inputs, pairs=cursor.pairs), key
    if rank < 8:
        for key in ("generator_rms_epilogue", "generator_modulation", "generator_gate"):
            broken = deepcopy(value)
            counts = broken[key]["scope_forward_calls"]
            # Reorder two individually legal counts, preserving their totals:
            # only the real fresh row-derived validator should reject this.
            other = next(i for i, count in enumerate(counts) if count != counts[0])
            counts[0], counts[other] = counts[other], counts[0]
            assert runtime_receipt_errors(broken, rank=rank, cycles=2) == []
            assert any("fresh row-derived scope" in e for e in continuation_receipt_errors(
                broken, rank=rank, inputs=cursor.inputs, pairs=cursor.pairs)), key
        for damage in ("reorder", "replay", "nonfinite", "missing"):
            broken = deepcopy(value); outputs = broken["output_fingerprints"]
            if damage == "reorder": outputs.reverse()
            elif damage == "replay": outputs[-1] = deepcopy(outputs[0])
            elif damage == "nonfinite": outputs[0]["finite"] = False
            else: outputs.pop()
            assert any("rollout output" in e for e in continuation_receipt_errors(
                broken, rank=rank, inputs=cursor.inputs, pairs=cursor.pairs)), damage
        for key in ("student/nograd/plain", "student/grad/checkpoint"):
            broken = deepcopy(value); broken["physical_call_inventory"][key] += 1
            assert any("Generator actual" in e for e in continuation_receipt_errors(
                broken, rank=rank, inputs=cursor.inputs, pairs=cursor.pairs)), key
    else:
        for key in ("teacher/nograd/plain", "critic/grad/checkpoint"):
            broken = deepcopy(value); broken["physical_call_inventory"][key] += 1
            score_runtime_census(broken)
            assert runtime_receipt_errors(broken, rank=rank, cycles=2) == []
            assert any("Score physical" in e for e in continuation_receipt_errors(
                broken, rank=rank, inputs=cursor.inputs, pairs=cursor.pairs)), key
    record_property("scope", value["fixture_scope"])


@pytest.mark.parametrize("request_change", ["inventory", "horizon", "seed", "unchanged"])
def test_actual_loader_reconciles_public_identity_before_continuation_or_models(
        stream, original_source, tmp_path, monkeypatch, request_change):
    """Two independently valid inventories; atomically replace A with B."""
    from contextlib import nullcontext
    from distillschedule.data.livewan_inputs import prepare_c51_inputs
    from distillschedule.runtime import livewan_crossscale_worker as worker
    from distillschedule.search import resources
    from distillschedule.training import distributed, livewan_c51_training as training, livewan_training as reference
    path = tmp_path/"prepared.json"
    path.write_text(json.dumps(stream[1]))
    admitted_inputs = training.input_metadata(path, cycles=4, seed=31415, source=stream[1]["source"])
    replacement = tmp_path/"replacement.json"
    if request_change == "inventory":
        data = tmp_path/"other-valid-inventory"; data.mkdir()
        for record in stream[1]["records"]:
            os.link(record["path"], data/Path(record["path"]).name)
        second = prepare_c51_inputs(data, stream[1]["prompts"]["path"], replacement,
            source_root=original_source, cycles=4, seed=31415, evidence_kind=stream[1]["evidence_kind"])
        assert second["input_sha256"] != admitted_inputs["input_sha256"]
        actual_loader = worker.InputLoader  # retain real source/record admission
        def swapped_assets(_):
            os.replace(replacement, path)
            return object()
    else:
        swapped_assets = lambda _: object()
        actual_loader = worker.InputLoader
    requested = dict(cycles=4, seed=31415, checkpoint_every=2)
    if request_change == "horizon": requested["cycles"] = 3
    elif request_change == "seed": requested["seed"] += 1
    config = SimpleNamespace(trainer=requested, data=dict(prepared=path))
    admitted = dict(inputs=admitted_inputs, assets=dict(native=dict(sources=dict(livewan=stream[1]["source"]))), capacity_policy={})
    allocations, checked = [], []
    original_checked = distributed.checked
    def exchange(fn, label):
        checked.append(label)
        return original_checked(fn, label)
    class ConstructionReached(RuntimeError):
        pass
    def construct(**kwargs):
        allocations.append("continuation")
        if request_change == "unchanged": raise ConstructionReached("reconciled before construction")
        pytest.fail("unreconciled actual loader reached Continuation/model allocation")
    monkeypatch.setattr(distributed, "world16_reference_process_group", lambda _: nullcontext())
    monkeypatch.setattr(distributed, "admit_run_controls", lambda _: None)
    monkeypatch.setattr(distributed, "checked", exchange)
    monkeypatch.setattr(reference, "_reserve_output", lambda _: None)
    monkeypatch.setattr(reference, "_bind_devices", lambda *a, **k: {})
    monkeypatch.setattr(reference, "_capacity", lambda *a, **k: {})
    monkeypatch.setattr(reference, "_admit_model_layout", lambda *a: pytest.fail("model layout reached"))
    monkeypatch.setattr(resources, "ResourceObserver", lambda *a, **k: object())
    monkeypatch.setattr(training, "admit_runtime", lambda _: {})
    monkeypatch.setattr(training.C51Assets, "load", swapped_assets)
    monkeypatch.setattr(training, "Continuation", construct)
    monkeypatch.setattr(worker, "InputLoader", actual_loader)
    error = ConstructionReached if request_change == "unchanged" else ValueError
    message = "reconciled before construction" if request_change == "unchanged" else "actual InputLoader identity/horizon/seed differs"
    with pytest.raises(error, match=message):
        training.run_training(config, admit=lambda: (None, dict(identity=dict(head="a"*40, dirty=False)),
            dict(output=tmp_path/"output"), admitted))
    assert allocations == (["continuation"] if request_change == "unchanged" else [])
    assert "C51 actual InputLoader immutable identity" in checked
    assert not list(tmp_path.rglob("_SUCCESS"))


@pytest.mark.parametrize("reject", [False, True], ids=["admit", "reject"])
def test_checkpoint_prepublication_guard_sees_live_cpu_snapshot_before_metadata(tmp_path, monkeypatch, reject):
    """Real pack/write/publication boundary, no memory measurement claim."""
    import weakref
    import torch
    from distillschedule.training import checkpoint, distributed
    model = torch.nn.Linear(3, 2)
    state = {"model": model}
    path = tmp_path/"checkpoint"
    references, labels = [], []
    original_pack, original_checked = checkpoint.pack_state, distributed.checked
    def packing(value):
        packed = original_pack(value)
        references.append(weakref.ref(packed["model"]["value"]["weight"]))
        return packed
    def exchange(fn, label):
        labels.append(label)
        return original_checked(fn, label)
    def guard():
        snapshot = references[-1]()
        assert snapshot is not None and snapshot.device.type == "cpu"
        assert snapshot.data_ptr() != model.weight.data_ptr() and torch.equal(snapshot, model.weight)
        staging, = tmp_path.glob(".checkpoint.tmp-*")
        assert (staging/"state.pt").is_file() and not path.exists()
        assert not (staging/"manifest.json").exists() and not (staging/"_SUCCESS").exists()
        if reject: raise ValueError("synthetic host staging budget/reserve exceeded")
    monkeypatch.setattr(checkpoint, "pack_state", packing)
    monkeypatch.setattr(distributed, "checked", exchange)
    kwargs = dict(identity=dict(scope="SYNTHETIC_TEST_ONLY_CPU_SNAPSHOT_GUARD"), cycle=1,
                  cursor=5, versions=dict(generator=1), pre_publication_guard=guard)
    if reject:
        with pytest.raises(ValueError, match="host staging budget/reserve"):
            checkpoint.save_checkpoint(path, state, **kwargs)
        assert not path.exists() and not list(tmp_path.rglob("_SUCCESS"))
    else:
        assert checkpoint.save_checkpoint(path, state, **kwargs) == path.resolve()
        assert (path/"_SUCCESS").is_file()
    assert labels.count("checkpoint pre-publication guard") == 1
    assert not torch.cuda.is_initialized()


def score_continuation_fixture(stream, original_source, original_lora, tmp_path, monkeypatch,
                               *, start=0, end=2, cadence=2, capacity_check=None):
    """SYNTHETIC_TEST_ONLY CPU orchestration with real named Critic state.

    Source/TeacherLatents/LoRA/Adam/PoolInputs/Progress and save/restore are real.
    Only CUDA synchronize, shadow, reducer, transport and peer metadata are
    stand-ins. Real full-bundle and continuation validators remain enabled.
    """
    import torch
    from distillschedule.runtime import livewan_crossscale_worker as worker
    from distillschedule.training import checkpoint, distributed, livewan_c51_training as training
    from distillschedule.training.livewan_c51_state import Progress, owner_state, physical_versions
    from test_training_livewan_state import update
    cursor = pool(stream, 8)
    identity = dict(schema_version=training.SCHEMA, horizon=cursor.inputs.cycles,
        input_sha256=cursor.inputs.input_sha256, scope="SYNTHETIC_TEST_ONLY_SCORE_CPU_ORCHESTRATION")
    header = None
    if start:
        previous = owner(original_lora, "critic")
        actual = physical_versions(8, 0)
        state = owner_state(previous.bundle, source_root=original_source, progress=Progress(cursor, actual))
        for cycle in range(start):
            for _ in range(5): update(previous); actual["critic"] += 1
            consume(cursor, cycle)
        resume = checkpoint.save_checkpoint(tmp_path/"resume", state, identity=identity, cycle=start,
            cursor=5*start, versions=dict(generator=start, critic=5*start, ema=0), policy=training.checkpoint_policy(identity))
        header = checkpoint.read_manifest(resume, identity=identity)
    current = owner(original_lora, "critic")
    config = SimpleNamespace(trainer=dict(cycles=cursor.inputs.cycles, stop_after_cycle=end, checkpoint_every=cadence,
        **(dict(resume_from=resume) if start else {})))
    timeline = []
    check = capacity_check or (lambda: timeline.append(("capacity",)))
    assets = dict(native=dict(sources=dict(livewan=dict(root=str(original_source)))))
    cont = training.Continuation(config=config, admitted=dict(assets=assets),
        identity=identity, output=tmp_path/"output", bootstrap=None, loader=cursor.loader, header=header, capacity_check=check)
    from distillschedule.runtime.livewan_grad_reducer import GradReadyAsyncGroupReducer
    # Constructorless CPU stand-in for only the drained reducer boundary. No
    # CUDA stream, hook or all-reduce is claimed; native zero_grad and the exact
    # owner/pending/view checks stay real instead of disabling their validator.
    reducer = object.__new__(GradReadyAsyncGroupReducer)
    reducer._active_backward = reducer._pending_sync = False
    reducer._parameter_view = {id(p): p.grad for p in current.parameters}
    reducer._buckets = [SimpleNamespace(flat=current.flat, parameters=current.parameters,
                                       views=[p.grad for p in current.parameters])]
    native_clear = reducer.zero_grad
    def clear():
        timeline.append(("clear",))
        native_clear()
    reducer.zero_grad = clear
    current.bundle.critic_gradient_reducer = reducer
    actual = physical_versions(8, 0)
    loader = cont.bind(torch=torch, rank=8, bundle=current.bundle, versions=actual, pairs=cursor.pairs)
    transport = SimpleNamespace(drain_through=lambda cycle: timeline.append(("drain", cycle)))
    shadow = SimpleNamespace(pending_source_work=None, pending_source_version=None, events=[],
                             begin_source=lambda **kw: timeline.append(("prime", kw["version"])))
    evidence, inventory = worker.RunEvidence(), worker.CallInventory()
    cont.mechanism_receipt = lambda: continuation_runtime_fixture(
        cont.inputs, cursor.pairs, rank=8, start=start, end=loader.cycle)
    original_gather = distributed.gather
    def gather(value):
        if type(value) is dict and set(value) == {"rank", "cycle", "versions", "slots", "rows", "quiescent"}:
            # Explicit synthetic peer plane; local rank8 rows are actual loads.
            rows = []
            for rank in range(16):
                view = object.__new__(type(loader)); view.__dict__.update(loader.__dict__)
                view.rank, view.slots = rank, cursor.pairs[rank % 8]
                rows.append(dict(rank=rank, cycle=value["cycle"], versions=physical_versions(rank, value["cycle"]),
                    slots=list(view.slots), rows=view.expected(value["cycle"]-1), quiescent=True))
            assert rows[8] == value
            return rows
        return original_gather(value)
    monkeypatch.setattr(distributed, "gather", gather)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: timeline.append(("synchronize",)))
    streamers = [SimpleNamespace(ctx=object(), ctx_lens=object()) for _ in range(2)]
    def step():
        cycle = loader.cycle
        for s in range(5):
            for slot in loader.slots:
                loader.context_from_trace(cont.inputs.row(5*cycle+s, slot), device="cpu")
            update(current)
            actual["critic"] += 1
        inventory.counts["critic/grad/checkpoint"] = inventory.counts.get("critic/grad/checkpoint", 0)+5
        inventory.counts["teacher/nograd/plain"] = inventory.counts.get("teacher/nograd/plain", 0)+2
        evidence.physical_critic_backward_calls += 5
        evidence.loss_fingerprints.extend(dict(finite=True, mean=1., norm=1., max_abs=1.) for _ in range(10))
        evidence.optimizer_diagnostics.extend(dict(clip_input_norm=1.) for _ in range(5))
    def complete():
        cont.cycle_complete(torch=torch, rank=8, cycle=loader.cycle+1, versions=actual, bundle=current.bundle,
            transport=transport, shadow=shadow, evidence=evidence, inventory=inventory, streamers=streamers, device="cpu")
    return SimpleNamespace(cont=cont, current=current, loader=loader, actual=actual, evidence=evidence,
        inventory=inventory, transport=transport, shadow=shadow, timeline=timeline, step=step, complete=complete, streamers=streamers)


@pytest.mark.parametrize("start", [0, 2], ids=["initial", "resumed"])
def test_score_actual_owner_publication_drain_and_nonpublication_overlap(
        stream, original_source, original_lora, tmp_path, monkeypatch, start):
    import torch
    from distillschedule.training.checkpoint import read_checkpoint
    value = score_continuation_fixture(stream, original_source, original_lora, tmp_path, monkeypatch, start=start, end=start+2)
    gradient_ids = [id(p.grad) for p in value.current.parameters]
    if start:
        assert value.loader.cycle == start and value.actual["critic"] == 5*start
        assert {slot["step"].item() for slot in value.current.optimizer.state.values()} == {5*start}
        assert value.cont.prime(torch=torch, rank=8, trace=value.cont.inputs, bundle=value.current.bundle,
            streamers=value.streamers, loader=value.loader, device="cpu", transport=value.transport,
            shadow=value.shadow, evidence=value.evidence, elision=True) is None
        assert value.timeline == [("prime", 5*start)]
    pending = object()
    value.shadow.pending_source_work, value.shadow.pending_source_version = pending, 5*(start+1)
    value.step(); value.complete()
    assert value.shadow.pending_source_work is pending
    assert value.shadow.pending_source_version == 5*(start+1)
    assert not any(row[0] in ("drain", "clear", "synchronize") for row in value.timeline)
    assert value.cont.progress.quiescent is False
    assert not list((tmp_path/"output").rglob("_SUCCESS"))
    value.shadow.pending_source_work = value.shadow.pending_source_version = None
    value.step(); value.complete()
    assert value.timeline.index(("drain", start+2)) < value.timeline.index(("clear",)) < value.timeline.index(("synchronize",))
    assert [id(p.grad) for p in value.current.parameters] == gradient_ids
    assert value.cont.progress.quiescent is True
    path = tmp_path/"output/checkpoints"/f"cycle-{start+2:06d}"
    payload = read_checkpoint(path, identity=value.cont.identity)
    assert payload["cycle"] == start+2 and payload["versions"] == dict(generator=start+2, critic=5*(start+2), ema=0)
    assert payload["state"]["pool.progress"]["value"]["data"]["cycle"] == start+2
    assert all(s.ctx is s.ctx_lens is None for s in value.streamers)
    receipt = json.loads((tmp_path/"output/continuation/rank-00008.json").read_text())
    assert receipt["physical_versions"] == dict(generator=0, critic=5*(start+2), ema=0)
    assert receipt["native_cuda_numerical_qualification"] == "NOT_RUN" and not receipt["claimable_speedup"]


@pytest.mark.parametrize("damage", ["missing_reducer", "pending_work", "pending_version", "critic_calls", "teacher_calls",
                                    "backward_calls", "loss_count", "norm_count", "nonfinite_loss", "nonfinite_norm", "receipt_source"])
def test_score_invalid_boundary_never_publishes_checkpoint(
        stream, original_source, original_lora, tmp_path, monkeypatch, damage):
    value = score_continuation_fixture(stream, original_source, original_lora, tmp_path, monkeypatch, end=1, cadence=1)
    value.step()
    if damage == "missing_reducer": value.current.bundle.critic_gradient_reducer = None
    elif damage == "pending_work": value.shadow.pending_source_work = object()
    elif damage == "pending_version": value.shadow.pending_source_version = 5
    elif damage == "critic_calls": value.inventory.counts["critic/grad/checkpoint"] -= 1
    elif damage == "teacher_calls": value.inventory.counts["teacher/nograd/plain"] -= 1
    elif damage == "backward_calls": value.evidence.physical_critic_backward_calls -= 1
    elif damage == "loss_count": value.evidence.loss_fingerprints.pop()
    elif damage == "norm_count": value.evidence.optimizer_diagnostics.pop()
    elif damage == "nonfinite_loss": value.evidence.loss_fingerprints[0]["mean"] = float("nan")
    elif damage == "nonfinite_norm": value.evidence.optimizer_diagnostics[0]["clip_input_norm"] = float("inf")
    else:
        receipt = value.cont.mechanism_receipt
        def corrupt_source():
            result = receipt(); result["planner_c51_bundle"]["source_template_sha256"] = "0"*64
            return result
        value.cont.mechanism_receipt = corrupt_source
    message = ("async reducer" if damage == "missing_reducer" else "unmatched critic-shadow" if damage.startswith("pending_")
               else "backward/Teacher census" if damage.endswith("calls") else "full-bundle runtime counters/source" if damage == "receipt_source"
               else "loss/clip evidence")
    with pytest.raises(ValueError, match=message): value.complete()
    assert not list((tmp_path/"output").rglob("_SUCCESS"))
    assert not (tmp_path/"output/continuation/rank-00008.json").exists()


def test_c51_staging_capacity_breach_prevents_checkpoint_and_progress_publication(
        stream, original_source, original_lora, tmp_path, monkeypatch):
    import weakref
    from distillschedule.training import checkpoint
    references, checks = [], []
    original_pack = checkpoint.pack_state
    def packing(state):
        result = original_pack(state)
        tensor = next(iter(result["critic.weights"]["value"]["tensors"].values()))
        references.append(weakref.ref(tensor))
        return result
    def check():
        checks.append(bool(references))
        if references:
            assert references[-1]() is not None  # CPU copies still retained
            staging, = (tmp_path/"output/checkpoints").glob(".cycle-000001.tmp-*")
            assert (staging/"state.pt").is_file() and not (staging/"_SUCCESS").exists()
            raise ValueError("synthetic C51 host snapshot reserve exceeded")
    value = score_continuation_fixture(stream, original_source, original_lora, tmp_path, monkeypatch,
                                       end=1, cadence=1, capacity_check=check)
    monkeypatch.setattr(checkpoint, "pack_state", packing)
    value.step()
    with pytest.raises(ValueError, match="host snapshot reserve"): value.complete()
    assert checks == [False, True]
    assert not list((tmp_path/"output").rglob("_SUCCESS"))
    assert not (tmp_path/"output/continuation/rank-00008.json").exists()


@pytest.mark.parametrize("end", [1, 4], ids=["stop", "completed"])
def test_c51_final_receipt_rechecks_capacity_without_checkpoint_cadence(
        stream, original_source, original_lora, tmp_path, monkeypatch, end):
    checks = []
    def check():
        checks.append(True)
        if len(checks) == end+1: raise ValueError("synthetic final receipt host peak exceeded")
    value = score_continuation_fixture(stream, original_source, original_lora, tmp_path, monkeypatch,
                                       end=end, cadence=0, capacity_check=check)
    for _ in range(end-1):
        value.step(); value.complete()
    value.step()
    with pytest.raises(ValueError, match="final receipt host peak"): value.complete()
    assert len(checks) == end+1 and not list((tmp_path/"output").rglob("_SUCCESS"))
    assert not (tmp_path/"output/continuation/rank-00008.json").exists()


@pytest.fixture
def c51_hf_assets(tmp_path, monkeypatch):
    """SYNTHETIC_TEST_ONLY: tiny files, NOT actual HF/native/GPU qualification."""
    from test_training_livewan_reference import asset_document
    from distillschedule.training import livewan_reference_assets as reference, livewan_c51_assets as assets
    from distillschedule.workloads.best_known.registry import file_sha256
    sources = {name: dict(root=str(tmp_path/name), head="0"*40, files={"LICENSE": "0"*64})
               for name in ("livewan", "wan21")}
    monkeypatch.setattr(reference, "source_identity", lambda root, *, project: deepcopy(sources[project]))
    native = asset_document(tmp_path, sources)
    native["runtime"]["dependency_versions"].pop("flash-attn")
    extension = tmp_path/"SYNTHETIC_TEST_ONLY_torch_C.so"
    extension.write_bytes(b"SYNTHETIC_TEST_ONLY_NOT_A_NATIVE_EXTENSION")
    files, kernels = {}, {}
    names = {"fa2": ("metadata.json", "__init__.py", "_ops.py", "bert_padding.py", "flash_attn_interface.py",
                       "flash_attn2/__init__.py", "_flash_attn2_cuda_a511a4c.abi3.so"),
             "fa3": ("metadata.json", "__init__.py", "_ops.py", "flash_attn_interface.py",
                       "flash_attn_config.py", "flash_attn3/__init__.py", "_flash_attn3_cuda_373a45c.abi3.so")}
    for kind, paths in names.items():
        root = tmp_path/("SYNTHETIC_TEST_ONLY_"+kind); root.mkdir()
        for name in paths:
            path = root/name; path.parent.mkdir(exist_ok=True)
            path.write_text("# SYNTHETIC_TEST_ONLY " + kind + ":" + name + "\n")
        files[kind] = {name: file_sha256(root/name) for name in paths}
        kernels[kind] = dict(root=str(root), files=deepcopy(files[kind]))
    monkeypatch.setattr(assets, "kernel_files", lambda **kwargs: deepcopy(files))
    try:
        from distillschedule.runtime import livewan_c51_provider as provider
    except ImportError:
        pass  # The red test precedes the new production module.
    else:
        monkeypatch.setattr(provider, "provider_files", lambda: deepcopy(files))
    overlay = dict(root=str(tmp_path/"SYNTHETIC_TEST_ONLY_overlay"), files={"wan/modules/attention.py": "0"*64})
    monkeypatch.setattr(assets, "overlay_inventory", lambda native, root: deepcopy(overlay))
    value = dict(schema_version="distillschedule.livewan_c51_assets.v2", native=native, kernels=kernels,
        wan21_overlay=overlay, attention_provider=dict(schema_version="distillschedule.livewan_c51_hf_provider.v1",
            abi=dict(torch="2.10.0+cu128", torch_cuda="12.8", cxx11_abi=True, system="Linux", machine="x86_64",
                     torch_extension_sha256=file_sha256(extension))))
    return SimpleNamespace(value=value, extension=extension, files=files)


def test_c51_hf_v2_is_explicit_and_keeps_reference_and_c51_v1_strict(c51_hf_assets):
    from distillschedule.training import livewan_c51_assets as assets, livewan_reference_assets as reference
    value = c51_hf_assets.value
    assert assets.asset_config(value) == value
    with pytest.raises(ValueError, match="complete original dependency"):
        reference.asset_config(value["native"])
    legacy = deepcopy(value); legacy["schema_version"] = assets.SCHEMA; legacy.pop("attention_provider")
    with pytest.raises(ValueError, match="complete original dependency"):
        assets.asset_config(legacy)
    missing = deepcopy(value); missing.pop("attention_provider")
    with pytest.raises(ValueError):
        assets.asset_config(missing)


@pytest.mark.parametrize("damage", ["provider", "abi", "hash", "missing_file", "dependency"])
def test_c51_hf_v2_rejects_missing_changed_or_unbound_assets(c51_hf_assets, damage):
    from distillschedule.training.livewan_c51_assets import asset_config
    value = deepcopy(c51_hf_assets.value)
    if damage == "provider": value["attention_provider"]["schema_version"] = "unregistered"
    elif damage == "abi": value["attention_provider"]["abi"]["torch_cuda"] = "13.0"
    elif damage == "hash": value["kernels"]["fa2"]["files"]["bert_padding.py"] = "0"*64
    elif damage == "missing_file": (Path(value["kernels"]["fa3"]["root"])/"_ops.py").unlink()
    else: value["native"]["runtime"]["dependency_versions"].pop("numpy")
    with pytest.raises(ValueError):
        asset_config(value)


def test_c51_hf_owner_and_native_plan_bind_the_complete_provider(c51_hf_assets):
    from distillschedule.training.livewan_c51_assets import C51Assets
    from distillschedule.training.livewan_c51_training import make_plan
    from distillschedule.workloads.best_known.registry import fingerprint
    value = c51_hf_assets.value
    owner = C51Assets.load(value)
    assert owner.sha256 == fingerprint(value)
    marker = owner.value["attention_provider"]
    assert marker["provider"] == value["attention_provider"] and marker["kernels"] == value["kernels"]
    assert owner.native_plan()["evidence"]["attention_provider"] == marker
    assert make_plan(value, "/SYNTHETIC_TEST_ONLY/output")["evidence"]["attention_provider"] == marker
    owner.unchanged()
    (Path(value["kernels"]["fa2"]["root"])/"bert_padding.py").write_text("# SYNTHETIC_TEST_ONLY changed\n")
    with pytest.raises(ValueError):
        owner.unchanged()


@pytest.fixture
def c51_hf_runtime(c51_hf_assets, monkeypatch):
    """SYNTHETIC_TEST_ONLY native loader/mapping and Torch ABI stand-ins.

    Real production asset/runtime/bytecode/owner/checkpoint code is exercised;
    these tiny Python files are NOT the actual HF providers or extensions.
    """
    import importlib.metadata
    import importlib.util
    import types
    import torch
    from distillschedule.runtime import livewan_c51_provider as provider
    from distillschedule.runtime import livewan_attention_backend as fa2, livewan_score_fa3 as fa3
    from distillschedule.training import livewan_c51_assets as assets
    from distillschedule.workloads.best_known.registry import file_sha256
    value = c51_hf_assets.value
    modules, loads = {}, []
    for kind in ("fa2", "fa3"):
        root = Path(value["kernels"][kind]["root"])
        text = '''# SYNTHETIC_TEST_ONLY; NOT an HF/native implementation.
import torch
calls = []
class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q): return q
    @staticmethod
    def backward(ctx, gradient): return gradient
class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q): return q
    @staticmethod
    def backward(ctx, gradient): return gradient
def flash_attn_func(q, k=None, v=None):
    calls.append("dense")
    return q
def flash_attn_varlen_func(q, k=None, v=None):
    calls.append("varlen")
    return q
'''
        source = root/"flash_attn_interface.py"; source.write_text(text)
        sha = file_sha256(source)
        value["kernels"][kind]["files"][source.name] = c51_hf_assets.files[kind][source.name] = sha
        name = "SYNTHETIC_TEST_ONLY_"+kind+".flash_attn_interface"
        spec = importlib.util.spec_from_file_location(name, source)
        interface = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, interface); spec.loader.exec_module(interface)
        package = types.ModuleType("flash_attn"+kind[-1])
        package.__file__ = str(root/(package.__name__+"/__init__.py"))
        package.flash_attn_func, package.flash_attn_varlen_func = interface.flash_attn_func, interface.flash_attn_varlen_func
        monkeypatch.setitem(sys.modules, package.__name__, package)
        modules[kind] = package
    def load(kind):
        loads.append(kind)
        return modules[kind], Path(value["kernels"][kind]["root"])
    monkeypatch.setattr(fa2, "_load_flash_attn2", lambda: load("fa2"))
    monkeypatch.setattr(fa3, "load_flash_attn3", lambda: load("fa3"))
    monkeypatch.setattr(provider, "_extension_identity", lambda kind, root: dict(
        path=str(root/(provider.EXTENSIONS[kind]+".abi3.so")),
        sha256=c51_hf_assets.files[kind][provider.EXTENSIONS[kind]+".abi3.so"],
        evidence_scope="SYNTHETIC_TEST_ONLY_NOT_A_NATIVE_EXTENSION"))
    monkeypatch.setattr(provider, "_dispatcher_identity", lambda kind, module, torch: dict(
        evidence_scope="SYNTHETIC_TEST_ONLY_NOT_REGISTERED_NATIVE_DISPATCHERS"))
    facade = SimpleNamespace(__version__="2.10.0+cu128", version=SimpleNamespace(cuda="12.8"),
        _C=SimpleNamespace(__file__=str(c51_hf_assets.extension), _GLIBCXX_USE_CXX11_ABI=True), autograd=torch.autograd,
        get_default_dtype=lambda: "torch.float32", get_float32_matmul_precision=lambda: "highest",
        are_deterministic_algorithms_enabled=lambda: False,
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(allow_tf32=True, benchmark=False)))
    monkeypatch.setitem(sys.modules, "torch", facade)
    versions = value["native"]["runtime"]["dependency_versions"]
    def version(name):
        if name not in versions: raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]
    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setenv(assets.KERNEL_ENV["fa2"], value["kernels"]["fa2"]["root"])
    monkeypatch.setenv(assets.KERNEL_ENV["fa3"], value["kernels"]["fa3"]["root"])
    # Provider runtime installs only this explicit alias; restore it per test.
    monkeypatch.setitem(sys.modules, "flash_attn", None)
    monkeypatch.delitem(sys.modules, "flash_attn")
    return SimpleNamespace(value=value, modules=modules, loads=loads, facade=facade, version=version, real_torch=torch)


def test_c51_hf_synthetic_runtime_chain_reaches_both_role_callables(c51_hf_runtime):
    from distillschedule.training.livewan_c51_assets import admit_runtime
    runtime = admit_runtime(c51_hf_runtime.value)
    provider = runtime["attention_provider"]
    assert c51_hf_runtime.loads == ["fa2", "fa3"]
    assert set(runtime["dependencies"]) == set(c51_hf_runtime.value["native"]["runtime"]["dependency_versions"])
    assert "flash-attn" not in runtime["dependencies"]
    assert set(provider["components"]) == {"fa2", "fa3"}
    assert set(provider["role_bindings"]) == {"generator", "score"}
    assert sys.modules["flash_attn"] is c51_hf_runtime.modules["fa2"]
    for role, kind in (("generator", "fa2"), ("score", "fa3")):
        assert kind in provider["role_bindings"][role]
        assert c51_hf_runtime.modules[kind].flash_attn_func("SYNTHETIC_TEST_ONLY_"+role) == "SYNTHETIC_TEST_ONLY_"+role
        assert provider["components"][kind]["functions"]["flash_attn_func"]["original_public_and_autograd_bytecode_attested"]
    assert admit_runtime(c51_hf_runtime.value) == runtime


@pytest.mark.parametrize("change", ["added", "changed", "removed"])
def test_c51_environment_drift_reports_changed_control_before_provider_load(c51_hf_runtime, monkeypatch, change):
    from distillschedule.training.livewan_c51_assets import admit_runtime
    key = "CUBLAS_WORKSPACE_CONFIG"
    expected = c51_hf_runtime.value["native"]["runtime"]["environment"]
    if change == "added":
        expected.pop(key, None)
        monkeypatch.setenv(key, ":4096:8")
    elif change == "changed":
        expected[key] = ":16:8"
        monkeypatch.setenv(key, ":4096:8")
    else:
        expected[key] = ":16:8"
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="effective C51 precision/environment controls changed.*" + key):
        admit_runtime(c51_hf_runtime.value)
    assert c51_hf_runtime.loads == []


@pytest.mark.parametrize("name", ["flash_attn", "flash_attn2", "flash_attn3", "flash_attn_interface", "_flash_attn2_cuda_a511a4c"])
def test_c51_hf_foreign_preloads_fail_before_native_load(c51_hf_runtime, monkeypatch, name):
    from distillschedule.training.livewan_c51_assets import admit_runtime
    foreign = SimpleNamespace(__file__=__file__)
    monkeypatch.setitem(sys.modules, name, foreign)
    with pytest.raises(ValueError, match="origin|foreign"):
        admit_runtime(c51_hf_runtime.value)
    assert c51_hf_runtime.loads == []


@pytest.mark.parametrize("damage", ["torch_cuda", "cxx11", "torch_extension", "callable", "autograd", "default"])
def test_c51_hf_observed_abi_and_loaded_code_fail_closed(c51_hf_runtime, damage):
    from distillschedule.training.livewan_c51_assets import admit_runtime
    value = c51_hf_runtime
    if damage == "torch_cuda": value.facade.version.cuda = "13.0"
    elif damage == "cxx11": value.facade._C._GLIBCXX_USE_CXX11_ABI = False
    elif damage == "torch_extension": Path(value.facade._C.__file__).write_bytes(b"SYNTHETIC_TEST_ONLY_CHANGED")
    else:
        fn = value.modules["fa2"].flash_attn_func
        if damage == "callable": value.modules["fa2"].flash_attn_func = lambda *args: args[0]
        elif damage == "default": fn.__defaults__ = (1, None)
        else: fn.__globals__["FlashAttnFunc"].forward = staticmethod(lambda ctx, q: q)
    with pytest.raises(ValueError, match="ABI|FUNCTION|SOURCE|DEFAULT|autograd"):
        admit_runtime(value.value)


def test_c51_hf_unchanged_rejects_a_late_foreign_preload(c51_hf_assets, monkeypatch):
    from distillschedule.training.livewan_c51_assets import C51Assets
    owner = C51Assets.load(c51_hf_assets.value)
    monkeypatch.setitem(sys.modules, "flash_attn2", SimpleNamespace(__file__=__file__))
    with pytest.raises(ValueError, match="origin"):
        owner.unchanged()


def test_c51_hf_original_v1_runtime_still_requires_distribution(c51_hf_runtime):
    import importlib.metadata
    from distillschedule.training import livewan_c51_assets as assets
    value = deepcopy(c51_hf_runtime.value)
    value["schema_version"] = assets.SCHEMA; value.pop("attention_provider")
    value["native"]["runtime"]["dependency_versions"]["flash-attn"] = "MUST_BE_REAL_DISTRIBUTION_NOT_PROVIDER_VERSION"
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        assets.admit_runtime(value)
    assert c51_hf_runtime.loads == []


def test_c51_hf_continuation_consumes_owner_provider_and_observed_identity(c51_hf_runtime, tmp_path, monkeypatch):
    from distillschedule.training import livewan_c51_training as training, livewan_c51_assets as assets
    admitted = dict(assets=c51_hf_runtime.value, inputs=dict(input_sha256="1"*64), capacity_policy={})
    config = SimpleNamespace(trainer=dict(cycles=4, seed=31, checkpoint_every=1))
    with pytest.raises(ValueError, match="observed runtime"):
        training.training_identity("a"*40, admitted, config)
    runtime = assets.admit_runtime(admitted["assets"])
    identity = training.training_identity("a"*40, admitted, config, runtime=runtime)
    assert identity["attention_provider"] == runtime["attention_provider"]
    monkeypatch.setattr(sys, "path", list(sys.path))
    owner = assets.C51Assets.load(admitted["assets"])
    continuation = training.Continuation(config=config, admitted=admitted, identity=identity, output=tmp_path,
        bootstrap=owner, loader=SimpleNamespace(inputs=SimpleNamespace(cycles=4)), header=None, capacity_check=lambda: None)
    receipt = continuation.configure_sources()
    assert receipt["attention_provider"] == identity["attention_provider"]
    continuation.validate_constructed()
    continuation.identity = deepcopy(identity)
    continuation.identity["attention_provider"]["role_bindings"]["score"] = "foreign"
    with pytest.raises(ValueError, match="constructed provider/checkpoint"):
        continuation.validate_constructed()


def test_c51_hf_provider_observation_is_in_real_checkpoint_restore_identity(c51_hf_assets, tmp_path):
    """Real one-CPU checkpoint, synthetic provider receipt; NOT native world16."""
    import torch
    from distillschedule.training import checkpoint, livewan_c51_training as training
    from distillschedule.training.livewan_c51_assets import provider_binding
    admitted = dict(assets=c51_hf_assets.value, inputs=dict(input_sha256="1"*64), capacity_policy={})
    config = SimpleNamespace(trainer=dict(cycles=4, seed=31, checkpoint_every=1))
    observed = dict(**provider_binding(admitted["assets"]), components=dict(evidence_scope="SYNTHETIC_TEST_ONLY"))
    identity = training.training_identity("a"*40, admitted, config, runtime=dict(attention_provider=observed))
    state = dict(tiny=torch.nn.Linear(1, 1))
    saved = checkpoint.save_checkpoint(tmp_path/"SYNTHETIC_TEST_ONLY-checkpoint", state, identity=identity,
        cycle=1, cursor=5, versions=dict(generator=1, critic=5, ema=0), policy=training.checkpoint_policy(identity))
    assert checkpoint.read_manifest(saved, identity=identity)["identity"]["attention_provider"] == observed
    restored = dict(tiny=torch.nn.Linear(1, 1))
    checkpoint.restore_checkpoint(saved, restored, identity=identity)
    assert torch.equal(state["tiny"].weight, restored["tiny"].weight)
    changed = deepcopy(identity)
    changed["attention_provider"]["provider"]["abi"]["torch_extension_sha256"] = "f"*64
    with pytest.raises(ValueError, match="identity changed"):
        checkpoint.read_manifest(saved, identity=changed)


def test_c51_hf_wrapper_process_salted_names_are_not_checkpoint_identity(c51_hf_runtime, monkeypatch):
    """Same attested bytes, different process-local HF wrapper module names."""
    import importlib.util
    from distillschedule.training.livewan_c51_assets import admit_runtime
    baseline = admit_runtime(c51_hf_runtime.value)
    for kind, package in c51_hf_runtime.modules.items():
        source = Path(c51_hf_runtime.value["kernels"][kind]["root"])/"flash_attn_interface.py"
        # Import the identical tiny file under the other synthetic namespace.
        # The production wrapper derives this namespace from a salted hash.
        name = "SYNTHETIC_TEST_ONLY_other_process_"+kind+".flash_attn_interface"
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        # Use the already real Torch import, not the synthetic ABI facade.
        with monkeypatch.context() as context:
            context.setitem(sys.modules, "torch", c51_hf_runtime.real_torch)
            spec.loader.exec_module(module)
        package.flash_attn_func = module.flash_attn_func
        package.flash_attn_varlen_func = module.flash_attn_varlen_func
    assert admit_runtime(c51_hf_runtime.value) == baseline


def test_c51_hf_native_extension_cannot_be_attested_from_a_filename_or_spec(c51_hf_assets, monkeypatch):
    """SYNTHETIC_TEST_ONLY fake .so cannot pass real loader/map attestation."""
    import importlib.machinery
    from distillschedule.runtime import livewan_c51_provider as provider
    root = Path(c51_hf_assets.value["kernels"]["fa2"]["root"])
    name = provider.EXTENSIONS["fa2"]
    path = root/(name+".abi3.so")
    module = SimpleNamespace(__file__=str(path), __spec__=None)
    monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(ValueError, match="extension loader"):
        provider._extension_identity("fa2", root)
    module.__spec__ = SimpleNamespace(loader=importlib.machinery.ExtensionFileLoader(name, str(path)))
    with pytest.raises(ValueError, match="unmapped"):
        provider._extension_identity("fa2", root)


def test_c51_hf_dispatcher_cannot_be_attested_from_a_python_callable(c51_hf_assets):
    import torch
    from distillschedule.runtime import livewan_c51_provider as provider
    module = SimpleNamespace(flash_attn_func=lambda q: q)
    with pytest.raises(ValueError, match="actual registered native CustomOpDefs"):
        provider._dispatcher_identity("fa2", module, torch)


def test_c51_hf_default_kernel_contract_remains_the_v1_three_and_five_files():
    from distillschedule.training.livewan_c51_assets import kernel_files, SCHEMA
    from distillschedule.runtime.livewan_c51_provider import provider_files
    assert SCHEMA == "distillschedule.livewan_c51_assets.v1"
    assert {name: len(files) for name, files in kernel_files().items()} == dict(fa2=3, fa3=5)
    assert {name: len(files) for name, files in provider_files().items()} == dict(fa2=7, fa3=7)


def test_c51_hf_v1_owner_value_plan_and_digest_are_unchanged(c51_hf_assets):
    from distillschedule.training.livewan_c51_assets import C51Assets, SCHEMA
    from distillschedule.training.livewan_reference_assets import ReferenceAssets
    value = deepcopy(c51_hf_assets.value)
    value["schema_version"] = SCHEMA; value.pop("attention_provider")
    value["native"]["runtime"]["dependency_versions"]["flash-attn"] = "SYNTHETIC_TEST_ONLY"
    legacy = C51Assets.load(value)
    reference = ReferenceAssets.load(value["native"])
    assert legacy.value == reference.value and legacy.native_plan() == reference.native_plan()
    assert legacy.sha256 == reference.sha256


def test_c51_hf_all_eight_real_distribution_versions_stay_mandatory(c51_hf_runtime, monkeypatch):
    import importlib.metadata
    from distillschedule.training.livewan_c51_assets import admit_runtime
    def changed(name):
        return "SYNTHETIC_TEST_ONLY_WRONG_VERSION" if name == "numpy" else c51_hf_runtime.version(name)
    monkeypatch.setattr(importlib.metadata, "version", changed)
    with pytest.raises(ValueError, match="dependency changed: numpy"):
        admit_runtime(c51_hf_runtime.value)
    assert c51_hf_runtime.loads == []


def test_c51_hf_metadata_provider_import_is_torch_free():
    script = '''import sys
from distillschedule.runtime.livewan_c51_provider import provider_files
from distillschedule.training.livewan_c51_assets import HF_SCHEMA, SCHEMA
assert {name: len(files) for name, files in provider_files().items()} == {"fa2": 7, "fa3": 7}
assert HF_SCHEMA != SCHEMA
assert "torch" not in sys.modules
assert not any(name.startswith(("flash_attn", "wan.", "wanstreamer.")) for name in sys.modules)
'''
    subprocess.run([sys.executable, "-B", "-c", script], check=True)
