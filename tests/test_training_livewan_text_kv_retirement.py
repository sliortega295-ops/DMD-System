"""Public-only C51 dead-text-payload lifecycle; CPU/source, not CUDA capacity."""
from copy import deepcopy
import gc
import inspect
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import weakref

import pytest


PRESET = "self_forcing.c51_stream_retire_textkv.v1"
CONTROL = "self_forcing.c51_stream.v1"


def test_retirement_requires_distinct_explicit_preset_and_preserves_control():
    from distillschedule.training.native_presets import describe_preset, preset_implementation
    from distillschedule.training.livewan_c51_training import descriptor

    control = describe_preset("W4", CONTROL)
    selected = describe_preset("W4", PRESET)
    assert selected["id"] == PRESET and selected["continuous_training"] is True
    assert selected["gpu_execution"] == "NOT_RUN"
    assert selected["text_kv_payload_retirement"]["boundary"] == "prepare_rollout_after_nograd_scope"
    assert "text_kv_payload_retirement" not in control
    assert "text_kv_payload_retirement" not in descriptor()
    assert preset_implementation("W4", PRESET) == preset_implementation("W4", CONTROL)
    with pytest.raises(ValueError, match="unknown or mismatched"):
        describe_preset("W4", PRESET.replace("v1", "v2"))


def config(preset=PRESET):
    return SimpleNamespace(execution=dict(preset=preset), trainer=dict(cycles=4, seed=31415, checkpoint_every=2))


def admitted():
    return dict(assets=dict(native=dict(models={role: dict(root="/explicit/"+role)
                for role in ("student", "critic", "teacher")})),
                inputs=dict(input_sha256="a"*64), capacity_policy=dict(unchanged_explicit_reserve=512*1024**2))


def continuation(preset=PRESET):
    from distillschedule.training import livewan_c51_training as training
    conf, admission = config(preset), admitted()
    identity = training.training_identity("a"*40, admission, conf)
    value = training.Continuation(config=conf, admitted=admission, identity=identity, output="/unused",
        bootstrap=None, loader=SimpleNamespace(inputs=SimpleNamespace(cycles=4)), header=None, capacity_check=lambda: None)
    return value, training.make_plan(admission["assets"], "/unused", identity=identity, stop=4)


def test_policy_is_exact_checkpoint_identity_and_same_frozen_c51_lowering(tmp_path):
    from distillschedule.training import livewan_c51_training as training
    from distillschedule.runtime.livewan_c51_bundle import source_template, TEMPLATE_SHA256, validate_plan
    from distillschedule.workloads.best_known.registry import fingerprint
    from distillschedule.training.checkpoint import save_checkpoint, read_manifest
    import torch

    old, old_plan = continuation(CONTROL)
    new, new_plan = continuation()
    policy = new.identity["text_kv_payload_retirement"]
    assert "text_kv_payload_retirement" not in old.identity
    assert new_plan["metadata"] == old_plan["metadata"]
    assert new_plan["evidence"]["text_kv_payload_retirement"] == policy
    assert validate_plan(new_plan, mode="pipeline")
    assert fingerprint(source_template()) == TEMPLATE_SHA256
    new.validate_text_kv_plan(new_plan)
    for bad in (old_plan, {**new_plan, "evidence": {**new_plan["evidence"], "text_kv_payload_retirement": {**policy, "preset": CONTROL}}}):
        with pytest.raises(ValueError, match="lowered plan/identity"):
            new.validate_text_kv_plan(bad)
    bad = deepcopy(new.identity); bad["text_kv_payload_retirement"]["schema_version"] = "v999"
    with pytest.raises(ValueError, match="policy/version"):
        training.make_plan(admitted()["assets"], "/unused", identity=bad)
    # Real checkpoint serialization and identity validation, not a hash-only assertion.
    model = torch.nn.Linear(2, 2)
    state = {"generator": model}
    saved = save_checkpoint(tmp_path/"checkpoint", state, identity=new.identity, cycle=2, cursor=10,
        versions=dict(generator=2, critic=10, ema=0), policy=training.checkpoint_policy(new.identity))
    assert read_manifest(saved, identity=new.identity)["identity"]["text_kv_payload_retirement"] == policy
    with pytest.raises(ValueError, match="identity"):
        read_manifest(saved, identity=old.identity)
    with pytest.raises(ValueError, match="identity"):
        read_manifest(saved, identity=bad)


def test_public_options_description_and_recipe_require_new_explicit_selection(tmp_path):
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training.native import _options, describe_native
    root = Path(__file__).resolve().parents[1]
    control = json.loads((root/"recipes/self_forcing/training.c51-stream.json").read_text())
    raw = deepcopy(control); raw["execution"]["preset"] = PRESET
    conf = TrainingConfig.from_dict(raw, source=tmp_path/"training.json")
    assert _options(conf, "W4") == "manual"
    assert describe_native(conf)["preset"]["text_kv_payload_retirement"]["preset"] == PRESET
    assert describe_native(conf)["capabilities"]["resume"]["supported"] is True
    assert control["execution"]["preset"] == CONTROL
    variant = json.loads((root/"recipes/self_forcing/training.c51-stream-retire-textkv.json").read_text())
    assert variant == {**control, "execution": {**control["execution"], "preset": PRESET}}
    raw["execution"]["mode"] = "reference"
    with pytest.raises(ValueError):
        _options(TrainingConfig.from_dict(raw, source=conf.source), "W4")
    raw["execution"]["mode"] = "manual"; raw["execution"]["preset"] += ".unknown"
    with pytest.raises(ValueError):
        _options(TrainingConfig.from_dict(raw, source=conf.source), "W4")


def test_untyped_benchmark_and_wrong_plan_cannot_enable_retirement_before_setup(tmp_path, monkeypatch):
    from distillschedule.runtime import livewan_crossscale_worker as worker
    current, plan = continuation()
    path = tmp_path/"plan.json"; path.write_text(json.dumps(plan))
    monkeypatch.setattr(worker, "request_from_plan", lambda *_: pytest.fail("geometry/model setup must not begin"))
    for mode, kind in (("pipeline", "correctness"), ("baseline", "correctness"), ("pipeline", "trajectory")):
        with pytest.raises(RuntimeError, match="continuation-only"):
            worker.run_worker(plan_path=path, mode=mode, run_kind=kind, repo_root=tmp_path)
    plan["evidence"]["text_kv_payload_retirement"]["schema_version"] = "unknown"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="lowered plan/identity"):
        worker.run_worker(plan_path=path, mode="pipeline", run_kind="continuation", repo_root=tmp_path, continuation=current)


@pytest.fixture
def runtime(monkeypatch):
    """TEST ONLY: original tiny classes, CPU BF16 SDPA, pretend-CUDA tensors.

    The exact full-original source provider is loaded before aliases/patching.
    Geometry constants and the FlashAttention source pin are substituted ONLY
    for this test fixture. Real CPU tensors, autograd, non-reentrant checkpoint,
    AdamW and original set_text/cross-forward bodies execute. This is neither
    native CUDA numerics/capacity nor a replacement production dependency.
    """
    import torch
    import torch.nn.functional as functional
    from distillschedule.reference import livewan_wan21 as provider
    from distillschedule.runtime import livewan_cross_kv_cache as cache
    api = provider.load_cpu(backend="torch-sdpa-cpu", precision="float32")
    before = (torch.get_rng_state().clone(), random.getstate(), torch.get_num_threads())
    torch.set_num_threads(1)

    class CPUTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    class TorchCPUFixture:
        def __getattr__(self, name):
            return getattr(torch, name)
        amp = SimpleNamespace(autocast=lambda *args, **kwargs: torch.autocast("cpu", dtype=torch.bfloat16))

    def cpu_attention(q, k, v, k_lens=None):
        assert k_lens is None
        return functional.scaled_dot_product_attention(q.transpose(1, 2).to(v.dtype),
            k.transpose(1, 2).to(v.dtype), v.transpose(1, 2)).transpose(1, 2).contiguous()

    monkeypatch.setitem(sys.modules, "wan", SimpleNamespace(modules=SimpleNamespace(model=api.model)))
    monkeypatch.setitem(sys.modules, "wan.modules", SimpleNamespace(model=api.model))
    monkeypatch.setitem(sys.modules, "wan.modules.model", api.model)
    monkeypatch.setitem(sys.modules, "wanstreamer", SimpleNamespace(stream=api.stream))
    monkeypatch.setitem(sys.modules, "wanstreamer.stream", api.stream)
    # Original source bodies remain genuinely pinned; only the CUDA op differs.
    assert cache._source_sha256(api.stream.FewStepStreamer.set_text) == cache.EXPECTED_SET_TEXT_SHA256
    assert cache._source_sha256(api.model.WanT2VCrossAttention.forward) == cache.EXPECTED_CROSS_FORWARD_SHA256
    monkeypatch.setattr(api.model, "flash_attention", cpu_attention)
    monkeypatch.setattr(cache, "EXPECTED_FLASH_ATTENTION_SHA256", cache._source_sha256(cpu_attention))
    for name, value in dict(CONTEXT_SHAPE=(1, 512, 16), BLOCK_COUNT=2, NUM_HEADS=1, HEAD_DIM=16,
                            CACHE_BYTES=2*512*16*6).items():
        monkeypatch.setattr(cache, name, value)
    facade = TorchCPUFixture()
    receipt = cache.install_generator_cross_kv_cache(facade)
    receipt.update(role="generator", execution_status="PASS")

    def model():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(184)
            return api.model.WanModel(model_type="t2v", patch_size=(1, 2, 2), text_len=512,
                in_dim=16, dim=16, ffn_dim=32, freq_dim=16, text_dim=4096, out_dim=16,
                num_heads=1, num_layers=2).float().eval()

    def streamer(student, retire=True):
        value = api.stream.FewStepStreamer(student, width=32, height=32, max_frames=12,
            cache_frames=9, block_frames=3, num_steps=4, device="cpu", dtype=torch.float32, cuda_graphs=False)
        if retire:
            cache.bind_generator_text_kv_retirement(facade, streamer=value, student=student,
                                                   policy=cache.payload_retirement_policy())
        return value

    yield SimpleNamespace(torch=facade, real_torch=torch, api=api, cache=cache, receipt=receipt,
                          model=model, streamer=streamer, tensor=lambda x: x.as_subclass(CPUTensor))
    cache.uninstall_generator_cross_kv_cache()
    torch.set_rng_state(before[0]); random.setstate(before[1]); torch.set_num_threads(before[2])
    assert not torch.cuda.is_initialized()


def build(runtime, streamer, *, seed=301, generator=None):
    torch = runtime.real_torch
    generator = generator if generator is not None else torch.Generator().manual_seed(seed)
    context = runtime.tensor(torch.randn(runtime.cache.CONTEXT_SHAPE, generator=generator))
    query = runtime.tensor(torch.randn(1, 3, 16, generator=generator))
    with torch.no_grad():
        streamer.set_text(context, None)
        output = streamer.model.blocks[0].cross_attn(query, context, None)
    return context, query, output


def test_actual_payload_weakrefs_marker_grad_bypass_two_streamers_and_next_set_text(runtime):
    r, torch = runtime, runtime.real_torch
    model = r.model(); first, second = r.streamer(model), r.streamer(model)
    native = (first.cache, first.kv, second.cache, second.kv)
    with r.cache.generator_text_kv_payload_lifetime(r.torch, first):
        ctx, query, _ = build(r, first)
        marker = getattr(ctx, r.cache.CACHE_ATTRIBUTE)
        refs = [weakref.ref(entry[name]) for entry in marker.values() for name in ("key", "value")]
        snapshot = ctx.clone()
        with pytest.raises(RuntimeError, match="active rollout"):
            r.cache.retire_generator_text_kv_payload(r.torch, streamer=first)
    gc.collect()
    assert marker is getattr(ctx, r.cache.CACHE_ATTRIBUTE) and marker == {}
    assert all(ref() is None for ref in refs) and torch.equal(ctx, snapshot)
    assert native == (first.cache, first.kv, second.cache, second.kv)
    with r.cache.generator_text_kv_payload_lifetime(r.torch, second):
        other, _, _ = build(r, second, seed=401)
        assert getattr(other, r.cache.CACHE_ATTRIBUTE) and not getattr(ctx, r.cache.CACHE_ATTRIBUTE)
    # Empty marker still takes the ORIGINAL grad branch before entry lookup.
    before = r.receipt["grad_bypass_count"]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model.blocks[0].cross_attn(query.requires_grad_(), ctx, None)
    output.float().square().sum().backward()
    assert r.receipt["grad_bypass_count"] == before+1
    assert model.blocks[0].cross_attn.k.weight.grad is not None
    assert r.receipt["uncached_fallback_count"] == 0
    invalidations = r.receipt["cache_invalidation_count"]
    with r.cache.generator_text_kv_payload_lifetime(r.torch, first):
        replacement, _, _ = build(r, first, seed=501)
        assert not hasattr(ctx, r.cache.CACHE_ATTRIBUTE)
        assert getattr(replacement, r.cache.CACHE_ATTRIBUTE) is not marker
    assert r.receipt["cache_invalidation_count"] == invalidations+1
    assert r.receipt["cache_build_context_count"] == 3
    assert r.receipt["cache_build_block_count"] == 6
    assert r.receipt["cache_build_bytes"] == 3*r.cache.CACHE_BYTES
    assert r.receipt["text_kv_payload_retirement"]["retired_context_count"] == 3


@pytest.mark.parametrize("damage", ["owner", "policy", "missing_marker", "blocks", "parameter", "geometry", "duplicate", "nograd"])
def test_retirement_rejects_invalid_lifetime_ownership_geometry_and_version(runtime, damage):
    r, torch = runtime, runtime.real_torch
    model = r.model(); streamer = r.streamer(model)
    with r.cache.generator_text_kv_payload_lifetime(r.torch, streamer):
        context, query, _ = build(r, streamer)
    if damage == "nograd":
        with torch.no_grad(), pytest.raises(RuntimeError, match="retired context.*no-grad"):
            model.blocks[0].cross_attn(query, context, None)
    elif damage == "duplicate":
        with pytest.raises(RuntimeError, match="duplicate retirement"):
            r.cache.retire_generator_text_kv_payload(r.torch, streamer=streamer)
    else:
        # Rebuild a live payload without invoking the successful boundary yet.
        binding = getattr(streamer, r.cache._RETIREMENT_BINDING)
        binding["active"] = True
        context, query, _ = build(r, streamer, seed=601)
        binding["active"] = False
        marker = getattr(context, r.cache.CACHE_ATTRIBUTE)
        owner = getattr(context, r.cache._RETIREMENT_CONTEXT)
        if damage == "owner": owner["streamer"] += 1
        elif damage == "policy": owner["policy"]["schema_version"] = "unknown"
        elif damage == "missing_marker": delattr(context, r.cache.CACHE_ATTRIBUTE)
        elif damage == "blocks": marker.pop(next(iter(marker)))
        elif damage == "parameter":
            with torch.no_grad(): model.blocks[0].cross_attn.k.weight.add_(.01)
        else: marker[next(iter(marker))]["key"] = torch.zeros(1)
        with pytest.raises(RuntimeError, match="retirement contract changed"):
            r.cache.retire_generator_text_kv_payload(r.torch, streamer=streamer)
        assert r.receipt["text_kv_payload_retirement"]["retired_context_count"] == 1
    assert r.receipt["text_kv_payload_retirement"]["contract_failure_count"] == 1


def equal(left, right, torch):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k], torch) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b, torch) for a, b in zip(left, right))
    return left == right


def test_real_nonreentrant_checkpoint_output_all_grads_adam_rng_and_counters_equal(runtime):
    r, torch = runtime, runtime.real_torch
    from torch.utils.checkpoint import checkpoint
    from distillschedule.runtime.livewan_crossscale_worker import materialize_adam_state, _bind_trajectory_optimizer_parameter_names

    def run(retire):
        torch.manual_seed(720); random.seed(720)
        student = r.model()
        names = ("cache_build_context_count", "cache_build_block_count", "cache_build_bytes", "cache_hit_count",
                 "grad_bypass_count", "cache_invalidation_count", "uncached_fallback_count")
        before = {k: r.receipt[k] for k in names}
        optimizer = torch.optim.AdamW(student.parameters(), lr=2e-6, betas=(0., .99), weight_decay=0., foreach=True)
        parameter_names = _bind_trajectory_optimizer_parameter_names(named_parameters=list(student.named_parameters()),
            optimizer=optimizer, expected_parameter_ids={id(p) for p in student.parameters()})
        materialize_adam_state(torch, optimizer)
        streamers = [r.streamer(student, retire=retire) for _ in range(2)]
        prepared = []
        local = torch.Generator().manual_seed(916); local_before = local.get_state().clone()
        for wave, streamer in enumerate(streamers):
            with r.cache.generator_text_kv_payload_lifetime(r.torch, streamer):
                context, query, output = build(r, streamer, generator=local)
                refs = [weakref.ref(entry[k]) for entry in getattr(context, r.cache.CACHE_ATTRIBUTE).values() for k in ("key", "value")]
            assert all(ref() is None for ref in refs) is retire
            prepared.append((context, query, output.clone()))
        outputs = []
        for context, query, _ in prepared:
            query = query.clone().requires_grad_()
            def block(x, ctx=context):
                for layer in student.blocks:
                    x = layer.cross_attn(x, ctx, None)
                # Non-vacuous real checkpoint RNG replay; global/local RNG are
                # compared separately, not asserted to be upstream sequential RNG.
                return x + torch.rand_like(x)*.01
            with torch.autocast("cpu", dtype=torch.bfloat16):
                output = checkpoint(block, query, use_reentrant=False)
            outputs.append(output.detach().clone()); output.float().square().sum().backward()
        gradients = {name: None if p.grad is None else p.grad.clone() for name, p in student.named_parameters()}
        optimizer.step()
        # Retirement itself never changes either global or operation-local RNG.
        for streamer in streamers:
            assert getattr(streamer.ctx, r.cache.CACHE_ATTRIBUTE) == {} if retire else bool(getattr(streamer.ctx, r.cache.CACHE_ATTRIBUTE))
        return dict(outputs=outputs, rollout_outputs=[v[2] for v in prepared], grads=gradients,
                    weights=deepcopy(student.state_dict()), adam=deepcopy(optimizer.state_dict()),
                    named_adam=deepcopy({parameter_names[id(p)]: state for p, state in optimizer.state.items()}),
                    rng=torch.get_rng_state().clone(), python_rng=random.getstate(), local_rng=local.get_state(),
                    local_before=local_before, counters={k:r.receipt[k]-before[k] for k in names})

    control, retired = run(False), run(True)
    assert equal(control, retired, torch)
    assert retired["counters"]["grad_bypass_count"] == 8  # 2 waves x 2 blocks x first+replay
    assert retired["counters"]["cache_hit_count"] == 2
    assert not equal(retired["local_before"], retired["local_rng"], torch)
    assert any(v is not None and torch.count_nonzero(v) for k,v in retired["grads"].items() if ".cross_attn.k." in k)


def test_unbound_default_lifetime_does_not_access_torch_or_change_any_owner():
    from distillschedule.runtime.livewan_cross_kv_cache import generator_text_kv_payload_lifetime
    streamer = SimpleNamespace(ctx=object(), cache=object(), kv=object())
    before = vars(streamer).copy()
    with generator_text_kv_payload_lifetime(None, streamer):
        pass
    assert vars(streamer) == before


def test_actual_prepare_rollout_runahead_and_fresh_resume_prime_retire_same_payload(runtime, monkeypatch, record_property):
    """Real tiny original rollout/recompute; native scheduling/capacity NOT_RUN.

    The native worker's prepare_rollout is NOT replaced. Its CUDA geometry check
    is scaled only in this CPU fixture; all original set_text/set_world/rollout
    calls, records, native self-KV and operation-local sampler RNG execute.
    """
    r, torch = runtime, runtime.real_torch
    from distillschedule.runtime import livewan_crossscale_worker as worker
    student = r.model()
    # Original generate_block allocates fresh ordinary CPU noise. Alias only
    # the cross-attention query Tensor subclass for this fixture's is_cuda
    # metadata substitution; do not replace a forward or modify any values.
    for block in student.blocks:
        block.cross_attn.register_forward_pre_hook(lambda module, args: (r.tensor(args[0]), *args[1:]))
    with torch.no_grad():
        student.head.head.weight.copy_(torch.randn(student.head.head.weight.shape,
                                                  generator=torch.Generator().manual_seed(111))*.01)
    streamers = [r.streamer(student, retire=False) for _ in range(2)]
    current, plan = continuation()
    current.configure_text_kv_retirement(torch=r.torch, rank=0, student=student, streamers=streamers,
                                        cache_receipt=r.receipt, plan=plan)
    with pytest.raises(ValueError, match="duplicate"):
        current.configure_text_kv_retirement(torch=r.torch, rank=0, student=student, streamers=streamers,
                                            cache_receipt=r.receipt, plan=plan)
    monkeypatch.setattr(worker, "latent_shape", lambda: (1, 16, 6, 4, 4))
    local = torch.Generator().manual_seed(112)
    latent = r.tensor(torch.randn(16, 6, 4, 4, generator=local))
    raw_context = r.tensor(torch.randn(512, 4096, generator=local))
    loader = SimpleNamespace(load=lambda row: (latent, raw_context, 0))
    def row(index, slot=0):
        return SimpleNamespace(operation_key=f"cpu-original-{index}-{slot}", world_frames=3, new_blocks=1,
                               rope_gap=0, seeds=dict(rollout=113+index))
    before_rng = torch.get_rng_state().clone()
    first = worker.prepare_rollout(r.torch, row=row(0), streamer=streamers[0], student=student,
                                  loader=loader, device="cpu", elide_full_context_lens=True)
    assert getattr(first.streamer.ctx, r.cache.CACHE_ATTRIBUTE) == {}
    assert first.raw_context is not first.streamer.ctx and first.recs
    assert torch.equal(before_rng, torch.get_rng_state())
    cache = first.streamer.cache
    original_k, original_v = cache.k.clone(), cache.v.clone()
    prefix = r.api.blockcausal.BufferPrefixKV(cache)
    replay = first.streamer.recompute_step(first.recs[0][0], prefix)
    replay.square().sum().backward()
    assert student.blocks[0].cross_attn.k.weight.grad is not None
    assert torch.equal(cache.k, original_k) and torch.equal(cache.v, original_v)
    # Remaining wave/step1--4 use the very same prepare function, matching every
    # actual native call site; two contexts retire per logical step.
    for index in range(1, 10):
        worker.prepare_rollout(r.torch, row=row(index), streamer=streamers[index % 2], student=student,
                               loader=loader, device="cpu", elide_full_context_lens=True)
    delta = current.text_kv_retirement_delta(rank=0, rows=list(range(10)))
    assert delta["delta"] == dict(retired_context_count=10, retired_block_count=20,
                                  retired_payload_bytes=10*r.cache.CACHE_BYTES)
    # Execute the REAL resume-prime owner, not a reconstruction of its call.
    current.start_cycle = 2; current.pairs = ((0, 1),)*8
    sent = []
    monkeypatch.setattr(worker, "_record_output", lambda *args, **kwargs: None)
    evidence = SimpleNamespace(transport_bytes=0)
    primed = current.prime(torch=r.torch, rank=0, trace=SimpleNamespace(row=row), bundle=SimpleNamespace(student=student),
        streamers=streamers, loader=loader, device="cpu", transport=SimpleNamespace(enqueue_send=lambda **kw: sent.append(kw)),
        shadow=None, evidence=evidence, elision=True)
    assert sent[0]["tensor"] is primed.x0 and sent[0]["step"] == 10
    assert getattr(primed.streamer.ctx, r.cache.CACHE_ATTRIBUTE) == {}
    for index in range(11, 20):
        worker.prepare_rollout(r.torch, row=row(index), streamer=streamers[index % 2], student=student,
                               loader=loader, device="cpu", elide_full_context_lens=True)
    resumed_delta = current.text_kv_retirement_delta(rank=0, rows=list(range(10)))
    assert resumed_delta["delta"] == delta["delta"]
    with pytest.raises(ValueError, match="census"):
        current.text_kv_retirement_delta(rank=0, rows=list(range(10)))
    assert r.receipt["text_kv_payload_retirement"]["retired_context_count"] == 20
    assert r.receipt["cache_build_context_count"] == 20 and r.receipt["cache_invalidation_count"] == 18
    assert r.receipt["contract_failure_count"] == 0 and r.receipt["uncached_fallback_count"] == 0
    record_property("retirement_evidence_scope", "ACTUAL_CPU_TINY_ORIGINAL_PREPARE_AND_REPLAY_NOT_CUDA_CAPACITY")
    record_property("actual_retired_context_count", 20)
    record_property("actual_retired_payload_bytes_cpu_fixture", 20*r.cache.CACHE_BYTES)


def test_failure_cleans_built_payload_without_forward_retry_or_success_counter(runtime):
    r = runtime
    student = r.model(); streamer = r.streamer(student)
    with pytest.raises(ValueError, match="original rollout failure"):
        with r.cache.generator_text_kv_payload_lifetime(r.torch, streamer):
            context, _, _ = build(r, streamer)
            marker = getattr(context, r.cache.CACHE_ATTRIBUTE)
            refs = [weakref.ref(entry[k]) for entry in marker.values() for k in ("key", "value")]
            raise ValueError("original rollout failure")
    assert marker == {} and all(ref() is None for ref in refs)
    counters = r.receipt["text_kv_payload_retirement"]
    assert counters["aborted_context_count"] == 1 and counters["retired_context_count"] == 0
    assert r.receipt["cache_hit_count"] == 1 and r.receipt["cache_build_context_count"] == 1
    assert not getattr(streamer, r.cache._RETIREMENT_BINDING)["active"]
    with r.cache.generator_text_kv_payload_lifetime(r.torch, streamer):
        build(r, streamer, seed=1001)
    assert counters["retired_context_count"] == 1 and r.receipt["cache_invalidation_count"] == 1


def test_unbound_score_missing_cache_and_unknown_binding_version_refuse(runtime):
    r = runtime
    student = r.model(); unbound = r.streamer(student, retire=False)
    with pytest.raises(RuntimeError, match="explicit bound owner"):
        r.cache.retire_generator_text_kv_payload(r.torch, streamer=unbound)
    bad = r.cache.payload_retirement_policy(); bad["schema_version"] = "unknown"
    with pytest.raises(RuntimeError, match="policy/version"):
        r.cache.bind_generator_text_kv_retirement(r.torch, streamer=unbound, student=student, policy=bad)
    with pytest.raises(RuntimeError, match="original owner"):
        r.cache.bind_generator_text_kv_retirement(r.torch, streamer=unbound, student=r.model(), policy=r.cache.payload_retirement_policy())
    r.cache.bind_generator_text_kv_retirement(r.torch, streamer=unbound, student=student, policy=r.cache.payload_retirement_policy())
    with pytest.raises(RuntimeError, match="duplicate binding"):
        r.cache.bind_generator_text_kv_retirement(r.torch, streamer=unbound, student=student, policy=r.cache.payload_retirement_policy())
    with pytest.raises(RuntimeError, match="missing marker"):
        r.cache.retire_generator_text_kv_payload(r.torch, streamer=unbound)
    with pytest.raises(RuntimeError, match="outside the bound rollout"):
        build(r, unbound)
    current, plan = continuation()
    current.configure_text_kv_retirement(torch=r.torch, rank=8, student=None, streamers=[],
                                        cache_receipt=dict(role="score"), plan=plan)
    snapshot = current.text_kv_retirement_snapshot(rank=8)
    assert snapshot["execution_status"] == "NOT_APPLICABLE" and "retired_context_count" not in snapshot
    # Unmarked Score contexts still follow the original fallback, not retirement.
    with r.real_torch.no_grad():
        context = r.real_torch.randn(r.cache.CONTEXT_SHAPE)
        student.blocks[0].cross_attn(r.real_torch.randn(1, 3, 16), context, None)
    assert not hasattr(context, r.cache.CACHE_ATTRIBUTE)
    assert r.receipt["uncached_fallback_count"] == 1
    binding = getattr(unbound, r.cache._RETIREMENT_BINDING)
    binding["policy"]["schema_version"] = "unknown"
    with pytest.raises(RuntimeError, match="policy/version"):
        with r.cache.generator_text_kv_payload_lifetime(r.torch, unbound):
            pass


def test_retirement_has_no_value_reads_rng_allocator_sync_or_model_calls():
    from distillschedule.runtime.livewan_cross_kv_cache import _retire_generator_text_kv_payload
    source = inspect.getsource(_retire_generator_text_kv_payload)
    for forbidden in (".cpu(", ".item(", ".numpy(", "synchronize(", "empty_cache(", ".detach(", ".to(", "manual_seed(", ".forward(", ".step("):
        assert forbidden not in source
