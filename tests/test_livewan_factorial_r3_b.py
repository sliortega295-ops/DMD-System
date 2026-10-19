"""R3 helper tests: real CPU autograd/lifetimes, NOT target kernel evidence."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from threading import get_ident
from types import SimpleNamespace as NS

import pytest

from distillschedule.runtime import livewan_factorial_r3_b as r3


def test_capture_restores_pristine_functions_and_installer_markers_on_success_and_failure():
    def native(x):
        return x
    owner = NS(op=native, _distillschedule_existing="keep")

    def installer():
        owner.op = lambda x: x + 1
        owner._distillschedule_new = True
        owner._distillschedule_existing = "changed"
        return {"calls": 0}

    functions, receipt = r3.capture_installer(installer, [(owner, "op")])
    assert functions[0](2) == 3 and receipt == {"calls": 0}
    assert owner.op is native and owner._distillschedule_existing == "keep"
    assert not hasattr(owner, "_distillschedule_new")

    def broken():
        installer()
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        r3.capture_installer(broken, [(owner, "op")])
    assert owner.op is native and not hasattr(owner, "_distillschedule_new")


@pytest.mark.parametrize("role,replay_role", [("critic", None), (None, "critic"),
                                            ("student", "teacher")])
def test_real_cross_thread_checkpoint_captures_role_gradients_and_rng(role, replay_role):
    torch = pytest.importorskip("torch")
    from torch.utils import checkpoint
    seen = []
    class SavedSquare(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x, torch.ones(1))
            return x.square()

        @staticmethod
        def backward(ctx, grad):
            x, _ = ctx.saved_tensors
            return grad * (2 * x)

    def native(x):
        seen.append(("native", get_ident()))
        return x.square()

    def score(x):
        seen.append(("score", get_ident()))
        return SavedSquare.apply(x)

    owner = NS(op=native)
    dispatcher = r3.RoleDispatcher(torch)
    dispatcher.add("square", owner, "op", score=score)
    proxy = r3.RoleCheckpointProxy(checkpoint, torch, dispatcher)
    torch.manual_seed(13)
    x = torch.randn(4, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_(True)
    rng = torch.get_rng_state()
    expected = (expected_x.square() * torch.rand_like(expected_x)).sum()
    expected.backward()
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    with dispatcher.role(role):
        y = proxy.checkpoint(lambda z: owner.op(z) * torch.rand_like(z), x,
                             use_reentrant=False)
    origin_thread = get_ident()

    def backward():
        with dispatcher.role(replay_role):
            y.sum().backward()
            assert dispatcher.current_role == replay_role

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(backward).result()
    assert torch.equal(x.grad, expected_x.grad)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert dispatcher.current_role is None
    assert {label for label, _ in seen} == ({"score"} if role == "critic" else {"native"})
    assert any(thread != origin_thread for _, thread in seen)
    assert proxy.receipt()["replay_entries"] == 1
    first, second = torch.optim.AdamW([x]), torch.optim.AdamW([expected_x])
    first.step(); second.step()
    assert torch.equal(x, expected_x)
    assert all(torch.equal(first.state[x][key], second.state[expected_x][key])
               for key in first.state[x])
    dispatcher.restore()
    assert owner.op is native and not torch.cuda.is_initialized()


def test_dispatch_nograd_student_selection_and_nested_exception_restore():
    torch = pytest.importorskip("torch")
    owner = NS(op=lambda: "native")
    dispatcher = r3.RoleDispatcher(torch)
    dispatcher.add("label", owner, "op", score=lambda: "score",
                   student=lambda: "G-nograd", student_nograd_only=True)
    assert owner.op() == "native"
    with dispatcher.role("student"):
        assert owner.op() == "native"
        with torch.no_grad():
            assert owner.op() == "G-nograd"
    with pytest.raises(RuntimeError, match="synthetic"):
        with dispatcher.role("teacher"):
            assert owner.op() == "score"
            with dispatcher.role("student"), torch.no_grad():
                assert owner.op() == "G-nograd"
            assert dispatcher.current_role == "teacher"
            raise RuntimeError("synthetic")
    assert dispatcher.current_role is None
    receipt = dispatcher.receipt()
    assert receipt["scope_failures"] == 1
    assert receipt["dispatch_completed"]["student"]["label"] == 2


def test_real_cpu_sac_is_per_checkpoint_mutation_guarded_and_student_bypassed():
    torch = pytest.importorskip("torch")
    from torch.utils import checkpoint
    from distillschedule.runtime.livewan_attention_checkpoint import AttentionCheckpointContexts
    # Real CPU custom-op execution is counted, NOT inferred from policy calls.
    # Torch2.14 no longer calls policy_fn during SAC replay; target2.10 does.
    counts = {"executions": 0}

    @torch.library.custom_op("livewan_r3_cpu_test::square", mutates_args=(),
                            schema="(Tensor a) -> Tensor")
    def cpu_square(a):
        counts["executions"] += 1
        return a.square()

    def setup(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    def backward(ctx, grad):
        return 2 * ctx.saved_tensors[0] * grad

    cpu_square.register_autograd(backward, setup_context=setup)
    contexts = AttentionCheckpointContexts(torch, torch.ops.livewan_r3_cpu_test.square.default,
        {"operator": "CPU_SYNTHETIC_SQUARE_ONLY"}, metadata_reader=lambda t, a, k:
        tuple(tuple(x.shape) for x in a))
    dispatcher = r3.RoleDispatcher(torch)
    proxy = r3.RoleCheckpointProxy(checkpoint, torch, dispatcher, score_contexts=contexts)
    x = torch.randn(3, 4, requires_grad=True)
    native_x = x.detach().clone().requires_grad_()
    expected = native_x.square().sin()
    expected.sum().backward()
    with dispatcher.role("critic"):
        first = proxy.checkpoint(lambda a: cpu_square(a).sin(), x, use_reentrant=False)
    first.sum().backward()
    assert torch.equal(first, expected) and torch.equal(x.grad, native_x.grad)
    assert counts["executions"] == 1  # No custom kernel execution on replay.
    with dispatcher.role("student"):
        proxy.checkpoint(lambda a: a.square(), x, use_reentrant=False).sum().backward()
    assert contexts.receipt()["checkpoint_calls"] == 1
    assert contexts.receipt()["original_fa_calls"] == 1
    with dispatcher.role("critic"):
        proxy.checkpoint(lambda a: cpu_square(a).sin(), x, use_reentrant=False).sum().backward()
    assert contexts.receipt()["checkpoint_calls"] == 2
    assert contexts.checkpoints[0] is not contexts.checkpoints[1]
    assert counts["executions"] == 2
    saved = []

    def changed(a):
        z = cpu_square(a)
        saved.append(z)
        return z.sin()

    with dispatcher.role("critic"):
        result = proxy.checkpoint(changed, x, use_reentrant=False)
    with torch.no_grad():
        saved[0].add_(1)
    with pytest.raises(RuntimeError, match="mutated|modified"):
        result.sum().backward()
    assert dispatcher.current_role is None


def _cpu_kv_fixture():
    torch = pytest.importorskip("torch")
    class Cross(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q, self.k, self.v, self.o = [torch.nn.Linear(4, 4) for _ in range(4)]
            self.norm_q = self.norm_k = torch.nn.Identity()
            self.num_heads, self.head_dim = 2, 2

        def forward(self, x, context, context_lens):
            q = self.q(x).view(1, -1, 2, 2)
            k = self.k(context).view(1, -1, 2, 2)
            v = self.v(context).view(1, -1, 2, 2)
            return self.o(attention(q, k, v, k_lens=context_lens).flatten(2))

    def attention(q, k, v, *, k_lens):
        assert k_lens is None
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    class CPUOnlyKV(r3.LazyTextKV):
        def _validate_context(self, context, lens):
            if tuple(context.shape) != (1, 3, 4) or lens is not None:
                raise RuntimeError("CPU synthetic context changed")

        def _validate_query(self, cross, x, context):
            assert x.shape == (1, 2, 4)

        def _validate_outputs(self, key, value):
            assert key.shape == value.shape == (1, 3, 2, 2)

        @contextmanager
        def _autocast(self):
            yield

    dispatcher = r3.RoleDispatcher(torch)
    cache = CPUOnlyKV(torch, dispatcher, Cross.forward, attention)
    cross = Cross()
    layer = NS(blk=NS(cross_attn=cross))
    context = torch.randn(1, 3, 4)
    x = torch.randn(1, 2, 4)
    return torch, dispatcher, cache, cross, layer, context, x


def test_lazy_kv_real_torch_only_builds_inside_entered_block_and_clears_on_exit():
    torch, dispatcher, cache, cross, layer, context, x = _cpu_kv_fixture()
    with torch.no_grad():
        expected = cross(x, context, None)
        with dispatcher.role("student"), cache.scope(generator_version=0):
            cache.bind(NS(), context, None)
            assert cache.receipt()["cache_build_block_count"] == 0
            with pytest.raises(RuntimeError, match="gathered"):
                cache.forward(cross, x, context, None)
            with cache.gathered(layer):
                a = cache.forward(cross, x, context, None)
                b = cache.forward(cross, x, context, None)
                assert torch.equal(a, expected) and torch.equal(a, b)
                entry = next(iter(cache.entries.values()))
                assert not entry["key"].requires_grad and not entry["value"].requires_grad
                assert entry["key"]._base is None and entry["value"]._base is None
        assert not cache.entries and not cache.active
    r = cache.receipt()
    assert r["cache_build_block_count"] == 1 and r["cache_hit_count"] == 2
    assert r["retained_cache_bytes"] == 0
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("mutation", ["context", "parameter", "replacement"])
def test_lazy_kv_strict_context_and_parameter_invalidation(mutation):
    torch, dispatcher, cache, cross, layer, context, x = _cpu_kv_fixture()
    with torch.no_grad(), dispatcher.role("student"), cache.scope(generator_version=0):
        cache.bind(NS(), context, None)
        with cache.gathered(layer):
            cache.forward(cross, x, context, None)
            if mutation == "context":
                context.add_(1)
            elif mutation == "parameter":
                cross.k.weight.add_(1)
            else:
                cross.k.weight = torch.nn.Parameter(cross.k.weight.detach().clone())
            with pytest.raises(RuntimeError, match="changed"):
                cache.forward(cross, x, context, None)
            assert cache.entries == {}
    assert cache.receipt()["cache_invalidation_count"] == 1


def test_lazy_kv_grad_and_foreign_role_bypass_preserve_parameter_gradients():
    torch, dispatcher, cache, cross, layer, context, x = _cpu_kv_fixture()
    expected_model = deepcopy(cross)
    x = x.requires_grad_()
    y = x.detach().clone().requires_grad_()
    with dispatcher.role("student"), cache.scope(generator_version=0):
        cache.bind(NS(), context, None)
        with cache.gathered(layer):
            cache.forward(cross, x, context, None).sum().backward()
    expected_model(y, context, None).sum().backward()
    assert torch.equal(x.grad, y.grad)
    assert all(torch.equal(a.grad, b.grad) for a, b in zip(cross.parameters(), expected_model.parameters()))
    assert cache.receipt()["grad_bypass_count"] == 1
    assert cache.receipt()["cache_build_block_count"] == 0


def _retention_fixture():
    torch = pytest.importorskip("torch")
    events = []
    class CPUFSDPStandIn(torch.nn.Module):
        def __init__(self, label):
            super().__init__()
            self.label, self.reshard_after_forward = label, True
            self.weight = torch.nn.Parameter(torch.ones(2))

        def set_reshard_after_forward(self, value, *, recurse):
            assert recurse is False
            events.append((self.label, "set", value))
            self.reshard_after_forward = value

        def reshard(self):
            events.append((self.label, "reshard"))

        def unshard(self, *args, **kwargs):
            raise AssertionError("R3 must not eagerly all-gather")

    student = torch.nn.Sequential(CPUFSDPStandIn("a"), CPUFSDPStandIn("b"))
    retained = r3.FSDPRolloutRetention(torch, student, unit_type=CPUFSDPStandIn,
                                     device_type="cpu")
    return torch, student, retained, events


def test_materialization_retention_is_lazy_nograd_only_and_restored_after_error():
    torch, student, retained, events = _retention_fixture()
    with pytest.raises(RuntimeError, match="no_grad"):
        with retained.scope(torch, student):
            pass
    assert events == []
    with pytest.raises(RuntimeError, match="synthetic"):
        with torch.no_grad(), retained.scope(torch, student):
            assert all(not m.reshard_after_forward for m in student)
            assert not torch.is_autocast_enabled("cpu")
            with pytest.raises(RuntimeError, match="recursive"):
                with retained.scope(torch, student):
                    pass
            raise RuntimeError("synthetic")
    assert all(m.reshard_after_forward for m in student)
    assert events == [("a", "set", False), ("b", "set", False), ("a", "reshard"),
                      ("a", "set", True), ("b", "reshard"), ("b", "set", True)]
    r = retained.receipt()
    assert r["scope_entries"] == r["scope_exits"] == 1
    assert r["reshard_calls"] == 2 and r["autocast_cache_clear_count"] == 1
    assert not r["active"]


def test_retention_rejects_parameter_commit_inside_scope_but_still_restores():
    torch, student, retained, events = _retention_fixture()
    with pytest.raises(RuntimeError, match="parameter.*changed"):
        with torch.no_grad(), retained.scope(torch, student):
            student[0].weight.add_(1)
    assert all(m.reshard_after_forward for m in student) and not retained.active
    assert retained.receipt()["parameter_version_failures"] == 1


def test_retention_cleanup_failure_still_restores_later_units():
    torch, student, retained, events = _retention_fixture()

    def broken():
        raise RuntimeError("CPU synthetic reshard failure")

    student[0].reshard = broken
    with pytest.raises(RuntimeError, match="synthetic reshard failure"):
        with torch.no_grad(), retained.scope(torch, student):
            pass
    assert all(m.reshard_after_forward for m in student)
    assert ("b", "reshard") in events
    assert retained.receipt()["restore_policy_calls"] == 2
    assert not retained.active


def test_composed_rollout_scopes_do_not_cross_generator_commit(monkeypatch):
    torch, student, retained, events = _retention_fixture()
    monkeypatch.setattr(r3, "FSDPRolloutRetention", lambda t, s: retained)
    helper = r3.CompleteB(torch, student)
    _, _, kv, cross, layer, context, x = _cpu_kv_fixture()
    helper.kv, kv.dispatcher = kv, helper.dispatch
    with torch.no_grad(), helper.scope(torch, student, generator_version=0):
        kv.bind(NS(), context, None)
        with kv.gathered(layer):
            kv.forward(cross, x, context, None)
        assert helper.receipt()["generator_cross_attention_kv_cache"]["retained_cache_bytes"] > 0
        with pytest.raises(RuntimeError, match="commit attempted"):
            helper.generator_committed(1)
    helper.generator_committed(1)
    assert helper.dispatch.current_role is None
    assert helper.receipt()["generator_cross_attention_kv_cache"]["retained_cache_bytes"] == 0
    with torch.no_grad(), pytest.raises(RuntimeError, match="stale"):
        with helper.scope(torch, student, generator_version=0):
            pass
    with torch.no_grad(), helper.scope(torch, student):
        kv.bind(NS(), context, None)
        with kv.gathered(layer):
            kv.forward(cross, x, context, None)
    assert kv.receipt()["cache_build_block_count"] == 2
    assert helper.receipt()["observed_generator_commits"] == 1
    assert all(m.reshard_after_forward for m in student)


def test_import_and_helper_only_cpu_suite_never_initialize_cuda():
    torch = pytest.importorskip("torch")
    assert not torch.cuda.is_initialized()
