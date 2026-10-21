import copy
import pytest
from distillschedule.runtime.wan13b_cast_graph_composition import check, validate_build
from distillschedule.runtime import wan13b_lora_cast as impl


def test_admission_standalone_cache_still_rejected():
    p = "resident-async128-tree-chunk2048"
    assert check(p, cast_cache=False, fused=True, graph=True) is False
    assert check(p, cast_cache=True, fused=True, graph=True) is True
    for kwargs in (dict(fused=False, graph=True), dict(fused=True, graph=False),
                   dict(fused=True, graph=True, shadow=True),
                   dict(fused=True, graph=True, input_gemm=True)):
        with pytest.raises(ValueError):
            check(p, cast_cache=True, **kwargs)
    with pytest.raises(ValueError):
        check("world16-fsdp", cast_cache=True, fused=True, graph=True)


def test_fused_cast_composition_output_gradient_input_version_and_scope(monkeypatch):
    torch = pytest.importorskip("torch")

    class Linear(torch.nn.Module):
        disable_adapters = False
        merged = False
        active_adapters = ["default"]
        lora_variant = {}
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(4, 4, bias=False).to(torch.bfloat16)
            self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 2, bias=False)})
            self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 4, bias=False)})
            self.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Identity()})
            self.scaling = {"default": 0.5}
        def _check_forward_args(self, *args, **kwargs): pass
        def _cast_input_dtype(self, x, dtype): return x.to(dtype)
        def forward(self, x):
            base = self.base_layer(x)
            delta = self.lora_B["default"](self.lora_A["default"](self._cast_input_dtype(x, torch.float32)))
            return (base + delta * self.scaling["default"]).to(base.dtype)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = Linear(), Linear()
        def forward(self, x): return self.a(x) + self.b(x)

    model = Model(); reference = copy.deepcopy(model)
    from distillschedule.search.wan_candidate_bundle import INCUMBENT
    from distillschedule.search.wan_implementation import WanImplementation
    monkeypatch.setenv("DISTILLSCHEDULE_SEARCH_CANDIDATE_ID", WanImplementation("resident",2048,True,candidate_bundle=INCUMBENT).id)
    monkeypatch.setenv(impl.CAST_CACHE_ENV, "1")
    monkeypatch.setenv(impl.FUSED_EPILOGUE_ENV, "1")
    monkeypatch.setenv("DISTILLSCHEDULE_WAN13B_GENERATOR_ROLLOUT_CUDA_GRAPH", "1")
    cache = impl.install_generator_nograd_lora_cast_cache(model, role="generator", linear_type=Linear)
    impl.install_generator_nograd_lora_fused_epilogue(model, role="generator", linear_type=Linear,
        fused_epilogue=lambda a,b,s: (a+b*s).to(a.dtype))
    x = torch.randn(3, 4).to(torch.bfloat16)
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        assert torch.equal(model(x), reference(x))
        x.add_(1)
        assert torch.equal(model(x), reference(x))
    assert cache["cache_hit_count"] == 2 and cache["cache_miss_count"] == 2
    assert torch.equal(rng, torch.get_rng_state())
    before = dict(cache)
    model(x).float().sum().backward(); reference(x).float().sum().backward()
    assert cache == before  # grad path never enters cast scope
    for (_, a), (_, b) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        assert torch.equal(a.grad, b.grad)
    build = {"generator":{"metadata":{"generator_nograd_lora_cast_cache":cache}},
             "critic":{"metadata":{"generator_nograd_lora_cast_cache":{"enabled":False}}}}
    assert validate_build(build, enabled=True)["arithmetic_changed"] is False
    cache["cache_hit_count"] = 0
    with pytest.raises(AssertionError): validate_build(build, enabled=True)


def test_disabled_execution_gate():
    assert validate_build({}, enabled=False) == {"enabled":False}
    with pytest.raises(AssertionError):
        validate_build({"critic":{"metadata":{"generator_nograd_lora_cast_cache":{"enabled":True}}}}, enabled=False)
