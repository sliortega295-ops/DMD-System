import pytest
import torch

from runtime.rollout_trace import (
    ForceNoGradCalls,
    LightweightNestedCallAudit,
    NestedCallTracer,
    infer_pyramid_stage,
)


def test_infer_pyramid_stage_accepts_exact_power_of_two_shapes():
    for stage, shape in enumerate(((9, 12, 20), (9, 24, 40), (9, 48, 80))):
        tensor = torch.zeros(1, 16, *shape)
        assert infer_pyramid_stage(
            tensor, base_height=48, base_width=80, num_stages=3
        ) == stage


def test_infer_pyramid_stage_rejects_invalid_shape():
    with pytest.raises(ValueError, match="not a valid stage"):
        infer_pyramid_stage(
            torch.zeros(1, 16, 9, 18, 30),
            base_height=48,
            base_width=80,
            num_stages=3,
        )


def test_nested_call_tracer_preserves_graph_and_reports_selected_gradients():
    events = []
    tracer = NestedCallTracer(
        lambda **kwargs: (kwargs["hidden_states"] * 2,),
        base_height=48,
        base_width=80,
        num_stages=3,
        memory_snapshot=lambda: {"allocated": 7},
        on_event=events.append,
    )
    prefix = torch.ones(1, 1, 1, 12, 20, requires_grad=True)
    with torch.no_grad():
        prefix_output = tracer(hidden_states=prefix, timestep=torch.tensor([900]))[0]
    selected = torch.ones(1, 1, 1, 12, 20, requires_grad=True)
    selected_output = tracer(
        hidden_states=selected,
        timestep=torch.tensor([800]),
        is_first_denoising_step=True,
    )[0]
    selected_output.sum().backward()

    assert not prefix_output.requires_grad
    assert tracer.selected_call_indices() == [1]
    assert tracer.exit_call_by_stage() == {"0": 1}
    assert len(events) == 2
    assert events[0]["grad_enabled"] is False
    assert events[1]["output_requires_grad"] is True
    assert events[1]["is_first_denoising_step"] is True
    assert events[1]["memory_before"] == {"allocated": 7}
    assert tracer.retained_gradient_summary() == [
        {
            "call_index": 1,
            "gradient_present": True,
            "gradient_elements": selected_output.numel(),
            "gradient_abs_sum": float(selected_output.numel()),
            "gradient_abs_max": 1.0,
        }
    ]


def test_nested_call_tracer_records_failure():
    def fail(**_kwargs):
        raise RuntimeError("expected failure")

    tracer = NestedCallTracer(
        fail, base_height=48, base_width=80, num_stages=3
    )
    with pytest.raises(RuntimeError, match="expected failure"):
        tracer(hidden_states=torch.zeros(1, 1, 1, 24, 40))
    assert tracer.calls[0]["stage"] == 1
    assert tracer.calls[0]["status"] == "failed"
    assert tracer.calls[0]["error_type"] == "RuntimeError"


def test_inner_grad_context_reenables_graph_inside_outer_no_grad():
    tracer = NestedCallTracer(
        lambda **kwargs: (kwargs["hidden_states"] * 3,),
        base_height=48,
        base_width=80,
        num_stages=3,
    )
    value = torch.ones(1, 1, 1, 12, 20, requires_grad=True)
    with torch.no_grad():
        with torch.set_grad_enabled(True):
            output = tracer(hidden_states=value, timestep=torch.tensor([500]))[0]
    assert output.requires_grad
    assert tracer.calls[0]["grad_enabled"] is True
    assert tracer.calls[0]["output_requires_grad"] is True


def test_tracer_can_observe_grad_calls_without_retaining_activation_graphs():
    tracer = NestedCallTracer(
        lambda **kwargs: (kwargs["hidden_states"] * 5,),
        base_height=48,
        base_width=80,
        num_stages=3,
        retain_output_gradients=False,
    )
    value = torch.ones(1, 1, 1, 12, 20, requires_grad=True)
    output = tracer(hidden_states=value, timestep=torch.tensor([500]))[0]
    assert output.requires_grad
    assert tracer.selected_call_indices() == [0]
    assert tracer.retained_gradient_summary() == []


def test_force_no_grad_calls_preserves_values_and_suppresses_call_graph():
    baseline = NestedCallTracer(
        lambda **kwargs: (kwargs["hidden_states"] * 7,),
        base_height=48,
        base_width=80,
        num_stages=3,
        retain_output_gradients=False,
    )
    candidate = NestedCallTracer(
        lambda **kwargs: (kwargs["hidden_states"] * 7,),
        base_height=48,
        base_width=80,
        num_stages=3,
        retain_output_gradients=False,
    )
    value = torch.ones(1, 1, 1, 12, 20, requires_grad=True)
    source_output = baseline(hidden_states=value, timestep=torch.tensor([500]))[0]
    candidate_output = ForceNoGradCalls(candidate)(
        hidden_states=value, timestep=torch.tensor([500])
    )[0]
    assert torch.equal(source_output, candidate_output)
    assert source_output.requires_grad
    assert not candidate_output.requires_grad
    assert baseline.calls[0]["grad_enabled"] is True
    assert candidate.calls[0]["grad_enabled"] is False


def test_lightweight_audit_checks_stage_and_no_grad_without_tensor_retention():
    audit = LightweightNestedCallAudit(
        lambda **kwargs: (kwargs["hidden_states"] + 2,),
        base_height=48,
        base_width=80,
        num_stages=3,
    )
    value = torch.ones(1, 1, 1, 24, 40, requires_grad=True)
    output = ForceNoGradCalls(audit)(hidden_states=value)[0]
    assert not output.requires_grad
    assert audit.calls == [
        {
            "call_index": 0,
            "stage": 1,
            "denoise_call_in_stage": 0,
            "hidden_states_shape": [1, 1, 1, 24, 40],
            "grad_enabled": False,
            "output_shape": [1, 1, 1, 24, 40],
            "output_requires_grad": False,
            "status": "success",
        }
    ]
