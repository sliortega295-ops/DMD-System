"""Rank-local native samplers rendezvous only at equal-census B1 DiT calls."""
import pytest
import torch


def test_local_controls_and_two_chained_gradient_calls_match_sequential_B1():
    from distillschedule.runtime.static_native_cooperative import run_native_B1_cohort
    torch.manual_seed(216)
    weight = torch.nn.Parameter(torch.randn(5, 5))
    initial = torch.randn(8, 2, 5, requires_grad=True)
    timesteps = torch.arange(8, dtype=torch.float32) + 1
    history = torch.randn_like(initial)
    calls = []

    def model(**kw):
        calls.append((torch.is_grad_enabled(), kw['hidden_states'].shape[0]))
        return (torch.cat([(kw['hidden_states'][i:i+1] @ weight + kw['latents_history_short'][i:i+1]
                 * kw['timestep'][i:i+1].reshape(-1, 1, 1)).tanh()
                 for i in range(kw['hidden_states'].shape[0])]),)

    def sample(view, i):
        x = initial[i:i+1]
        kwargs = dict(hidden_states=x, timestep=timesteps[i:i+1],
            latents_history_short=history[i:i+1], return_dict=False)
        with torch.no_grad():
            first = view(**kwargs)[0]
        kwargs['hidden_states'] = x + first
        middle = view(**kwargs)[0]
        # This differing row-local host branch must not become rank0's branch.
        kwargs['hidden_states'] = middle * (1. if i % 2 else 2.)
        return view(**kwargs)[0]

    rng_before = torch.get_rng_state().clone()
    actual, receipt = run_native_B1_cohort(model, sample, device='cpu', timeout_seconds=5)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert receipt['logical_samples'] == 8
    assert receipt['rendezvous_calls'] == 3
    assert receipt['rank_local_controls_preserved']
    assert calls == [(False, 8), (True, 8), (True, 8)]
    torch.cat(actual).square().mean().backward()
    actual_grad = weight.grad.clone(); actual_input_grad = initial.grad.clone()
    weight.grad = None; initial.grad = None
    expected = [sample(model, i) for i in range(8)]
    assert torch.equal(torch.cat(actual), torch.cat(expected))
    torch.cat(expected).square().mean().backward()
    torch.testing.assert_close(weight.grad, actual_grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(initial.grad, actual_input_grad, rtol=1e-6, atol=1e-6)


def test_divergent_call_count_fails_without_hanging_workers():
    from distillschedule.runtime.static_native_cooperative import run_native_B1_cohort
    def sample(view, i):
        if i == 0:
            return torch.ones(1, 2)
        return view(hidden_states=torch.ones(1, 2), return_dict=False)[0]
    with pytest.raises(ValueError, match='call count'):
        run_native_B1_cohort(lambda **kw: (kw['hidden_states'],), sample,
                             device='cpu', timeout_seconds=3)


def test_mismatched_grad_scope_fails_before_model_execution():
    from distillschedule.runtime.static_native_cooperative import run_native_B1_cohort
    calls=[]
    def model(**kw):
        calls.append(1)
        return (kw['hidden_states'],)
    def sample(view, i):
        with torch.set_grad_enabled(i != 0):
            return view(hidden_states=torch.ones(1, 2), return_dict=False)[0]
    with pytest.raises(ValueError, match='gradient scope'):
        run_native_B1_cohort(model, sample, device='cpu', timeout_seconds=3)
    assert not calls
