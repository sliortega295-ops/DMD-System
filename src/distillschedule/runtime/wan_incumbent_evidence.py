"""Pure receipt gates for the typed coupled bundle, outside measured actions."""
from __future__ import annotations
from distillschedule.core.contracts import require
from distillschedule.search.wan_candidate_bundle import CAPABILITIES


def validate_bundle_build(bundle, build, *, executed=False, cycles=0):
    roles = {r: build[r]['metadata'] for r in ('generator', 'critic')}
    if not bundle.enabled:
        for role, metadata in roles.items():
            require(all(not metadata.get(name, {}).get('enabled', False) for name in CAPABILITIES),
                    'unrequested incumbent capability executed', path='wan.bundle.actual')
        return {'candidate_bundle': bundle.to_dict(), 'actual_compute_observed': executed}
    require(type(cycles) is int and (not executed or cycles == 2),
            'incumbent census supports exactly two original fixed-trace cycles', path='wan.bundle.actual')
    for role, metadata in roles.items():
        require(metadata.get('candidate_bundle') == bundle.to_dict()
                and metadata.get('candidate_bundle_sources') == bundle.source_identity(),
                'bundle source/capability receipt mismatch', path='wan.bundle.actual')
        expected = dict(grad_lora_fused_epilogue=True, modulated_norm_fusion=True,
            generator_rollout_cuda_graph=role == 'generator',
            generator_nograd_lora_cast_cache=role == 'generator',
            critic_grad_cuda_graph=role == 'critic')
        for name, enabled in expected.items():
            receipt = metadata.get(name, {})
            require(receipt.get('requested') is True and receipt.get('enabled') is enabled,
                    f'{role}: {name} installation mismatch', path='wan.bundle.actual')
    if not executed:
        return {'candidate_bundle': bundle.to_dict(), 'actual_compute_observed': False}
    from .wan13b_lora_grad_epilogue import validate_grad_epilogue_receipts
    from .wan13b_norm_policy import validate as validate_norm
    from .wan13b_cast_graph_composition import validate_build as validate_cast
    from .wan13b_rollout_cuda_graph import validate as validate_graph
    grad = validate_grad_epilogue_receipts(build, requested=True)
    norm = validate_norm(build, enabled=True)
    cast = validate_cast(build, enabled=True)
    rollout = validate_graph(build, enabled=True, cycles=cycles, diagnostic_reference=False)
    critic = roles['critic']['critic_grad_cuda_graph']
    expected = dict(replay_calls=5*cycles, backward_replay_calls=5*cycles,
                    reference_calls=0, weights_merged=False,
                    optimizer_reducer_ema_unchanged=True,
                    captured_parameter_gradients_return_to_original_accumulators=True)
    require(all(critic.get(k) == v for k, v in expected.items()),
            'Critic grad graph executed census/contract mismatch', path='wan.bundle.actual')
    entries = critic.get('capture_entries', 0)
    require(1 <= entries <= 3 and all(critic.get(k) == 4*entries for k in
                ('cold_original_forward_calls', 'cold_original_backward_calls')),
            'Critic cold capture census mismatch', path='wan.bundle.actual')
    return {'candidate_bundle': bundle.to_dict(), 'bundle_sources': bundle.source_identity(),
            'actual_compute_observed': True, 'grad_epilogue': grad,
            'norm': norm, 'cast': cast, 'rollout': rollout, 'critic': critic,
            'precision': 'CURRENT_GENERATED_POLICY_REQUIRES_MATCHED_CURRENT_CONTROL',
            'historical_time_used_as_cost': False}
