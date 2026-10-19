"""Versioned W1 four-cycle factorial policy, not a canonical workload override.

C1 is an observed semantic/mechanism gate within the same allocation. It is
not cross-arm numerical parity; R2 timings remain descriptive diagnostics.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

SCHEMA = 'distillschedule.wan13b_factorial_r2.v1'
PARENT = '45c4940d1ddba89bc65fd6080483296f511916cc'
FSDP = 'world16-fsdp'
RESIDENT = 'resident-async128-tree-chunk2048'
HELIOS_SHA256 = '5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886'
CYCLES = 4
INITIAL = {'generator': 2, 'critic': 10, 'ema': 10}
UPDATES = {'generator': 1, 'critic': 5, 'ema': 5}
SOURCE_PATHS = (
    'src/distillschedule/runtime/wan13b_factorial_r2.py',
    'scripts/distillschedule/probe_wan13b_factorial_r2_world16.py',
    'scripts/distillschedule/summarize_wan13b_factorial_r2_world16.py',
    'scripts/distillschedule/run_wan13b_fair_world16.py',
    'scripts/distillschedule/launch_wan13b_fair_world16_node.sh',
    'scripts/distillschedule/probe_wan13b_two_cycle_timing_world16.py',
    'scripts/distillschedule/probe_wan13b_fair_world16.py',
    'scripts/distillschedule/probe_wan13b_upper_bound_world16.py',
    'scripts/distillschedule/probe_static_fsdp2_p0_world16.py',
    'scripts/distillschedule/probe_static_fsdp2_full5_world16.py',
    'src/distillschedule/runtime/wan13b_fair.py',
    'src/distillschedule/runtime/fsdp2_p0.py',
    'src/distillschedule/runtime/wan13b_lora_cast.py',
    'src/distillschedule/runtime/wan13b_lora_grad_epilogue.py',
    'src/distillschedule/runtime/wan13b_norm_policy.py',
    'src/distillschedule/runtime/wan13b_rollout_cuda_graph.py',
    'src/distillschedule/runtime/wan13b_critic_grad_graph.py',
    'src/distillschedule/runtime/wan13b_cast_graph_composition.py',
    'runtime/gst_pool_helios.py',
    'runtime/static_world16_trace.py',
    'docs/contracts/wan13b_factorial_r2.md',
)
FLAGS = (
    'generator_nograd_lora_fused_epilogue', 'grad_lora_fused_epilogue',
    'modulated_norm_fusion', 'generator_rollout_cuda_graph',
    'critic_grad_cuda_graph', 'generator_nograd_lora_cast_cache',
    'score_cuda_graph', 'fused_input_lora_a', 'generator_rollout_merged_shadow',
)
MATCHED = {
    'bootstrap_hash': '6a713476d9839cf1ea37689d63cb1cc2b405b9f9915ca5314ccdadb67b3cd60d',
    'model_identity_sha256': 'd52cf9e2d04451c018ac50c4ee1043c4ceba4014e51549bfda2ea5a123c65f54',
    'trace_bundle_hash': '654d49015921b278ab82e2fc84ea1896692da3fd38ce82fbc500cb1ebfec46d3',
    'trace_binding_hash': 'bddcf32b379ea3d0cb36237b0c21f103ea59b1635473dd8080b7182cf3c1eded',
    'negative_prompt_manifest_hash': 'ca0de6928f83ebd450418d33f8990696763d3228069ec4ac0efb2444694cf78c',
    'trace_rollout_contract_sha256': 'd4964217e77fed2434feffbfc0c16b80a6bb9fb7b6adbdd9d9172a6ee0f2f6f8',
    'trace_replay_policy': 'FIXED_FIVE_STEP_COHORT_REPLAY',
    'seed': 20260903, 'global_batch': 16, 'rollout_batch_size': 1,
    'score_packing': False, 'updates': 'G1_C5_EMA5',
    'tf32_cublas_override': '1', 'float32_matmul_precision': 'high',
    'cuda_matmul_allow_tf32': 'true', 'cudnn_allow_tf32': 'true',
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_PATHS}


def make_configuration(cell: str, *, expected_head: str, source_hashes: Mapping[str, str]) -> dict[str, Any]:
    if cell == 'only_B':
        raise ValueError('only_B BLOCKED: user definition pending; full graphs/cast cache require Resident addresses')
    if cell not in {'baseline', 'only_A', 'full'}:
        raise ValueError('unregistered W1 R2 cell')
    b = cell == 'full'
    return {
        'schema_version': SCHEMA, 'cell': cell,
        'source': {'audited_parent': PARENT, 'expected_head': expected_head,
                   'files_sha256': dict(source_hashes)},
        'placement': FSDP if cell == 'baseline' else RESIDENT,
        'expected_helios_transformer_sha256': HELIOS_SHA256,
        'cycles': {'count': 4, 'warmup': [1], 'measurement': [2, 3, 4]},
        'geometry': {'video_height': 384, 'video_width': 640, 'video_frames': 33,
                     'latent_spatial_shapes': [[12, 20], [24, 40], [48, 80]], 'lora_rank': 256},
        'matched_contract': dict(MATCHED),
        'initial_versions': dict(INITIAL), 'updates_per_cycle': dict(UPDATES),
        'operator_flags': {name: b and index < 6 for index, name in enumerate(FLAGS)},
        'runtime_policy': {'clean_runtime': b, 'runtime_constants_cache': b,
                           'resident_clean_model_state': b},
        'optimizer_implementation': 'PYTORCH_DEFAULT', 'ema_implementation': 'reference',
        'retained_blocks': [], 'torch_compile': False,
        'qualification': 'C1_OBSERVED_SOURCE_INPUT_FINITE_CENSUS_VERSION_MECHANISM',
        'numerical_status': 'NOT_RUN', 'formal_timing_eligible': False,
    }


def validate_configuration(value: dict[str, Any], *, expected_head: str) -> None:
    source = value.get('source', {})
    hashes = source.get('files_sha256', {})
    if not re.fullmatch('[0-9a-f]{40}', expected_head) or expected_head == PARENT:
        raise ValueError('R2 requires a new, full reviewed source HEAD, not the audited parent')
    if set(hashes) != set(SOURCE_PATHS) or any(not re.fullmatch('[0-9a-f]{64}', str(h)) for h in hashes.values()):
        raise ValueError('R2 source file hash inventory changed')
    expected = make_configuration(value.get('cell', ''), expected_head=expected_head, source_hashes=hashes)
    if digest(value) != digest(expected):
        raise ValueError('W1 R2 exact configuration/source/arm/workload contract changed')


def load_configuration(path: Path, *, expected_hash: str, expected_head: str,
                       root: Path | None = None, verify_git: bool = True) -> dict[str, Any]:
    raw = path.read_bytes()
    if not re.fullmatch('[0-9a-f]{64}', expected_hash) or hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError('R2 configuration SHA-256 changed')
    value = json.loads(raw)
    validate_configuration(value, expected_head=expected_head)
    if root is not None:
        if source_hashes(root) != value['source']['files_sha256']:
            raise ValueError('R2 bound executable/source hash changed')
        if verify_git:
            head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
            dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=all'], text=True)
            if head != expected_head or dirty:
                raise ValueError('R2 source HEAD/clean state mismatch')
            subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', PARENT, head], check=True)
    return value


def lower_environment(cfg: dict[str, Any]) -> dict[str, str]:
    validate_configuration(cfg, expected_head=cfg['source']['expected_head'])
    result = {'DISTILLSCHEDULE_WAN13B_' + name.upper(): '1' if enabled else '0'
              for name, enabled in cfg['operator_flags'].items()}
    result.update(TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='1',
                  DISTILLSCHEDULE_WAN13B_EMA_IMPLEMENTATION='reference',
                  DISTILLSCHEDULE_WAN13B_LAYOUT_ONLY_ABLATION='1' if cfg['cell'] == 'only_A' else '0',
                  DISTILLSCHEDULE_WAN13B_LORA_OPERATOR_ATTRIBUTION='0')
    return result


def assert_runtime_environment(cfg: dict[str, Any], environ: Mapping[str, str] | None = None) -> None:
    env = os.environ if environ is None else environ
    for key, expected in lower_environment(cfg).items():
        if env.get(key) != expected:
            raise ValueError(f'R2 lowered runtime flag changed: {key}')
    if env.get('NCCL_ALGO') != ('Tree' if cfg['placement'] == RESIDENT else None):
        raise ValueError('R2 NCCL algorithm must be Tree or explicitly unset for FSDP')
    if env.get('DISTILLSCHEDULE_STATIC_FSDP2_NSYS_EXACT_RANGE', '0') not in {'', '0'}:
        raise ValueError('R2 native timer forbids profiling')
    if env.get('DISTILLSCHEDULE_WAN13B_COMPILE_CACHE_ROOT'):
        raise ValueError('R2 does not admit torch.compile/retention winners')


def builder_clean_model_state(args: Any, *, legacy_requested: bool) -> bool:
    """Only the versioned caller can override the historical placement bundle."""
    cfg = getattr(args, 'factorial_configuration', None)
    if cfg is None:
        return legacy_requested
    validate_configuration(cfg, expected_head=args.expected_head)
    if args.placement != cfg['placement']:
        raise ValueError('R2 builder placement differs from bound configuration')
    assert_runtime_environment(cfg)
    return cfg['runtime_policy']['resident_clean_model_state']


def expected_versions(cycle: int) -> dict[str, int]:
    return {key: INITIAL[key] + cycle * UPDATES[key] for key in INITIAL}


def validate_mechanisms(legacy: Any, build: dict[str, Any], cfg: dict[str, Any], *, cycles: int) -> dict[str, Any]:
    """Validate real mutable installer counters, never config-derived counters."""
    from .wan13b_lora_grad_epilogue import validate_grad_epilogue_receipts
    from .wan13b_norm_policy import validate as validate_norm
    from .wan13b_rollout_cuda_graph import validate as validate_graph
    from .wan13b_rollout_shadow import validate as validate_shadow
    from .wan13b_cast_graph_composition import validate_build as validate_cast
    from .wan13b_critic_grad_graph import validate_build as validate_critic
    flags = cfg['operator_flags']
    result = {
        'fused_epilogue': legacy._validate_fused_receipt(build, placement=cfg['placement'],
                        cycles=cycles, layout_only=cfg['cell'] == 'only_A'),
        'grad_epilogue': validate_grad_epilogue_receipts(build, requested=flags['grad_lora_fused_epilogue']),
        'norm': validate_norm(build, enabled=flags['modulated_norm_fusion']),
        'graph': validate_graph(build, enabled=flags['generator_rollout_cuda_graph'], cycles=cycles),
        'shadow': validate_shadow(build, enabled=False, cycles=cycles),
        'cast': validate_cast(build, enabled=flags['generator_nograd_lora_cast_cache']),
        'critic_graph': validate_critic(build, cycles=cycles, reference_checks=False,
                                       enabled=flags['critic_grad_cuda_graph']),
    }
    for role in ('generator', 'critic'):
        component = build[role]
        metadata, placement, engine = (component.get(k, {}) for k in ('metadata', 'placement', 'engine'))
        if metadata.get('score_cuda_graph', {}).get('enabled') or metadata.get('nograd_compile_runtime', {}).get('enabled'):
            raise AssertionError(f'{role}: unrequested score graph/compile executed')
        if bool(component.get('resident_clean_model_state_initial')) != cfg['runtime_policy']['resident_clean_model_state']:
            raise AssertionError(f'{role}: optional model-state hot path differs from B policy')
        resident = cfg['placement'] == RESIDENT
        if placement.get('backend') != ('FULL_REPLICA_GPU_RESIDENT_MANUAL_GRAD_SYNC' if resident else 'FSDP2_COMPOSABLE_FULL_SHARD'):
            raise AssertionError(f'{role}: actual placement backend changed')
        if placement.get('reshard_after_forward') is not (not resident):
            raise AssertionError(f'{role}: actual FSDP reshard policy changed')
        if engine.get('optimizer_implementation') != 'PYTORCH_DEFAULT':
            raise AssertionError(f'{role}: optimizer implementation changed')
        if not resident and engine.get('gradient_sync') != 'FSDP2_SUM_THEN_DIVIDE_REDUCE_SCATTER':
            raise AssertionError(f'{role}: original world16 FSDP reduction changed')
        modulation = metadata.get('modulation_chunk_runtime', {})
        if (modulation.get('effective_tokens') != [2048 if resident else 512]
                or modulation.get('block_count') != 30 or modulation.get('all_blocks_match') is not True):
            raise AssertionError(f'{role}: original geometry/chunk/block census changed')
        if resident and (engine.get('async_gradient_bucket_mb') != 128 or
                         engine.get('gradient_sync') != 'WORLD16_GRAD_READY_ASYNC_SUM_THEN_DIVIDE_ALL_REDUCE'):
            raise AssertionError(f'{role}: complete A async128 reducer changed')
    return result


def validate_census(census: dict[str, Any], *, cycle: int, rollout_contract: list[dict[str, Any]]) -> None:
    from .fsdp2_p0 import validate_exact_rollout_call_inventory
    required = {'generator_updates': 1, 'critic_updates': 5, 'ema_updates': 5,
                'generator_score_forwards': 3, 'critic_score_forwards': 5,
                'physical_rollout_batches': 5, 'rollout_batch_size': 1}
    if any(type(census.get(k)) is not int or census[k] != v for k, v in required.items()):
        raise AssertionError('R2 observed update/score/batch census changed')
    if digest(rollout_contract) != MATCHED['trace_rollout_contract_sha256']:
        raise AssertionError('R2 frozen trace rollout contract changed')
    if census.get('rollout_operation_keys') != [row['operation'] for row in rollout_contract]:
        raise AssertionError('R2 rollout operation order changed')
    calls = census.get('rollout_call_inventory', [])
    if len(calls) != 6:
        raise AssertionError('R2 rollout call census missing')
    for call, row in zip(calls, rollout_contract, strict=True):
        normalized = validate_exact_rollout_call_inventory(call, label='R2 observed rollout')
        if digest(normalized) != digest(row['inventory']):
            raise AssertionError('R2 trace-derived DiT census changed')
    identities = census.get('sample_payload_identities', [])
    if len(identities) != 6 or any(not re.fullmatch('[0-9a-f]{64}', str(v)) for v in identities):
        raise AssertionError('R2 consumed payload identities missing')
    if identities != [census.get('generator_trace_payload_identity'), *census.get('critic_trace_payload_identities', [])]:
        raise AssertionError('R2 Generator/Critic payload role routing changed')
    if census.get('sample_occurrence_identities') != [
            f'cycle={cycle}/slot={slot}/payload={v}' for slot, v in enumerate(identities)]:
        raise AssertionError('R2 cycle-qualified payload occurrences changed')


def qualify_c1(observations: list[dict[str, Any]], cfg: dict[str, Any], *,
               run_id: str, rollout_contract: list[dict[str, Any]]) -> dict[str, Any]:
    if len(observations) != 16 or [row.get('rank') for row in observations] != list(range(16)):
        raise AssertionError('C1 qualification requires every ordered world16 rank')
    payloads = []
    for row in observations:
        expected = {'run_id': run_id, 'configuration_fingerprint': digest(cfg),
                    'expected_head': cfg['source']['expected_head'], 'matched_contract': MATCHED,
                    'finite': True, 'versions': expected_versions(1),
                    'mechanism_status': 'OBSERVED_VALIDATED', 'numerical_status': 'NOT_RUN'}
        if any(digest(row.get(k)) != digest(v) for k, v in expected.items()):
            raise AssertionError('C1 observed source/input/finite/version/mechanism binding changed')
        if not re.fullmatch('[0-9a-f]{64}', str(row.get('mechanism_fingerprint', ''))):
            raise AssertionError('C1 actual mechanism fingerprint missing')
        validate_census(row['census'], cycle=1, rollout_contract=rollout_contract)
        payloads.extend(row['census']['sample_payload_identities'])
    if len(set(payloads)) != 96:
        raise AssertionError('C1 requires 96 distinct consumed rank-qualified payloads')
    return {'schema_version': SCHEMA + '.c1', 'status': 'OBSERVED_SEMANTIC_PASS',
            'scope': cfg['qualification'], 'rank_count': 16,
            'observations': observations, 'observations_sha256': digest(observations),
            'numerical_status': 'NOT_RUN', 'formal_timing_eligible': False}
