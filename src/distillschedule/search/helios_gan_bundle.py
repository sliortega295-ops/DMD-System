"""Explicit manual GAN incumbent cohort, never an automatic numerical promotion.

The pinned FA3/down40 implementation has historical diagnostic timings, not a
fresh cost or a strict-parity receipt. Missing mechanisms reject the whole tuple.
"""
from dataclasses import dataclass
from pathlib import Path

from distillschedule.workloads.best_known.registry import file_sha256, repository_root
from .contracts import require
from .helios_safe_bundle import (PINNED_HELPERS, LORA_COUNTERS, NORM_COUNTERS,
                                B_GEMM_IMPLEMENTATION, _counter)

GAN_BUNDLE_ID = 'gan-fa3-down40-nomerge-v1'
GAN_BUNDLE_ENV = 'DISTILLSCHEDULE_HELIOS_GAN_MANUAL_BUNDLE'
SOURCE_HEAD = 'ba52a2446b7f090cd31af9504679e68770d32603'
ROW_SHA256 = '924e758545a97e6e7e7de1ae7ad3db6733b00a7a6f4797c86ebc4699e886229b'
RUNTIME_DIR = 'src/distillschedule/runtime/helios_gan_manual/'
SOURCE_FILES = tuple(RUNTIME_DIR + p for p in (
    '__init__.py', 'runtime.py', 'backend_library.py', 'backend_abi.py',
    'down_contract.py', 'down_replay.py', 'down40_rows.json', 'fa3_pinned_interface.raw',
)) + tuple('src/distillschedule/runtime/' + p for p in PINNED_HELPERS) + (
    'src/distillschedule/runtime/fsdp2_static.py',
    'src/distillschedule/runtime/fsdp2_p0.py',
    'src/distillschedule/runtime/fsdp2_gan_head_groups.py',
    'src/distillschedule/runtime/helios_planner_cycle.py',
    'src/distillschedule/search/helios_gan_bundle.py',
    'src/distillschedule/search/helios_implementation.py',
    'src/distillschedule/search/helios_action_costs.py',
    'scripts/distillschedule/probe_static_fsdp2_p0_world16.py',
    'scripts/distillschedule/probe_static_fsdp2_p0_world16_hardened.py',
    'scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh',
)
LORA_INVENTORY = {
    'generator': {'linear': 406, 'conv3d_names': ['patch_long', 'patch_mid', 'patch_short']},
    'critic': {'linear': 406, 'conv3d_names': []},
}
LORA_LAYER_COUNT = sum(v['linear'] + len(v['conv3d_names']) for v in LORA_INVENTORY.values())


@dataclass(frozen=True)
class HeliosGANBundle:
    enabled: bool = False

    def __post_init__(self):
        require(type(self.enabled) is bool, 'GAN manual opt-in must be boolean')

    @property
    def id(self):
        return GAN_BUNDLE_ID if self.enabled else 'none'

    @classmethod
    def manual(cls):
        return cls(True)

    def validate_scope(self, impl):
        require(self.enabled, 'use legacy empty bundle for the off path')
        require(impl.request is None and (
            impl.variant, impl.mesh, impl.chunk_tokens, impl.rollout_window,
            impl.persistent_unshard) == ('gan', 'world16-fsdp', 2048, 5, True),
            'GAN manual bundle requires canonical GAN/FSDP/C2048/p1/w5; no new shape or H100 admission')

    def environment(self):
        require(self.enabled, 'GAN manual bundle is explicit opt-in')
        return {
            GAN_BUNDLE_ENV: GAN_BUNDLE_ID,
            'DISTILLSCHEDULE_HELIOS_SELECTIVE_FLASH_CHECKPOINT_STRIDE': '2',
            'DISTILLSCHEDULE_HELIOS_LORA_B_GEMM_EPILOGUE': 'on',
            'DISTILLSCHEDULE_HELIOS_LORA_CHECKPOINT_REPLAY_SCOPE': 'on',
            'DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_MERGED_EXECUTION': 'off',
        }

    def source_identity(self, *, root: Path | None = None):
        root = root or repository_root()
        for name, expected in PINNED_HELPERS.items():
            require(file_sha256(root / 'src/distillschedule/runtime' / name) == expected,
                    'manual GAN helper drift: ' + name)
        require(file_sha256(root / RUNTIME_DIR / 'down40_rows.json') == ROW_SHA256,
                'manual GAN source-attested row table changed')
        return dict(authoritative_source_head=SOURCE_HEAD,
                    files={name: file_sha256(root / name) for name in SOURCE_FILES})

    def cost_identity(self, *, root=None):
        return dict(id=self.id, source=self.source_identity(root=root),
                    backend='pinned-sm80-fa3', critic_attention_stride=2,
                    frozen_down_replay_blocks=40, separate_gan_head_fsdp=True,
                    merged_lora=False, numerical_status='REVIEW_REQUIRES_FRESH_COMPARISON')

    def bind_effective(self, *, args, policy):
        require(args.variant == 'gan' and args.lora_checkpoint_replay_scope
                and policy.lora_b_gemm_epilogue and policy.selective_flash_checkpoint_stride == 2,
                'manual GAN effective checkpoint/LoRA controls differ')
        return self.cost_identity()

    def effect_snapshot(self, build):
        actual = build.get('gan_manual_runtime', {})
        require(actual.get('installed') is True and actual.get('effective_attention_backend') == 'pinned-sm80-fa3'
                and actual.get('unsupported_request_count') == 0,
                'manual GAN actual pinned backend not installed or unsupported request')
        from distillschedule.runtime.helios_gan_manual.backend_library import FA3_PINS
        identity = actual.get('actual_FA3_identity', {})
        require(identity.get('FA3_files_sha256') == FA3_PINS
                and identity.get('original_loaded_public_and_autograd_bytecode_attested') is True
                and identity.get('actual_mapped_FA3_library', {}).get('sha256') ==
                    FA3_PINS['_flash_attn3_cuda_373a45c.abi3.so'],
                'manual GAN actual public/autograd/library identity not attested')
        require(actual.get('merge_count') == 0 and actual.get('merged_layer_count') == 0
                and actual.get('lora_layer_count') == LORA_LAYER_COUNT
                and actual.get('lora_inventory') == LORA_INVENTORY
                and actual.get('merge_attempt_count') == 0 and actual.get('guarded_merge_methods') == LORA_LAYER_COUNT,
                'manual GAN merged execution or shadow is forbidden')
        require(build.get('critic', {}).get('fsdp2', {}).get('separate_gan_head_fsdp', {}).get('requested') is True,
                'manual GAN five independent head FSDP units missing')
        heads = build['critic']['fsdp2']['separate_gan_head_fsdp']
        require(heads.get('enabled') is True and heads.get('head_unit_count') == 5
                and heads.get('fsdp_unit_count') == 46 and heads.get('head_paths') ==
                    ['gan_heads.5','gan_heads.15','gan_heads.25','gan_heads.35','gan_final_head'],
                'manual GAN actual five-head placement census differs')
        checkpoint = build.get('critic', {}).get('checkpoint_policy', {})
        require(checkpoint.get('selective_flash_stride') == 2
                and checkpoint.get('selective_flash_op') == '_flash_attn3_cuda_373a45c._flash_attn_forward.default'
                and checkpoint.get('selective_flash_block_indices') == list(range(0,40,2)),
                'manual GAN actual Critic FA3 replay checkpoint selection differs')
        down = actual.get('down40', {})
        require(not down.get('audit_errors'), 'manual GAN replay audit failed')
        values = dict(public_backend_call_count=actual.get('public_backend_call_count'),
                      down_original=down.get('extra_down', {}).get('original'),
                      down_replay=down.get('extra_down', {}).get('replay'))
        require(all(type(v) is int and v >= 0 for v in values.values()),
                'manual GAN runtime call counters missing')
        for role in ('generator','critic'):
            lora = build.get(role, {}).get('all_role_lora_scale1_add', {})
            require(lora.get('enabled') is True and lora.get('scope') == 'ALL_ROLE_SCALE1_MODEL_FORWARD'
                    and lora.get('instrumented_module_count') == 406
                    and lora.get('checkpoint_replay_scope') is True
                    and lora.get('lora_b_gemm_epilogue_requested') is True
                    and lora.get('lora_b_gemm_epilogue_enabled') is True
                    and lora.get('checkpoint_missing_active_scope_count') == 0
                    and lora.get('lora_b_gemm_runtime_identity', {}).get('implementation') == B_GEMM_IMPLEMENTATION,
                    'manual GAN actual LoRA-B/replay mechanisms missing: ' + role)
            require(lora.get('checkpoint_scope_execution_count') == lora.get('checkpoint_scope_restore_count'),
                    'manual GAN checkpoint scope was not restored')
            values.update({role + '.' + key: _counter(lora,key) for key in LORA_COUNTERS})
            shapes = lora.get('lora_b_gemm_selected_shape_counts', {})
            wanted_shapes = ('M11040_N13824_R256','M11040_N5120_R256') + (
                ('M13680_N5120_R256',) if role == 'critic' else ())
            require(set(shapes) <= set(wanted_shapes), 'manual GAN unqualified selected LoRA shape')
            for shape in wanted_shapes:
                value = shapes.get(shape,0)
                require(type(value) is int and value >= 0, 'manual GAN shape counter invalid')
                values[role+'.shape.'+shape] = value
        generator_checkpoint = build['generator'].get('checkpoint_policy', {})
        require(generator_checkpoint.get('selective_flash_stride') == 0
                and generator_checkpoint.get('selective_flash_block_indices') == [],
                'manual GAN Generator must retain native recompute')
        for key in ('checkpoint_forward_count','selective_flash_context_count',
                    'selective_flash_original_save_count','selective_flash_recompute_reuse_count'):
            values['critic_checkpoint.' + key] = _counter(checkpoint,key)
        norm = build['generator'].get('exact_modulated_norm_epilogue', {})
        require(all(norm.get(k) is True for k in ('enabled','requested','materialization_requested',
                    'backward_requested','layernorm_requested','backward_kernel_requested'))
                and norm.get('reference_fallback_count') == 0
                and norm.get('backward_kernel_reference_fallback_count') == 0,
                'manual GAN exact norm mechanisms missing/fallback')
        values.update({'norm.' + key: _counter(norm,key) for key in NORM_COUNTERS})
        history = build['generator'].get('exact_history_current_fusion', {})
        kv = build['generator'].get('generator_nograd_cross_kv_cache', {})
        residual = build['generator'].get('generator_nograd_residual_fusion', {})
        require(history.get('enabled') is True and history.get('reference_fallback_count') == 0
                and kv.get('enabled') is True and kv.get('cross_rollout_reuse') is False
                and all(kv.get(k) == 0 for k in ('cache_miss_count','cached_tensor_mutation_count',
                    'encoder_signature_mismatch_count','source_encoder_identity_mismatch_count',
                    'missing_source_encoder_count','unexpected_cross_attention_count'))
                and residual.get('enabled') is True and residual.get('exact_grad_requested') is True
                and residual.get('reference_fallback_count') == residual.get('grad_reference_fallback_count') == 0,
                'manual GAN history/KV/residual mechanisms missing or fallback')
        for prefix, owner, keys in (
                ('history',history,('call_count',)),
                ('kv',kv,('cache_fill_count','cache_reuse_count','rollout_scope_count','model_call_count')),
                ('residual',residual,('fused_call_count','grad_fused_call_count','grad_backward_call_count'))):
            values.update({prefix+'.'+key:_counter(owner,key) for key in keys})
        return values

    def executed_effects(self, build, before):
        return self.effects_between(before, self.effect_snapshot(build))

    def effects_between(self, before, after):
        expected = dict(public_backend_call_count=4720, down_original=400, down_replay=400)
        for role, selected, grad, nograd, wraps, scope in (
                ('generator',2400,480,1920,132,263), ('critic',4640,4400,240,514,985)):
            expected.update({role + '.' + k:selected for k in (
                'lora_b_gemm_selected_invocation_count','lora_b_gemm_selected_attempt_count',
                'lora_b_gemm_completed_invocation_count','lora_b_gemm_kernel_call_count')})
            expected.update({role+'.lora_b_gemm_selected_grad_invocation_count':grad,
                             role+'.lora_b_gemm_selected_nograd_invocation_count':nograd,
                             role+'.checkpoint_wrap_count':wraps,
                             role+'.checkpoint_scope_execution_count':scope,
                             role+'.checkpoint_scope_restore_count':scope})
        expected.update({'critic_checkpoint.' + k:v for k,v in (
            ('checkpoint_forward_count',440),('selective_flash_context_count',220),
            ('selective_flash_original_save_count',440),('selective_flash_recompute_reuse_count',440))})
        expected.update({'norm.'+k:(1120 if k.startswith('backward') else 4720) for k in NORM_COUNTERS})
        expected.update({'history.call_count':2360, 'kv.cache_fill_count':200,
            'kv.cache_reuse_count':880, 'kv.rollout_scope_count':5, 'kv.model_call_count':27,
            'residual.fused_call_count':2160, 'residual.grad_fused_call_count':2240,
            'residual.grad_backward_call_count':1120})
        expected.update({'generator.shape.M11040_N13824_R256':400,
            'generator.shape.M11040_N5120_R256':2000,
            'critic.shape.M11040_N13824_R256':440,
            'critic.shape.M11040_N5120_R256':2200,
            'critic.shape.M13680_N5120_R256':2000})
        inventory = set(expected) | {role+'.'+k for role in ('generator','critic') for k in LORA_COUNTERS}
        inventory |= {'norm.'+k for k in NORM_COUNTERS}
        require(set(before) == set(after) == inventory, 'manual GAN effect inventory differs')
        require(all(type(x) is int and x >= 0 for row in (before, after) for x in row.values()),
                'manual GAN actual effect values invalid')
        delta = {k: after[k] - before[k] for k in inventory}
        cycles = delta['down_original'] // 400
        require(cycles in (1, 2) and all(delta[k] == v * cycles for k,v in expected.items())
                and all(v > 0 for v in delta.values()),
                'manual GAN mechanisms did not execute the exact observed cycle inventory')
        return dict(status='PASS_EXECUTED_GAN_MANUAL_BUNDLE', **self.cost_identity(),
                    before=before, after=after, call_deltas=delta,
                    scope='ACTUAL_NATIVE_COUNTERS_NOT_NUMERICAL_PARITY')
