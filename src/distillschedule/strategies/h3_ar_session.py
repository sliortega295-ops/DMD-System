"""Persistent native H3 session, using original losses/sampler/ZeRO3 step.

The planner owns only an admitted implementation and whole-update dispatch.
It does not lower opaque native stages to fictitious independent G/F costs.
Imports and model construction occur only after worker admission.
"""
from collections import Counter
import contextlib
import copy
import hashlib
import json
import math
from pathlib import Path

from distillschedule.core.contracts import require
from distillschedule.runtime.role_runner import StaticRoleRunner
from distillschedule.runtime.version_sync import VersionLedger
from .h3_ar_task import native_config, native_graph
from .h3_ar_assets import checked
from .h3_ar_inputs import prepare_rank_input


def deepspeed_config_receipt(config):
    """Snapshot metadata only; keep Accelerate's disabled-print infinity live.

    Accelerate mutates the shared effective config during engine preparation.
    Only its known logging sentinel is tagged; every other nonfinite value
    remains a strict JSON error, never a silently accepted training metric.
    """
    result = copy.deepcopy(config)
    value = result.get('steps_per_print')
    if type(value) is float and value == math.inf:
        result['steps_per_print'] = dict(value_type='float', value='+Infinity', meaning='logging_disabled')
    json.dumps(result, allow_nan=False)
    return result


def move(value, device, torch):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device, torch) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(v, device, torch) for v in value)
    return value


def configure_native_execution(implementation, config):
    """Consume plan choices in the ORIGINAL native DeepSpeed config object."""
    require(config['zero_optimization']['stage'] == 3 and config['zero_optimization']['overlap_comm'] is False,
            'original native ZeRO3 communication control drifted', path='h3_ar.session')
    if implementation.optimized:
        config['zero_optimization'].update(overlap_comm=True, stage3_prefetch_bucket_size=657499392,
                                           stage3_max_live_parameters=1400000000)
    return config


def install_native_attention_cache(implementation, assets, native, module, adapter, trainer):
    if not implementation.optimized:
        return None, None, None
    epoch = assets.helper('native_selected_cycle_scope_r1')
    pool, view = epoch.install(native, module, adapter, trainer)
    return epoch, pool, view


class H3ARSession:
    def __init__(self, task, assets, implementation, *, rank, local_rank, torch, accelerator,
                 input_provider=None, seed=42):
        self.task, self.assets, self.implementation = task, assets, implementation
        if input_provider is not None:
            from distillschedule.training.h3_ar_inputs import H3ARInputStream
            require(isinstance(input_provider, H3ARInputStream) and seed == input_provider.seed,
                    'explicit finite W6 input owner/seed required', path='h3_ar.session')
        else:
            require(type(seed) is int and seed == 42, 'legacy native seed is immutable', path='h3_ar.session')
        self.input_provider = input_provider
        self.segment_origin, self.local_completed, self.pool_ready = 0, 0, False
        self.rank, self.torch, self.device = rank, torch, torch.device('cuda', local_rank)
        self.ledger = VersionLedger(dict(student=0, fake=0, ema=0))
        self.census, self.updates, self.adam = Counter(), Counter(), Counter()
        self.native = assets.source_module('h3_dmd2')
        training = assets.source_module('train_dmd2')
        zero = assets.source_module('dmd2_zero3')
        module = assets.source_module('diffsynth.models.minimax_h3_dit')
        self.config = self.native.DMD2Config(**native_config())
        # The ORIGINAL deterministic parser runs before the large construction:
        # reject wrong history/geometry/masks without loading a 50-block model.
        prepare_input = prepare_rank_input if input_provider is None else input_provider.prepare_rank_input
        positive, negative, self.input_identity, input_validation = prepare_input(
            task, assets.cache_manifest, self.native, rank, torch, teacher_cfg_scale=self.config.teacher_cfg_scale)
        shapes = input_validation['input_shapes']
        self.mechanism = assets.helper('native_mechanisms_r3').NativeMechanisms(lora_inplace=implementation.optimized)
        self.mechanism.install(module)
        replay = assets.helper('zero3_checkpoint_tensor_identity_r2').install()
        self.replay_compatibility = replay
        self.build = dict(checkpoint_compatibility={k: v for k, v in replay.items() if k not in ('counters', 'original_class')},
                          implementation=implementation.to_dict(), native_source='ORIGINAL_DIFFSYNTH_NOT_GRID')
        if implementation.optimized:
            torch.ops.load_library(str(checked(assets.libraries['pinned'])))
            require(list(torch.ops.h3_native_pinned_cfg_r1.metadata()) == [1, 4, 1, 0],
                    'native host-register4 allocator not active', path='h3_ar.session')
            self.pointwise = assets.helper('native_pointwise_fusion_r1').install(module)
            self.swiglu = assets.helper('native_swiglu_fusion_r1').install(module, enabled=True)
            for name, key in (('native_rms_fusion_r3', 'rms'), ('native_cutlass_lora_r1', 'cutlass')):
                ref = assets.libraries[key]
                assets.helper(name).load_verified(str(checked(ref)), ref['sha256'])
        training.seed_all(seed)
        ds_config = configure_native_execution(implementation, zero.configure(accelerator, self.config))
        self.dit, architecture, checkpoint = training.load_teacher(Path(assets.checkpoint_root), self.device, ds_config=ds_config)
        require(len(self.dit.blocks) == 50, 'real native H3 50-block model required', path='h3_ar.session')
        self.adapter = self.native.H3Adapter(self.dit, self.config)
        self.discriminator = self.native.VelocityDiscriminator(self.config)
        self.leaf = self.pool = self.view = None
        self.rms = self.cutlass = None
        if implementation.optimized:
            self.leaf_api = assets.helper('zero3_block_leaf_r3')
            self.leaf = self.leaf_api.mark(self.dit, 50)
            for name, key in (('native_rms_fusion_r3', 'rms'), ('native_cutlass_lora_r1', 'cutlass')):
                ref = assets.libraries[key]
                state = assets.helper(name).install(self.adapter, module, enabled=True, allow_tiny=False,
                                                    library=ref['path'], library_sha=ref['sha256'])
                setattr(self, key, state)
        self.engine, self.optimizer = zero.initialize(accelerator, self.adapter, self.discriminator, self.config)
        require(self.engine.zero_optimization_stage() == 3, 'actual engine not ZeRO3', path='h3_ar.session')
        streams = [v for k, v in vars(self.engine.optimizer.parameter_offload).items() if k.endswith('__allgather_stream')]
        require(len(streams) == 1 and (streams[0].cuda_stream != torch.cuda.default_stream(self.device).cuda_stream)
                == implementation.optimized, 'native communication branch not installed', path='h3_ar.session')
        if implementation.optimized:
            self.leaf_api.attach_runtime_audit(self.dit, self.leaf)
            self.leaf_api.install_replay_lifetime(module, self.engine, self.leaf)
            self.prefetch_api = assets.helper('zero3_unit_prefetch_audit_r1')
            self.prefetch = self.prefetch_api.install(self.engine)
            require(all(b['logical_parameter_numel'] == 657499392 for b in self.leaf['leaf_blocks']),
                    'native prefetch block identity changed', path='h3_ar.session')
        old_adam = self.optimizer.step

        def observed_adam(*args, **kwargs):
            role = self.optimizer.active_role
            groups = [g for g in self.optimizer.param_groups if g['role'] == role
                      and not (g.get('head', False) and not self.optimizer.train_heads)
                      and any(p.grad is not None for p in g['params'])]
            result = old_adam(*args, **kwargs)
            self.adam[role] += len(groups)
            return result

        self.optimizer.step = observed_adam
        native_predict = self.adapter.predict

        def predict(*args, **kwargs):
            roles = {layer.role for _, layer in self.adapter.layers}
            require(len(roles) == 1, 'native adapter role race', path='h3_ar.session')
            role = next(iter(roles)); self.census[role] += 1
            if torch.is_grad_enabled():
                self.census[role + '_grad'] += 1
            return native_predict(*args, **kwargs)

        self.adapter.predict = predict

        def step(role, loss):
            expected = 'student' if role == 'student' else 'fake_score'
            require(all(layer.role == expected for _, layer in self.adapter.layers), 'native backward adapter mismatch', path='h3_ar.session')
            previous = self.engine.global_steps
            zero.step(accelerator, self.engine, self.optimizer, self.discriminator, role, loss, self.config)
            require(self.engine.global_steps == previous + 1, 'native optimizer double/missing step', path='h3_ar.session')
            self.updates[role] += 1; self.census[role + '_backward'] += 1

        self.trainer = self.native.DMD2Trainer(self.adapter, self.discriminator, self.config, step)
        self.epoch, self.pool, self.view = install_native_attention_cache(
            implementation, assets, self.native, module, self.adapter, self.trainer)
        self.positive, self.negative = move(positive, self.device, torch), move(negative, self.device, torch)
        self.real = (self.positive.pop('input_latents'), self.positive.pop('audio_input_latents'))
        self.build.update(input_shapes=shapes, input_validation=input_validation, architecture=architecture, checkpoint=checkpoint,
                          native_implementation_installed=True)
        self.build['deepspeed_config'] = deepspeed_config_receipt(ds_config)
        training.seed_all(seed + rank)
        self.static_bytes = torch.cuda.memory_allocated(self.device)
        self.previous_leaf, self.previous_lifetime = Counter(), Counter()
        self.observer_baseline = self._observer_snapshot()

    def _observer_snapshot(self):
        if self.leaf is None:
            return {}
        return dict(swiglu=self.swiglu['counters'].copy(), cutlass=self.cutlass['counters'].copy(),
                    cutlass_operator=self.cutlass['operator_counters'].copy(), rms=self.rms['counters'].copy())

    def rebase_segment(self, cycle):
        """Only read-only epoch/observer origins change; never native clocks."""
        require(self.input_provider is not None and self.local_completed == 0
                and self.versions() == dict(student=cycle, fake=cycle, ema=0)
                and self.engine.global_steps == 2*cycle,
                'segment rebase requires restored complete native clocks', path='h3_ar.session')
        self.segment_origin = cycle
        if self.view is not None:
            require(self.pool.selected is None and self.view.origin == 0,
                    'restore must rebuild an empty cold native attention pool', path='h3_ar.session')
            self.view.origin = cycle
        self.ledger = VersionLedger(self.versions())
        self.observer_baseline = self._observer_snapshot()

    def drain_for_checkpoint(self):
        require(self.input_provider is not None and self.local_completed > 0,
                'only completed stream updates can drain', path='h3_ar.session')
        if self.pool is not None and not self.pool_ready:
            # Original helper proves every consumer/event is complete, advances
            # only its read-only origin, and drops metadata, not pinned storage.
            self.epoch.begin_next_cycle(self.pool, self.view)
            self.pool_ready = True

    def versions(self):
        return dict(student=self.trainer.generator_step, fake=self.trainer.critic_step, ema=0)

    def prepare(self, cycle, order=('native_update',)):
        require(self.versions() == dict(student=cycle-1, fake=cycle-1, ema=0),
                'actual native versions diverged before dispatch', path='h3_ar.session')
        if self.input_provider is not None:
            require(cycle == self.input_provider.completed+1 and cycle <= self.input_provider.cycles,
                    'fresh input cursor differs from native cycle', path='h3_ar.session')
            if self.local_completed and self.pool is not None and not self.pool_ready:
                self.epoch.begin_next_cycle(self.pool, self.view)
            self.pool_ready = False
            if self.input_identity.get('cycle') != cycle:
                positive, negative, self.input_identity, witness = self.input_provider.prepare_rank_input(
                    self.task, self.assets.cache_manifest, self.native, self.rank, self.torch,
                    teacher_cfg_scale=self.config.teacher_cfg_scale)
                self.positive, self.negative = move(positive, self.device, self.torch), move(negative, self.device, self.torch)
                self.real = (self.positive.pop('input_latents'), self.positive.pop('audio_input_latents'))
                self.build.update(input_validation=witness, input_shapes=witness['input_shapes'])
        elif cycle == 2 and self.pool is not None:
            self.epoch.begin_next_cycle(self.pool, self.view)
        self.before = (self.census.copy(), self.updates.copy(), self.adam.copy())
        self.metrics = None

        def update(context):
            require(context.operation.required_versions == self.versions(), 'native version mismatch', path='h3_ar.session')
            scope = self.leaf_api.replay_scope(True) if self.leaf is not None else contextlib.nullcontext()
            with scope, self.torch.autocast(device_type='cuda', dtype=self.torch.bfloat16, cache_enabled=False):
                self.metrics = self.trainer.update(self.real, self.positive, self.negative)
            require(self.versions() == {**context.operation.produced_versions, 'ema': 0},
                    'native version commit mismatch', path='h3_ar.session')
            return self.metrics

        graph = (native_graph(self.task, cycle=cycle) if self.input_provider is None
                 else self.input_provider.graph(self.task, cycle))
        return StaticRoleRunner('native_runtime', {'call': update}, versions=self.ledger).prepare(graph, order=tuple(order))

    def finish(self, cycle):
        census, updates, adam = (dict(a-b) for a, b in zip((self.census, self.updates, self.adam), self.before))
        require(census == self.task.census and updates == adam == dict(student=1, critic=1),
                'actual native forward/backward/Adam census mismatch', path='h3_ar.session')
        require(self.engine.global_steps == cycle*2 and not self.metrics['gan_active']
                and self.metrics['generator_step'] == self.metrics['critic_step'] == cycle,
                'native GAN/engine/update contract changed', path='h3_ar.session')
        cache = self.epoch.finish_cycle(self.pool, self.view) if self.pool is not None else None
        elapsed = self.local_completed+1 if self.input_provider is not None else cycle
        if cache is not None and self.input_provider is not None:
            historical = {key: cache[key] for key in ('cold_cache_staging_inside_timer', 'same_process_pinned_pool_reuse')}
            cache.update(source_helper_epoch_metadata=historical, segment_origin=self.segment_origin,
                process_local_cycle=elapsed, cold_cache_staging_inside_timer=elapsed == 1,
                same_process_pinned_pool_reuse=elapsed > 1)
        leaf = None
        backend_effects = dict(implementation=self.implementation.to_dict(), actual_compute_observed=True)
        if self.leaf is not None:
            view = dict(self.leaf)
            view['counters'] = self.leaf['counters'] - self.previous_leaf
            view['replay_lifetime_counters'] = self.leaf['replay_lifetime_counters'] - self.previous_lifetime
            view['replay_lifetime_counters']['max_pending_leaf_blocks'] = self.leaf['replay_lifetime_counters']['max_pending_leaf_blocks']
            leaf = self.leaf_api.finalize_lifetime(self.dit, view)
            self.previous_leaf = self.leaf['counters'].copy()
            self.previous_lifetime = self.leaf['replay_lifetime_counters'].copy()
            baseline = self.observer_baseline if self.input_provider is not None else {}
            delta = lambda family, key: (getattr(self, family)['counters'][key]
                - baseline.get(family, {}).get(key, 0))
            require(delta('swiglu', 'physical_swiglu_forward') == 724*elapsed
                    and delta('swiglu', 'physical_swiglu_backward') == 104*elapsed,
                    'native SwiGLU runtime census differs', path='h3_ar.session')
            require(delta('cutlass', 'physical_B_module_forward') == 2600*elapsed
                    and delta('cutlass', 'fused_custom_backward') == 400*elapsed
                    and self.cutlass['operator_counters']['physical_CUDA_B_GEMM_round_then_add']
                        - baseline.get('cutlass_operator', {}).get('physical_CUDA_B_GEMM_round_then_add', 0) == 2600*elapsed,
                    'native cutlass LoRA runtime/fallback differs', path='h3_ar.session')
            rc = self.rms['counters']; blocks = 50*elapsed
            expected = dict(physical_norm_module_forward=56*blocks, logical_mod_calls=28*blocks,
                logical_rope_calls=28*blocks, fused_mod_forward=28*blocks, fused_rope_forward=28*blocks,
                fused_mod_backward=4*blocks, fused_rope_backward=4*blocks)
            require(all(rc.get(k, 0)-baseline.get('rms', {}).get(k, 0) == v for k, v in expected.items())
                    and rc.get('TINY_EXPLICIT_NATIVE_FALLBACK', 0) == 0,
                    'native RMS runtime/fallback differs', path='h3_ar.session')
            pointwise = self.pointwise['counters']
            families = {n: sum(v for k, v in pointwise.items() if k.startswith(n))
                        for n in ('_modulate_scale_shift', '_modulate_gate', '_apply_rope')}
            families['_modulate_scale_shift'] += rc['logical_mod_calls']
            families['_apply_rope'] += rc['logical_rope_calls']
            require(all(v > 0 for v in families.values()) and self.mechanism.counters['lora_inplace_completed_sums'] > 0,
                    'native pointwise/LoRA runtime missing', path='h3_ar.session')
            backend_effects.update(pointwise=dict(pointwise), rms=dict(rc), swiglu=dict(self.swiglu['counters']),
                cutlass=dict(self.cutlass['counters']), unit_prefetch=self.prefetch_api.receipt(self.prefetch))
        require(self.replay_compatibility['counters']['detached_frozen_recompute_tensors'] > 0,
                'native checkpoint zero-copy replay compatibility not observed', path='h3_ar.session')
        if self.input_provider is not None:
            self.input_provider.commit(cycle)
            self.local_completed += 1
            backend_effects.update(segment_origin=self.segment_origin, process_local_completed=self.local_completed)
        return dict(census=census, updates=updates, adam_updates=adam, final_versions=self.versions(),
                    metrics=dict(self.metrics), selected_cache=cache, leaf_runtime=leaf,
                    backend_effects=backend_effects)

    def state_metadata(self, cycle):
        # Same boundary as the accepted worker: never in a training timer.
        torch = self.torch
        digest = lambda x: hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        masters = []
        for subgroup, param in enumerate(self.engine.optimizer.fp32_partitioned_groups_flat):
            group = self.optimizer.param_groups[self.engine.optimizer.sub_group_to_group_id[subgroup]]
            state = self.optimizer.state.get(param, {})
            require(float(state.get('step', 0)) == (0 if group.get('head', False) else cycle),
                    'native FP32 master Adam clock mismatch', path='h3_ar.session')
            masters.append(dict(role=group['role'], head=group.get('head', False), master_sha256=digest(param),
                                state={k: digest(v) for k, v in state.items() if isinstance(v, torch.Tensor)}))
        parameters = [*self.adapter.role_parameters('student'), *self.adapter.role_parameters('fake_score')]
        return dict(master_optimizer=masters, trainable_partitions=[digest(p.ds_tensor) for p in parameters],
                    cpu_rng_sha256=digest(torch.get_rng_state()), cuda_rng_sha256=digest(torch.cuda.get_rng_state(self.device)))
