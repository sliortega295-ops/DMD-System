"""Measured implementation helper, relocated from r2_fsdp2_baselines/scripts/sf_phase_retention.py.

The public owner validates capabilities and lifecycle; this module does not
launch experiments or claim framework GPU acceptance. Source lock records all
original and relocated executable definitions.
"""
import types

def require(value, reason):
    if not value:
        raise ValueError('SF_PHASE_RETENTION:' + reason)

class PhaseRetention:
    """RolloutHold's legacy scope ABI plus an explicit physical-state receipt.

`assert_released` here means no ACTIVE ROLLOUT SCOPE, not that BF16 parameter
buffers were evicted. Physical residency is separately tracked and validated.
The legacy logical counters are actual scope entries/exits, never fake evictions.
"""

    def __init__(self, trainer, max_retained_bytes):
        import torch
        from torch.distributed.fsdp import FSDPModule
        self.torch = torch
        self.trainer = trainer
        self.max_bytes = max_retained_bytes
        self.roles = {r: getattr(trainer.model, r) for r in ('generator', 'fake_score')}
        self.units = {r: [m for m in root.modules() if isinstance(m, FSDPModule)] for r, root in self.roles.items()}
        require(all((isinstance(root, FSDPModule) for root in self.roles.values())), 'ACTUAL_FSDP2_REQUIRED')
        require(all((hasattr(root, 'set_reshard_after_backward') for root in self.roles.values())), 'PUBLIC_BACKWARD_RETENTION_UNAVAILABLE')
        self.bytes = {r: sum((p.numel() for p in root.parameters())) * 2 for r, root in self.roles.items()}
        require(all((0 < v <= max_retained_bytes for v in self.bytes.values())), 'PER_ROLE_BF16_CAPACITY')
        self.expected_bytes = self.bytes['generator']
        self.g_version = self.c_version = 0
        self.active = False
        self.live = {r: False for r in self.roles}
        self.closed = False
        self.counts = dict(hold_entries=0, hold_exits=0, skipped_grad_rollouts=0, generator_commits=0, critic_commits=0, failed_rollouts=0)
        self.record = dict(physical_releases={r: 0 for r in self.roles}, clip_releases={r: 0 for r in self.roles}, root_forwards={r: 0 for r in self.roles}, finalization_flushes=0, master_parameter_checks=0, contract_failures=0)
        self.handles = []
        self.clips = {}
        for role, root in self.roles.items():
            root.set_reshard_after_forward(False, recurse=True)
            root.set_reshard_after_backward(False, recurse=True)

            def forward(module, args, role=role):
                require(not self.closed, 'FORWARD_AFTER_FINALIZATION')
                self.live[role] = True
                self.record['root_forwards'][role] += 1
            self.handles.append(root.register_forward_pre_hook(forward, prepend=True))
            original = root.clip_grad_norm_
            self.clips[role] = original

            def clip(module, *args, role=role, original=original, **kwargs):
                require(self.live[role], 'CLIP_WITHOUT_ROLE_FORWARD')
                self._release(role)
                self.record['clip_releases'][role] += 1
                self._master_check(role)
                return original(*args, **kwargs)
            root.clip_grad_norm_ = types.MethodType(clip, root)
        self.original = trainer.model._consistency_backward_simulation

        def rollout(_model, *args, **kwargs):
            self.assert_released()
            if torch.is_grad_enabled():
                self.counts['skipped_grad_rollouts'] += 1
                return self.original(*args, **kwargs)
            version = self.g_version
            self.active = True
            self.counts['hold_entries'] += 1
            try:
                value = self.original(*args, **kwargs)
                require(version == self.g_version, 'GENERATOR_VERSION_CHANGED_IN_ROLLOUT')
                return value
            except BaseException:
                self.counts['failed_rollouts'] += 1
                raise
            finally:
                self.active = False
                self.counts['hold_exits'] += 1
        trainer.model._consistency_backward_simulation = types.MethodType(rollout, trainer.model)
        for name, role, clock in [('generator_optimizer', 'generator', 'generator'), ('critic_optimizer', 'fake_score', 'critic')]:
            optimizer = getattr(trainer, name)

            def before(opt, args, kwargs, role=role):
                self.assert_released()
                require(not self.live[role], 'OPTIMIZER_BEFORE_NATIVE_CLIP_RESHARD')
                self._master_check(role)

            def after(opt, args, kwargs, clock=clock):
                self.counts[clock + '_commits'] += 1
                if clock == 'generator':
                    self.g_version += 1
                else:
                    self.c_version += 1
            self.handles.extend([optimizer.register_step_pre_hook(before), optimizer.register_step_post_hook(after)])

    def _master_check(self, role):
        from torch.distributed.tensor import DTensor
        require(all((isinstance(p, DTensor) and p.dtype == self.torch.float32 for p in self.roles[role].parameters())), 'SHARDED_FP32_MASTER_NOT_RESTORED')
        self.record['master_parameter_checks'] += 1

    def _release(self, role):
        if not self.live[role]:
            return
        for unit in reversed(self.units[role]):
            unit.reshard()
        self.live[role] = False
        self.record['physical_releases'][role] += 1

    def assert_released(self):
        require(not self.active, 'ACTIVE_ROLLOUT_SCOPE')

    def finalize(self):
        self.assert_released()
        if self.closed:
            return
        for role, root in self.roles.items():
            self._release(role)
            root.set_reshard_after_forward(True, recurse=True)
            root.set_reshard_after_backward(True, recurse=True)
            self._master_check(role)
        self.closed = True
        self.record['finalization_flushes'] += 1

    def receipt(self):
        require(self.trainer.step == 10, 'FINAL_RECEIPT_ONLY_AFTER_TWO_NATIVE_CYCLES')
        self.finalize()
        return dict(mechanism='FSDP2_TRAINABLE_EXECUTION_WEIGHT_PHASE_RETENTION', counters=dict(self.counts), physical_state=dict(self.record), roles=list(self.roles), retained_compute_weight_bytes=self.bytes, maximum_retained_bytes_per_role=self.max_bytes, maximum_total_retained_bytes=2 * self.max_bytes, generator_version=self.g_version, critic_version=self.c_version, active=False, live_at_final=dict(self.live), closed=self.closed, merged_weights=False, dtype_changed=False, clip_and_optimizer='ORIGINAL_SHARDED_FP32_STATE_AFTER_ROLE_RESHARD', reduction='UNCHANGED_NATIVE_FSDP2_MEAN', logical_scope_exits_are_not_evictions=True)

    def close(self):
        self.finalize()
        self.trainer.model._consistency_backward_simulation = self.original
        for role, root in self.roles.items():
            root.clip_grad_norm_ = self.clips[role]
        for handle in self.handles:
            handle.remove()
