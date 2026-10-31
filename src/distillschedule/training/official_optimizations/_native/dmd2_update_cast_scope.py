"""Measured implementation helper, relocated from r4_researchboost_transfer/scripts/dmd2_update_cast_scope.py.

The public owner validates capabilities and lifecycle; this module does not
launch experiments or claim framework GPU acceptance. Source lock records all
original and relocated executable definitions.
"""
def require(ok, why):
    if not ok:
        raise ValueError('DMD2_UPDATE_CAST_SCOPE:' + why)

class UpdateCastScope:

    def __init__(self, guidance, torch, *, cpu_test_only=False):
        self.guidance, self.torch = (guidance, torch)
        self.original = guidance.guidance_forward
        self.active = False
        self.device_type = 'cpu' if cpu_test_only else 'cuda'
        require(not cpu_test_only or not torch.cuda.is_initialized(), 'CPU_ONLY')
        require(not guidance.sdxl, 'PINNED_SD15_ONLY')
        self.parameters = tuple(guidance.fake_unet.parameters())
        require(all((p.is_leaf and p.dtype == torch.float32 for p in self.parameters)), 'FP32_NATIVE_PARAMETERS')
        self.record = dict(mechanism='ONE_CRITIC_FORWARD_AUTOCAST_WEIGHT_LIFETIME', entered=0, exited=0, exception_exits=0, clear_calls=0, epsilon_calls=0, classifier_trunk_calls=0, fp32_head_calls=0, maximum_nesting=0, parameter_mutations=0, stale_cache_crosses_commit=False, precision_changed=False, cpu_test_only=cpu_test_only, rounding_boundary='SHARED_BF16_CAST_GRADIENT_REASSOCIATION_REQUIRES_REVIEW')

        def fake_before(module, args, kwargs):
            if not self.active:
                return
            require(torch.is_grad_enabled() and torch.is_autocast_enabled(self.device_type), 'NATIVE_BF16_REGION')
            require(torch.get_autocast_dtype(self.device_type) == torch.bfloat16, 'NATIVE_BF16_DTYPE')
            self.record['classifier_trunk_calls' if kwargs.get('classify_mode', False) else 'epsilon_calls'] += 1

        def head_before(module, args):
            if not self.active:
                return
            require(not torch.is_autocast_enabled(self.device_type) and args[0].dtype == torch.float32, 'HEAD_MUST_STAY_FP32')
            self.record['fp32_head_calls'] += 1
        self.handles = [guidance.fake_unet.register_forward_pre_hook(fake_before, with_kwargs=True), guidance.cls_pred_branch.register_forward_pre_hook(head_before)]

        def forward(*args, **kwargs):
            require(not self.active and torch.is_grad_enabled(), 'REENTRANCY_OR_NOGRAD')
            require(not torch.is_autocast_enabled(self.device_type), 'OUTER_AUTOCAST_FORBIDDEN')
            require(torch.is_autocast_cache_enabled(), 'AUTOCAST_CACHE_DISABLED')
            require(all((p.requires_grad for p in self.parameters)), 'FAKE_MUST_BE_TRAINABLE')
            versions = tuple((p._version for p in self.parameters))
            depth = torch.autocast_increment_nesting()
            if depth != 1:
                torch.autocast_decrement_nesting()
                raise ValueError('DMD2_UPDATE_CAST_SCOPE:FOREIGN_AUTOCAST_NESTING')
            self.record['maximum_nesting'] = max(self.record['maximum_nesting'], depth)
            torch.clear_autocast_cache()
            self.record['clear_calls'] += 1
            self.active = True
            self.record['entered'] += 1
            try:
                result = self.original(*args, **kwargs)
                changed = versions != tuple((p._version for p in self.parameters))
                self.record['parameter_mutations'] += int(changed)
                require(not changed, 'PARAMETERS_CHANGED_DURING_FORWARD')
                return result
            except BaseException:
                self.record['exception_exits'] += 1
                raise
            finally:
                self.active = False
                final_depth = torch.autocast_decrement_nesting()
                torch.clear_autocast_cache()
                self.record['clear_calls'] += 1
                self.record['exited'] += 1
                require(final_depth == 0, 'UNBALANCED_NATIVE_AUTOCAST')
        self.installed = forward
        guidance.guidance_forward = forward

    def close(self):
        require(not self.active and self.guidance.guidance_forward is self.installed, 'PATCH_OR_ACTIVE_LIFETIME')
        self.guidance.guidance_forward = self.original
        for handle in self.handles:
            handle.remove()
