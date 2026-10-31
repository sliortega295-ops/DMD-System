"""Measured DMD2 mechanisms in the existing continuous native-cycle owner.

No experiment runner or alternative training loop. Native mixed-precision
exceptions, source losses and private RNG remain the upstream model's work.
This public integration needs its own GPU qualification; historical timing is
not a guarantee for new data, weights, logging or checkpoint frequency.
"""
from contextlib import ExitStack
from copy import deepcopy
import os
from types import SimpleNamespace

from ..dmd2_collective import DMD2CollectiveOwner
from ..native_roles import unwrap_role
from ..official_profiles import resolve, verify_sources
from ..sd import require

PROFILE = 'dmd2-sd15-resident-tree-v1'


def validate_environment():
    for name, value in resolve(PROFILE)['environment'].items():
        require(os.environ.get(name) == value,
                f'optimized profile requires explicit {name}={value} before process-group initialization')


def admit(parameters):
    """No allocation, model downloads or changes to the collective environment."""
    import torch
    from ..single_device import admit_fp32
    require(parameters.get('optimization_profile') == PROFILE, 'unknown DMD2 optimization profile')
    require(str(parameters.get('device', 'cpu')).startswith('cuda')
            and torch.distributed.is_initialized() and torch.distributed.get_world_size() == 16,
            'optimized profile requires actual CUDA world16 ownership')
    device = admit_fp32(parameters, require=require, allow_replicated=True)
    require(device.type == 'cuda' and 'A100' in torch.cuda.get_device_name(device),
            'optimized profile is source-qualified for A100 CUDA world16 only')
    require(str(torch.__version__).split('+')[0] == '2.10.0', 'optimized profile requires Torch2.10')
    from ..sd import runtime_identity
    runtime = runtime_identity()
    require(all(runtime[name] == value for name, value in
                dict(diffusers='0.37.1', transformers='4.51.3', peft='0.17.1').items()),
            'optimized profile numerical dependencies changed')
    validate_environment()
    verify_sources(PROFILE)
    return device


def validate_settings(args):
    # These are mechanism predicates, not silent configuration rewrites.
    expected = dict(sdxl=False, batch_size=1, resolution=512, latent_resolution=64,
        latent_channel=4, dfake_gen_update_ratio=10, num_denoising_step=1,
        use_fp16=True, gradient_checkpointing=True, cls_on_clean_image=True,
        gen_cls_loss=True, denoising=False, backward_simulation=False,
        generator_lora=False, gan_alone=False)
    require(all(getattr(args, key) == value and type(getattr(args, key)) is type(value)
                for key, value in expected.items())
            and args.gen_cls_loss_weight > 0 and args.guidance_cls_loss_weight > 0,
            'optimized profile settings/shape/update/precision/checkpoint contract changed')


class Bundle:
    """One cycle's hooks; cast and classifier state cannot cross a commit."""
    def __init__(self, owner, *, cpu_test_only=False):
        from ._native.dmd2_activation_retention import ActivationRetention
        from ._native.dmd2_classifier_pack import PackedClassifier
        from ._native.dmd2_update_cast_scope import UpdateCastScope
        import torch
        verify_sources(PROFILE)
        self.owner, self.stack = owner, ExitStack()
        guidance = unwrap_role(owner.model.guidance_model)
        try:
            self.activation = ActivationRetention(SimpleNamespace(
                model=owner.model, accelerator=owner.model.accelerator))
            self.stack.callback(self.activation.close)
            self.classifier = PackedClassifier(guidance, torch, cpu_test_only=cpu_test_only)
            self.stack.callback(self.classifier.close)
            self.cast = UpdateCastScope(guidance, torch, cpu_test_only=cpu_test_only)
            self.stack.callback(self.cast.close)
        except BaseException:
            self.stack.close()
            raise

    def receipt(self):
        require(self.classifier.active is None and not self.cast.active,
                'optimized profile cannot checkpoint active cast/classifier state')
        a, p, c = self.activation.receipt(), self.classifier.record, self.cast.record
        n = self.owner.ratio
        require(a['calls'] == a['native_checkpoint_enabled'] == a['checkpoint_disabled'] == n
                and p['pair_calls'] == p['physical_packed_trunk_calls'] == n
                and p['logical_classifier_calls'] == p['physical_head_calls'] == 2 * n
                and p['contract_failures'] == 0
                and c['entered'] == c['exited'] == c['epsilon_calls'] == c['classifier_trunk_calls'] == n
                and c['fp32_head_calls'] == 2 * n and c['exception_exits'] == c['parameter_mutations'] == 0,
                'optimized profile actual execution census changed')
        return dict(profile=PROFILE, scope='THIS_COMPLETED_NATIVE_MACROCYCLE',
            activation=deepcopy(a), classifier=deepcopy(p), cast=deepcopy(c),
            transient_state_empty=True, numerical_status='REVIEW', integration_gpu_validation='NOT_RUN')

    def close(self):
        self.stack.close()


class DMD2OptimizedOwner(DMD2CollectiveOwner):
    """Native role DDP, native precision exceptions and full restart state."""
    def _optimizer_options(self):
        return dict(fused=True)

    def _validate_precision(self):
        import torch
        validate_settings(self.args)
        validate_environment()
        require(self.identity['optimization_profile'] == resolve(PROFILE),
                'optimized profile identity changed')
        require(not torch.is_autocast_enabled(self.device.type), 'optimized profile forbids outer AMP')
        for kind in ('parameters', 'buffers'):
            for name, tensor in self._model_named(kind):
                dtype = (torch.bfloat16 if name.startswith('guidance_model.real_unet.') else
                         torch.float16 if name.startswith('vae.') else torch.float32)
                require(tensor.device == self.device and (not tensor.is_floating_point() or tensor.dtype == dtype),
                        'optimized profile native precision exception changed: ' + name)
        require(all(t.device == self.device for t in self.runtime_tensors()),
                'optimized profile compute/Adam state left its owned device')
        for optimizer in self.optimizers.values():
            require(all(group.get('fused') is True for group in optimizer.param_groups),
                    'optimized profile fused AdamW policy changed')
            for parameter, state in optimizer.state.items():
                require(parameter.dtype == torch.float32
                        and all(t.dtype == torch.float32 and t.device == self.device for t in state.values()),
                        'optimized profile AdamW master/moment/step placement changed')

    def run(self, feeds, *, cycle):
        self._validate_live()
        require(self._role_replicas is not None and bool(self._role_replicas.wrappers),
                'optimized profile requires native role preparation before training')
        bundle = None
        self._last_optimization_receipt = None
        try:
            bundle = Bundle(self)
            value = super().run(feeds, cycle=cycle)
            value['optimization_effects'] = bundle.receipt()
            self._last_optimization_receipt = deepcopy(value['optimization_effects'])
            return value
        except BaseException:
            self._resident_trial_poisoned = True
            raise
        finally:
            if bundle is not None:
                bundle.close()

    def validate_collective_execution(self):
        value = super().validate_collective_execution()
        value.update(precision='NATIVE_MIXED_EXCEPTIONS_FP32_MASTER', numerical_status='REVIEW',
                     optimization_profile=PROFILE, gpu='NOT_RUN')
        if getattr(self, '_last_optimization_receipt', None) is not None:
            value['optimization_effects'] = deepcopy(self._last_optimization_receipt)
        return value

    def execution_options(self):
        value = super().execution_options()
        value['identity']['scope'] = 'NATIVE_MIXED_ROLE_DDP_SOURCE_BOUND_BUNDLE_NO_LOCAL_OVERRIDES'
        return value
