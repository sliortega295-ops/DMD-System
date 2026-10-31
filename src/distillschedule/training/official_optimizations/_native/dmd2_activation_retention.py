"""Measured implementation helper, relocated from r2_fsdp2_baselines/scripts/dmd2_activation_retention.py.

The public owner validates capabilities and lifecycle; this module does not
launch experiments or claim framework GPU acceptance. Source lock records all
original and relocated executable definitions.
"""
class ActivationRetention:

    def __init__(self, trainer):
        import torch
        guidance = trainer.accelerator.unwrap_model(trainer.model.guidance_model)
        self.fake = guidance.fake_unet
        self.calls = self.observed_enabled = self.observed_disabled = 0

        def before(module, args, kwargs):
            if torch.is_grad_enabled() and (not kwargs.get('classify_mode', False)):
                if not module.is_gradient_checkpointing:
                    raise ValueError('NATIVE_EPSILON_CHECKPOINT_ENABLE_MISSING')
                self.observed_enabled += 1
                module.disable_gradient_checkpointing()
                if module.is_gradient_checkpointing:
                    raise ValueError('EPSILON_ACTIVATION_RETENTION_NOT_ACTIVE')
                self.observed_disabled += 1
                self.calls += 1
        self.handle = self.fake.register_forward_pre_hook(before, with_kwargs=True)

    def receipt(self):
        return dict(mechanism='RETAIN_FAKE_EPSILON_ACTIVATIONS_INSTEAD_OF_REPLAY', calls=self.calls, native_checkpoint_enabled=self.observed_enabled, checkpoint_disabled=self.observed_disabled, generator_checkpoint_unchanged=True, precision_unchanged=True, loss_or_rng_changes=False)

    def close(self):
        self.handle.remove()
