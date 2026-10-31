"""Measured implementation helper, relocated from r4_researchboost_transfer/scripts/dmd2_classifier_pack.py.

The public owner validates capabilities and lifecycle; this module does not
launch experiments or claim framework GPU acceptance. Source lock records all
original and relocated executable definitions.
"""
def require(ok, why):
    if not ok:
        raise ValueError('DMD2_CLASSIFIER_PACK:' + why)

class PackedClassifier:

    def __init__(self, guidance, torch, *, cpu_test_only=False):
        self.guidance = guidance
        self.torch = torch
        self.cpu = cpu_test_only
        self.original_pair = guidance.compute_guidance_clean_cls_loss
        self.original_classify = guidance.compute_cls_logits
        require(not guidance.sdxl, 'PINNED_SD15_ONLY')
        require(not any((isinstance(m, torch.nn.modules.batchnorm._BatchNorm) or (isinstance(m, torch.nn.Dropout) and m.p != 0) for root in (guidance.fake_unet, guidance.cls_pred_branch) for m in root.modules())), 'CROSS_SAMPLE_OR_STOCHASTIC_LAYER')
        require(not cpu_test_only or not torch.cuda.is_initialized(), 'CPU_ONLY')
        self.active = None
        self.record = dict(mechanism='SD15_REAL_FAKE_PACKED_CLASSIFIER_TRUNK', pair_calls=0, logical_classifier_calls=0, physical_packed_trunk_calls=0, physical_head_calls=0, reference_classifier_calls=0, packed_shapes={}, contract_failures=0, rng_draw_order='REAL_THEN_FAKE_NATIVE', unchanged_fp32_heads=True, two_independent_loss_means=True, cpu_test_only=cpu_test_only, precision_changed=False)

        def signature(t):
            return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)

        def prepare(image):
            if guidance.diffusion_gan:
                time = torch.randint(0, guidance.diffusion_gan_max_timestep, [image.shape[0]], device=image.device, dtype=torch.long)
                image = guidance.scheduler.add_noise(image, torch.randn_like(image), time)
            else:
                time = torch.zeros([image.shape[0]], device=image.device, dtype=torch.long)
            return (image, time)

        def trunk_before(module, args, kwargs):
            if self.active is None or not kwargs.get('classify_mode', False):
                return
            require(self.active['phase'] == 'trunk' and args[0].shape[0] == 2, 'ACTUAL_PACKED_TRUNK_SCOPE')
            self.record['physical_packed_trunk_calls'] += 1
            key = str(tuple(args[0].shape))
            self.record['packed_shapes'][key] = self.record['packed_shapes'].get(key, 0) + 1

        def head_before(module, args):
            if self.active is not None:
                require(self.active['phase'] == 'heads' and args[0].shape[0] == 1 and (args[0].dtype == torch.float32), 'ACTUAL_FP32_SEPARATE_HEAD')
                self.record['physical_head_calls'] += 1
        self.handles = [guidance.fake_unet.register_forward_pre_hook(trunk_before, with_kwargs=True), guidance.cls_pred_branch.register_forward_pre_hook(head_before)]

        def classify(image, text_embedding, unet_added_conditions):
            if self.active is None:
                self.record['reference_classifier_calls'] += 1
                return self.original_classify(image, text_embedding, unet_added_conditions)
            state = self.active
            index = state['reads']
            require(state['phase'] == 'heads' and index < 2 and (not image.requires_grad) and (unet_added_conditions is None), 'PAIR_HEAD_SCOPE')
            require((signature(image), signature(text_embedding)) == state['identities'][index], 'STALE_OR_DIFFERENT_CLASSIFIER_INPUT')
            state['reads'] += 1
            self.record['logical_classifier_calls'] += 1
            return guidance.cls_pred_branch(state['representations'][index]).squeeze(dim=[2, 3])

        def pair(real_image, fake_image, real_text_embedding, fake_text_embedding, real_unet_added_conditions=None, fake_unet_added_conditions=None):
            require(self.active is None and torch.is_grad_enabled(), 'REENTRANCY_OR_GRAD_MODE')
            require(real_unet_added_conditions is fake_unet_added_conditions is None, 'SD15_CONDITIONS_ONLY')
            require(real_image.shape == fake_image.shape and real_image.shape[0] == 1 and (real_image.dtype == fake_image.dtype == torch.float32) and (real_text_embedding.shape == fake_text_embedding.shape), 'SAME_NATIVE_B1_SHAPE_PRECISION')
            require(not guidance.fake_unet.is_gradient_checkpointing, 'PARENT_ACTIVATION_POLICY')
            require(next(guidance.fake_unet.parameters()).requires_grad, 'FAKE_PARAMETERS_FROZEN')
            state = dict(phase='trunk', reads=0, identities=[(signature(real_image), signature(real_text_embedding)), (signature(fake_image), signature(fake_text_embedding))])
            self.active = state
            try:
                real, rt = prepare(real_image.detach())
                fake, ft = prepare(fake_image.detach())
                with guidance.network_context_manager:
                    representation = guidance.fake_unet(torch.cat((real, fake)), torch.cat((rt, ft)), torch.cat((real_text_embedding, fake_text_embedding)), added_cond_kwargs=None, classify_mode=True)[-1].float()
                state['representations'] = representation.chunk(2)
                state['phase'] = 'heads'
                result = self.original_pair(real_image, fake_image, real_text_embedding, fake_text_embedding, None, None)
                require(state['reads'] == 2, 'MISSING_REAL_OR_FAKE_BRANCH')
                self.record['pair_calls'] += 1
                return result
            finally:
                self.active = None
        self.installed_pair, self.installed_classify = (pair, classify)
        guidance.compute_guidance_clean_cls_loss = pair
        guidance.compute_cls_logits = classify

    def close(self):
        require(self.guidance.compute_guidance_clean_cls_loss is self.installed_pair and self.guidance.compute_cls_logits is self.installed_classify, 'PATCH_OWNERSHIP')
        self.guidance.compute_guidance_clean_cls_loss = self.original_pair
        self.guidance.compute_cls_logits = self.original_classify
        for handle in self.handles:
            handle.remove()
