"""Original Qwen-Image T2I native ABI; edit/new-generation weights are rejected.

Native text uses the actual QwenImagePipeline template and crop34, its mask and
Qwen2.5-VL model. Native CFG norm-rescaling is NOT the common algorithm's linear
CFG; the first admitted recipe therefore requires teacher_cfg=fake_cfg=1.
"""
import sys

import torch

from . import flow_image as common
from .flow_image import require

FAMILY = 'qwen-image'
MODEL_ID = 'Qwen/Qwen-Image'
REVISION = '75e0b4be04f60ec59a75f475837eced720f823b6'
ENTRY = 'distillschedule.training.qwen_image'
CODEC_POLICY = 'NATIVE_QWEN2_5_VL_TEMPLATE_CROP34_MASK; VAE_MODE_MEAN_STD_PACK2; FIXED_RIGHT_PAD_NATIVE_MASK_NO_TRUNCATION; T2I_ONLY'


def classes():
    from diffusers import QwenImageTransformer2DModel, AutoencoderKLQwenImage, QwenImagePipeline
    from transformers import Qwen2_5_VLForConditionalGeneration
    return QwenImageTransformer2DModel, AutoencoderKLQwenImage, Qwen2_5_VLForConditionalGeneration, QwenImagePipeline


def validate_architecture(config, media):
    text = common.architecture(classes()[2], media['text_encoder']).text_config
    require(config.get('_class_name') == 'QwenImageTransformer2DModel' and config.get('patch_size') == 2
            and all(config.get(key) is False for key in ('guidance_embeds', 'zero_cond_t', 'use_additional_t_cond', 'use_layer3d_rope'))
            and config['in_channels'] == media['vae']['z_dim'] * 4
            and config['out_channels'] == media['vae']['z_dim']
            and config['joint_attention_dim'] == text.hidden_size,
            'original T2I Qwen-Image with matched VAE/VL widths only; edit/extra time/causal variants rejected')
    require(type(media['text_max_length']) is int and 1 <= media['text_max_length'] <= 1024
            and len(media['vae']['latents_mean']) == len(media['vae']['latents_std']) == media['vae']['z_dim']
            and all(type(x) in (int, float) and 0 < x < float('inf') for x in media['vae']['latents_std']),
            'explicit native text length and actual finite channel mean/std required')


def empty_shape(media):
    # Actual empty-prompt length is tokenizer-dependent, not max_sequence_length.
    # It is persisted explicitly at projection time, never guessed here.
    return tuple(media['empty_context_shape'])


def unpack(tokens, shape):
    b, c, h, w = shape
    require(tokens.shape == (b, (h // 2) * (w // 2), c * 4), 'invalid Qwen native packed geometry')
    pipe = classes()[3]
    return pipe._unpack_latents(tokens, h, w, 1)[:, :, 0]


class Media(common.ImageMedia):
    def pipeline(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        return classes()[3](scheduler=FlowMatchEulerDiscreteScheduler(), vae=self.vae,
            text_encoder=self.text_encoder, tokenizer=self.tokenizer, transformer=None)

    def encode_text(self, prompts):
        require(isinstance(prompts, list) and prompts and all(isinstance(p, str) for p in prompts), 'explicit native text prompts required')
        pipe = self.pipeline()
        for prompt in prompts:
            full = self.tokenizer(pipe.prompt_template_encode.format(prompt), truncation=False)['input_ids']
            require(len(full) <= pipe.tokenizer_max_length + pipe.prompt_template_encode_start_idx,
                    'native Qwen prompt exceeds tokenizer length; truncation refused')
        with torch.no_grad():
            context, mask = pipe.encode_prompt(prompts, device=torch.device('cpu'), max_sequence_length=1024)
        if mask is None:
            mask = torch.ones(context.shape[:2], dtype=torch.bool)
        limit = self.architecture['text_max_length']
        require(0 < context.shape[1] <= limit, 'native Qwen features exceed explicit length; truncation refused')
        # Native encoder already right-pads multi-prompt batches. Make that
        # explicit fixed cohort width for the existing shape-bound CyclePolicy.
        # Only masked padding is zero; no conditional feature is invented.
        from torch.nn import functional as F
        count = limit - context.shape[1]
        return F.pad(context.float(), (0, 0, 0, count)).detach(), F.pad(mask.bool(), (0, count), value=False).detach()

    def encode_image(self, pixels):
        pixels = self.image(pixels)
        scale = 2 ** len(self.vae.temperal_downsample)
        require(pixels.shape[-2] % (2 * scale) == pixels.shape[-1] % (2 * scale) == 0,
                'native image geometry must divide VAE scale times 2; implicit resize forbidden')
        with torch.no_grad():
            raw = self.vae.encode(pixels.unsqueeze(2)).latent_dist.mode()
            mean = raw.new_tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
            std = raw.new_tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
            normalized = (raw - mean) / std
            shape = list(raw[:, :, 0].shape)
            tokens = classes()[3]._pack_latents(normalized, *shape)
        return tokens.detach(), shape

    def decode_tokens(self, generated):
        latent = generated['image_latents'].unsqueeze(2)
        mean = latent.new_tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = latent.new_tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        return self.vae.decode(latent * std + mean, return_dict=False)[0][:, :, 0]


def geometry(row, *, shape, context, null_context):
    b, _, h, w = shape
    return {'img_shapes': torch.tensor([[1, h // 2, w // 2]] * b, dtype=torch.int64)}


class Role(common.ImageRole):
    def denoise(self, sample, sigma, feeds, *, conditional):
        self.check_denoise(sample, sigma, feeds)
        context = feeds['context'] if conditional else feeds['null_context']
        mask = feeds['context_mask'] if conditional else feeds['null_context_mask']
        shapes = [[tuple(value)] for value in feeds['img_shapes'].tolist()]
        value = self.native(hidden_states=sample['image'], encoder_hidden_states=context,
            encoder_hidden_states_mask=mask, timestep=sigma, img_shapes=shapes, guidance=None, return_dict=False)[0]
        require(value.shape == sample['image'].shape and bool(torch.isfinite(value).all()), 'invalid native Qwen flow output')
        return {'image': value}


def build_model(parameters):
    return common.build_family(parameters, sys.modules[__name__])


def build_inference_model(parameters):
    return common.build_inference_family(parameters, sys.modules[__name__])


def admit_training(**kwargs):
    return common.admit_training(sys.modules[__name__], **kwargs)


def generate(state, feeds, parameters):
    return common.generate(sys.modules[__name__], state, feeds, parameters)


def project_current_generator(components, effective_training_parameters):
    return common.project_current_generator(sys.modules[__name__], components, effective_training_parameters)
