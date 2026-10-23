"""FLUX.2 klein Base 4B T2I native ABI; not FLUX.1 or distilled klein.

Source-qualified pinned Diffusers implementation; CPU1 FP32 fixture evidence is
not pretrained/GPU validation. Native flow output is epsilon-x0, without sign
inversion. Klein Base has no guidance embedding input.
"""
import sys

import torch

from . import flow_image as common
from .flow_image import require

FAMILY = 'flux2-klein-base'
MODEL_ID = 'black-forest-labs/FLUX.2-klein-base-4B'
REVISION = 'a3b4f4849157f664bdbc776fd7453c2783562f4d'
ENTRY = 'distillschedule.training.flux'
CODEC_POLICY = 'NATIVE_QWEN3_LAYERS_9_18_27; VAE_MODE_PATCH2_BN; T2I_ONLY'


def classes():
    from diffusers import Flux2Transformer2DModel, AutoencoderKLFlux2, Flux2KleinPipeline
    from transformers import Qwen3ForCausalLM
    return Flux2Transformer2DModel, AutoencoderKLFlux2, Qwen3ForCausalLM, Flux2KleinPipeline


def validate_architecture(config, media):
    text = common.architecture(classes()[2], media['text_encoder'])
    require(config.get('_class_name') == 'Flux2Transformer2DModel' and config.get('patch_size') == 1
            and config.get('guidance_embeds') is False and config.get('out_channels') == config.get('in_channels')
            and config['in_channels'] == media['vae']['latent_channels'] * 4
            and config['joint_attention_dim'] == text.hidden_size * 3,
            'only native klein Base, no embedded guidance, and matched VAE/Qwen3 width admitted')
    require(text.num_hidden_layers >= 27
            and media['vae'].get('patch_size') == [2, 2]
            and type(media['text_max_length']) is int and 1 <= media['text_max_length'] <= 512,
            'native klein text layers 9/18/27, 2x2 codec and explicit max length <=512 required')


def empty_shape(media):
    text = common.architecture(classes()[2], media['text_encoder'])
    return (1, media['text_max_length'], text.hidden_size * 3)


def unpack(tokens, shape):
    b, c, h, w = shape
    require(tokens.shape == (b, (h // 2) * (w // 2), c * 4), 'invalid native klein packed geometry')
    # Native pack inverses to PATCHIFIED BCHW; BN inversion precedes unpatchify.
    return tokens.transpose(1, 2).reshape(b, c * 4, h // 2, w // 2)


class Media(common.ImageMedia):
    def pipeline(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        return classes()[3](scheduler=FlowMatchEulerDiscreteScheduler(), vae=self.vae,
            text_encoder=self.text_encoder, tokenizer=self.tokenizer, transformer=None, is_distilled=False)

    def encode_text(self, prompts):
        require(isinstance(prompts, list) and prompts and all(isinstance(p, str) for p in prompts), 'explicit native text prompts required')
        for prompt in prompts:
            text = self.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            count = len(self.tokenizer(text, truncation=False)['input_ids'])
            require(count <= self.architecture['text_max_length'], 'native klein prompt exceeds explicit length; truncation refused')
        with torch.no_grad():
            context, _ = self.pipeline().encode_prompt(prompts, device=torch.device('cpu'),
                max_sequence_length=self.architecture['text_max_length'])
        return context.float().detach(), torch.ones(context.shape[:2], dtype=torch.bool)

    def encode_image(self, pixels):
        pixels = self.image(pixels)
        scale = 2 ** (len(self.vae.config.block_out_channels) - 1)
        require(pixels.shape[-2] % (2 * scale) == pixels.shape[-1] % (2 * scale) == 0,
                'native image geometry must divide VAE scale times 2; implicit resize forbidden')
        with torch.no_grad():
            raw = self.vae.encode(pixels).latent_dist.mode()
            patched = classes()[3]._patchify_latents(raw)
            normalized = (patched - self.vae.bn.running_mean.view(1, -1, 1, 1)) / torch.sqrt(
                self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps)
        return classes()[3]._pack_latents(normalized).detach(), list(raw.shape)

    def decode_tokens(self, generated):
        patched = generated['image_latents']
        restored = patched * torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps)
        restored = restored + self.vae.bn.running_mean.view(1, -1, 1, 1)
        return self.vae.decode(classes()[3]._unpatchify_latents(restored), return_dict=False)[0]


def geometry(row, *, shape, context, null_context):
    b, c, h, w = shape
    pipe = classes()[3]
    template = torch.empty(b, c * 4, h // 2, w // 2)
    return dict(img_ids=pipe._prepare_latent_ids(template), txt_ids=pipe._prepare_text_ids(context),
                null_txt_ids=pipe._prepare_text_ids(null_context))


class Role(common.ImageRole):
    def denoise(self, sample, sigma, feeds, *, conditional):
        self.check_denoise(sample, sigma, feeds)
        context = feeds['context'] if conditional else feeds['null_context']
        ids = feeds['txt_ids'] if conditional else feeds['null_txt_ids']
        # Native forward multiplies sigma by 1000 itself. Never pass 1000*sigma
        # or the embedded-guidance tensor used by other FLUX variants.
        value = self.native(hidden_states=sample['image'], encoder_hidden_states=context, timestep=sigma,
            img_ids=feeds['img_ids'], txt_ids=ids, guidance=None, return_dict=False)[0]
        require(value.shape == sample['image'].shape and bool(torch.isfinite(value).all()), 'invalid native klein flow output')
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
