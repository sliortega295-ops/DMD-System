"""Explicit RANDOM tiny actual Diffusers/CLIP/VAE fixture, NOT pretrained SD.

This is a CPU integration starter, not a production model or quality result.
No network requests, weights downloads, or architecture substitution occur in
the production factory. Pass --variant sdxl to exercise real dual CLIP/time IDs.
"""
import argparse
import json
from pathlib import Path


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def build_fixture(root, source_root, *, variant='sd15', cycles=3, ratio=None, lora=False):
    import torch
    from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from transformers.models.clip.tokenization_clip import bytes_to_unicode
    from distillschedule.data.sd_inputs import prepare_inputs, SCHEMA
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    assets = root / 'RANDOM_TINY_ACTUAL_DIFFUSERS_NOT_PRETRAINED'
    assets.mkdir()
    sdxl = variant == 'sdxl'
    ratio = ratio if ratio is not None else (5 if sdxl else 10)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        vocabulary = {'<|startoftext|>': 0, '<|endoftext|>': 1}
        for token in bytes_to_unicode().values():
            for suffix in ('', '</w>'):
                vocabulary[token + suffix] = len(vocabulary)
        for folder in (('tokenizer', 'tokenizer_2') if sdxl else ('tokenizer',)):
            target = assets / folder; target.mkdir()
            _json(target / 'vocab.json', vocabulary)
            (target / 'merges.txt').write_text('#version: 0.2\n')
            tokenizer = CLIPTokenizer(str(target / 'vocab.json'), str(target / 'merges.txt'), model_max_length=8)
            tokenizer.save_pretrained(target)
        clip_config = CLIPTextConfig(vocab_size=len(vocabulary), hidden_size=8, intermediate_size=16,
            num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=8,
            projection_dim=8, bos_token_id=0, eos_token_id=1, pad_token_id=1, attention_dropout=0.)
        CLIPTextModel(clip_config).save_pretrained(assets / 'text_encoder')
        if sdxl:
            CLIPTextModelWithProjection(clip_config).save_pretrained(assets / 'text_encoder_2')
        unet_settings = dict(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
            block_out_channels=(8, 16), down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
            up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'), cross_attention_dim=16 if sdxl else 8,
            attention_head_dim=2, norm_num_groups=4)
        if sdxl:
            unet_settings.update(addition_embed_type='text_time', addition_time_embed_dim=2,
                                 projection_class_embeddings_input_dim=20)
        UNet2DConditionModel(**unet_settings).save_pretrained(assets / 'unet')
        AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(8, 16),
            down_block_types=('DownEncoderBlock2D', 'DownEncoderBlock2D'),
            up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D'), layers_per_block=1,
            norm_num_groups=4, sample_size=16).save_pretrained(assets / 'vae')
        DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
            beta_schedule='scaled_linear', clip_sample=False, set_alpha_to_one=False,
            prediction_type='epsilon').save_pretrained(assets / 'scheduler')
    source = root / 'prompts.json'
    # Distinct prompts and native input records; no cached rollout is accepted.
    _json(source, dict(schema_version=SCHEMA, variant=variant, model_id=str(assets),
        records=[dict(id=f'prompt-{index:04d}', prompt=f'a different image {index}')
                 for index in range(max(cycles * ratio, 8))]))
    prepared = prepare_inputs(source, root / 'prepared')
    counts = {'generator': 1}
    if ratio > 1:
        counts['guidance'] = ratio - 1
    if sdxl:
        counts['denoising'] = ratio
    settings = dict(batch_size=1, resolution=16, latent_resolution=8,
        dfake_gen_update_ratio=ratio, generator_lr=1e-4, guidance_lr=2e-4,
        warmup_step=2, train_iters=max(100, cycles * ratio), generator_lora=lora,
        lora_rank=2, lora_alpha=2.)
    config = dict(schema_version='distillschedule.training.v1',
        model=dict(name=variant, parameters=dict(source_root=str(Path(source_root).resolve()),
            model_id=str(assets), settings=settings)), algorithm=dict(name='dmd2'),
        data=dict(streams={name: dict(prepared=str(prepared), samples_per_cycle=count,
            sampling=dict(kind='epochs', shuffle=True)) for name, count in counts.items()}),
        execution=dict(adapter='program', mode='reference'),
        trainer=dict(cycles=cycles, seed=17, device='cpu', output_dir=str(root / 'run'), checkpoint_every=1))
    _json(root / 'training.json', config)
    _json(root / 'EVIDENCE.json', dict(evidence_kind='RANDOM_TINY_ACTUAL_DIFFUSERS_CPU_INTEGRATION_ONLY',
        quality='NOT_RUN', pretrained_weights='NONE', source_root=str(Path(source_root).resolve()), variant=variant))
    return dict(root=str(root), assets=str(assets), prepared=str(prepared), config=str(root / 'training.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output'); parser.add_argument('--source-root', required=True)
    parser.add_argument('--variant', choices=('sd15', 'sdxl'), default='sd15')
    parser.add_argument('--cycles', type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(build_fixture(args.output, args.source_root, variant=args.variant, cycles=args.cycles), indent=2))
