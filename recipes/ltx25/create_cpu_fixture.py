"""Create RANDOM tiny ACTUAL native LTX-2.5 assets and real offline media files.

Never pretrained, never production/quality/performance validation. No network,
CUDA, source modification, substitute model, PEFT install or merged LoRA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import string


def tiny_architectures():
    transformer = dict(model_type='ltx av model', num_attention_heads=1, attention_head_dim=24,
        audio_num_attention_heads=1, audio_attention_head_dim=24, in_channels=128, out_channels=128,
        audio_in_channels=128, audio_out_channels=128, cross_attention_dim=24, audio_cross_attention_dim=24,
        num_layers=1, cross_attention_adaln=True, use_prompt_adaln_single=False, ff_bias=False,
        audio_ff_bias=False, use_middle_indices_grid=True, rope_type='split', double_precision_rope=False,
        use_keyframes_abs_pos_embedding=False)
    common_audio = dict(ch=128, ch_mult=[1, 1, 1], num_res_blocks=1, attn_resolutions=[], resolution=64,
        z_channels=8, norm_type='pixel', causality_axis='height', mid_block_add_attention=True,
        sample_rate=16000, mel_hop_length=160, is_causal=True, mel_bins=64)
    connector = dict(attention_head_dim=24, num_attention_heads=1, num_layers=1,
                     num_learnable_registers=4, apply_gated_attention=True, ff_bias=False)
    vocab = {symbol: index for index, symbol in enumerate([
        '<pad>', '<eos>', '<bos>', '<unk>', '<mask>', '▁', *string.ascii_letters, *string.digits, '.', ',', '!', '?'])}
    media = dict(
        video_encoder=dict(convolution_dimensions=3, in_channels=3, out_channels=128,
            encoder_blocks=[['res_x', 1], ['compress_all', 1]], patch_size=2,
            norm_layer='pixel_norm', latent_log_var='uniform', encoder_spatial_padding_mode='zeros'),
        video_decoder=dict(convolution_dimensions=3, in_channels=128, out_channels=3, base_channels=16,
            decoder_blocks=[['res_x', 1], ['compress_all', {'multiplier': 1}]], patch_size=2,
            norm_layer='pixel_norm', decoder_spatial_padding_mode='zeros', causal=False, timestep_conditioning=False),
        audio_encoder=dict(in_channels=2, double_z=True, n_fft=256, **common_audio),
        audio_decoder=dict(out_ch=2, **common_audio),
        vocoder=dict(upsample_initial_channel=32, resblock_kernel_sizes=[3],
            resblock_dilation_sizes=[[1, 3, 5]], upsample_rates=[6, 5, 2, 2, 2],
            upsample_kernel_sizes=[16, 15, 8, 4, 4], output_sampling_rate=24000,
            resblock='1', activation='snake', use_tanh_at_final=True, apply_final_activation=True, use_bias_at_final=True),
        text_encoder=dict(text=dict(vocab_size=128, hidden_size=24, intermediate_size=48, num_hidden_layers=1,
            num_attention_heads=1, num_key_value_heads=1, head_dim=24, global_head_dim=24,
            max_position_embeddings=32, sliding_window=16, layer_types=['full_attention'], use_cache=False),
            unified=dict(vision_config=None, audio_config=None, boi_token_id=120, eoi_token_id=121,
                image_token_id=122, video_token_id=123, boa_token_id=124, eoa_token_index=125, audio_token_id=126),
            tokenizer=dict(vocab=vocab, merges=[], max_length=16)),
        text_processor=dict(embedding_dim=24, hidden_features=48, video_dim=24, audio_dim=24,
            projection_bias=True, video_connector=dict(connector), audio_connector=dict(connector)))
    return transformer, media


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def build_fixture(output, source_root, *, cycles=3):
    import numpy as np
    import torch
    import wave
    from PIL import Image
    from distillschedule.training.ltx import native_transformer
    from distillschedule.training.ltx_media import NativeMedia, ASSET_SCHEMA, file_sha256
    from distillschedule.training.ltx_source import original, require
    from distillschedule.data.ltx_inputs import prepare_inputs, SCHEMA
    require(type(cycles) is int and 1 <= cycles <= 10, 'CPU fixture cycles must be 1..10')
    root = Path(output).expanduser().resolve(); root.mkdir(parents=True, exist_ok=False)
    assets = root / 'RANDOM_TINY_NATIVE_NOT_PRETRAINED'; assets.mkdir()
    raw = root / 'raw'; raw.mkdir()
    api = original(source_root); transformer, media_architecture = tiny_architectures()
    manifest = dict(schema_version=ASSET_SCHEMA,
        evidence_kind='RANDOM_TINY_ACTUAL_NATIVE_CLASSES_NOT_PRETRAINED', roles={}, media={})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261023)
        for role in ('student', 'fake', 'teacher'):
            native = native_transformer(api.source, transformer)
            for name, value in native.named_parameters():
                if 'scale_shift_table' in name:
                    torch.nn.init.normal_(value, std=.02)
            path = assets / (role + '.pt'); torch.save(native.state_dict(), path)
            manifest['roles'][role] = dict(config=transformer, weights=dict(path=path.name, sha256=file_sha256(path)))
        media = NativeMedia(api.source, media_architecture)
        for name in sorted(media_architecture):
            module = getattr(media, name)
            require(all(bool(torch.isfinite(value).all()) for value in module.state_dict().values()),
                    'fixture native module has uninitialized nonfinite state: ' + name)
            path = assets / (name + '.pt'); torch.save(module.state_dict(), path)
            manifest['media'][name] = dict(config=media_architecture[name], weights=dict(path=path.name, sha256=file_sha256(path)))
    _json(assets / 'assets.json', manifest)
    frames = raw / 'frames'; frames.mkdir()
    grid = torch.arange(8 * 8 * 3).reshape(8, 8, 3)
    for index in range(3):
        pixels = ((grid * 3 + index * 43) % 256).to(torch.uint8).numpy()
        Image.fromarray(pixels).save(frames / f'{index:03d}.png')
    Image.fromarray(np.flip(pixels, axis=1).copy()).save(raw / 'image.png')
    samples = torch.arange(6400) / 16000
    audio = torch.stack((.2 * torch.sin(2 * torch.pi * 220 * samples),
                         .15 * torch.sin(2 * torch.pi * 330 * samples)), dim=1)
    with wave.open(str(raw / 'audio.wav'), 'wb') as stream:
        stream.setnchannels(2); stream.setsampwidth(2); stream.setframerate(16000)
        stream.writeframes(np.rint(audio.numpy() * 32767).astype('<i2').tobytes())
    records = []
    for index in range(cycles):
        records.append(dict(id=f'native-av-{index:04d}', image='raw/image.png', video='raw/frames', audio='raw/audio.wav',
            text={'value': ['a cat runs', 'a dog runs', 'a cat rests'][index % 3]},
            metadata={'geometry': {'fps': 8., 'conditioning': {
                'video_prefix_latent_frames': 1, 'audio_prefix_latent_frames': 1}}}))
    source = root / 'inputs.json'
    _json(source, dict(schema_version=SCHEMA, mode='raw', source_root=str(api.source.root),
                       assets=str(assets / 'assets.json'), records=records))
    prepared = prepare_inputs(source, root / 'prepared')
    policy = dict(fake_updates=5, fake_sample_policy='reuse_student', rollout_sigmas=[1., .75, .5, .25, 0.],
        gradient_policy='all_steps', sigma_min=.1, sigma_max=.9, teacher_cfg=2., fake_cfg=1.,
        student_loss='flow_x0_surrogate', fake_loss='flow_velocity_mse', normalizer='per_modality_mean_abs',
        normalizer_epsilon=1e-5, modality_weights={'video': 1., 'audio': 1.}, max_grad_norm=1.)
    parameters = dict(source_root=str(api.source.root), assets=str(assets / 'assets.json'), trainable=dict(mode='full'),
        student_lr=1e-4, fake_lr=2e-4, scheduler_gamma=.98, ema_decay=.9, generation_sigmas=policy['rollout_sigmas'])
    training = dict(schema_version='distillschedule.training.v1', model=dict(name='ltx25', parameters=parameters),
        algorithm=dict(name='dmd', parameters=policy), data=dict(prepared=str(prepared)),
        execution=dict(adapter='program', mode='reference'),
        trainer=dict(cycles=cycles, seed=17, device='cpu', checkpoint_every=1, output_dir=str(root / 'run')))
    _json(root / 'training.json', training)
    training['model']['parameters']['trainable'] = dict(mode='lora', rank=2, alpha=2.,
        targets=['transformer_blocks.0.attn1.to_q', 'transformer_blocks.0.audio_attn1.to_q'])
    training['trainer']['output_dir'] = str(root / 'run-lora')
    _json(root / 'training.lora.json', training)
    _json(root / 'EVIDENCE.json', dict(evidence_kind=manifest['evidence_kind'], source=api.identity,
        pretrained_weights='NONE', quality_validation='NOT_RUN', gpu_execution='NOT_RUN',
        inputs='REAL_PNG_FRAMES_IMAGE_PCM_WAV_AND_TEXT_THROUGH_NATIVE_CODECS_GEMMA4',
        raw_content='PROCEDURAL_FIXTURE_NOT_REAL_WORLD_QUALITY_DATA', scope='FP32_CPU1_NATIVE_CLASSES_ONLY'))
    require(torch.cuda.is_initialized() is False, 'fixture unexpectedly initialized CUDA')
    return {'root': str(root), 'assets': str(assets), 'prepared': str(prepared), 'config': str(root / 'training.json'),
            'lora_config': str(root / 'training.lora.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output'); parser.add_argument('--source-root', required=True)
    parser.add_argument('--cycles', type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(build_fixture(args.output, args.source_root, cycles=args.cycles), indent=2))
