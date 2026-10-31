"""Offline native LTX codecs and Gemma4 conditioning, strictly local FP32.

This module never downloads models and never replaces encoders with synthetic
embeddings. The supported bundle format carries complete constructor configs
and hash-locked native state dicts. Tiny fixtures are real classes, not pretrained
quality evidence. Official monolithic 22B checkpoint conversion is NOT provided.
"""
from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path

import torch
from torch import nn
from distillschedule.workloads.best_known.registry import file_sha256 as _file_sha256

from .ltx_source import require

ASSET_SCHEMA = 'distillschedule.ltx25_local_assets.v1'
MEDIA_KEYS = {'video_encoder', 'video_decoder', 'audio_encoder', 'audio_decoder', 'vocoder',
              'text_encoder', 'text_processor'}


def place_module(module, device):
    """Keep the native LTX public alias and error prefix for shared ownership."""
    from .single_device import place_module as place
    return place(module, device, require=require)


def validate_placement(module, device):
    from .single_device import validate_placement as validate
    return validate(module, device, require=require)


def encoding_sources():
    """Shared helper identity for prepared features and portable exports."""
    paths = (Path(__file__), Path(__file__).with_name('ltx_source.py'),
             Path(__file__).with_name('single_device.py'),
             Path(__file__).parents[1] / 'data' / 'ltx_inputs.py')
    return {path.name: file_sha256(path) for path in paths}


def media_binding(manifest, source):
    from distillschedule.workloads.best_known.registry import fingerprint
    # A prepared encoding must not survive a numerical change in our intake
    # helpers merely because the external source/weights stayed unchanged.
    return fingerprint({'native_source': source.identity, 'media': manifest['media'],
                        'adapter_sources': encoding_sources()})


def file_sha256(path):
    """Stream even multi-GB local assets; never read a checkpoint into one bytes object."""
    return _file_sha256(Path(path))


def json_object(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(), object_pairs_hook=unique,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + value)))
    require(isinstance(value, dict), 'asset/config JSON must contain an object')
    return value


def asset_file(root, spec):
    root = Path(root).resolve(strict=True)
    require(isinstance(spec, dict) and set(spec) == {'path', 'sha256'} and isinstance(spec['path'], str)
            and isinstance(spec['sha256'], str) and len(spec['sha256']) == 64
            and all(c in '0123456789abcdef' for c in spec['sha256']), 'asset requires explicit path and SHA256')
    path = (root / spec['path']).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), 'asset escapes its local bundle or is not a file')
    require(file_sha256(path) == spec['sha256'], 'local asset SHA256 mismatch: ' + str(path))
    return path


def strict_load(module, path, expected_sha256):
    from .checkpoint import _validate_layout
    require(file_sha256(path) == expected_sha256, 'local state hash changed before load')
    value = torch.load(path, map_location='cpu', weights_only=True)
    require(isinstance(value, dict) and value and all(isinstance(k, str) and type(t) is torch.Tensor
        and not t.is_meta and t.device.type == 'cpu' and t.layout == torch.strided and t.numel() > 0
        and bool(torch.isfinite(t).all()) for k, t in value.items()),
        'native weights must be nonempty finite dense non-meta tensors, not an empty/inference shell')
    _validate_layout(module.state_dict(), value, 'LTX native local checkpoint')
    module.load_state_dict(value, strict=True)
    require(file_sha256(path) == expected_sha256, 'local state hash changed during load')
    require(all(not p.is_meta and p.numel() and bool(torch.isfinite(p).all()) for p in module.parameters()),
            'native checkpoint left empty/meta/nonfinite parameters')


def constructor(cls, config, required):
    require(isinstance(config, dict) and set(required) <= config.keys(),
            'explicit native constructor configuration missing for ' + cls.__name__)
    allowed = {name for name, parameter in inspect.signature(cls.__init__).parameters.items()
               if name != 'self' and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)}
    require(config.keys() <= allowed, 'unknown native constructor fields for ' + cls.__name__)
    return cls(**config)


def _video_config(source, config):
    config = deepcopy(config)
    enums = source.import_module('ltx_core.model.video_vae.enums')
    for key, cls in (('norm_layer', enums.NormLayerType), ('latent_log_var', enums.LogVarianceType),
                     ('encoder_spatial_padding_mode', enums.PaddingModeType),
                     ('decoder_spatial_padding_mode', enums.PaddingModeType)):
        if key in config:
            config[key] = cls(config[key])
    require('attention' not in config, 'external callable/automatic codec attention is not admitted by this CPU bundle')
    return config


def _audio_config(source, config):
    config = deepcopy(config)
    norm = source.import_module('ltx_core.model.common.normalization')
    causal = source.import_module('ltx_core.model.audio_vae.causality_axis')
    require({'norm_type', 'causality_axis', 'sample_rate', 'mel_hop_length', 'is_causal', 'mel_bins',
             'ch_mult', 'ch', 'z_channels', 'attn_resolutions'} <= config.keys(), 'explicit native audio geometry required')
    config['norm_type'] = norm.NormType(config['norm_type'])
    config['causality_axis'] = causal.CausalityAxis(config['causality_axis'])
    config['ch_mult'] = tuple(config['ch_mult']); config['attn_resolutions'] = set(config['attn_resolutions'])
    require(config['norm_type'] is norm.NormType.PIXEL and config['causality_axis'] is causal.CausalityAxis.HEIGHT,
            'this admitted audio route requires native causal HEIGHT PixelNorm, not incompatible GroupNorm')
    require(config['is_causal'] is True and 1 <= config['ch'] <= 256,
            'explicit bounded causal CPU audio architecture required')
    return config


class NativeMedia(nn.Module):
    """One owned immutable module containing actual VAE, Gemma and connectors."""
    def __init__(self, source, architecture):
        super().__init__()
        require(isinstance(architecture, dict) and architecture.keys() == MEDIA_KEYS,
                'all native video/audio/text codec configurations are required')
        self.source = source
        self.architecture = deepcopy(architecture)
        video = source.import_module('ltx_core.model.video_vae.video_vae')
        decoder = source.import_module('ltx_core.model.video_vae.conv_video_decoder')
        audio = source.import_module('ltx_core.model.audio_vae.audio_vae')
        vocoder = source.import_module('ltx_core.model.audio_vae.vocoder')
        enc_config = _video_config(source, architecture['video_encoder'])
        dec_config = _video_config(source, architecture['video_decoder'])
        require(enc_config.get('out_channels') == dec_config.get('in_channels') == 128
                and enc_config.get('in_channels') == dec_config.get('out_channels') == 3,
                'actual native RGB/128-channel video VAE required')
        require(type(dec_config.get('base_channels')) is int and 1 <= dec_config['base_channels'] <= 256
                and len(enc_config.get('encoder_blocks', [])) <= 16 and len(dec_config.get('decoder_blocks', [])) <= 16,
                'bounded explicit native CPU video codec architecture required')
        self.video_encoder = constructor(video.VideoEncoder, enc_config,
            ('encoder_blocks', 'patch_size', 'norm_layer', 'latent_log_var'))
        self.video_decoder = constructor(decoder.ConvVideoDecoder, dec_config,
            ('decoder_blocks', 'patch_size', 'base_channels', 'norm_layer', 'decoder_spatial_padding_mode'))
        require(self.video_encoder.video_scale_factors == self.video_decoder.video_downscale_factors,
                'native video encoder/decoder scale factors disagree')
        audio_enc = _audio_config(source, architecture['audio_encoder'])
        audio_dec = _audio_config(source, architecture['audio_decoder'])
        self.audio_encoder = constructor(audio.AudioEncoder, audio_enc, ('n_fft', 'in_channels'))
        self.audio_decoder = constructor(audio.AudioDecoder, audio_dec, ('out_ch',))
        require(all(audio_enc[key] == audio_dec[key] for key in ('sample_rate', 'mel_hop_length', 'mel_bins',
                'ch_mult', 'z_channels', 'is_causal')), 'native audio encoder/decoder geometry differs')
        vocoder_config = architecture['vocoder']
        require(type(vocoder_config.get('upsample_initial_channel')) is int
                and 1 <= vocoder_config['upsample_initial_channel'] <= 1024,
                'bounded explicit native CPU vocoder width required')
        self.vocoder = constructor(vocoder.Vocoder, deepcopy(vocoder_config),
            ('upsample_initial_channel', 'upsample_rates', 'upsample_kernel_sizes', 'resblock_kernel_sizes',
             'resblock_dilation_sizes', 'output_sampling_rate'))
        self._build_text(architecture['text_encoder'], architecture['text_processor'])
        self.float().eval().requires_grad_(False)

    def _build_text(self, encoder_config, processor_config):
        from transformers import (GemmaTokenizer, Gemma4UnifiedTextConfig,
                                  Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration)
        source = self.source
        require(isinstance(encoder_config, dict) and set(encoder_config) == {'text', 'unified', 'tokenizer'},
                'explicit native Gemma4 text/unified/tokenizer configuration required')
        text = encoder_config['text']
        essential = {'vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads',
                     'num_key_value_heads', 'head_dim', 'global_head_dim', 'max_position_embeddings',
                     'sliding_window', 'layer_types', 'use_cache'}
        require(isinstance(text, dict) and essential <= text.keys() and text['use_cache'] is False,
                'complete cache-disabled Gemma4 shape configuration required')
        require(text['vocab_size'] * text['hidden_size']
                + text['num_hidden_layers'] * 24 * text['hidden_size'] ** 2 < 50_000_000,
                'this explicit CPU local bundle is bounded; full 22B construction is not admitted')
        token = encoder_config['tokenizer']
        require(isinstance(token, dict) and set(token) == {'vocab', 'merges', 'max_length'},
                'explicit local Gemma tokenizer vocabulary/merges/length required; no online auto-loader')
        require(isinstance(token['vocab'], dict) and token['vocab'] and type(token['max_length']) is int
                and 2 <= token['max_length'] <= text['max_position_embeddings']
                and max(token['vocab'].values()) < text['vocab_size'], 'tokenizer/Gemma vocabulary or length mismatch')
        tokenizer_api = source.import_module('ltx_core.text_encoders.gemma.tokenizer')
        encoder_api = source.import_module('ltx_core.text_encoders.gemma.encoders.base_encoder')
        tokenizer = GemmaTokenizer(vocab=token['vocab'], merges=token['merges'])
        native_tokenizer = tokenizer_api.LTXGemmaTokenizer(tokenizer, max_length=token['max_length'],
                                                           padding_side=tokenizer_api.PaddingSide.LEFT)
        unified = deepcopy(encoder_config['unified'])
        require({'vision_config', 'audio_config'} <= unified.keys()
                and unified['vision_config'] is None and unified['audio_config'] is None,
                'this native text intake uses an explicit text-only Gemma4 unified configuration')
        model = Gemma4UnifiedForConditionalGeneration(Gemma4UnifiedConfig(
            text_config=Gemma4UnifiedTextConfig(**text), **unified))
        model.set_attn_implementation('eager')
        self.text_encoder = encoder_api.LTXGemmaTextEncoder(model=model, tokenizer=native_tokenizer, dtype=torch.float32)
        p = processor_config
        require(isinstance(p, dict) and set(p) == {'embedding_dim', 'hidden_features', 'video_dim', 'audio_dim',
                'projection_bias', 'video_connector', 'audio_connector'}, 'explicit native dual feature/connector configuration required')
        require(p['embedding_dim'] == text['hidden_size']
                and p['hidden_features'] == (text['num_hidden_layers'] + 1) * text['hidden_size'],
                'native FeatureExtractorV2 input dimensions differ from Gemma4 hidden states')
        feature = source.import_module('ltx_core.text_encoders.gemma.feature_extractor')
        connector = source.import_module('ltx_core.text_encoders.gemma.embeddings_connector')
        processor = source.import_module('ltx_core.text_encoders.gemma.embeddings_processor')
        feature = feature.FeatureExtractorV2(
            video_aggregate_embed=nn.Linear(p['hidden_features'], p['video_dim'], bias=p['projection_bias']),
            embedding_dim=p['embedding_dim'],
            audio_aggregate_embed=nn.Linear(p['hidden_features'], p['audio_dim'], bias=p['projection_bias']))
        self.text_processor = processor.EmbeddingsProcessor(feature_extractor=feature,
            video_connector=constructor(connector.Embeddings1DConnector, p['video_connector'],
                                         ('attention_head_dim', 'num_attention_heads', 'num_layers', 'num_learnable_registers')),
            audio_connector=constructor(connector.Embeddings1DConnector, p['audio_connector'],
                                         ('attention_head_dim', 'num_attention_heads', 'num_layers', 'num_learnable_registers')))

    def encode_text(self, prompts):
        require(isinstance(prompts, list) and prompts and all(isinstance(p, str) for p in prompts),
                'prompt strings required; unconditioned context must be a real encoded empty prompt')
        self.source.validate()
        with torch.no_grad():
            encoded = self.text_encoder.encode(prompts)
            processed = [self.text_processor.process_hidden_states(hidden, mask, padding_side='left')
                         for hidden, mask in encoded]
        return {'video_context': torch.cat([value.video_encoding for value in processed]),
                'audio_context': torch.cat([value.audio_encoding for value in processed]),
                'context_mask': torch.cat([value.attention_mask.bool() for value in processed])}

    def encode_video(self, video):
        require(type(video) is torch.Tensor and video.dtype == torch.uint8 and video.ndim == 4
                and video.shape[-1] == 3, 'raw native video must be exact FHWC RGB uint8')
        scales = self.video_encoder.video_scale_factors
        require((video.shape[0] - 1) % scales.time == 0 and video.shape[1] % scales.height == 0
                and video.shape[2] % scales.width == 0, 'raw video dimensions violate actual native VAE scale/causal geometry')
        with torch.no_grad():
            video = video.to(next(self.video_encoder.parameters()).device)
            return self.video_encoder(video.permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1.)

    def encode_audio(self, audio):
        api = self.source.import_module('ltx_core.model.audio_vae.audio_vae')
        types = self.source.import_module('ltx_core.types')
        require(isinstance(audio, dict) and set(audio) == {'samples', 'sample_rate'}
                and type(audio['sample_rate']) is int and audio['sample_rate'] == self.audio_encoder.sample_rate,
                'raw audio requires the explicit native sampling rate; no implicit resample')
        samples = audio['samples']
        require(type(samples) is torch.Tensor and samples.dtype == torch.float32 and samples.ndim == 2
                and samples.shape[1] == self.architecture['audio_encoder']['in_channels']
                and bool(torch.isfinite(samples).all()) and bool((samples.abs() <= 1).all()),
                'raw audio must be finite normalized TC FP32 with native channels')
        with torch.no_grad():
            samples = samples.to(next(self.audio_encoder.parameters()).device)
            return api.encode_audio(types.Audio(waveform=samples.T.unsqueeze(0), sampling_rate=audio['sample_rate']),
                                    self.audio_encoder)

    def decode(self, generated, parameters):
        require(not parameters or set(parameters) <= {'rollout_sigmas'}, 'unsupported native generation parameters')
        self.source.validate()
        video, audio = generated['video_latents'], generated['audio_latents']
        device = next(self.parameters()).device
        validate_placement(self, device)
        require(video.device == audio.device == device, 'native decode inputs must match the owned codec device')
        require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('cuda'),
                'explicit FP32 decoding with no autocast required')
        require(video.shape[0] == audio.shape[0] == 1, 'media artifact output currently requires batch one')
        frames = torch.cat(list(self.video_decoder.decode_video(video)), dim=0)
        api = self.source.import_module('ltx_core.model.audio_vae.audio_vae')
        waveform = api.decode_audio(audio, self.audio_decoder, self.vocoder)
        require(frames.ndim == 4 and frames.shape[-1] == 3 and bool(torch.isfinite(frames).all())
                and bool(((frames >= 0) & (frames <= 1)).all()), 'native decoder did not return FHWC [0,1] chunks')
        require(waveform.waveform.ndim == 2 and bool(torch.isfinite(waveform.waveform).all())
                and bool((waveform.waveform.abs() <= 1).all()), 'native vocoder returned invalid normalized audio')
        return {'video': (frames * 255).round().to(torch.uint8).contiguous(),
                'audio': {'samples': waveform.waveform.T.contiguous(), 'sample_rate': waveform.sampling_rate}}


def load_media(source, manifest_path):
    path = Path(manifest_path).expanduser().resolve(strict=True)
    value = json_object(path)
    require(value.get('schema_version') == ASSET_SCHEMA and isinstance(value.get('media'), dict)
            and value['media'].keys() == MEDIA_KEYS, 'native local media manifest missing or incompatible')
    architecture = {}
    files = {}
    for name, row in value['media'].items():
        require(isinstance(row, dict) and set(row) == {'config', 'weights'}, 'native media entry needs exact config/weights')
        architecture[name] = row['config']
        files[name] = asset_file(path.parent, row['weights'])
    module = NativeMedia(source, architecture)
    for name, file in files.items():
        strict_load(getattr(module, name), file, value['media'][name]['weights']['sha256'])
    return module, value, {'manifest': file_sha256(path), 'files': {
        name: value['media'][name]['weights']['sha256'] for name in sorted(files)}}
