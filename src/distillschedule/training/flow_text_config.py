"""Strict TF5 admission for the pinned native text configs' legacy JSON ABI.

Only config objects are constructed here. Native classes own all migrations and
math; raw documents are never normalized for storage or reconstruction. In
particular T5's legacy tie_word_embeddings flag also controls decoder scaling.
"""
from copy import deepcopy
import inspect
import math


def _require(condition, message):
    if not condition:
        raise ValueError('native flow image: ' + message)


def _rope_identity(value):
    # The native parser retains the old spelling as metadata after interpreting
    # it. Compare effective native fields, not the redundant spelling.
    return {key: _rope_identity(item) if isinstance(item, dict) else item
            for key, item in value.items() if key != 'type'}


def _check_rope(config_class, config):
    from transformers.modeling_rope_utils import RopeParameters
    allowed = set(RopeParameters.__annotations__) | {'type'} | config_class.ignore_keys_at_rope_validation

    def check(value):
        _require(isinstance(value, dict), 'explicit native RoPE object required')
        if value and all(isinstance(item, dict) for item in value.values()):
            _require(set(value) <= set(config.get('layer_types') or config_class().layer_types),
                     'unknown native RoPE layer type')
            for item in value.values():
                check(item)
            return
        _require(set(value) <= allowed, 'unknown native RoPE fields: ' + ', '.join(sorted(set(value) - allowed)))
        _require(not {'type', 'rope_type'} <= value.keys() or value['type'] == value['rope_type'],
                 'conflicting native RoPE type aliases')
        if 'rope_theta' in value:
            theta = value['rope_theta']
            _require(type(theta) in (float, int) and math.isfinite(theta) and theta > 0,
                     'positive finite native rope_theta required')
            _require('rope_theta' not in config or theta == config['rope_theta'], 'conflicting native rope_theta aliases')

    if 'rope_theta' in config:
        check({'rope_theta': config['rope_theta']})
    for name in ('rope_scaling', 'rope_parameters'):
        if config.get(name) is not None:
            check(config[name])
    if config.get('rope_scaling') is not None and config.get('rope_parameters') is not None:
        old = deepcopy(config); old.pop('rope_parameters')
        new = deepcopy(config); new.pop('rope_scaling')
        _require(_rope_identity(config_class.from_dict(old).rope_parameters)
                 == _rope_identity(config_class.from_dict(new).rope_parameters), 'conflicting native RoPE aliases')


def text_config(config_class, config):
    _require(isinstance(config, dict), 'explicit native text config object required')
    kind = config_class.model_type
    known = set(config_class().to_dict()) | {'torch_dtype', '_commit_hash'}
    legacy = set()
    if kind in ('qwen3', 'qwen2_5_vl_text'):
        legacy |= {'rope_theta', 'rope_scaling'}
    if kind in ('qwen2_5_vl', 'qwen2_5_vl_text'):
        # These are retained native metadata, not inputs to TextModel.forward.
        tokens = {'vision_token_id', 'image_token_id', 'video_token_id', 'vision_start_token_id', 'vision_end_token_id'}
        legacy |= tokens
        if kind == 'qwen2_5_vl_text':
            legacy.add('tie_word_embeddings')
        if kind == 'qwen2_5_vl_text' and 'tie_word_embeddings' in config:
            _require(config['tie_word_embeddings'] is False, 'unsupported text-only tied output-head metadata')
        for key in tokens:
            if key in config:
                _require(config[key] is None or type(config[key]) is int and config[key] >= 0,
                         'invalid native text token metadata: ' + key)
    if kind == 'qwen2_5_vl_vision':
        legacy |= {'in_chans', 'spatial_patch_size'}
        for old, new in (('in_chans', 'in_channels'), ('spatial_patch_size', 'patch_size')):
            if old in config:
                _require(new in config and type(config[old]) is type(config[new]) and config[old] == config[new],
                         'conflicting native vision aliases: ' + old)
    if kind in ('t5', 'umt5'):
        # The native EncoderModels never read output_past/scalable_attention or
        # tokenizer_class. Only the audited saved metadata values are admitted;
        # accepting arbitrary kwargs would turn unsupported switches into no-ops.
        exact = dict(tokenizer_class='T5Tokenizer') if kind == 'umt5' else dict(
            tokenizer_class='ByT5Tokenizer', decoder_start_token_id=0, gradient_checkpointing=False)
        if kind == 'umt5':
            exact.update(output_past=True, scalable_attention=True)
        legacy |= exact.keys()
        for key, value in exact.items():
            if key in config:
                _require(type(config[key]) is type(value) and config[key] == value,
                         'unsupported native legacy encoder metadata: ' + key)
    if kind == 'qwen2_5_vl':
        text_class = config_class.sub_configs['text_config']
        # The pinned native parent __post_init__ migrates these flat fields.
        flat_keys = (set(inspect.signature(text_class.__init__).parameters) - {'self', 'kwargs'}) | {'rope_theta', 'rope_scaling'}
        legacy |= flat_keys | {'vision_token_id'}
    _require(set(config) <= known | legacy,
             'unknown native text config fields: ' + ', '.join(sorted(set(config) - known - legacy)))
    expected_types = {kind, 'qwen2_5_vl'} if kind == 'qwen2_5_vl_vision' else {kind}
    _require(config.get('model_type', kind) in expected_types, 'wrong native text submodel type')
    if 'torch_dtype' in config and 'dtype' in config and config['torch_dtype'] is not None and config['dtype'] is not None:
        _require(config['torch_dtype'] == config['dtype'], 'conflicting native dtype aliases')
    if kind in ('qwen3', 'qwen2_5_vl_text'):
        _check_rope(config_class, config)
    for name, child in getattr(config_class, 'sub_configs', {}).items():
        if name in config:
            text_config(child, config[name])
    if kind == 'qwen2_5_vl':
        text_class = config_class.sub_configs['text_config']
        flat = {key: deepcopy(value) for key, value in config.items() if key in flat_keys}
        # Validate flat fields even when a nested document would make the native
        # parent ignore them. Compare their *native effective* fields: e.g. the
        # public sliding_window=32768 and nested None agree when disabled.
        flat_native = text_config(text_class, flat)
        if 'text_config' in config:
            nested = text_config(text_class, config['text_config'])
            for key in flat:
                if key in ('architectures', 'transformers_version', 'dtype'):
                    # Parent/child serialization provenance and declared storage
                    # dtype are independent, not conflicting text architecture.
                    continue
                field = 'rope_parameters' if key in ('rope_theta', 'rope_scaling', 'rope_parameters') else key
                left, right = getattr(flat_native, field), getattr(nested, field)
                if field == 'rope_parameters':
                    left, right = _rope_identity(left), _rope_identity(right)
                _require(left == right, 'conflicting flat/nested native text field: ' + key)
    # from_dict and native RoPE handling mutate nested dictionaries. Give them a
    # private copy, and never feed to_dict() back into a construction path.
    return config_class.from_dict(deepcopy(config))
