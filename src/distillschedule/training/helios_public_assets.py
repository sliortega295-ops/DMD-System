"""Pinned local cold-zero Helios assets; never Hub fallback or random bases.

This is the CPU1 FP32 reference contract, not the Step10/world16 asset ABI.
All model bytes are admitted before materializing the original meta model.
"""
from __future__ import annotations
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path

from distillschedule.workloads.best_known.registry import file_sha256

SCHEMA = 'distillschedule.helios_cpu_reference_assets.v1'
FIXTURE = 'FULL_ORIGINAL_TINY_CPU_FIXTURE_NOT_PRETRAINED_OR_QUALITY'
REAL = 'LOCAL_PRETRAINED_NATIVE_FEATURES_UNVERIFIED_QUALITY'
EVIDENCE_KINDS = (FIXTURE, REAL)


def require(ok, message):
    if not ok:
        raise ValueError('Helios CPU reference assets: ' + message)


def read_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, 'duplicate JSON key: ' + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=pairs)
    json.dumps(value, allow_nan=False)
    return value


def local_file(path):
    require(isinstance(path, str) and '://' not in path and Path(path).is_absolute(),
            'explicit absolute local file path required')
    candidate = Path(path)
    require(candidate.resolve() == candidate and candidate.is_file(),
            'missing file or symlink/noncanonical path: ' + path)
    return candidate


def checked_pin(value):
    require(isinstance(value, dict) and set(value) == {'path','bytes','sha256'}, 'exact file pin required')
    path = local_file(value['path'])
    require(type(value['bytes']) is int and value['bytes'] > 0 and path.stat().st_size == value['bytes']
            and file_sha256(path) == value['sha256'], 'file pin changed: ' + str(path))
    return path


def runtime_identity():
    result = {}
    for name in ('torch','diffusers','peft','numpy','omegaconf','transformers','safetensors'):
        module = importlib.import_module(name)
        entry = Path(module.__file__).resolve()
        result[name] = dict(version=importlib.metadata.version(name),imported_version=str(module.__version__),
                            path=str(entry),entry_sha256=file_sha256(entry))
    return result


def read_args(pin):
    from omegaconf import OmegaConf
    from distillschedule.strategies.wan_capture import native_primitives
    native_primitives()
    from helios.utils.train_config import Args
    value = read_json(checked_pin(pin))
    args = OmegaConf.merge(OmegaConf.structured(Args), value)
    require(OmegaConf.to_container(args, resolve=True) == value,
            'effective complete original Args content required (no implicit defaults)')
    return args


def inspect_safe(pin, *, evidence_kind):
    from safetensors import safe_open, SafetensorError
    path = checked_pin(pin)
    require(path.suffix == '.safetensors', 'safe local safetensors weights required')
    try:
        with safe_open(path, framework='pt', device='cpu') as handle:
            require((handle.metadata() or {}).get('evidence_kind') == evidence_kind,
                    'weight evidence kind disagrees with immutable payload metadata')
            result = {}
            for name in handle.keys():
                view = handle.get_slice(name)
                require(view.get_dtype() == 'F32', 'CPU reference weights must already be FP32')
                result[name] = tuple(view.get_shape())
    except SafetensorError as exc:
        raise ValueError('invalid safe local weight file: ' + str(path)) from exc
    return path, result


def admit_assets(parameters):
    from distillschedule.adapters.helios.reference_model import reference_model_identity
    from .helios_step_owner import HeliosStepOwner
    require(set(parameters) <= {'asset_manifest','seed','device'}, 'unsupported factory parameters')
    require(parameters.get('device','cpu') == 'cpu', 'CPU1 only; no CUDA substitution')
    HeliosStepOwner._validate_topology()
    path = local_file(parameters.get('asset_manifest'))
    value = read_json(path)
    require(isinstance(value,dict) and set(value) == {'schema_version','evidence_kind','source','runtime',
        'args','negative','generator','critic','geometry','budget'}, 'asset manifest fields changed')
    require(value['schema_version'] == SCHEMA and value['evidence_kind'] in EVIDENCE_KINDS,
            'unsupported asset schema/evidence kind')
    require(value['source'] == reference_model_identity(backend='torch-sdpa'), 'original source/provider pin changed')
    require(value['runtime'] == runtime_identity(), 'dependency identity changed')
    budget = value['budget']
    require(isinstance(budget,dict) and set(budget)=={'max_parameters','max_loaded_bytes'}
            and type(budget['max_parameters']) is int and 0 < budget['max_parameters'] <= 100000000
            and type(budget['max_loaded_bytes']) is int and 0 < budget['max_loaded_bytes'] <= 2**31,
            'explicit bounded CPU parameter/loaded-byte budget required')
    pins, count = [], 0
    for role in ('generator','critic'):
        spec = value[role]
        require(isinstance(spec,dict) and set(spec)=={'config','weights'} and isinstance(spec['weights'],list)
                and spec['weights'], 'role needs explicit config and every safetensors shard (no implicit index lookup)')
        checked_pin(spec['config'])
        shapes = {}
        for pin in spec['weights']:
            weight_path, layout = inspect_safe(pin, evidence_kind=value['evidence_kind'])
            require(not shapes.keys() & layout.keys(), 'duplicate weight keys across shards')
            shapes.update(layout)
            pins.append(str(weight_path))
        require(shapes, 'empty base weights')
        import math
        count += sum(math.prod(shape) for shape in shapes.values())
    _, negative = inspect_safe(value['negative'], evidence_kind=value['evidence_kind'])
    require(negative == {'negative_prompt_embeds':tuple(value['geometry']['prompt_shape'])},
            'negative conditioning keys/dimensions changed')
    pins.append(value['negative']['path'])
    require(len(set(pins)) == len(pins), 'duplicate asset payload paths')
    total = sum(pin['bytes'] for role in ('generator','critic') for pin in value[role]['weights'])
    total += value['negative']['bytes']
    require(count <= budget['max_parameters'] and total <= budget['max_loaded_bytes'], 'CPU asset size budget exceeded')
    args = read_args(value['args'])
    # Validate the source branch before constructing a potentially large model.
    probe = object.__new__(HeliosStepOwner)
    probe.args = args
    probe._validate_config()
    c = args.training_config
    require(list(c.history_sizes)==[16,2,1] and c.is_use_gt_history and not c.is_amplify_first_chunk,
            'native producer requires GT history [16,2,1] without amplified first chunk')
    require(type(c.max_train_steps) is int and c.max_train_steps > 0 and c.max_train_steps % 5 == 0,
            'fixed original scheduler horizon must be complete five-step cycles')
    geometry = value['geometry']
    require(isinstance(geometry,dict) and set(geometry)=={'noise_shape','prompt_shape'}, 'invalid geometry fields')
    noise, prompt = geometry['noise_shape'], geometry['prompt_shape']
    require(isinstance(noise,list) and len(noise)==5 and noise[:2]==[1,16]
            and noise[2]==c.num_critic_input_frames and all(type(x) is int and x>0 for x in noise)
            and noise[-2]%16==noise[-1]%16==0
            and isinstance(prompt,list) and len(prompt)==3 and prompt[0]==1
            and all(type(x) is int and x>0 for x in prompt), 'native CPU geometry is unsupported')
    return path, value, args


def load_base(cls, spec, *, evidence_kind, remaining_parameters):
    import torch
    from safetensors.torch import load_file
    config_path = checked_pin(spec['config'])
    config = read_json(config_path)
    allowed = set(inspect.signature(cls.__init__).parameters) - {'self'}
    require(all(key in allowed or key.startswith('_') for key in config), 'unknown original architecture config key')
    # A meta instance is the actual model owner, not a second hidden model.
    with torch.device('meta'):
        model = cls.from_config(config)
    expected = model.state_dict()
    require(sum(p.numel() for p in model.parameters()) <= remaining_parameters, 'CPU architecture parameter budget exceeded')
    paths, shapes = [], {}
    for pin in spec['weights']:
        path, layout = inspect_safe(pin, evidence_kind=evidence_kind)
        require(not shapes.keys() & layout.keys(), 'duplicate base keys')
        paths.append(path)
        shapes.update(layout)
    require(shapes == {name:tuple(t.shape) for name,t in expected.items()},
            'missing/extra base keys or architecture/weight layout changed; no random fallback')
    weights = {}
    for path in paths:
        weights.update(load_file(path, device='cpu'))
    require(all(t.dtype==torch.float32 and t.isfinite().all().item() for t in weights.values()), 'nonfinite base weights')
    # Strict assignment materializes only the admitted base tensors. to_empty
    # would ALSO erase nonpersistent native RoPE buffers, absent from the file.
    nonpersistent = {name:t for name,t in model.named_buffers() if name not in expected}
    rope_buffers = dict(model.rope.named_buffers())
    require(set(nonpersistent)=={'rope.'+name for name in rope_buffers}
            and not any(True for _ in model.rope.parameters()),
            'unadmitted nonpersistent original model buffers; no guessed initialization')
    require(sum(t.numel() for t in nonpersistent.values()) <= remaining_parameters,
            'nonpersistent original buffer construction exceeds CPU budget')
    model.load_state_dict(weights,strict=True,assign=True)
    from .checkpoint import rng_state
    from .helios_step_owner import _content
    before = _content(rng_state())
    # The actual original tiny positional constructor owns its deterministic
    # frequency math. Do not copy the formula or create another G/C model.
    rope = type(model.rope)(rope_dim=model.config.rope_dim,theta=model.config.rope_theta)
    require(_content(rng_state())==before, 'original positional constructor unexpectedly consumed training RNG')
    require({name:(tuple(t.shape),t.dtype) for name,t in rope.named_buffers()} ==
            {name:(tuple(t.shape),t.dtype) for name,t in rope_buffers.items()},
            'original nonpersistent buffer layout/dtype changed')
    model.rope = rope
    require(all(t.device.type=='cpu' and t.dtype==torch.float32 and t.isfinite().all().item()
                for t in [*model.parameters(),*model.buffers()]), 'meta/uninitialized/nonfinite model tensors remain')
    materialized = model.state_dict()
    require(all(torch.equal(materialized[name],tensor) for name,tensor in weights.items()),
            'base tensor values changed during original buffer initialization')
    for pin in [spec['config'],*spec['weights']]:
        checked_pin(pin)
    return model
