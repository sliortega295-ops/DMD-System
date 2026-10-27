#!/usr/bin/env python3
"""Exercise the wheel outside the checkout, reusing installed CPU dependencies.

No downloads or model assets. The isolated package installation is new; Torch,
NumPy and Pillow are reused from the invoking validated environment through a
dependency-only .pth path. Actual framework imports must be in the new venv.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
import site
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def dependency_roots(checkout, *, paths=None):
    """Keep dependency-only import roots in the invoking interpreter's order.

    An overlay may intentionally provide Torch 2.10 before a base environment's
    Torch 2.14. site.getsitepackages() alone discards that qualification. Never
    expose the checkout or another framework/runtime tree to an installed-wheel
    acceptance, and never emit executable or multi-line .pth directives.
    """
    checkout = Path(checkout).resolve()
    ordered = list(sys.path if paths is None else paths)
    if paths is None:
        ordered.extend(site.getsitepackages())
    result = []
    for item in ordered:
        if not isinstance(item, str) or not item or any(c in item for c in ('\n', '\r')):
            continue
        root = Path(item).resolve()
        if (not root.is_dir() or root.is_relative_to(checkout) or
                (root / 'distillschedule').exists() or (root / 'runtime').exists()):
            continue
        if not (root.name in ('site-packages', 'dist-packages') or
                next(root.glob('*.dist-info'), None) is not None):
            continue
        if str(root) not in result:
            result.append(str(root))
    return result


def qualified_dependencies(roots, *, modern=False, official=False):
    """Record the real imported numerical stack, rather than guessing versions."""
    import importlib
    import importlib.metadata
    packages = {'torch':'torch', 'numpy':'numpy', 'Pillow':'PIL', 'safetensors':'safetensors'}
    if modern:
        packages.update({'diffusers':'diffusers', 'transformers':'transformers', 'peft':'peft',
                         'tokenizers':'tokenizers', 'accelerate':'accelerate', 'huggingface-hub':'huggingface_hub'})
    if official:
        if modern:
            raise ValueError('official-workflows and modern-flow require separate environments')
        packages.update({'diffusers':'diffusers', 'transformers':'transformers', 'peft':'peft',
                         'accelerate':'accelerate', 'ftfy':'ftfy', 'einops':'einops',
                         'omegaconf':'omegaconf', 'regex':'regex', 'sentencepiece':'sentencepiece'})
        from packaging.specifiers import SpecifierSet
        official_requirements = dict(torch='==2.10.0', diffusers='==0.37.1', transformers='==4.51.3',
            peft='==0.17.1', accelerate='>=1.1,<2', ftfy='==6.3.1', einops='==0.8.1',
            omegaconf='>=2.3', regex='>=2024.1', sentencepiece='==0.2.2', safetensors='>=0.4', numpy='>=1.24')
    result = {}
    modern_pins = dict(torch='2.10.0', diffusers='0.37.1', transformers='5.14.1', peft='0.18.1',
                       tokenizers='0.22.2', accelerate='1.14.0', huggingface_hub='1.7.1', safetensors='0.8.0')
    for distribution, name in packages.items():
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        if not any(origin.is_relative_to(Path(root)) for root in roots):
            raise RuntimeError('dependency origin is not a reusable package root: ' + str(origin))
        version = str(module.__version__) if hasattr(module, '__version__') else importlib.metadata.version(distribution)
        if modern and name in modern_pins and version.split('+')[0] != modern_pins[name]:
            raise RuntimeError('modern-flow requires ' + name + '==' + modern_pins[name] + ', found ' + version)
        if official and name in official_requirements and version not in SpecifierSet(official_requirements[name]):
            raise RuntimeError('official-workflows requires ' + name + official_requirements[name] + ', found ' + version)
        result[name] = {'version':version, 'origin':str(origin)}
    return result


# The verifier observes real CPU input values, not independently logged IDs.
# This instrumentation exists only in the explicit installed tiny-test lane.
OFFICIAL_INPUT_WITNESS = r'''
def materialize_cpu_reference_storage(*modules):
    """Post-training CPU reference only: match fresh export-owned storage.

    Safetensor views can have nonzero SIMD alignment. CPU Linear reductions
    then differ by an ulp despite identical weights, inputs and strides. A
    portable loader owns fresh allocations. Keep the strict tensor comparison
    by matching that allocation policy, never by changing values/tolerances.
    """
    import torch
    seen = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            assert parameter.device.type == 'cpu' and not parameter.is_meta
            original = parameter.detach()
            parameter.data = original.clone(memory_format=torch.preserve_format)
            assert torch.equal(parameter, original) and parameter.stride() == original.stride()

import contextlib, hashlib, json
def feed_digest(value):
    import torch
    def encode(item):
        if isinstance(item, torch.Tensor):
            assert item.device.type == 'cpu', 'installed CPU feed witness received a non-CPU tensor'
            raw = item.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            return ['tensor',str(item.dtype),list(item.shape),hashlib.sha256(raw).hexdigest()]
        if isinstance(item, dict):
            return ['dict',[[key,encode(item[key])] for key in sorted(item)]]
        if isinstance(item, (tuple,list)):
            return [type(item).__name__,[encode(child) for child in item]]
        assert item is None or type(item) in (str,int,float,bool), type(item)
        return [type(item).__name__,item]
    return hashlib.sha256(json.dumps(encode(value),separators=(',',':')).encode()).hexdigest()
def require_distinct_sd_conditions(rows):
    values = [{key:value for key,value in row.items() if key.startswith('text_input_ids_')} for row in rows]
    assert values and all(values), 'missing native SD conditions'
    digests = [feed_digest(value) for value in values]
    assert len(set(digests)) == len(digests), 'installed verifier requires distinguishable SD conditions'
@contextlib.contextmanager
def observe_sd_inputs(owner_type):
    original = owner_type._native_iteration
    observed = []
    def record(owner, prompt, denoising_dict, real_train_dict):
        observed.append(dict(native_step=owner.native_step,prompt=feed_digest(prompt),
            denoising=feed_digest(denoising_dict),real=feed_digest(real_train_dict)))
        return original(owner,prompt,denoising_dict,real_train_dict)
    owner_type._native_iteration = record
    try:
        yield observed
    finally:
        owner_type._native_iteration = original
def validate_sd_consumption(observed, cycles, *, ratio, start=0):
    expected = []
    for cycle in range(start,len(cycles)):
        feeds = cycles[cycle]
        for offset in range(ratio):
            expected.append(dict(native_step=cycle*ratio+offset,
                prompt=feed_digest(feeds['generator'][0] if offset==0 else feeds['guidance'][offset-1]),
                denoising=feed_digest(feeds['denoising'][offset] if 'denoising' in feeds else None),
                real=feed_digest(feeds['real'][offset] if 'real' in feeds else None)))
    assert observed == expected, 'actual native input consumption differs from independently sampled conditions'
'''

# Every official CLI/fixture child checks its own real import origins, including
# transitive imports after the operation. -I alone does not prove provenance.
OFFICIAL_CHILD_BOOTSTRAP = OFFICIAL_INPUT_WITNESS + r'''
import contextlib, hashlib, json, runpy, sys, sysconfig
from pathlib import Path
checkout = Path(sys.argv[1]).resolve()
receipt = Path(sys.argv[2])
mode, arguments = sys.argv[3], sys.argv[4:]
site = Path(sysconfig.get_paths()['purelib']).resolve()
import distillschedule, runtime
from distillschedule.recipes import directory
recipes = directory().resolve()
def installed():
    assert sys.flags.isolated == 1
    assert not Path.cwd().resolve().is_relative_to(checkout)
    assert recipes.is_relative_to(Path(sys.prefix).resolve()) and not recipes.is_relative_to(checkout)
    assert all(not Path(path).resolve().is_relative_to(checkout) for path in sys.path if path)
    paths = {}
    for name, module in tuple(sys.modules.items()):
        if (name == 'distillschedule' or name.startswith('distillschedule.') or
                name == 'runtime' or name.startswith('runtime.')) and getattr(module, '__file__', None):
            origin = Path(module.__file__).resolve()
            assert origin.is_relative_to(site), (name, str(origin), str(site))
            paths[name] = str(origin)
    return paths
installed()
query = mode == 'query'
def audit(event, args):
    assert event not in ('socket.connect', 'socket.getaddrinfo'), 'offline installed acceptance: ' + event
    if query and event == 'import':
        assert args[0].split('.')[0] not in ('torch','diffusers','transformers','peft'), args[0]
    if query and event == 'open' and isinstance(args[0], (str, bytes)):
        assert Path(str(args[0])).suffix not in ('.pt','.bin','.safetensors'), args[0]
sys.addaudithook(audit)
native_inputs = []
if mode in ('query', 'cli'):
    from distillschedule.cli import main
    sd_training = (mode == 'cli' and arguments[0] == 'train'
        and json.loads(Path(arguments[1]).read_text())['model'].get('name') in ('sd15','sdxl'))
    observer = contextlib.nullcontext(native_inputs)
    if sd_training:
        from distillschedule.training.dmd2_owner import DMD2Owner
        observer = observe_sd_inputs(DMD2Owner)
    with observer as native_inputs:
        assert main(arguments) == 0
elif mode == 'sources':
    # The existing loader validates the exact upstream closure without importing
    # numerical definitions, constructing a model, or requiring system Git.
    from distillschedule.training import sd, self_forcing_source
    from distillschedule.training.upstream import PinnedSource
    identities = {}
    for name, root, provider in (('dmd2',arguments[0],sd), ('self_forcing',arguments[1],self_forcing_source)):
        source = PinnedSource(root, provider.SOURCE_FILES)
        identities[name] = dict(root=str(source.root), repository=provider.REPOSITORY,
                                revision=provider.REVISION, identity=source.validate())
    assert 'torch' not in sys.modules
    print(json.dumps(identities))
elif mode == 'fixture':
    name, output, source_root = arguments
    recipe = 'self_forcing_official' if name == 'self_forcing_official' else 'dmd2_sd15'
    helper = (recipes/recipe/'create_cpu_fixture.py').resolve(strict=True)
    assert helper.is_relative_to(recipes)
    with contextlib.redirect_stdout(sys.stderr):
        builder = runpy.run_path(str(helper))['build_fixture']
        options = dict(cycles=3, ratio=2)
        if recipe == 'dmd2_sd15':
            options['variant'] = 'sdxl' if name == 'dmd2_sdxl' else 'sd15'
        assets = builder(output, source_root, **options)
    config = Path(assets['config']); request = json.loads(config.read_text())
    if recipe == 'self_forcing_official':
        # Explicit tiny-fixture policy: witness real EMA initialization AND
        # updates within three ratio2 cycles, not the production warmup budget.
        request['model']['parameters']['settings']['ema_start_step'] = 2
        for stream in request['data']['streams'].values():
            stream['sampling'] = dict(kind='epochs', shuffle=True)
        config.write_text(json.dumps(request, indent=2)+'\n')
    else:
        # The recipe's longer shared prefix is truncated by its tiny tokenizer.
        # Only this opt-in fixture uses short, actually distinct leading tokens.
        prompts = Path(output)/'prompts.json'
        document = json.loads(prompts.read_text())
        for index,record in enumerate(document['records']):
            record['prompt'] = 'p'+str(index)
        prompts.write_text(json.dumps(document,indent=2)+'\n')
    assets['fixture_contract'] = dict(complete_cycles=3,native_ratio=2,
        ema_start_step=2 if recipe == 'self_forcing_official' else None)
    assets['recipe'] = str(helper)
    assets['recipe_sha256'] = hashlib.sha256(helper.read_bytes()).hexdigest()
    print(json.dumps(assets))
else:
    raise ValueError('unknown installed official child operation: ' + mode)
if query:
    assert not any(name.split('.')[0] in ('torch','diffusers','transformers','peft') for name in sys.modules)
    assert not any(name in sys.modules for name in ('distillschedule.training.sd','distillschedule.training.self_forcing'))
if 'torch' in sys.modules:
    assert not sys.modules['torch'].cuda.is_initialized()
receipt.write_text(json.dumps(dict(isolated=True,working_directory_outside_checkout=True,
    site_packages=str(site),recipes=str(recipes),module_paths=installed(),
    python_search_path=sys.path,queries_without_torch_cuda_or_weights=query,
    native_input_consumption=native_inputs), indent=2)+'\n')
'''


OFFICIAL_WORKFLOWS_ACCEPTANCE = 'CHILD_BOOTSTRAP = ' + repr(OFFICIAL_CHILD_BOOTSTRAP) + '\n' + OFFICIAL_INPUT_WITNESS + r'''
import hashlib, importlib, importlib.metadata, json, shutil, subprocess, sys, sysconfig
from pathlib import Path
def child_argv(python, checkout, receipt, mode, arguments):
    return [str(python), '-I', '-c', CHILD_BOOTSTRAP, str(checkout), str(receipt), mode, *map(str,arguments)]

# BEGIN OFFICIAL ACCEPTANCE
import distillschedule, runtime
from distillschedule.recipes import directory
site = Path(sysconfig.get_paths()['purelib']).resolve()
checkout = Path(sys.argv[1]).resolve()
recipes = directory().resolve()
expected_dependencies = json.loads(sys.argv[2])
source_roots = json.loads(sys.argv[3])
def installed_parent():
    assert sys.flags.isolated == 1 and not Path.cwd().resolve().is_relative_to(checkout)
    assert recipes.is_relative_to(Path(sys.prefix).resolve()) and not recipes.is_relative_to(checkout)
    assert all(not Path(path).resolve().is_relative_to(checkout) for path in sys.path if path)
    paths = {}
    for name,module in tuple(sys.modules.items()):
        if (name == 'distillschedule' or name.startswith('distillschedule.') or
                name == 'runtime' or name.startswith('runtime.')) and getattr(module,'__file__',None):
            origin = Path(module.__file__).resolve()
            assert origin.is_relative_to(site), (name,str(origin))
            paths[name] = str(origin)
    return paths
installed_parent()
def offline(event,args):
    assert event not in ('socket.connect','socket.getaddrinfo'), 'offline installed acceptance: ' + event
sys.addaudithook(offline)
children = Path('children').resolve(); children.mkdir()
commands = []
def command(mode, arguments):
    receipt = children/(str(len(commands)).zfill(3)+'-'+mode+'.json')
    argv = child_argv(sys.executable,checkout,receipt,mode,arguments)
    result = subprocess.run(argv,cwd=Path.cwd(),capture_output=True,text=True)
    row = dict(argv=argv,cwd=str(Path.cwd()),returncode=result.returncode)
    commands.append(row)
    with Path('subprocess.log').open('a') as stream:
        stream.write(json.dumps(row)+'\n'+result.stdout+result.stderr+'\n')
    if result.returncode:
        raise RuntimeError('installed official '+mode+' failed:\n'+result.stdout+result.stderr)
    row['origin'] = json.loads(receipt.read_text())
    return result.stdout if mode == 'query' else json.loads(result.stdout)
def cli(*arguments):
    return command('cli',arguments)
queries = []
for arguments in (['doctor'],['workloads','list'],['models','list'],['capabilities','list'],
        ['config','validate',str(recipes/'dmd2_sd15/training.json')],
        ['config','validate',str(recipes/'self_forcing_official/training.json')]):
    queries.append(dict(arguments=arguments,output=command('query',arguments)))
assert 'torch' not in sys.modules
sources = command('sources',[source_roots['dmd2'],source_roots['self_forcing']])
import torch
import numpy as np
from PIL import Image
from distillschedule.data import PreparedDataset
from distillschedule.training.checkpoint import cpu_tree
from distillschedule.training.config import load_training_config
from distillschedule.training.session import construct, _load_for_output
from distillschedule.training.data_streams import InputStreams
from distillschedule.workloads.best_known.registry import file_sha256
for name,expected in expected_dependencies.items():
    module = importlib.import_module(name)
    version = str(module.__version__) if hasattr(module,'__version__') else importlib.metadata.version(name)
    assert version == expected['version'], (name,version,expected)
    assert str(Path(module.__file__).resolve()) == expected['origin'], (name,module.__file__,expected)
assert not torch.cuda.is_initialized()
def equal(x,y):
    if isinstance(x,torch.Tensor): return isinstance(y,torch.Tensor) and x.dtype==y.dtype and x.shape==y.shape and torch.equal(x,y)
    if isinstance(x,dict): return isinstance(y,dict) and x.keys()==y.keys() and all(equal(v,y[k]) for k,v in x.items())
    if isinstance(x,(tuple,list)): return type(x) is type(y) and len(x)==len(y) and all(equal(a,b) for a,b in zip(x,y))
    return type(x) is type(y) and x==y
def changed_parameters(before,after,ownership):
    return all(any(not equal(before[name],after[name]) for group in names for name in group)
               for names in ownership.values())
workflows = {}
for name in ('dmd2_sd15','dmd2_sdxl','self_forcing_official'):
    video = name == 'self_forcing_official'
    root = Path(name).resolve(); root.mkdir()
    assets = command('fixture',[name,root/'training-inputs',source_roots['self_forcing' if video else 'dmd2']])
    config = Path(assets['config']); request = json.loads(config.read_text())
    preparer = 'distillschedule.data.'+('self_forcing_inputs' if video else 'sd_inputs')+':prepare_inputs'
    prepared_paths = {}
    if video:
        for role in ('generator','critic'):
            prepared_paths[role] = cli('data','prepare',root/'training-inputs'/(role+'-inputs.json'),
                '--output',root/'training-inputs/public-prepared'/role,'--preparer',preparer)['manifest']
    else:
        prepared = cli('data','prepare',root/'training-inputs/prompts.json',
            '--output',root/'training-inputs/public-prepared','--preparer',preparer)['manifest']
        prepared_paths = dict.fromkeys(request['data']['streams'],prepared)
    for role,path in prepared_paths.items():
        request['data']['streams'][role]['prepared'] = path
    config.write_text(json.dumps(request,indent=2)+'\n')
    command('query',['config','validate',config])
    for path in set(prepared_paths.values()):
        cli('data','validate',path,'--sources')
    training_config = load_training_config(config)
    baseline_components,data,_,_ = construct(training_config)
    assert isinstance(data,InputStreams)
    with baseline_components.managed():
        initial = cpu_tree(baseline_components.state['native_cycle'].state_dict())
    expected_ids = [data.sample_id(cycle) for cycle in range(3)]
    training_conditions = [data.cycle_sample(cycle) for cycle in range(3)]
    if not video:
        for dataset in data.datasets.values():
            require_distinct_sd_conditions([dataset.sample(index) for index in range(len(dataset))])
        # Read the real prepared rows at independently derived per-stream draws;
        # do not rely on the training cycle's sample_id/progress bookkeeping.
        expected_feeds = [{role:[data.datasets[role].sample(data.samplers[role].position(cycle*count+draw).sample_index)
            for draw in range(count)] for role,count in data.counts.items()} for cycle in range(3)]
    first = cli('train',config,'--cycles',1,'--output',root/'partial')
    first_inputs = commands[-1]['origin']['native_input_consumption']
    resumed = cli('train',config,'--cycles',3,'--resume',first['checkpoint'],'--output',root/'resumed')
    resumed_inputs = commands[-1]['origin']['native_input_consumption']
    whole = cli('train',config,'--cycles',3,'--output',root/'whole')
    whole_inputs = commands[-1]['origin']['native_input_consumption']
    if not video:
        validate_sd_consumption(first_inputs,expected_feeds[:1],ratio=2)
        validate_sd_consumption(resumed_inputs,expected_feeds,ratio=2,start=1)
        validate_sd_consumption(whole_inputs,expected_feeds,ratio=2)
    assert resumed['resume_cycle']==1 and resumed['cycles_this_process']==2
    assert whole['completed_cycles']==3 and first['completed_cycles']==1
    a = torch.load(Path(resumed['checkpoint'])/'state.pt',weights_only=True)
    b = torch.load(Path(whole['checkpoint'])/'state.pt',weights_only=True)
    assert equal(a,b), name+': ALL packed model/Adam/scheduler/EMA/native+outer RNG/data/policy/versions must match'
    assert a['cycle']==a['cursor']==3 and a['versions']=={'native_cycle':3}
    assert a['data_progress']==data.progress(3)
    native = a['state']['native_cycle']['value']
    assert native['native_step']==6 and native['generator_updates']==3
    assert native['consumed']=={role:3*spec['samples_per_cycle'] for role,spec in request['data']['streams'].items()}
    assert changed_parameters(initial['model'],native['model'],native['optimizer_ownership'])
    assert all(optimizer['state'] for optimizer in native['optimizers'].values())
    assert native['rng'] and a['rng'] and not equal(initial['rng'],native['rng'])
    if video:
        assert native['critic_updates']==6 and native['ema_initialized_step']==2 and native['ema_updates']==2
        assert native['ema'] is not None and 'schedulers' not in native
        ema,schedulers = dict(initialized_step=2,updates=2),'NOT_CONFIGURED'
        frozen = ('real_score.','vae.')
    else:
        assert native['guidance_updates']==6 and 'ema' not in native
        schedulers = {role:value['last_epoch'] for role,value in native['schedulers'].items()}
        assert schedulers==dict(generator=6,guidance=6)
        ema = 'NOT_CONFIGURED'; frozen = ('guidance_model.real_unet.','vae.')
    assert all(equal(value,native['model'][key]) for key,value in initial['model'].items() if key.startswith(frozen))
    rows = [json.loads(line) for line in (root/'whole/metrics.jsonl').read_text().splitlines()]
    continued = [json.loads(line) for line in (root/'resumed/metrics.jsonl').read_text().splitlines()]
    assert [row['sample_id'] for row in rows]==expected_ids
    assert [row['sample_id'] for row in continued]==expected_ids[1:]
    for row in rows:
        assert row['data_progress']==data.progress(row['cycle'])
        assert row['versions']=={'native_cycle':row['cycle']}
    exported = cli('export',config,'--checkpoint',resumed['checkpoint'],'--output',root/'export')
    assert exported['checkpoint_cycle']==3
    assert set(exported['model']['modules'])==({'generator','vae'} if video else {'generator','text_encoder','vae'})
    assert not {'model_id','assets'} & exported['model']['parameters'].keys()
    weights = torch.load(root/'export/weights.pt',weights_only=True)
    current = _load_for_output(training_config,resumed['checkpoint'])[0]
    with current.managed():
        owner = current.state['native_cycle']
        source_key = 'self_forcing' if video else 'dmd2'
        admitted = sources[source_key]
        assert owner.source.root==Path(admitted['root'])
        assert owner.source.identity['files']==admitted['identity']['files']
        assert owner.source.identity['importer_sha256']==admitted['identity']['importer_sha256']
        assert exported['model']['parameters']['upstream']==owner.source.identity
        if 'native_identity' in admitted:
            assert admitted['native_identity']==owner.source.identity
        admitted['native_identity'] = owner.source.identity
        owner.source.validate()
        if video:
            from distillschedule.training.self_forcing import original
            api = original(source_roots['self_forcing'])
            assert isinstance(owner.model,api.dmd.DMD)
            assert owner.model.generator_loss.__func__ is api.dmd.DMD.generator_loss
            assert owner.model.critic_loss.__func__ is api.dmd.DMD.critic_loss
            assert type(owner.model.generator.model) is api.causal.CausalWanModel
            assert type(owner.model.real_score.model) is type(owner.model.fake_score.model) is api.wan.WanModel
            assert type(owner.model.vae.model) is api.vae.WanVAE_
            native_classes = [api.dmd.DMD,type(owner.model.generator.model),type(owner.model.fake_score.model),type(owner.model.vae.model)]
            assert equal(weights['generator'],cpu_tree(owner.model.generator.state_dict()))
            new_tensor = root/'new-embedding.pt'
            torch.save(torch.linspace(-.8,.9,128).reshape(1,8,16),new_tensor)
            from distillschedule.data.self_forcing_inputs import SCHEMA
            document = dict(schema_version=SCHEMA,mode='preencoded',records=[dict(id='NEW_NATIVE_CONDITION',
                embedding=dict(path=new_tensor.name,sha256=file_sha256(new_tensor)))])
        else:
            from distillschedule.training.sd import original
            from diffusers import UNet2DConditionModel, AutoencoderKL
            from transformers import CLIPTextModel
            api = original(source_roots['dmd2'])
            assert type(owner.model) is api.unified.SDUniModel
            assert type(owner.model.guidance_model) is api.guidance.SDGuidance
            assert type(owner.model.feedforward_model) is UNet2DConditionModel
            assert type(owner.model.vae) is AutoencoderKL
            assert type(owner.model.text_encoder) is (api.text.SDXLTextEncoder if name=='dmd2_sdxl' else CLIPTextModel)
            native_classes = [type(owner.model),type(owner.model.guidance_model),type(owner.model.feedforward_model),type(owner.model.text_encoder),type(owner.model.vae)]
            assert equal(weights['generator'],cpu_tree(owner.model.feedforward_model.state_dict()))
            from distillschedule.data.sd_inputs import SCHEMA
            document = dict(schema_version=SCHEMA,variant='sdxl' if name=='dmd2_sdxl' else 'sd15',
                model_id=assets['assets'],records=[dict(id='NEW_NATIVE_CONDITION',prompt='new zebra on the moon')])
        class_evidence = []
        for cls in native_classes:
            path = Path(importlib.import_module(cls.__module__).__file__).resolve()
            class_evidence.append(dict(name=cls.__name__,module=cls.__module__,source=str(path),sha256=file_sha256(path)))
        # Public preparation gets genuinely new native conditions, never saved
        # rollout tensors or training sample IDs. SD tokenization still needs
        # its legitimate local tokenizer assets AT PREPARATION TIME only.
        new_source = root/'new-inputs.json'; new_source.write_text(json.dumps(document)+'\n')
        prepared = cli('data','prepare',new_source,'--output',root/'new-prepared','--preparer',preparer)['manifest']
        cli('data','validate',prepared,'--sources')
        feeds = PreparedDataset(prepared).sample(0)
        assert all(not equal(feeds,row) for cycle in training_conditions for stream in cycle.values() for row in stream)
        materialize_cpu_reference_storage(*(
            (owner.model.generator,owner.model.vae) if video else
            (owner.model.feedforward_model,owner.model.text_encoder,owner.model.vae)))
        torch.random.default_generator.manual_seed(909)
        with torch.no_grad():
            generated = current.generate(current.state,feeds,{})
            expected = cpu_tree(dict(generated=generated,decoded=current.decode(generated,{})))
    inference = dict(schema_version='distillschedule.training.v1',
        model=dict(factory=exported['model']['factory'],parameters=exported['model']['parameters']),
        algorithm=({'factory':'distillschedule.training.self_forcing:build_self_forcing'} if video else {'name':'dmd2'}),
        data=dict(prepared=prepared),execution=dict(adapter='program',mode='reference'),
        trainer=dict(cycles=1,seed=909,device='cpu',output_dir=str(root/'unused')))
    inference_path = root/'inference.json'; inference_path.write_text(json.dumps(inference)+'\n')
    # Source integrations remain: export is bound to their exact original code.
    # Base model assets, all training conditions, configs and checkpoints do not.
    shutil.rmtree(root/'training-inputs')
    for folder in ('partial','resumed','whole'):
        shutil.rmtree(root/folder)
    assert not config.exists() and not Path(assets['assets']).exists() and not list(root.rglob('state.pt'))
    actual = cli('generate',inference_path,'--weights',root/'export','--prepared',prepared,
                 '--output',root/'generation','--output-kind','video' if video else 'image')
    artifact = torch.load(actual['artifact'],weights_only=True)
    assert equal(artifact,expected), name+': fresh export output must equal current native Student, not EMA/base/replayed data'
    assert actual['sample_id']=='NEW_NATIVE_CONDITION' and actual['checkpoint_cycle']==3
    assert actual['loaded_factory']==exported['model']['source']
    if video:
        pixels = artifact['decoded']['video']; assert pixels.shape==(81,32,32,3) and pixels.dtype==torch.uint8
        paths = sorted(Path(actual['video']['frames']).glob('*.png')); assert len(paths)==81
    else:
        pixels = artifact['decoded']['image']; assert pixels.shape==(16,16,3) and pixels.dtype==torch.uint8
        paths = [Path(actual['image']['path'])]; pixels = pixels.unsqueeze(0)
    for index,path in enumerate(paths):
        with Image.open(path) as image:
            assert torch.equal(torch.from_numpy(np.array(image)),pixels[index])
    workflows[name] = dict(evidence_kind='RANDOM_TINY_ACTUAL_NATIVE_CPU_ONLY',
        complete_packed_state_equal=True,fresh_process_resume=True,independent_stream_sampling_equal=True,
        real_native_classes=True,native_classes=class_evidence,source_identity_equal=True,
        complete_cycles=3,native_iterations=6,native_ratio=2,
        native_optimizer_roles_updated=True,teacher_unchanged=True,ema=ema,schedulers=schedulers,
        sample_ids=expected_ids,input_progress=a['data_progress'],
        actual_native_input_consumption=({'partial':first_inputs,'resumed':resumed_inputs,'whole':whole_inputs}
            if not video else 'NOT_APPLICABLE_SD_ITERATION_WITNESS'),
        training_assets_checkpoints_deleted=True,new_conditions_differ_from_training=True,
        fresh_process_generation=True,exact_current_student_output=True,generated=actual,
        recipe=assets['recipe'],recipe_sha256=assets['recipe_sha256'],fixture_contract=assets['fixture_contract'],
        conditioning='EXPLICIT_RANDOM_NATIVE_EMBEDDINGS_NOT_T5_OUTPUT' if video else 'ACTUAL_CLIP_TOKENIZERS_AND_ENCODERS',
        retained_assets='EXACT_EXTERNAL_SOURCE_ONLY; NEW_PREPARED_CONDITIONS; EXPORT',
        new_inputs_prepared_before_training_asset_deletion=True)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_INSTALLED_OFFICIAL_WORKFLOWS_CPU',workflows=workflows,commands=commands,
    queries_without_torch_cuda_or_weights=True,queries=queries,external_sources=sources,
    scope='RANDOM_TINY_ACTUAL_NATIVE_CPU_ONLY',
    source=dict(working_directory_outside_checkout=True,site_packages=str(site),recipes=str(recipes),
        module_paths=installed_parent(),python_search_path=sys.path,python_version=sys.version.split()[0],
        qualified_dependencies=expected_dependencies),
    gpu_execution='NOT_RUN',pretrained_quality='NOT_RUN',capacity_validation='NOT_RUN',quality_validation='NOT_RUN')))
'''


MODERN_FLOW_ACCEPTANCE = r'''
import contextlib, hashlib, importlib, io, json, shutil, subprocess, sys, sysconfig
from pathlib import Path
import distillschedule, runtime
site = Path(sysconfig.get_paths()['purelib']).resolve()
checkout = Path(sys.argv[1]).resolve()
assert not Path.cwd().resolve().is_relative_to(checkout)
assert Path(distillschedule.__file__).resolve().is_relative_to(site)
assert Path(runtime.__file__).resolve().is_relative_to(site)
assert all(not Path(path).resolve().is_relative_to(checkout) for path in sys.path if path)
from distillschedule.recipes import directory
recipes = directory()
assert recipes.is_relative_to(Path(sys.prefix)) and not recipes.is_relative_to(checkout)
queries = []
query_phase = True
def audit(event, args):
    if query_phase and event == 'open' and isinstance(args[0], (str, bytes)):
        assert Path(str(args[0])).suffix not in ('.pt', '.bin', '.safetensors'), args[0]
sys.addaudithook(audit)
from distillschedule.cli import main
for arguments in (['doctor'], ['workloads','list'], ['models','list'], ['capabilities','list'],
                  ['config','validate',str(recipes/'flux_dmd/training.template.json')]):
    with contextlib.redirect_stdout(io.StringIO()) as output:
        assert main(arguments) == 0
    queries.append(dict(arguments=arguments, output=output.getvalue()))
assert not any(name in sys.modules for name in ('torch','diffusers','transformers','peft'))
query_phase = False
import numpy as np
import torch
from PIL import Image
from distillschedule.data import PreparedDataset
from distillschedule.data.manifest import source_identity
from distillschedule.training.config import load_training_config
from distillschedule.training.session import _load_for_output
expected_dependencies = json.loads(sys.argv[2])
for name, expected in expected_dependencies.items():
    module = importlib.import_module(name)
    assert str(module.__version__) == expected['version'], (name, module.__version__, expected)
    assert str(Path(module.__file__).resolve()) == expected['origin'], (name, module.__file__, expected)
assert not torch.cuda.is_initialized()
commands = []
def command(arguments):
    argv = [sys.executable, '-I', *map(str, arguments)]
    result = subprocess.run(argv, text=True, capture_output=True)
    commands.append(dict(argv=argv, cwd=str(Path.cwd()), returncode=result.returncode))
    with Path('subprocess.log').open('a') as stream:
        stream.write(json.dumps(commands[-1])+'\n'+result.stdout+result.stderr+'\n')
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return json.loads(result.stdout)
def cli(*arguments):
    return command(['-m','distillschedule',*arguments])
def equal(x,y):
    if isinstance(x,torch.Tensor): return isinstance(y,torch.Tensor) and torch.equal(x,y)
    if isinstance(x,dict): return isinstance(y,dict) and x.keys()==y.keys() and all(equal(v,y[k]) for k,v in x.items())
    if isinstance(x,(tuple,list)): return isinstance(y,type(x)) and len(x)==len(y) and all(equal(a,b) for a,b in zip(x,y))
    return x == y
families = {}
for family, recipe, shape, kind, size, preparer in (
        ('flux2-klein-base','flow_image',[1,4,4,4],'image',8,'flow_image_inputs'),
        ('wan22-ti2v5b-t2v','wan22_dmd',[1,4,2,2,2],'video',32,'wan22_inputs')):
    root = Path(family).resolve(); root.mkdir()
    helper = recipes/recipe/'create_cpu_fixture.py'
    arguments = [helper,root/'assets','--cycles',3]
    if kind == 'image': arguments += ['--family',family]
    assets = command(arguments)
    config = Path(assets['config'])
    prepared = cli('data','prepare',root/'assets/raw.json','--output',root/'assets/prepared-cli',
        '--preparer','distillschedule.data.'+preparer+':prepare_inputs')
    assets['prepared'] = prepared['manifest']
    request = json.loads(config.read_text()); request['data']['prepared'] = assets['prepared']
    config.write_text(json.dumps(request))
    cli('data','validate',assets['prepared'],'--sources')
    first = cli('train',config,'--cycles',1,'--output',root/'partial')
    resumed = cli('train',config,'--cycles',3,'--resume',first['checkpoint'],'--output',root/'resumed')
    whole = cli('train',config,'--cycles',3,'--output',root/'whole')
    a = torch.load(Path(resumed['checkpoint'])/'state.pt',weights_only=True)
    b = torch.load(Path(whole['checkpoint'])/'state.pt',weights_only=True)
    assert equal(a,b), 'complete state/RNG/Adam/schedulers/EMA/data/versions must match'
    assert a['state']['dmd_counters']['value']['counters'].tolist() == [3,3,15]
    rows = [json.loads(line) for line in (root/'whole/metrics.jsonl').read_text().splitlines()]
    continued = [json.loads(line) for line in (root/'resumed/metrics.jsonl').read_text().splitlines()]
    assert [row['sample_id'] for row in continued] == [row['sample_id'] for row in rows][1:]
    current = _load_for_output(load_training_config(config),resumed['checkpoint'])[0]
    training_feeds = PreparedDataset(assets['prepared']).sample(0)
    with torch.no_grad():
        sigma = torch.tensor([.5])
        student = current.state['student'].denoise(training_feeds['latents'],sigma,training_feeds,conditional=True)
        teacher = current.state['teacher'].denoise(training_feeds['latents'],sigma,training_feeds,conditional=True)
    assert not equal(student,teacher), 'current Student must be an actual updated model'
    exported = cli('export',config,'--checkpoint',resumed['checkpoint'],'--output',root/'export')
    assert set(exported['model']['modules']) == {'student','media'}
    native_class = type(current.state['student'].native)
    source_path = Path(importlib.import_module(native_class.__module__).__file__).resolve()
    for folder in ('partial','resumed','whole'):
        shutil.rmtree(root/folder)
    shutil.rmtree(root/'assets')
    assert not list(root.rglob('state.pt')) and not config.exists()
    source = dict(schema_version='distillschedule.export_inputs.v1', weights='export',mode='prompts',
        records=[dict(id='NEW_AFTER_DELETION',prompt='a violet kite',latent_shape=shape)])
    (root/'new.json').write_text(json.dumps(source))
    prepared_result = cli('data','prepare',root/'new.json','--output',root/'new-prepared',
        '--preparer','distillschedule.data.export_inputs:prepare_inputs')
    dataset = PreparedDataset(prepared_result['manifest']); assert dataset.validate(sources=True)['status']=='PASS'
    feeds = dataset.sample(0)
    assert not equal(feeds['context'],training_feeds['context'])
    assert equal(tuple(feeds[key] for key in ('context','context_mask')),current.state['media'].encode_text(['a violet kite']))
    inference = dict(schema_version='distillschedule.training.v1',
        model=dict(factory=exported['model']['factory'],parameters=exported['model']['parameters']),
        algorithm=dict(name='dmd'),data=dict(prepared=str(dataset.path)),execution=dict(mode='reference'),
        trainer=dict(cycles=1,seed=909,device='cpu',output_dir=str(root/'unused')))
    (root/'inference.json').write_text(json.dumps(inference))
    actual = cli('generate',root/'inference.json','--weights',root/'export','--prepared',prepared_result['manifest'],
        '--output',root/'generation','--output-kind',kind)
    for value in current.state.values():
        if isinstance(value,torch.nn.Module): value.eval()
    torch.random.default_generator.manual_seed(909)
    with torch.no_grad():
        generated=current.generate(current.state,feeds,{})
        expected=dict(generated=generated,decoded=current.decode(generated,{}))
    artifact=torch.load(actual['artifact'],weights_only=True)
    assert equal(expected,artifact), 'fresh portable output must equal native CURRENT Student'
    pixels=((np.arange(size*size*3).reshape(size,size,3)*11+37)%256).astype('uint8')
    if kind=='image':
        media=root/'new-image.png'; Image.fromarray(pixels).save(media)
        assert Path(actual['image']['path']).is_file() and artifact['decoded']['image'].shape==(8,8,3)
    else:
        media=root/'new-frames'; media.mkdir()
        for i in range(5): Image.fromarray(np.roll(pixels,i,axis=0)).save(media/f'{i:04d}.png')
        assert len(list(Path(actual['video']['frames']).glob('*.png')))==5
        assert artifact['decoded']['video'].shape==(5,32,32,3)
    source.update(mode='raw',records=[dict(id='NEW_RAW_AFTER_DELETION',prompt='a teal bird',
        **{kind:dict(path=media.name,sha256=source_identity(media,frames=kind=='video')['sha256'])})])
    (root/'raw.json').write_text(json.dumps(source))
    raw=cli('data','prepare',root/'raw.json','--output',root/'raw-prepared',
        '--preparer','distillschedule.data.export_inputs:prepare_inputs')
    raw_actual=cli('generate',root/'inference.json','--weights',root/'export','--prepared',raw['manifest'],
        '--output',root/'raw-generation','--output-kind',kind)
    assert raw_actual['sample_id']=='NEW_RAW_AFTER_DELETION'
    families[family]=dict(evidence_kind='SYNTHETIC_TEST_ONLY', native_class=native_class.__name__,
        native_source=str(source_path),native_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        recipe=str(helper),recipe_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        complete_state_rng_data_equal=True,fresh_process_resume=True,clocks=[3,3,15],
        sample_ids=[row['sample_id'] for row in rows],training_assets_checkpoints_deleted=True,
        current_student_differs_from_teacher=True,public_export_only_new_prompt=True,
        public_export_only_new_raw_media=True,exact_current_student_decode=True,
        generated=actual,raw_generated=raw_actual,export_manifest_sha256=actual['export_manifest_sha256'])
for name,module in tuple(sys.modules.items()):
    if (name=='distillschedule' or name.startswith('distillschedule.') or name=='runtime' or name.startswith('runtime.')) and getattr(module,'__file__',None):
        assert Path(module.__file__).resolve().is_relative_to(site), (name,module.__file__)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_INSTALLED_MODERN_FLOW_CPU',families=families,commands=commands,
    queries_without_torch_cuda_or_weights=True,queries=queries,scope='SYNTHETIC_TEST_ONLY',
    source=dict(working_directory_outside_checkout=True,site_packages=str(site),recipes=str(recipes),
        python_search_path=sys.path,
        module_paths=dict(distillschedule=str(Path(distillschedule.__file__).resolve()),runtime=str(Path(runtime.__file__).resolve())),
        python_version=sys.version.split()[0],qualified_dependencies=expected_dependencies),
    gpu_execution='NOT_RUN',pretrained_quality='NOT_RUN',capacity_validation='NOT_RUN',quality_validation='NOT_RUN')))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--modern-flow", action="store_true", help="actual tiny native image and video CPU acceptance")
    modes.add_argument("--official-workflows", action="store_true", help="actual tiny official DMD2 SD1.5/SDXL and Self-Forcing CPU acceptance")
    parser.add_argument("--dmd2-source", type=Path, help="explicit local pinned original DMD2 source tree; requires --official-workflows")
    parser.add_argument("--self-forcing-source", type=Path, help="explicit local pinned official Self-Forcing source tree; requires --official-workflows")
    parser.add_argument("--wheel", type=Path, help="validate this already-built wheel without rebuilding")
    parser.add_argument("--work-dir", type=Path, help="preserve installation/media/logs in a NEW directory outside checkout")
    args = parser.parse_args()
    if args.official_workflows:
        if args.dmd2_source is None or args.self_forcing_source is None:
            parser.error('--official-workflows requires both --dmd2-source and --self-forcing-source')
        for name in ('dmd2_source', 'self_forcing_source'):
            path = getattr(args, name).expanduser().resolve()
            if not path.is_dir():
                parser.error('--'+name.replace('_','-')+' must be an existing local directory')
            setattr(args, name, path)
    elif args.dmd2_source is not None or args.self_forcing_source is not None:
        parser.error('--dmd2-source and --self-forcing-source require --official-workflows')
    root = Path(__file__).resolve().parents[1]
    dependency_sites = dependency_roots(root)
    qualified = (qualified_dependencies(dependency_sites, official=True) if args.official_workflows else
                 qualified_dependencies(dependency_sites, modern=args.modern_flow))
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if args.work_dir:
        args.work_dir = args.work_dir.resolve()
        if args.work_dir.is_relative_to(root):
            raise ValueError('installed acceptance work directory must be outside the checkout')
        args.work_dir.mkdir(parents=True, exist_ok=False)
    context = nullcontext(str(args.work_dir)) if args.work_dir else tempfile.TemporaryDirectory(prefix="dmd-framework-wheel-")
    with context as scratch:
        work = Path(scratch)
        wheels = work / "wheels"
        wheels.mkdir()
        # Invoke the already installed PEP 517 backend directly. Validated uv
        # environments need not carry pip, and this must not modify them just
        # to build a wheel or fetch isolated build dependencies.
        if args.wheel:
            import shutil
            shutil.copy2(args.wheel.resolve(), wheels / args.wheel.name)
        else:
            subprocess.run([sys.executable, "-c", ("import sys; from setuptools.build_meta import build_wheel; "
                        "build_wheel(sys.argv[1])"), str(wheels)],
                       cwd=root, env=env, check=True, stdout=subprocess.PIPE, text=True)
        files = list(wheels.glob("distillschedule-*.whl"))
        if len(files) != 1:
            raise RuntimeError("expected one framework wheel")
        target = work / "isolated"
        venv.EnvBuilder(with_pip=True).create(target)
        python = target / "bin/python"
        subprocess.run([str(python), "-I", "-m", "pip", "install", "--no-index", "--no-deps", str(files[0])],
                       cwd=work, env=env, check=True, stdout=subprocess.PIPE, text=True)
        package_site = Path(subprocess.check_output([str(python), "-I", "-c",
                            "import sysconfig; print(sysconfig.get_paths()['purelib'])"], text=True).strip())
        (package_site / "reused-dependencies.pth").write_text("\n".join(dependency_sites) + "\n")
        script = r'''
import json, os, shutil, subprocess, sys, sysconfig
from pathlib import Path
import distillschedule, runtime
site = Path(sysconfig.get_paths()['purelib']).resolve()
checkout = Path(sys.argv[1]).resolve()
assert not Path.cwd().resolve().is_relative_to(checkout)
assert Path(distillschedule.__file__).resolve().is_relative_to(site)
assert Path(runtime.__file__).resolve().is_relative_to(site)
from distillschedule.cli import main
assert main(['doctor']) == 0
assert 'torch' not in sys.modules
from distillschedule.examples.tiny_av import QUALITY_STATUS, build_fixture
from distillschedule.data import PreparedDataset, prepare
from distillschedule.training.config import digest, load_training_config, TrainingConfig
from distillschedule.training.session import train, evaluate, export
import torch
expected_dependencies = json.loads(sys.argv[2])
import importlib
for name, expected in expected_dependencies.items():
    module = importlib.import_module(name)
    assert str(module.__version__) == expected['version'], (name, module.__version__, expected)
    assert str(Path(module.__file__).resolve()) == expected['origin'], (name, module.__file__, expected)
assert not torch.cuda.is_initialized()
def run_cli(*args):
    result = subprocess.run([sys.executable, '-I', '-m', 'distillschedule', *map(str, args)],
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return json.loads(result.stdout)
root = Path('demo').resolve()
build_fixture(root, samples=4)
prepare(root/'raw/manifest.json', root/'prepared')
training_inputs = PreparedDataset(root/'prepared/manifest.json')
config = load_training_config(root/'train.json')
raw = config.to_dict(); raw['trainer']['cycles'] = 2
first = train(TrainingConfig.from_dict(raw, source=config.source))
resumed = run_cli('train', root/'train.json', '--resume', first['checkpoint'],
                  '--cycles', 4, '--output', root/'resumed')
assert resumed['resume_cycle'] == 2 and resumed['cycles_this_process'] == 2
full = train(TrainingConfig.from_dict({**config.to_dict(),
    'trainer': {**dict(config.trainer), 'output_dir':str(root/'full')}}, source=config.source))
a = torch.load(Path(resumed['checkpoint'])/'state.pt', weights_only=True)
b = torch.load(Path(full['checkpoint'])/'state.pt', weights_only=True)
def equal(x,y):
    if isinstance(x,torch.Tensor): return isinstance(y,torch.Tensor) and torch.equal(x,y)
    if isinstance(x,dict): return x.keys()==y.keys() and all(equal(v,y[k]) for k,v in x.items())
    if isinstance(x,(tuple,list)): return len(x)==len(y) and all(equal(v,w) for v,w in zip(x,y))
    return x==y
assert all(equal(a[k],b[k]) for k in ('state','rng','cycle','cursor','versions','policy'))
evaluate(config, first['checkpoint'], root/'generated')
weights = Path('exported').resolve()
exported = export(config, first['checkpoint'], weights)
# Installed-source costs are measured with the same existing planner too.
auto = config.to_dict(); auto['trainer'].update(cycles=2, output_dir=str(root/'auto'))
auto['execution'] = dict(mode='auto', memory_limit_bytes=100000000,
    max_candidates=2, samples=1, max_program_runs=8)
planned = train(TrainingConfig.from_dict(auto, source=config.source))
assert planned['policy']['plan_sha256']
assert planned['policy']['initial_reference_gate']['status']=='PASS'
plan_path = root/'auto/planning/selected-plan.json'
saved = train(TrainingConfig.from_dict({**auto,
    'trainer': {**auto['trainer'], 'output_dir':str(root/'saved-plan')},
    'execution': {'mode':'plan', 'plan':str(plan_path)}}, source=config.source))
assert saved['policy']['plan_sha256'] == planned['policy']['plan_sha256']
assert saved['policy']['implementation'] == planned['policy']['implementation']
assert saved['final_versions'] == planned['final_versions']
assert not torch.cuda.is_initialized()
# A wheel installation must not require the development Git executable.
auto['trainer']['output_dir'] = str(root/'auto-no-git')
os.environ['PATH'] = str(root/'no-system-executables')
without_git = train(TrainingConfig.from_dict(auto, source=config.source))
assert without_git['policy']['initial_reference_gate']['status']=='PASS'
# The new record has genuinely different video/audio/text, not a deterministic
# copy of training records 0..3 at another path. Discard those unused copies.
new = Path('new-inputs').resolve()
build_fixture(new, samples=5)
raw_path = new/'raw/manifest.json'
document = json.loads(raw_path.read_text())
for record in document['records'][:-1]:
    shutil.rmtree(new/'raw'/record['id'])
record = document['records'][-1]
record['id'] = 'new-input-000004'
document['records'] = [record]
raw_path.write_text(json.dumps(document) + '\n')
(new/'train.json').unlink()
prepare(raw_path, new/'prepared')
new_inputs = PreparedDataset(new/'prepared/manifest.json')
training_hashes = {row['features']['sha256'] for row in training_inputs.manifest['records']}
new_hashes = {row['features']['sha256'] for row in new_inputs.manifest['records']}
assert training_hashes.isdisjoint(new_hashes)
assert all(not equal(new_inputs.sample(0), training_inputs.sample(index))
           for index in range(len(training_inputs)))
request = Path('inference-request.json').resolve()
request.write_text(json.dumps(config.to_dict()) + '\n')
# Keep only the resolved inference request and portable weights. A fresh Python
# process cannot recover any original raw/prepared data or checkpoint in memory.
shutil.rmtree(root)
assert not root.exists() and not Path(config.data['prepared']).exists()
generated = run_cli('generate', request, '--weights', weights,
                    '--prepared', new/'prepared/manifest.json', '--output', Path('new-generated').resolve())
assert generated['kind'] == 'INFERENCE_FROM_EXPORTED_WEIGHTS'
assert generated['sample_id'] == 'new-input-000004'
assert generated['input_identity'] == new_inputs.identity != training_inputs.identity
assert generated['model']['source'] == exported['model']['source']
assert generated['loaded_factory'] == exported['model']['source']
assert generated['model']['modules'] == exported['model']['modules']
assert generated['quality_validation'] == exported['quality_validation'] == 'NOT_RUN'
values = torch.load(generated['artifact'], weights_only=True)
video = values['decoded']['video']
assert values['decoded']['quality_status'] == QUALITY_STATUS
assert generated['video']['layout'] == 'THWC' and video.dtype == torch.uint8
assert list(video.shape) == generated['video']['shape']
import numpy as np
from PIL import Image
frames = sorted(Path(generated['video']['frames']).glob('*.png'))
assert len(frames) == len(video)
for index, path in enumerate(frames):
    with Image.open(path) as image:
        assert torch.equal(torch.from_numpy(np.array(image)), video[index])
for name, module in tuple(sys.modules.items()):
    if (name == 'distillschedule' or name.startswith('distillschedule.')
            or name == 'runtime' or name.startswith('runtime.')) and getattr(module, '__file__', None):
        assert Path(module.__file__).resolve().is_relative_to(site), name
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_INSTALLED_FRAMEWORK_CPU',
    raw_media=True, real_autograd=True, restart_full_state_equal=True,
    generate=True, export=True, measured_plan_executed=True, git_optional=True,
    fresh_process_resume=True, saved_plan_executed=True,
    exported_weights_new_inputs=True, training_artifacts_unavailable=not root.exists(),
    fresh_process_generation=True, scope='LOCAL_RANDOM_TINY_MEDIA_FIXTURE',
    source=dict(working_directory_outside_checkout=True, site_packages=str(site),
        module_paths={'distillschedule':str(Path(distillschedule.__file__).resolve()),
                      'runtime':str(Path(runtime.__file__).resolve())},
        python_version=sys.version.split()[0], torch_version=torch.__version__,
        qualified_dependencies=expected_dependencies),
    inference=dict(kind=generated['kind'], sample_id=generated['sample_id'],
        input_identity=generated['input_identity'], training_input_identity=training_inputs.identity,
        prepared_features_disjoint=True, encoded_feeds_disjoint=True,
        video_layout=generated['video']['layout'], source_bound=True, module_layout_bound=True,
        model_source_sha256=generated['loaded_factory']['sha256'],
        module_layout_sha256=digest(generated['model']['modules']),
        export_manifest_sha256=generated['export_manifest_sha256'],
        quality_validation=generated['quality_validation']),
    gpu_execution='NOT_RUN', pretrained_quality='NOT_RUN', quality_validation='NOT_RUN',
    native_pretrained_lifecycles='NOT_RUN')))
'''
        if args.modern_flow:
            script = MODERN_FLOW_ACCEPTANCE
        arguments = [str(root), json.dumps(qualified)]
        if args.official_workflows:
            script = OFFICIAL_WORKFLOWS_ACCEPTANCE
            arguments.append(json.dumps(dict(dmd2=str(args.dmd2_source), self_forcing=str(args.self_forcing_source))))
        result = subprocess.run([str(python), "-I", "-c", script, *arguments], cwd=work, env=env,
                                text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError("installed framework acceptance failed:\n" + result.stdout + result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        report["source"]["wheel_sha256"] = hashlib.sha256(files[0].read_bytes()).hexdigest()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
