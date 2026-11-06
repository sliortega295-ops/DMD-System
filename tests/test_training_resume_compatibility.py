"""Minor framework fixes preserve complete-cycle state, not just weights."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from distillschedule.training.config import TrainingConfig
from distillschedule.training.session import train
from test_training_end_to_end import fixture, equal


def test_framework_and_planning_only_changes_resume_the_real_owned_state(tmp_path, monkeypatch):
    from distillschedule.training import session
    from distillschedule.open_workload import planning
    root, config = fixture(tmp_path, samples=6)
    raw = config.to_dict(); raw['trainer'].update(cycles=1, output_dir=str(root / 'first'))
    first = train(TrainingConfig.from_dict(raw, source=config.source))
    old_identity = first['identity']
    old_policy = first['policy']
    monkeypatch.setattr(session, '_source', lambda: 'a' * 64)
    monkeypatch.setattr(planning, 'source_identity', lambda: 'b' * 64)
    raw['trainer'].update(cycles=4, resume_from=first['checkpoint'], output_dir=str(root / 'resumed'))
    resumed = train(TrainingConfig.from_dict(raw, source=config.source))
    raw['trainer'].pop('resume_from'); raw['trainer']['output_dir'] = str(root / 'whole')
    whole = train(TrainingConfig.from_dict(raw, source=config.source))
    a = torch.load(Path(resumed['checkpoint']) / 'state.pt', weights_only=True)
    b = torch.load(Path(whole['checkpoint']) / 'state.pt', weights_only=True)
    for key in ('state', 'rng', 'versions', 'cycle', 'cursor'):
        assert equal(a[key], b[key]), key
    assert resumed['identity']['framework'] != old_identity['framework']
    assert resumed['policy'] == old_policy  # Original source provenance is retained.


@pytest.mark.parametrize('field', ['config', 'dataset', 'model', 'algorithm', 'model_metadata', 'resume_contract'])
def test_minor_fix_compatibility_does_not_relax_model_or_training_identity(field, math_descriptor):
    from distillschedule.training.compatibility import same_training_identity, TRAINING_CONTRACT
    original = dict(framework='a' * 64, config='config', dataset='dataset', model={},
                    algorithm={}, model_metadata={}, resume_contract=TRAINING_CONTRACT,
                    training_semantics=math_descriptor)
    current = deepcopy(original); current['framework'] = 'b' * 64
    assert same_training_identity(original, current)
    current[field] = 'changed'
    assert not same_training_identity(original, current)


def test_unversioned_unknown_source_is_not_implicitly_compatible():
    from distillschedule.training.compatibility import same_training_identity, execution_source_compatible
    assert not same_training_identity({'framework': 'a' * 64}, {'framework': 'b' * 64})
    assert not execution_source_compatible({'source': 'a' * 64}, 'b' * 64)


@pytest.mark.parametrize('source', [[], {}, None, 4, 'bad-hash'])
def test_malformed_policy_source_returns_incompatible(source):
    from distillschedule.training.compatibility import execution_source_compatible
    assert execution_source_compatible({'source': source}, 'b' * 64) is False


def test_old_release_contract_is_not_promoted_by_a_future_epoch(monkeypatch):
    from distillschedule.training import compatibility as c
    old_training = 'b3a92414c8e360ea9f231962e67a0fdf65ec4abaed12d929b8c5a5c9d692e69a'
    old_execution = 'fe1678279c8af3d8e41a890e898be99e5a448f4df91fa5b1df46df20dba20ccb'
    assert c._LEGACY_TRAINING[old_training] == 'distillschedule.training_resume.v1'
    assert c._LEGACY_EXECUTION[old_execution] == 'distillschedule.program_resume.v1'
    monkeypatch.setattr(c, 'TRAINING_CONTRACT', 'distillschedule.training_resume.v2')
    monkeypatch.setattr(c, 'EXECUTION_CONTRACT', 'distillschedule.program_resume.v2')
    assert not c.same_training_identity({'framework': old_training},
        {'framework': 'b' * 64, 'resume_contract': c.TRAINING_CONTRACT})
    assert not c.execution_source_compatible({'source': old_execution}, 'b' * 64)
    assert not c.execution_source_compatible({'source': old_execution,
        'resume_contract': 'distillschedule.program_resume.v1'}, old_execution)


def test_runtime_cannot_change_again_mid_process_after_resume(tmp_path, monkeypatch):
    from distillschedule.examples.tiny_av import build_algorithm, build_model
    from distillschedule.open_workload import planning
    from distillschedule.training.policy import CyclePolicy

    def build():
        return build_algorithm(build_model({}), {
            'latent': torch.zeros(1, 4, 1, 2, 2),
            'condition': torch.zeros(1, 4, 1, 1, 1),
        }, {'fake_updates': 1}, cycle=0, seed=7)

    policy = CyclePolicy({'mode': 'reference'})
    policy.run(build(), tmp_path)
    restored = CyclePolicy({'mode': 'reference'}); restored.restore(policy.bound)
    monkeypatch.setattr(planning, 'source_identity', lambda: 'a' * 64)
    restored.prepare_resume(build())
    monkeypatch.setattr(planning, 'source_identity', lambda: 'b' * 64)
    with pytest.raises(ValueError, match='source'):
        restored.run(build(), tmp_path)


def test_real_resume_rejects_changed_imported_math_before_loading_state(tmp_path, monkeypatch):
    """The entry factory and declared v1 remain unchanged; its helper does not."""
    import importlib
    from distillschedule.training import checkpoint
    root, config = fixture(tmp_path, samples=4)
    package = tmp_path / 'user_math_adapter'
    package.mkdir()
    (package / '__init__.py').write_text('')
    (package / 'entry.py').write_text('''from distillschedule.examples.tiny_av import build_model

def build_algorithm(components, feeds, parameters, cycle, seed):
    from .helper import build as make_cycle
    return make_cycle(components, feeds, parameters, cycle, seed)
''')
    helper = package / 'helper.py'
    helper.write_text('''from distillschedule.examples.tiny_av import build_algorithm as base

def build(components, feeds, parameters, cycle, seed):
    return base(components, feeds, parameters, cycle, seed)
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    raw = config.to_dict()
    raw['model']['factory'] = 'user_math_adapter.entry:build_model'
    raw['algorithm']['factory'] = 'user_math_adapter.entry:build_algorithm'
    raw['trainer'].update(cycles=1, output_dir=str(root / 'first'))
    first = train(TrainingConfig.from_dict(raw, source=config.source))
    entry = first['identity']['algorithm']
    helper.write_text(helper.read_text().replace('return base(components, feeds, parameters, cycle, seed)',
        "return base(components, feeds, {**parameters, 'guidance_scale': 2.0}, cycle, seed)"))
    raw['trainer'].update(cycles=2, resume_from=first['checkpoint'], output_dir=str(root / 'resume'))
    def forbid_state_load(*args, **kwargs):
        raise AssertionError('incompatible helper reached checkpoint tensor restoration')
    monkeypatch.setattr(checkpoint, 'unpack_state', forbid_state_load)
    with pytest.raises(ValueError, match='identity|semantics|source'):
        train(TrainingConfig.from_dict(raw, source=config.source))
    from distillschedule.training.plugins import load_callable
    assert load_callable(raw['algorithm']['factory'])[1] == entry


def test_partial_algorithm_factory_binds_its_actual_helper_before_restore(tmp_path, monkeypatch):
    import importlib
    from distillschedule.training import checkpoint
    package = tmp_path / 'partial_math_adapter'
    package.mkdir()
    (package / '__init__.py').write_text('')
    (package / 'entry.py').write_text('''from functools import partial
from distillschedule.examples.tiny_av import build_model
from .helper import build
build_algorithm = partial(build)
''')
    helper = package / 'helper.py'
    helper.write_text('''from distillschedule.examples.tiny_av import build_algorithm as base
def build(components, feeds, parameters, cycle, seed):
    return base(components, feeds, parameters, cycle, seed)
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    root, config = fixture(tmp_path / 'training', samples=4)
    raw = config.to_dict()
    raw['model']['factory'] = 'partial_math_adapter.entry:build_model'
    raw['algorithm']['factory'] = 'partial_math_adapter.entry:build_algorithm'
    raw['trainer'].update(cycles=1, output_dir=str(root / 'first'))
    first = train(TrainingConfig.from_dict(raw, source=config.source))
    helper.write_text(helper.read_text().replace('return base(components, feeds, parameters, cycle, seed)',
        "return base(components, feeds, {**parameters, 'guidance_scale': 2.0}, cycle, seed)"))
    raw['trainer'].update(cycles=2, resume_from=first['checkpoint'], output_dir=str(root / 'resume'))
    def forbidden(*args, **kwargs):
        raise AssertionError('partial factory helper change reached tensor restoration')
    monkeypatch.setattr(checkpoint, 'unpack_state', forbidden)
    with pytest.raises(ValueError, match='identity|semantics|source'):
        train(TrainingConfig.from_dict(raw, source=config.source))


@pytest.mark.parametrize('nested', [False, True])
def test_partial_bound_callbacks_keep_source_dependencies(tmp_path, monkeypatch, nested):
    import importlib
    from functools import partial
    from distillschedule.training.semantics import _Dependencies
    module_name = 'partial_callback_math_' + str(nested).lower()
    helper = tmp_path / (module_name + '.py')
    helper.write_text('def loss(x):\n    return x * 2\n')
    entry = tmp_path / 'partial_callback_entry.py'
    entry.write_text('def invoke(callback, x):\n    return callback(x)\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    math = importlib.import_module(module_name)
    invoke = importlib.import_module('partial_callback_entry').invoke
    callback = {'callbacks': [math.loss]} if nested else math.loss
    wrapped = partial(invoke, callback)
    # Cyclic metadata must not recurse indefinitely during source inspection.
    wrapped.keywords['recursive'] = wrapped
    def descriptor():
        collector = _Dependencies()
        collector.callable(wrapped, owned=True)
        return collector.descriptor()
    first = descriptor()
    assert any(key.startswith(module_name + ':loss') for key in first['dependencies'])
    helper.write_text(helper.read_text().replace('x * 2', 'x * 3'))
    assert descriptor() != first


def test_editable_package_children_resolve_without_importing_them(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    from distillschedule.training.semantics import _Dependencies, _source_path
    root = tmp_path / 'off_path_package'
    root.mkdir()
    helper = root / 'math.py'
    marker = tmp_path / 'unexpected-import'
    helper.write_text(f'''from pathlib import Path
Path({str(marker)!r}).touch()
def loss(x):
    return x * 2
''')
    package = ModuleType('editable_math_package')
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    # PEP 660/namespace packages can be importable without their source root
    # appearing in sys.path. The actual parent package owns that search path.
    name = package.__name__ + '.math'
    assert _source_path(name) == helper
    collector = _Dependencies()
    collector.owned.add(package.__name__)
    collector.module(name, 'loss')
    assert name + ':loss' in collector.descriptor()['dependencies']
    assert name not in sys.modules and not marker.exists()


@pytest.fixture(scope='module')
def math_descriptor():
    from distillschedule.training.semantics import _Dependencies
    collector = _Dependencies()
    collector.module('runtime.dmd_boundary')
    return collector.descriptor()


@pytest.mark.parametrize('same_source', [False, True])
@pytest.mark.parametrize('damage', ['missing', 'digest', 'epoch', 'dependencies', 'external', 'schema'])
def test_semantic_descriptor_damage_is_not_an_epoch_bypass(math_descriptor, damage, same_source):
    from distillschedule.training.compatibility import (same_training_identity,
        execution_source_compatible, TRAINING_CONTRACT, EXECUTION_CONTRACT)
    original = dict(framework='a' * 64, config='x', resume_contract=TRAINING_CONTRACT,
                    training_semantics=deepcopy(math_descriptor))
    current = deepcopy(original)
    current['framework'] = original['framework'] if same_source else 'b' * 64
    altered = deepcopy(math_descriptor)
    if damage == 'missing':
        altered = None
    elif damage == 'digest':
        altered['sha256'] = 'a' * 64
    elif damage == 'epoch':
        altered['epochs']['mathematics'] += 1
    elif damage == 'dependencies':
        altered['dependencies'] = {'bad': []}
    elif damage == 'external':
        altered['external'] = []
    elif damage == 'schema':
        altered['schema_version'] = 'future'
    current['training_semantics'] = altered
    assert not same_training_identity(original, current)
    saved = dict(source=original['framework'], resume_contract=EXECUTION_CONTRACT,
                 execution_semantics=math_descriptor)
    assert not execution_source_compatible(saved, current['framework'], current_semantics=altered)


def test_legacy_without_descriptor_only_accepts_exact_original_source(math_descriptor):
    from distillschedule.training import compatibility as c
    for source in ('a' * 64, *c._LEGACY_TRAINING):
        old = dict(framework=source, config='same')
        current = dict(framework=source, config='same', resume_contract=c.TRAINING_CONTRACT,
                       training_semantics=math_descriptor)
        assert c.same_training_identity(old, current)
        current['framework'] = 'b' * 64
        assert not c.same_training_identity(old, current)
    for source in ('a' * 64, *c._LEGACY_EXECUTION):
        old = dict(source=source)
        assert c.execution_source_compatible(old, source, current_semantics=math_descriptor)
        assert not c.execution_source_compatible(old, 'b' * 64, current_semantics=math_descriptor)


def test_normalized_helpers_ignore_only_docstrings_comments_and_formatting(tmp_path, monkeypatch):
    from distillschedule.training.semantics import _Dependencies
    helper = tmp_path / 'semantic_helper.py'
    monkeypatch.syspath_prepend(str(tmp_path))
    def descriptor():
        collector = _Dependencies()
        collector.module('semantic_helper', 'loss')
        return collector.descriptor()
    helper.write_text('"module docs"\nimport torch\ndef loss(x):\n    "function docs"\n    return (x + 1).square()\n')
    before = descriptor()
    helper.write_text("'''other module docs'''\n# explanation\nimport torch\n\ndef loss( x ):\n    '''other docs'''\n    return ( x + 1 ).square() # same math\n")
    assert descriptor() == before
    helper.write_text(helper.read_text().replace('x + 1', 'x + 2'))
    assert descriptor() != before
    # Calls described as logging are not deleted: arguments may consume RNG.
    helper.write_text('import torch\ndef loss(x):\n    print(torch.rand(()))\n    return (x + 1).square()\n')
    assert descriptor() != before


def test_semantic_import_aliases_share_only_immutable_ast_analysis(monkeypatch):
    import ast
    from distillschedule.training import semantics
    scope = ast.parse('import math as left\nimport json as right\nleft.sin(1)\nright.dumps(2)\n')
    walks = []
    original = ast.walk
    def observed(node):
        if node is scope: walks.append(node)
        return original(node)
    monkeypatch.setattr(semantics.ast, 'walk', observed)
    collector = semantics._Dependencies()
    collector._import(scope.body[0], 'left', scope, '')
    collector._import(scope.body[1], 'right', scope, '')
    assert len(walks) == 1


def test_installed_first_party_and_user_adapter_helpers_keep_ast_binding(tmp_path, monkeypatch):
    import ast
    import importlib
    from distillschedule.training import semantics
    from distillschedule.examples.tiny_av import build_model
    purelib = tmp_path / 'site-packages'
    package = purelib / 'installed_dmd_adapter'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'entry.py').write_text('''from distillschedule.examples.tiny_av import build_model
import installed_dmd_adapter.helper as math_helper

def build_algorithm(*args):
    return math_helper.apply(*args)
''')
    helper = package / 'helper.py'
    helper.write_text('''from distillschedule.examples.tiny_av import build_algorithm as build

def apply(*args):
    return build(*args)
''')
    monkeypatch.syspath_prepend(str(purelib))
    importlib.invalidate_caches()
    entry = importlib.import_module('installed_dmd_adapter.entry')
    original_paths = semantics.sysconfig.get_paths
    monkeypatch.setattr(semantics.sysconfig, 'get_paths',
                        lambda: {**original_paths(), 'purelib': str(purelib)})
    # Simulate the shipped top-level runtime installed in that same purelib.
    actual_resolve = semantics._source_path
    runtime = purelib / 'runtime/dmd_boundary.py'
    runtime.parent.mkdir()
    runtime.write_text(actual_resolve('runtime.dmd_boundary').read_text())
    monkeypatch.setattr(semantics, '_source_path',
        lambda name: runtime if name == 'runtime.dmd_boundary' else actual_resolve(name))
    def descriptor():
        return semantics.training_semantics(entry.build_model, entry.build_algorithm, build_model({}).state)
    first = descriptor()
    assert any(key.startswith('runtime.dmd_boundary:') for key in first['dependencies'])
    assert any(key.startswith('installed_dmd_adapter.helper:') for key in first['dependencies'])
    helper.write_text(helper.read_text().replace('return build(*args)', 'return build(*args) + 1'))
    assert descriptor() != first
    helper.write_text(helper.read_text().replace('return build(*args) + 1', 'return build(*args)'))
    tree = ast.parse(runtime.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'detached_dmd_surrogate_loss')
    function.body.insert(1, ast.Expr(ast.Call(ast.Name('print', ast.Load()), [ast.Constant('changed math')], [])))
    ast.fix_missing_locations(tree)
    runtime.write_text(ast.unparse(tree))
    assert descriptor() != first


def test_policy_independently_rejects_helper_change_before_layout_installation(tmp_path, monkeypatch):
    import ast
    from distillschedule.examples.tiny_av import build_algorithm, build_model
    from distillschedule.training import semantics
    from distillschedule.training.policy import CyclePolicy
    from distillschedule.open_workload.execution import ExecutionBackend
    def build():
        return build_algorithm(build_model({}), {
            'latent': torch.zeros(1, 4, 1, 2, 2),
            'condition': torch.zeros(1, 4, 1, 1, 1),
        }, {'fake_updates': 1}, cycle=0, seed=7)
    policy = CyclePolicy({'mode': 'reference'})
    policy.run(build(), tmp_path)
    assert semantics.valid(policy.bound['execution_semantics'])
    restored = CyclePolicy({'mode': 'reference'}); restored.restore(policy.bound)
    original = semantics._source_path
    modified = tmp_path / 'dmd_boundary.py'
    tree = ast.parse(original('runtime.dmd_boundary').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'detached_dmd_surrogate_loss')
    function.body.append(ast.Return(ast.Constant(999)))
    ast.fix_missing_locations(tree)
    modified.write_text(ast.unparse(tree))
    monkeypatch.setattr(semantics, '_source_path',
        lambda name: modified if name == 'runtime.dmd_boundary' else original(name))
    def forbidden(*args, **kwargs):
        raise AssertionError('incompatible math reached physical layout installation')
    monkeypatch.setattr(ExecutionBackend, 'session', forbidden)
    with pytest.raises(ValueError, match='source|structure|semantics'):
        restored.prepare_resume(build())


def test_exact_native_identity_still_roundtrips_real_checkpoint(tmp_path):
    from distillschedule.training.checkpoint import save_checkpoint, read_manifest, restore_checkpoint
    module = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(module.parameters(), lr=.01)
    module(torch.ones(1, 2)).sum().backward()
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    state = {'student': module, 'optimizer': optimizer}
    expected = deepcopy(module.state_dict())
    expected_optimizer = deepcopy(optimizer.state_dict())
    identity = {'source_head': 'a' * 40, 'source_files': {'worker.py': 'b' * 64},
                'assets': {'student': 'c' * 64}, 'contract': 'livewan.native.same-topology'}
    path = tmp_path / 'cycle-000001'
    save_checkpoint(path, state, identity=identity, cycle=1, cursor=16,
                    versions={'student': 1, 'optimizer': 1})
    assert read_manifest(path, identity=identity)['identity'] == identity
    with torch.no_grad():
        module.weight.zero_()
    optimizer.state.clear()
    restore_checkpoint(path, state, identity=identity)
    assert equal(module.state_dict(), expected)
    assert equal(optimizer.state_dict(), expected_optimizer)
    altered = deepcopy(identity); altered['source_files']['worker.py'] = 'd' * 64
    with pytest.raises(ValueError, match='identity'):
        read_manifest(path, identity=altered)


def test_semantic_source_graph_is_not_rebuilt_each_cycle(tmp_path, monkeypatch):
    from distillschedule.training import semantics
    calls = {'training': 0, 'execution': 0}
    for name in calls:
        original = getattr(semantics, name + '_semantics')
        def observed(*args, _name=name, _function=original, **kwargs):
            calls[_name] += 1
            return _function(*args, **kwargs)
        monkeypatch.setattr(semantics, name + '_semantics', observed)
    root, config = fixture(tmp_path, samples=4)
    raw = config.to_dict(); raw['trainer'].update(cycles=3, output_dir=str(root / 'run'))
    result = train(TrainingConfig.from_dict(raw, source=config.source))
    assert result['completed_cycles'] == 3
    assert calls == {'training': 1, 'execution': 1}
    dependencies = result['identity']['training_semantics']['dependencies']
    assert any(name.startswith('runtime.dmd_boundary:') for name in dependencies)
    assert not any('CheckpointPolicy' in name or '.planning:' in name or 'prune_checkpoints' in name
                   for name in dependencies)


def test_separate_method_local_aliases_bind_both_helpers(tmp_path, monkeypatch):
    from distillschedule.training.semantics import _Dependencies
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / 'alias_owner.py').write_text('''class Owner:
    def first(self, x):
        from first_helper import update as transform
        return transform(x)
    def second(self, x):
        from second_helper import update as transform
        return transform(x)
''')
    first = tmp_path / 'first_helper.py'
    first.write_text('def update(x):\n    return x + 1\n')
    (tmp_path / 'second_helper.py').write_text('def update(x):\n    return x * 2\n')
    def descriptor():
        collector = _Dependencies(); collector.module('alias_owner', 'Owner')
        return collector.descriptor()
    before = descriptor()
    first.write_text('def update(x):\n    return x + 3\n')
    assert descriptor() != before


def test_conditional_math_definitions_keep_all_branches_and_predicate(tmp_path, monkeypatch):
    from distillschedule.training.semantics import _Dependencies
    monkeypatch.syspath_prepend(str(tmp_path))
    helper = tmp_path / 'conditional_math.py'
    helper.write_text('''USE_FIRST = True
if USE_FIRST:
    def loss(x):
        return x + 1
else:
    def loss(x):
        return x * 2
''')
    def descriptor():
        collector = _Dependencies(); collector.module('conditional_math', 'loss')
        return collector.descriptor()
    before = descriptor()
    helper.write_text(helper.read_text().replace('x + 1', 'x + 3'))
    assert descriptor() != before


def test_changed_cycle_cursor_routing_rejects_before_tensor_restore(tmp_path, monkeypatch):
    from distillschedule.training import checkpoint, semantics
    root, config = fixture(tmp_path, samples=4)
    raw = config.to_dict(); raw['trainer'].update(cycles=1, output_dir=str(root / 'first'))
    first = train(TrainingConfig.from_dict(raw, source=config.source))
    original = semantics._source_path
    modified = tmp_path / 'session.py'
    text = original('distillschedule.training.session').read_text()
    assert 'cursor += world' in text
    modified.write_text(text.replace('cursor += world', 'cursor += 2 * world'))
    monkeypatch.setattr(semantics, '_source_path',
        lambda name: modified if name == 'distillschedule.training.session' else original(name))
    raw['trainer'].update(cycles=2, resume_from=first['checkpoint'], output_dir=str(root / 'resume'))
    def forbidden(*args, **kwargs):
        raise AssertionError('changed clock routing reached tensor restoration')
    monkeypatch.setattr(checkpoint, 'unpack_state', forbidden)
    with pytest.raises(ValueError, match='source|structure|semantics'):
        train(TrainingConfig.from_dict(raw, source=config.source))


@pytest.mark.parametrize('initialization', [
    'SCALE = 1\nSCALE *= FACTOR\nCONFIG = {"scale": SCALE}\n',
    'CONFIG = {"scale": 1}\nCONFIG.update(scale=FACTOR)\n',
    'CONFIG = {"scale": 1}\nALIAS = CONFIG\nALIAS["scale"] *= FACTOR\n',
    'CONFIG = {"scale": 1}\ndef initialize(config):\n    config["scale"] *= FACTOR\ninitialize(CONFIG)\n',
    'CONFIG = {"scale": 1}\ndef initialize():\n    CONFIG["scale"] *= FACTOR\ninitialize()\n',
    'CONFIG = {"scale": 1}\nfor factor in (FACTOR,):\n    CONFIG["scale"] *= factor\n',
    'CONFIG = {"scale": 1}\ndef unrelated(default=CONFIG.update(scale=FACTOR)):\n    return 0\n',
], ids=['augassign', 'mapping-call', 'alias-mutation', 'helper-call', 'global-helper-call',
        'loop', 'definition-default'])
def test_real_resume_rejects_changed_module_initialization_before_state_load(
        tmp_path, monkeypatch, initialization):
    import importlib
    from distillschedule.training import checkpoint
    package_name = 'init_math_' + tmp_path.name.replace('-', '_')
    package = tmp_path / package_name
    package.mkdir()
    (package / '__init__.py').write_text('')
    (package / 'entry.py').write_text('''from distillschedule.examples.tiny_av import build_model

def build_algorithm(components, feeds, parameters, cycle, seed):
    from .helper import build
    return build(components, feeds, parameters, cycle, seed)
''')
    helper = package / 'helper.py'
    helper.write_text('from distillschedule.examples.tiny_av import build_algorithm as base\n'
        + initialization.replace('FACTOR', '2') + '''
def build(components, feeds, parameters, cycle, seed):
    return base(components, feeds, {**parameters, 'guidance_scale': CONFIG['scale']}, cycle, seed)
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    root, config = fixture(tmp_path / 'training', samples=4)
    raw = config.to_dict()
    raw['model']['factory'] = package_name + '.entry:build_model'
    raw['algorithm']['factory'] = package_name + '.entry:build_algorithm'
    raw['trainer'].update(cycles=1, output_dir=str(root / 'first'))
    first = train(TrainingConfig.from_dict(raw, source=config.source))
    helper.write_text(helper.read_text().replace(initialization.replace('FACTOR', '2'),
                                                 initialization.replace('FACTOR', '3')))
    raw['trainer'].update(cycles=2, resume_from=first['checkpoint'], output_dir=str(root / 'resume'))
    def forbidden(*args, **kwargs):
        raise AssertionError('changed initialization reached checkpoint restoration')
    monkeypatch.setattr(checkpoint, 'unpack_state', forbidden)
    from distillschedule.open_workload.execution import ExecutionBackend
    monkeypatch.setattr(ExecutionBackend, 'session', forbidden)
    with pytest.raises(ValueError, match='identity|semantics|source'):
        train(TrainingConfig.from_dict(raw, source=config.source))


@pytest.mark.parametrize('initialization_call', ['initialize(CONFIG)', '(lambda: initialize(CONFIG))()'])
def test_imported_initialization_dependencies_bind_both_resume_gates(
        tmp_path, monkeypatch, initialization_call):
    from distillschedule.training.compatibility import (same_training_identity,
        execution_source_compatible, TRAINING_CONTRACT, EXECUTION_CONTRACT)
    from distillschedule.training.semantics import _Dependencies
    monkeypatch.syspath_prepend(str(tmp_path))
    helper = tmp_path / 'init_dependency.py'
    helper.write_text('def initialize(config):\n    config["scale"] *= 2\n')
    (tmp_path / 'init_entry.py').write_text('''from init_dependency import initialize
CONFIG = {'scale': 1}
initialize(CONFIG)
def loss(x):
    return x * CONFIG['scale']
def unused(x):
    import missing_optional_model_dependency
    return missing_optional_model_dependency.predict(x)
unused_callback = lambda x: unused(x)
'''.replace('initialize(CONFIG)', initialization_call, 1))
    def descriptor():
        collector = _Dependencies()
        collector.module('init_entry', 'loss')
        return collector.descriptor()
    original = descriptor()
    # An unrelated body is not executed during import, and cannot pull a model
    # dependency into a restore check for this loss.
    entry = tmp_path / 'init_entry.py'
    entry.write_text(entry.read_text().replace('.predict(x)', '.predict(x) + 1'))
    assert descriptor() == original
    helper.write_text(helper.read_text().replace('*= 2', '*= 3'))
    current = descriptor()
    assert current != original
    assert any(key.startswith('init_dependency:initialize') for key in current['dependencies'])
    assert not same_training_identity(
        dict(framework='a' * 64, resume_contract=TRAINING_CONTRACT, training_semantics=original),
        dict(framework='a' * 64, resume_contract=TRAINING_CONTRACT, training_semantics=current))
    assert not execution_source_compatible(
        dict(source='a' * 64, resume_contract=EXECUTION_CONTRACT, execution_semantics=original),
        'a' * 64, current_semantics=current)


def test_initialization_projection_preserves_explicit_lifecycle_scope(tmp_path, monkeypatch):
    from distillschedule.training import semantics
    original = semantics._source_path
    checkpoint = tmp_path / 'checkpoint.py'
    registry = tmp_path / 'registry.py'
    checkpoint.write_text(original('distillschedule.training.checkpoint').read_text())
    registry.write_text(original('distillschedule.workloads.best_known.registry').read_text())
    paths = {'distillschedule.training.checkpoint': checkpoint,
             'distillschedule.workloads.best_known.registry': registry}
    monkeypatch.setattr(semantics, '_source_path', lambda name: paths.get(name) or original(name))
    def descriptor():
        collector = semantics._Dependencies()
        collector.module('distillschedule.training.checkpoint')
        collector.module('distillschedule.workloads.best_known.registry')
        return collector.descriptor()
    before = descriptor()
    checkpoint.write_text(checkpoint.read_text().replace('interval_seconds: float = 1800.',
                                                       'interval_seconds: float = 900.'))
    registry.write_text(registry.read_text().replace('H3_SYSTEM_TOOLCHAIN_PATH = "/usr/bin:/bin"',
                                                    'H3_SYSTEM_TOOLCHAIN_PATH = "/another/toolchain"'))
    assert descriptor() == before


@pytest.mark.parametrize('layout', ['fsdp', 'hsdp', 'ddp'])
def test_real_cpu_wrappers_preserve_source_semantics_and_resume_gate(tmp_path, monkeypatch, layout):
    """Actual CPU wrapper metadata; this is not CUDA FSDP training coverage."""
    import importlib
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard, FSDPModule
    from torch.nn.parallel import DistributedDataParallel
    from distillschedule.examples.tiny_av import build_algorithm, build_model
    from distillschedule.open_workload.execution import ExecutionBackend
    from distillschedule.open_workload.planning import source_identity
    from distillschedule.training.compatibility import EXECUTION_CONTRACT
    from distillschedule.training.policy import CyclePolicy, cycle_structure
    from distillschedule.training.semantics import execution_semantics

    assert not torch.cuda.is_initialized()
    module_name = 'actual_wrapped_child_' + layout
    (tmp_path / (module_name + '.py')).write_text('''from torch import nn
class Projection(nn.Linear):
    def forward(self, value):
        return super().forward(value)
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    child_type = importlib.import_module(module_name).Projection
    def build():
        components = build_model({})
        components.state['student'].layers[0] = child_type(9, 12)
        components.state['student_optimizer'] = torch.optim.AdamW(
            components.state['student'].parameters(), lr=.002)
        return build_algorithm(components, {
            'latent': torch.zeros(1, 4, 1, 2, 2),
            'condition': torch.zeros(1, 4, 1, 1, 1),
        }, {'fake_updates': 1}, cycle=0, seed=7)
    implementation = {'backend': 'torch-callable-resident-v1',
        'reduction': {'fsdp': 'fsdp2_reshard', 'hsdp': 'hsdp2_reshard', 'ddp': 'ddp'}[layout]}
    original = build()
    before = execution_semantics(original, implementation)
    dist.init_process_group('gloo', init_method='file://' + str(tmp_path / 'rendezvous'), rank=0, world_size=1)
    try:
        if layout == 'ddp':
            for name in ('student', 'fake'):
                original.state[name] = DistributedDataParallel(original.state[name])
        else:
            mesh = init_device_mesh('cpu', (1, 1) if layout == 'hsdp' else (1,),
                                    mesh_dim_names=('replicate', 'shard') if layout == 'hsdp' else ('shard',))
            child = original.state['student'].layers[0]
            fully_shard(child, mesh=mesh)
            assert isinstance(child, FSDPModule)
            for name in ('student', 'fake', 'teacher'):
                fully_shard(original.state[name], mesh=mesh)
                assert isinstance(original.state[name], FSDPModule)
        after = execution_semantics(original, implementation)
        assert after == before
        assert any(key.startswith(module_name + ':Projection') for key in after['dependencies'])
        assert any(key.startswith('distillschedule.examples.tiny_av:TinyAVNetwork')
                   for key in after['dependencies'])
        fresh = build()
        assert cycle_structure(original) == cycle_structure(fresh)
        restored = CyclePolicy({'mode': 'manual', 'implementation': implementation})
        restored.restore(dict(schema_version='distillschedule.cycle_policy.v1', mode='manual',
            implementation=implementation, source=source_identity(), structure=cycle_structure(original),
            resume_contract=EXECUTION_CONTRACT, execution_semantics=after))
        installations = []
        def installed(self, selected, **kwargs):
            installations.append(selected)
            return object()
        monkeypatch.setattr(ExecutionBackend, 'session', installed)
        restored.prepare_resume(fresh)
        assert installations == [implementation]
    finally:
        dist.destroy_process_group()
    assert not torch.cuda.is_initialized()
