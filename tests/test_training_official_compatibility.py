"""Metadata-only official request admission, not pretrained/GPU qualification."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from distillschedule.training.catalog import resolve, validate_compatibility
from distillschedule.training.config import TrainingConfig


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {'sd15': 'dmd2', 'sdxl': 'dmd2', 'self-forcing-wan': 'self-forcing'}
COLLECTIVE = dict(backend='torch-callable-native-owner-v1', local=dict(
    backend='torch-callable-resident-v1', share_immutable=False, checkpoint_bindings=[]))
RECIPES = sorted(path for folder in ('dmd2_sd15', 'dmd2_sdxl', 'self_forcing_official')
                 for path in (ROOT / 'recipes' / folder).glob('training*.json'))


def request(family, *, explicit=False, device='cpu', parameters=None, execution=None, distributed=None):
    model = dict(parameters=deepcopy(parameters or {}))
    algorithm = {}
    if explicit:
        model['factory'] = resolve('model', family)['factory']
        algorithm['factory'] = resolve('algorithm', FAMILIES[family])['factory']
    else:
        model['name'], algorithm['name'] = family, FAMILIES[family]
    trainer = dict(cycles=3, device=device, output_dir='NOT_CREATED')
    if distributed is not None:
        trainer['distributed'] = dict(backend=distributed)
    return dict(schema_version='distillschedule.training.v1', model=model, algorithm=algorithm,
                data=dict(prepared='MISSING/manifest.json'),
                execution=deepcopy(execution or dict(mode='reference')), trainer=trainer)


def parse(value):
    return TrainingConfig.from_dict(value, source=ROOT / 'metadata-only-training.json')


def resident_budget(device):
    trial = dict(host_limit_bytes=2**31, snapshot_limit_bytes=2**29, result_limit_bytes=2**29,
                 scratch_bytes=2**27, reserve_bytes=2**27)
    if device.startswith('cuda'):
        trial.update(device_limit_bytes=2**32, host_growth_bytes=2**27, device_state_growth_bytes=2**27,
                     device_gradient_bytes=2**26, device_activation_bytes=2**29,
                     device_workspace_bytes=2**27, device_restore_bytes=2**26, device_reserve_bytes=2**27)
    return dict(schema_version='distillschedule.training_resident.v1', construction_limit_bytes=2**27,
                host_staging_limit_bytes=2**27, trial=trial)


class TestOfficialCompatibility(unittest.TestCase):
    def test_wrong_algorithm_and_algorithm_overrides_fail_for_names_and_factories(self):
        for family in FAMILIES:
            for explicit in (False, True):
                for override in (False, True):
                    with self.subTest(family=family, explicit=explicit, override=override):
                        value = request(family, explicit=explicit)
                        if override:
                            value['algorithm']['parameters'] = {'ratio': 2}
                        else:
                            value['algorithm'] = {'factory': 'distillschedule.training.dmd:build_algorithm'}
                        with self.assertRaisesRegex(ValueError, 'official.*algorithm'):
                            parse(value)

    def test_known_backend_and_profile_conflicts_fail(self):
        cases = [
            ('sd15', {'native_role_backend': 'unknown'}),
            ('sd15', {'native_role_backend': None}),
            ('sd15', {'native_role_backend': []}),
            ('sdxl', {'native_role_backend': 'fsdp2-native-sd15-v1'}),
            ('sdxl', {'optimization_profile': 'dmd2-sd15-resident-tree-v1'}),
            ('sd15', {'optimization_profile': 'self-forcing-wan-fsdp2-r6-v1'}),
            ('self-forcing-wan', {'optimization_profile': 'dmd2-sd15-resident-tree-v1'}),
            ('self-forcing-wan', {'native_role_backend': 'single-device-v1'}),
            ('self-forcing-wan', {'native_role_backend': 'ddp-fp32-v1'}),
            ('sd15', {'optimization_profile': None}),
            ('sd15', {'optimization_profile': []}),
            ('sd15', {'optimization_profile': 'dmd2-sd15-resident-tree-v1',
                      'native_role_backend': 'single-device-v1'}),
            ('self-forcing-wan', {'optimization_profile': 'self-forcing-wan-fsdp2-r6-v1',
                                  'native_role_backend': 'fsdp2-native-sd15-v1'}),
        ]
        for family, parameters in cases:
            for explicit in (False, True):
                with self.subTest(family=family, parameters=parameters, explicit=explicit):
                    with self.assertRaisesRegex(ValueError, 'official.*(backend|profile)'):
                        parse(request(family, parameters=parameters, explicit=explicit,
                                      device='cuda', distributed='nccl'))
        for family in FAMILIES:
            with self.subTest(family=family, explicit_default=True):
                with self.assertRaisesRegex(ValueError, 'official.*backend'):
                    parse(request(family, parameters={'native_role_backend': 'default'}))

    def test_placement_and_distributed_contracts_fail_before_assets(self):
        cases = [request(family, distributed='gloo') for family in FAMILIES]
        cases += [request('sd15', parameters={'native_role_backend': 'ddp-fp32-v1'}),
                  request('sd15', parameters={'native_role_backend': 'ddp-fp32-v1'}, distributed='nccl'),
                  request('sd15', parameters={'native_role_backend': 'ddp-fp32-v1'},
                          device='cuda', distributed='gloo'),
                  request('sd15', parameters={'native_role_backend': 'fsdp2-native-sd15-v1'}, distributed='gloo'),
                  request('self-forcing-wan', parameters={'native_role_backend': 'fsdp2-native-bf16-v1'},
                          device='cuda'),
                  request('sd15', parameters={'optimization_profile': 'dmd2-sd15-resident-tree-v1'})]
        for family in FAMILIES:
            cases += [request(family, device=device) for device in ('mps', 'cpu:0', 'cuda:-1', 'cuda:bad')]
            cases += [request(family, parameters={'device': 'cpu'}, device='cuda'),
                      request(family, parameters={'device': 'cuda:1'}, device='cuda:0')]
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'official.*(device|distributed|backend)'):
                    parse(value)

    def test_collective_opaque_overrides_and_dmd2_order_are_rejected(self):
        for family, backend in (('sd15', 'ddp-fp32-v1'), ('sdxl', 'ddp-fp32-v1'),
                                ('self-forcing-wan', 'fsdp2-native-bf16-v1')):
            parameters = {'native_role_backend': backend}
            for execution in (dict(mode='auto'), dict(mode='plan', plan='MISSING/selected-plan.json'),
                              dict(mode='manual', implementation={}),
                              dict(mode='manual', implementation=dict(COLLECTIVE, owner_execution={})),
                              dict(mode='manual', implementation=dict(COLLECTIVE, local=dict(
                                  COLLECTIVE['local'], share_immutable=0)))):
                with self.subTest(family=family, execution=execution):
                    with self.assertRaisesRegex(ValueError, 'official.*execution'):
                        parse(request(family, parameters=parameters, device='cuda',
                                      distributed='nccl', execution=execution))
        with self.assertRaisesRegex(ValueError, 'official.*order'):
            parse(request('sd15', parameters={'native_role_backend': 'ddp-fp32-v1'}, distributed='gloo',
                          execution=dict(mode='manual', implementation=COLLECTIVE, order=[])))

    def test_collective_owned_policy_conflicts_are_rejected(self):
        for family, parameters in (
            ('sd15', dict(native_role_backend='ddp-fp32-v1', native_checkpoint_policy='component-v1')),
            ('sd15', dict(native_role_backend='fsdp2-native-sd15-v1', native_checkpoint_policy='component-v1')),
            ('sd15', dict(optimization_profile='dmd2-sd15-resident-tree-v1', native_checkpoint_policy='component-v1')),
            ('self-forcing-wan', dict(native_role_backend='fsdp2-native-bf16-v1',
                                      native_cache_lifetime='release-after-commit-v1')),
        ):
            with self.subTest(family=family, parameters=parameters):
                with self.assertRaisesRegex(ValueError, 'official.*policy'):
                    parse(request(family, parameters=parameters, device='cuda', distributed='nccl'))

    def test_unbudgeted_single_cuda_trials_fail_but_reference_remains_valid(self):
        for family in FAMILIES:
            for execution in (dict(mode='auto'), dict(mode='plan', plan='MISSING/selected-plan.json'),
                              dict(mode='manual', implementation={'owner_execution': {}})):
                with self.subTest(family=family, execution=execution):
                    with self.assertRaisesRegex(ValueError, 'official.*resident'):
                        parse(request(family, device='cuda:0', execution=execution))
            parse(request(family, device='cuda:0'))
            parse(request(family, device='cuda', parameters={'device': 'cuda:1'}))

    def test_existing_recipes_and_explicit_factory_forms_remain_valid(self):
        self.assertEqual(len(RECIPES), 12)
        for path in RECIPES:
            value = json.loads(path.read_text())
            with self.subTest(recipe=path.name, family=value['model']['name']):
                named = parse(value)
                for kind in ('model', 'algorithm'):
                    value[kind]['factory'] = resolve(kind, value[kind].pop('name'))['factory']
                explicit = parse(value)
                self.assertEqual(named.model['factory'], explicit.model['factory'])
                self.assertEqual(named.algorithm['factory'], explicit.algorithm['factory'])

    def test_cpu_ddp_and_sf_optional_policy_fields_are_not_overrestricted(self):
        for family in ('sd15', 'sdxl'):
            for device, backend in (('cpu', 'gloo'), ('cuda:0', 'nccl')):
                for mode in ('reference', 'manual'):
                    execution = dict(mode=mode, max_states=4, max_candidates=2)
                    if mode == 'manual':
                        execution['implementation'] = deepcopy(COLLECTIVE)
                    parse(request(family, device=device, distributed=backend, execution=execution,
                                  parameters={'native_role_backend': 'ddp-fp32-v1'}))
        parse(request('self-forcing-wan', device='cuda', distributed='nccl', parameters=dict(
            native_role_backend='fsdp2-native-bf16-v1', optimization_profile='self-forcing-wan-fsdp2-r6-v1'),
            execution=dict(mode='manual', implementation=COLLECTIVE, max_states=4,
                           order=['self_forcing.native_cycle', '__open_finish__'])))

    def test_r19_resident_manual_auto_plan_and_cpu_scope_remain_valid(self):
        for family in FAMILIES:
            key = 'fake_gradient_checkpointing' if family == 'self-forcing-wan' else 'gradient_checkpointing'
            manual = dict(mode='manual', implementation=dict(COLLECTIVE['local'],
                owner_execution={'native_cycle': {key: True}}))
            for device in ('cpu', 'cuda:0'):
                for execution in (manual, dict(mode='auto', memory_limit_bytes=2**31),
                                  dict(mode='plan', plan='MISSING/selected-plan.json')):
                    with self.subTest(family=family, device=device, mode=execution['mode']):
                        value = request(family, device=device, execution=execution)
                        value['execution']['resident_budget'] = resident_budget(device)
                        parsed = parse(value)
                        self.assertEqual(parse(parsed.to_dict()).identity, parsed.identity)
                # CPU native capability remains source-qualified at runtime, not rejected here.
                if device == 'cpu':
                    parse(request(family, execution=manual))

    def test_resident_budget_does_not_grant_other_initialization_profiles(self):
        for family, parameters in (
            ('sd15', {'native_role_backend': 'single-device-v1'}),
            ('sdxl', {'generator_checkpoint': 'MISSING/generator.pt'}),
        ):
            value = request(family, parameters=parameters)
            value['execution']['resident_budget'] = resident_budget('cpu')
            with self.subTest(family=family), self.assertRaisesRegex(ValueError, 'official.*resident'):
                parse(value)

    def test_replaced_requests_cannot_load_data_models_or_seed_rng(self):
        from distillschedule.training import session
        for family in FAMILIES:
            config = parse(request(family))
            invalid = replace(config, algorithm={'factory': 'distillschedule.training.dmd:build_algorithm'})
            with self.subTest(family=family), patch.object(session, 'training_data', side_effect=AssertionError('data loaded')), \
                    patch.object(session.distributed, 'rank_world', side_effect=AssertionError('runtime loaded')), \
                    patch.object(session, '_construct_model', side_effect=AssertionError('model loaded')), \
                    patch.object(session, '_seed', side_effect=AssertionError('RNG seeded')):
                with self.assertRaisesRegex(ValueError, 'official.*algorithm'):
                    session.construct(invalid)

    def test_unknown_plugins_keep_opt_in_compatibility(self):
        validate_compatibility(dict(factory='external_plugin.model:build', parameters={'device': 'private-device'}),
                               dict(factory='external_plugin.algorithm:build', parameters={'custom': True}),
                               dict(device='private-device', distributed={'backend': 'custom'}), dict(mode='auto'))

    def test_admission_in_fresh_dependency_blocked_process(self):
        script = '''
import importlib.abc, json, pathlib, sys
sys.path.insert(0, sys.argv[1])
class NoTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'diffusers', 'transformers', 'peft', 'safetensors', 'accelerate'} or fullname in {
            'distillschedule.training.sd', 'distillschedule.training.self_forcing',
            'distillschedule.open_workload.resident_trial'}:
            raise AssertionError('metadata imported numerical provider: ' + fullname)
sys.meta_path.insert(0, NoTraining())
from distillschedule.training.config import TrainingConfig
from distillschedule.training.catalog import describe
before = describe()
for filename in json.loads(sys.argv[2]):
    value = json.loads(pathlib.Path(filename).read_text())
    TrainingConfig.from_dict(value, source=filename)
    value['algorithm'] = {'factory': 'distillschedule.training.dmd:build_algorithm'}
    try: TrainingConfig.from_dict(value, source=filename)
    except ValueError as exc: assert 'official' in str(exc) and 'algorithm' in str(exc)
    else: raise AssertionError('wrong official algorithm survived public config validation')
assert describe() == before
print('PASS_METADATA_ONLY_OFFICIAL_RECIPES_AND_REJECTIONS')
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(ROOT / 'src'),
                                 json.dumps([str(path) for path in RECIPES])], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS_METADATA_ONLY', result.stdout)


if __name__ == '__main__':
    unittest.main()
