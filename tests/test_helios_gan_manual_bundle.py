"""CPU contract tests; synthetic objects are not target-GPU evidence."""
from dataclasses import replace
from copy import deepcopy
from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import patch
import os
import json
from pathlib import Path
import subprocess
import unittest

from distillschedule.search.helios_implementation import HeliosImplementation
from distillschedule.search.helios_gan_bundle import GAN_BUNDLE_ID, HeliosGANBundle


class GANManualBundleTests(unittest.TestCase):
    def test_roundtrip_and_explicit_optin(self):
        ordinary = HeliosImplementation('gan', 'world16-fsdp', 2048, 5, True)
        self.assertNotIn(GAN_BUNDLE_ID, ordinary.id)
        impl = replace(ordinary, candidate_bundle=HeliosGANBundle.manual())
        self.assertEqual(impl, HeliosImplementation.parse(impl.id))
        self.assertEqual(impl, HeliosImplementation.from_assignment(impl.assignment))
        self.assertEqual(impl.environment()['DISTILLSCHEDULE_HELIOS_SELECTIVE_FLASH_CHECKPOINT_STRIDE'], '2')
        self.assertEqual(impl.environment()['DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_MERGED_EXECUTION'], 'off')
        self.assertEqual(impl.environment()['DISTILLSCHEDULE_HELIOS_GAN_MANUAL_BUNDLE'], GAN_BUNDLE_ID)

    def test_scope_is_not_a_new_shape_or_h100_admission(self):
        impl = HeliosImplementation('gan', 'world16-fsdp', 2048, 5, True,
                                    HeliosGANBundle.manual())
        for changed in ({'variant': 'no-gan'}, {'chunk_tokens': 512},
                        {'persistent_unshard': False}, {'rollout_window': 4}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                replace(impl, **changed)

    def test_missing_and_stale_effects_cannot_be_claimed(self):
        bundle = HeliosGANBundle.manual()
        with self.assertRaises(ValueError):
            bundle.effect_snapshot({})
        before = {'public_backend_call_count': 0, 'down_original': 0, 'down_replay': 0}
        with self.assertRaises(ValueError):
            bundle.effects_between(before, before)

    def test_complete_counter_contract_and_incomplete_tuple_rejected(self):
        path = Path(__file__).resolve().parent / 'fixtures/helios_gan_manual_counters.json'
        fixture = json.loads(path.read_text())
        self.assertEqual(fixture['scope'], 'CPU_COUNTER_CONTRACT_FIXTURE_NOT_FRESH_GPU_EVIDENCE')
        build = fixture['build']; bundle = HeliosGANBundle.manual()
        after = bundle.effect_snapshot(build)
        result = bundle.effects_between(dict.fromkeys(after,0),after)
        self.assertEqual(result['call_deltas']['down_original'],800)
        self.assertEqual(result['call_deltas']['critic.shape.M13680_N5120_R256'],4000)
        for mutation in ('generator','lora','heads','backend','norm','kv'):
            bad = deepcopy(build)
            if mutation == 'generator': del bad['generator']
            elif mutation == 'lora': del bad['critic']['all_role_lora_scale1_add']
            elif mutation == 'heads': bad['critic']['fsdp2']['separate_gan_head_fsdp']={'requested':True}
            elif mutation == 'backend': bad['gan_manual_runtime']['actual_FA3_identity']={}
            elif mutation == 'norm': bad['generator']['exact_modulated_norm_epilogue']['reference_fallback_count']=1
            else: bad['generator']['generator_nograd_cross_kv_cache']['cached_tensor_mutation_count']=1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                bundle.effect_snapshot(bad)
        after['critic.shape.M13680_N5120_R256'] -= 1
        with self.assertRaises(ValueError):
            bundle.effects_between(dict.fromkeys(after,0),after)

    def test_no_import_or_fake_historical_costs(self):
        from distillschedule.search.helios_model_adapter import implementations
        from distillschedule.search.helios_implementation import GAN
        ordinary = implementations(GAN)
        expanded = implementations(GAN, include_gan_manual=True)
        self.assertEqual(expanded[:-1], ordinary)
        self.assertEqual(expanded[-1].candidate_bundle.id, GAN_BUNDLE_ID)
        self.assertNotIn('wall_seconds', expanded[-1].assignment)

    def test_source_only_lowering_resolves_actual_native_identity(self):
        from distillschedule.search.execution import resolve_candidate
        from distillschedule.search.helios_model_adapter import gan_manual_incumbent
        impl = gan_manual_incumbent()
        receipt = resolve_candidate(impl.workload, impl.id, mode='correctness', run_id='cpu-manual-only')
        self.assertEqual(receipt['search_candidate'], impl.candidate().to_dict())
        self.assertEqual(receipt['nodes'], [])  # No assets: no GPU launch authority.

    def test_off_scope_does_not_import_torch_or_touch_gpu(self):
        from distillschedule.runtime.helios_gan_manual.runtime import manual_runtime_scope
        with patch.dict(os.environ, {}, clear=True):
            with manual_runtime_scope(None):
                pass

    def test_each_role_has_406_layers_and_merge_guards_restore(self):
        from distillschedule.runtime.helios_gan_manual.runtime import _audit_unmerged, _forbid_merges
        class Layer:
            lora_A = lora_B = object()
            merged = False
            merged_adapters = ()
            def merge(self):
                return 'original'
        engines = [NS(module=NS(modules=lambda rows=[Layer() for _ in range(406)]: iter(rows)))
                   for _ in range(2)]
        state = {}
        _audit_unmerged(state, engines, 'AFTER_SETUP_BEFORE_BOTH_TIMERS')
        self.assertEqual(state['lora_layer_count'], 812)
        self.assertEqual([s['LoRA_count'] for s in state['model_metadata_samples']], [406,406])
        layer = next(engines[0].module.modules())
        with ExitStack() as stack:
            _forbid_merges(stack, state, engines)
            self.assertEqual(state['guarded_merge_methods'], 812)
            with self.assertRaises(ValueError):
                layer.merge()
            self.assertEqual(state['merge_attempt_count'], 1)
        self.assertEqual(layer.merge(), 'original')
        bad = [engines[0], NS(module=NS(modules=lambda: iter([Layer()] * 405)))]
        with self.assertRaises(ValueError):
            _audit_unmerged({}, bad, 'bad')

    def test_launch_gate_admits_only_explicit_complete_gan_tuple(self):
        from distillschedule.search.helios_model_adapter import gan_manual_incumbent
        script = Path(__file__).resolve().parents[1] / 'scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh'
        text = script.read_text()
        start = text.index('if [[ "${LORA_B_GEMM_EPILOGUE}" == on ]]; then')
        stop = text.index('[[ "${OPTIMIZER_DISPATCH_WARMUP}"', start)
        gate = text[start:stop]
        env = dict(os.environ, **gan_manual_incumbent().environment(),
            LORA_B_GEMM_EPILOGUE='on',VARIANT='gan',MESH_POLICY='world16-fsdp',
            ROLLOUT_BATCH_SIZE='1',PACK_REAL_SCORE_PAIR='off',ALL_ROLE_LORA_SCALE1_ADD='on',
            LORA_CHECKPOINT_REPLAY_SCOPE='on',ATTENTION_BACKEND='local-fa2',
            SELECTIVE_FLASH_CHECKPOINT_STRIDE='2',MODULATION_CHUNK_TOKENS='2048',
            GENERATOR_NOGRAD_PERSISTENT_UNSHARD='on')
        def run(changed):
            return subprocess.run(['bash','-c','set -eu\n'+gate],env=env|changed,
                                  capture_output=True).returncode
        self.assertEqual(run({}), 0)
        for change in ({'SELECTIVE_FLASH_CHECKPOINT_STRIDE':'0'}, {'MODULATION_CHUNK_TOKENS':'512'},
                       {'DISTILLSCHEDULE_HELIOS_GAN_MANUAL_BUNDLE':''},
                       {'DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_MERGED_EXECUTION':'on'}):
            self.assertNotEqual(run(change), 0)


if __name__ == '__main__':
    unittest.main()
