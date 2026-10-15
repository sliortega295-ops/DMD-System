import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace, ModuleType
import importlib.util
import unittest
from unittest.mock import patch

from runtime import h3_diffsynth_h100_fa3 as binding


class FA3BindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / 'attention.py'
        source.write_text('pinned diagnostic fixture\n')
        self.dispatcher = SimpleNamespace(__file__=str(source), ATTENTION_IMPLEMENTATION='torch')
        self.digest = hashlib.sha256(source.read_bytes()).hexdigest()

    def configure(self, backend='native', **kwargs):
        with patch.object(binding.importlib, 'import_module', return_value=self.dispatcher), \
             patch.object(binding, 'NATIVE_DISPATCH_SHA256', self.digest):
            return binding.configure_native_attention(backend, placement=kwargs.pop('placement', 'fsdp'), **kwargs)

    def test_default_does_not_patch_or_load_kernel(self):
        before = dict(vars(self.dispatcher))
        with patch.dict('sys.modules', {'kernels.utils': None}):
            receipt = self.configure()
        self.assertFalse(receipt['patch_applied'])
        self.assertEqual(vars(self.dispatcher), before)

    def test_existing_native_cache_remains_default(self):
        self.assertEqual(self.configure(placement='candidate')['attention_cache'], 'aten_flash')

    def test_FA3_rejects_old_cache_before_import(self):
        with patch.object(binding.importlib, 'import_module') as imported:
            with self.assertRaisesRegex(ValueError, 'nine-result'):
                binding.configure_native_attention('fa3', placement='candidate', asset_dir=self.root)
            imported.assert_not_called()

    def test_nograd_FA3_rejects_old_cache_before_import(self):
        with patch.object(binding.importlib, 'import_module') as imported:
            with self.assertRaisesRegex(ValueError, 'nine-result'):
                binding.configure_native_attention('fa3-nograd', placement='candidate', asset_dir=self.root)
            imported.assert_not_called()

    def test_required_asset_not_optional(self):
        with self.assertRaisesRegex(ValueError, 'explicit isolated'):
            self.configure('fa3')
        with self.assertRaisesRegex(ValueError, 'explicit isolated'):
            self.configure(asset_dir=self.root)

    def test_unknown_or_mutated_dispatcher_fails(self):
        with self.assertRaisesRegex(ValueError, 'unknown'):
            self.configure('anything')
        self.dispatcher.ATTENTION_IMPLEMENTATION = 'flash_attention_2'
        with self.assertRaisesRegex(ValueError, 'unexpected initial'):
            self.configure()

    def test_private_source_guard(self):
        with patch.object(binding.importlib, 'import_module', return_value=self.dispatcher):
            with self.assertRaisesRegex(ValueError, 'source mismatch'):
                binding.configure_native_attention(placement='fsdp')

    def test_asset_guard_does_not_patch_on_failure(self):
        p = self.root / 'metadata.json'; p.write_text('wrong')
        before = dict(vars(self.dispatcher))
        with patch.object(binding, 'FA3_ASSET_SHA256', {'metadata.json': '0' * 64}):
            with self.assertRaisesRegex(ValueError, 'asset mismatch'):
                self.configure('fa3', asset_dir=self.root)
        self.assertEqual(vars(self.dispatcher), before)

    def test_explicit_process_local_binding(self):
        p = self.root / 'flash_attn_interface.py'; p.write_text('test interface')
        calls = []
        def forward(*a, **kw):
            calls.append(kw)
            return 'output'
        interface = SimpleNamespace(__file__=str(p), flash_attn_func=forward,
                                    _flash_attn_backward=lambda *a, **kw: None)
        utils = ModuleType('kernels.utils')
        utils.get_local_kernel = lambda root, name: SimpleNamespace(flash_attn_interface=interface)
        fake_torch = SimpleNamespace()
        with patch.dict('sys.modules', {'torch':fake_torch, 'kernels': ModuleType('kernels'), 'kernels.utils': utils}), \
             patch.object(binding, 'FA3_ASSET_SHA256', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()}):
            receipt = self.configure('fa3', asset_dir=self.root)
        self.assertTrue(receipt['patch_applied'])
        self.assertIsNot(self.dispatcher.flash_attn_interface, interface)
        self.assertEqual(self.dispatcher.ATTENTION_IMPLEMENTATION, 'flash_attention_3')
        self.assertIs(interface.flash_attn_func, forward)
        self.assertEqual(self.dispatcher.flash_attn_interface.flash_attn_func(None,None,None), 'output')
        self.assertEqual(calls, [{'deterministic':False}])

    def test_explicit_deterministic_reaches_extension_and_conflict_fails(self):
        p = self.root / 'flash_attn_interface.py'; p.write_text('test interface')
        calls = []
        def forward(*a, **kw):
            calls.append(kw)
            return 'actual_call_boundary'
        interface = SimpleNamespace(__file__=str(p), flash_attn_func=forward,
                                    _flash_attn_backward=lambda *a, **kw:None)
        utils = ModuleType('kernels.utils')
        utils.get_local_kernel = lambda root,name:SimpleNamespace(flash_attn_interface=interface)
        torch = SimpleNamespace(are_deterministic_algorithms_enabled=lambda:True)
        with patch.dict('sys.modules', {'torch':torch, 'kernels':ModuleType('kernels'), 'kernels.utils':utils}), \
             patch.object(binding, 'FA3_ASSET_SHA256', {p.name:hashlib.sha256(p.read_bytes()).hexdigest()}):
            r = self.configure('fa3', asset_dir=self.root, deterministic_backward=True)
        call = self.dispatcher.flash_attn_interface.flash_attn_func
        self.assertEqual(call(None,None,None,causal=False), 'actual_call_boundary')
        self.assertTrue(calls[0]['deterministic'])
        self.assertTrue(r['FA3_deterministic_backward_requested'])
        with self.assertRaisesRegex(ValueError, 'CONFLICTING'):
            call(None,None,None,deterministic=False)
        self.assertEqual(len(calls), 1)
        self.assertIs(interface.flash_attn_func, forward)

    def test_deterministic_without_common_policy_is_rejected(self):
        p = self.root / 'flash_attn_interface.py'; p.write_text('test interface')
        interface=SimpleNamespace(__file__=str(p),flash_attn_func=lambda:None,_flash_attn_backward=lambda:None)
        utils=ModuleType('kernels.utils')
        utils.get_local_kernel=lambda root,name:SimpleNamespace(flash_attn_interface=interface)
        with patch.dict('sys.modules', {'torch':SimpleNamespace(are_deterministic_algorithms_enabled=lambda:False),
                       'kernels':ModuleType('kernels'),'kernels.utils':utils}), \
             patch.object(binding,'FA3_ASSET_SHA256',{p.name:hashlib.sha256(p.read_bytes()).hexdigest()}):
            with self.assertRaisesRegex(ValueError, 'COMMON_TORCH_POLICY'):
                self.configure('fa3',asset_dir=self.root,deterministic_backward=True)

    def test_nograd_scope_uses_native_for_grad_and_recompute(self):
        p = self.root / 'flash_attn_interface.py'; p.write_text('test interface')
        interface = SimpleNamespace(__file__=str(p), flash_attn_func=lambda *a, **kw: None,
                                    _flash_attn_backward=lambda *a, **kw: None)
        utils = ModuleType('kernels.utils')
        utils.get_local_kernel = lambda root, name: SimpleNamespace(flash_attn_interface=interface)
        fake_torch = ModuleType('torch'); mode = [False]
        fake_torch.is_grad_enabled = lambda: mode[0]
        calls = []
        def native(*args, **kw):
            calls.append(('native', args, kw)); return 'native_result'
        def fa3(*args, **kw):
            calls.append(('fa3', args, kw)); return 'fa3_result'
        self.dispatcher.attention_forward = native
        self.dispatcher.flash_attention_3 = fa3
        with patch.dict('sys.modules', {'torch': fake_torch, 'kernels': ModuleType('kernels'), 'kernels.utils': utils}), \
             patch.object(binding, 'FA3_ASSET_SHA256', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()}):
            receipt = self.configure('fa3-nograd', asset_dir=self.root)
            q = SimpleNamespace(shape=(1, 2, 31, 96))
            self.assertEqual(self.dispatcher.flash_attention_3(q, q, q, scale=0.125), 'fa3_result')
            mode[0] = True
            for _ in range(2):  # original grad forward and checkpoint replay
                self.assertEqual(self.dispatcher.flash_attention_3(q, q, q, scale=0.125), 'native_result')
        self.assertEqual([r[0] for r in calls], ['fa3', 'native', 'native'])
        self.assertNotIn('compatibility_mode', calls[0][2])
        self.assertTrue(all(r[2]['compatibility_mode'] for r in calls[1:]))
        self.assertTrue(all(r[2]['scale'] == 0.125 for r in calls))
        self.assertEqual(receipt['grad_and_checkpoint_replay_backend'], 'native_torch_SDPA')


class BackendProbeAdmissionTests(unittest.TestCase):
    def candidate(self, **changes):
        kw=dict(backend='fa3',placement='fsdp',asset_dir='pinned_asset',mode='correctness',
                export=True,one_cycle=False,deterministic=True,deterministic_backward=True,
                common_native_warmup=True,reference='actual_native_C1',retained_blocks=0)
        kw.update(changes)
        binding.validate_backend_probe(**kw)

    def test_explicit_common_C1_correctness_probe_is_admitted(self):
        self.candidate()
        self.candidate(backend='native',asset_dir=None,reference=None)

    def test_default_native_timing_is_unchanged(self):
        self.candidate(backend='native',asset_dir=None,reference=None,
            common_native_warmup=False,deterministic_backward=False,
            deterministic=False,export=False,mode='timing')

    def test_FA3_timing_and_unverified_C1_are_rejected(self):
        for changes in ({'mode':'timing'},{'reference':None},{'export':False},
                        {'one_cycle':True},{'common_native_warmup':False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.candidate(**changes)

    def test_no_combined_residency_or_aten_cache(self):
        for changes in ({'retained_blocks':40},{'placement':'candidate'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.candidate(**changes)

    def test_missing_policy_and_asset_are_rejected(self):
        for changes in ({'deterministic':False},{'deterministic_backward':False},
                        {'asset_dir':None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.candidate(**changes)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'actual Torch CPU required')
class RNGReceiptTests(unittest.TestCase):
    def test_real_cpu_rng_identity_without_advancing_any_generator(self):
        import torch
        import numpy as np
        import random
        state={'torch':torch.get_rng_state(),'cuda':None,
               'python':random.getstate(),'numpy':np.random.get_state()}
        before=torch.get_rng_state().clone()
        a=binding.snapshot_slot_rng({3:state,11:state})
        b=binding.snapshot_slot_rng({3:state,11:state})
        self.assertEqual(a,b)
        self.assertEqual(a['3'],a['11'])
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        changed={**state,'torch':state['torch'].clone()}
        changed['torch'][0]^=1
        self.assertNotEqual(a,binding.snapshot_slot_rng({3:changed,11:state}))
        self.assertFalse(torch.cuda.is_initialized())

    def test_non_cpu_rng_receipt_rejected_without_a_device_copy(self):
        state={'torch':SimpleNamespace(device=SimpleNamespace(type='cuda'))}
        with self.assertRaisesRegex(ValueError,'ALREADY_SAVED_CPU'):
            binding.snapshot_slot_rng({0:state})
