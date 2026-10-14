"""Real private native CPU tests; skip explicitly when its source is absent."""
import copy
import importlib.util
import unittest

from runtime.h3_diffsynth_logical_microbatches import staged_native_update, run_native_microbatches

HAS_NATIVE = importlib.util.find_spec('h3_dmd2') is not None


@unittest.skipUnless(HAS_NATIVE, 'pinned private DiffSynth source not installed')
class NativeBoundaryTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        from h3_dmd2 import DMD2Config, DMD2Trainer, H3Adapter, VelocityDiscriminator
        from diffsynth.models.minimax_h3_dit import MiniMaxH3DiT
        from test_train_dmd2 import tiny_architecture, sample
        from train_dmd2 import make_optimizers, seed_all
        self.torch, self.cls, self.sample, self.seed = torch, DMD2Trainer, sample, seed_all
        self.config = DMD2Config(num_inference_steps=8, critic_updates=1, lora_rank=2,
                                lora_alpha=3, hidden_channels=8, num_layers=2, gan_start_step=500)
        def construct():
            seed_all(42)
            adapter = H3Adapter(MiniMaxH3DiT(**tiny_architecture()), self.config)
            head = VelocityDiscriminator(self.config)
            optimizers, parameters = make_optimizers(adapter, head, self.config)
            return adapter, head, optimizers, parameters
        self.construct = construct

    def callbacks(self, adapter, optimizers, parameters):
        counts = {'critic': 0, 'student': 0}
        def backward(role, loss):
            selected = 'fake_score' if role == 'critic' else 'student'
            self.assertTrue(all(layer.role == selected for _, layer in adapter.layers))
            loss.backward()
        def commit(role):
            norm = self.torch.nn.utils.clip_grad_norm_(parameters[role],
                    self.config.max_grad_norm if role == 'student' else self.config.critic_max_grad_norm)
            self.assertTrue(self.torch.isfinite(norm))
            optimizers[role].step(); optimizers[role].zero_grad(set_to_none=True)
            counts[role] += 1
        return backward, commit, counts

    def test_one_row_is_bitwise_native_loss_parameters_and_adam(self):
        from h3_dmd2 import prepare_batch
        native_method = self.cls.update
        self.seed(123)
        positive, negative = prepare_batch(self.sample())
        real = (positive['input_latents'], positive['audio_input_latents'])
        a, d, opts, params = self.construct()
        bw, commit, counts = self.callbacks(a, opts, params)
        reference = self.cls(a, d, self.config, lambda role, loss: (bw(role, loss), commit(role)))
        self.seed(900)
        expected_metrics = reference.update(real, positive, negative)
        expected_state = copy.deepcopy(a.state_dict())
        expected_adam = {role: copy.deepcopy(opt.state_dict()) for role, opt in opts.items()}
        a, d, opts, params = self.construct()
        bw, commit, counts = self.callbacks(a, opts, params)
        actual = self.cls(a, d, self.config, lambda *args: self.fail('original callback must be yielded'))
        self.seed(900)
        observed = run_native_microbatches([actual], [(real, positive, negative)],
                                           backward=bw, commit=commit)
        self.assertEqual(observed, [expected_metrics]); self.assertEqual(counts, {'critic': 1, 'student': 1})
        self.assertIs(self.cls.update, native_method)
        for name, value in a.state_dict().items():
            self.torch.testing.assert_close(value, expected_state[name], rtol=0, atol=0)
        for role, optimizer in opts.items():
            state = optimizer.state_dict()
            self.assertEqual(state['param_groups'], expected_adam[role]['param_groups'])
            for pid, tensors in state['state'].items():
                for key, value in tensors.items():
                    self.torch.testing.assert_close(value, expected_adam[role]['state'][pid][key], rtol=0, atol=0)

    def test_two_independent_B1_rows_one_commit_each_and_replay_role(self):
        from h3_dmd2 import prepare_batch
        a, d, opts, params = self.construct()
        bw, commit, counts = self.callbacks(a, opts, params)
        trainers = [self.cls(a, d, self.config, lambda *x: self.fail('callback bypass')) for _ in range(2)]
        self.seed(901)
        batches = []
        for _ in range(2):
            p, n = prepare_batch(self.sample()); batches.append(((p['input_latents'], p['audio_input_latents']), p, n))
        self.assertFalse(self.torch.equal(batches[0][0][0], batches[1][0][0]))
        census = {'student': 0, 'fake_score': 0, 'teacher': 0}
        original = a.predict
        def predict(*args, **kwargs):
            role = next(iter({layer.role for _, layer in a.layers})); census[role] += 1
            return original(*args, **kwargs)
        a.predict = predict
        metrics = run_native_microbatches(trainers, batches, backward=bw, commit=commit, logical_slots=[0, 8])
        self.assertEqual(census, {'student': 18, 'fake_score': 4, 'teacher': 2})
        self.assertEqual(counts, {'critic': 1, 'student': 1})
        self.assertTrue(all(m['generator_step'] == m['critic_step'] == 1 and not m['gan_active'] for m in metrics))

    def test_native_source_mismatch_is_closed(self):
        def changed(self, real, inputs, negative=None):
            return None
        with self.assertRaisesRegex(ValueError, 'source identity'):
            staged_native_update(changed)

    def test_duplicate_logical_sample_is_closed(self):
        with self.assertRaisesRegex(ValueError, 'independent'):
            run_native_microbatches([None, None], [None, None], backward=None, commit=None,
                                   logical_slots=[0, 0])
