import unittest
from runtime.h3_diffsynth_fsdp_attention_reuse import FSDPAttentionReuse, ACTIVE


class DisabledAttentionReuseTests(unittest.TestCase):
    def test_default_disabled_does_not_import_or_mutate_native_modules(self):
        class Native:
            gradient_checkpoint_forward=object()
            rollout=object()
            predict=object()
        native=Native()
        original=dict(vars(Native))
        controller=FSDPAttentionReuse(native,native,native,[],[],enabled=False)
        with controller.scope(3):self.assertIsNone(ACTIVE.get())
        self.assertEqual(dict(vars(Native)),original)
        self.assertEqual(controller.finish_cycle(),{'enabled':False,'status':'DEFAULT_OFF_NO_PATCH'})
        controller.begin_next_cycle()

    def test_unregistered_slot_scope_fails_without_entering_GPU(self):
        controller=FSDPAttentionReuse(None,None,None,[],[],enabled=False)
        controller.enabled=True
        with self.assertRaisesRegex(RuntimeError,'identity'):
            with controller.scope(5):pass
