"""Both native arms must name the same preregistered file, not just budgets."""
import unittest

from distillschedule.search.manual_incumbent_validation import _require_protocol_reference


class ManualProtocolReferenceTests(unittest.TestCase):
    ref = {'path': '/fixture/preregistered.json', 'sha256': 'a' * 64}

    def test_manual_top_level_and_selected_nested_references(self):
        for receipt in ({'manual_match_protocol': dict(self.ref)},
                        {'model_plan': {'problem': {'lowering': {
                            'manual_match_protocol': dict(self.ref)}}}}):
            _require_protocol_reference(receipt, self.ref)

    def test_missing_reference_cannot_be_inferred_from_compatible_capacity(self):
        with self.assertRaisesRegex(ValueError, 'preregistration'):
            _require_protocol_reference({'capacity_policy': {}}, self.ref)

    def test_both_references_must_equal_the_registered_file(self):
        receipt = {'manual_match_protocol': dict(self.ref), 'model_plan': {
            'problem': {'lowering': {'manual_match_protocol': dict(self.ref)}}}}
        _require_protocol_reference(receipt, self.ref)
        for key, changed in (('path', '/fixture/other.json'), ('sha256', 'b' * 64)):
            wrong = dict(self.ref); wrong[key] = changed
            receipt['model_plan']['problem']['lowering']['manual_match_protocol'] = wrong
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'preregistration'):
                _require_protocol_reference(receipt, self.ref)


if __name__ == '__main__':
    unittest.main()
