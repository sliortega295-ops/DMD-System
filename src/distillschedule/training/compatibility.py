"""Versioned complete-cycle restore contracts, separate from source provenance.

Bump these epochs when training mathematics, state ownership/serialization,
update clocks or executable policy semantics change. Logging, retention and
CPU search-budget fixes do not change them. Model/plugin/data/topology identities
are still compared independently and exactly.
"""
TRAINING_CONTRACT = 'distillschedule.training_resume.v1'
EXECUTION_CONTRACT = 'distillschedule.program_resume.v1'

# Before explicit compatibility epochs, main 3ca90fa9 used exact source hashes.
# This is the reviewed predecessor, not an allow-all rule for old checkpoints.
_LEGACY_TRAINING = {'b3a92414c8e360ea9f231962e67a0fdf65ec4abaed12d929b8c5a5c9d692e69a':
                    'distillschedule.training_resume.v1'}
_LEGACY_EXECUTION = {'fe1678279c8af3d8e41a890e898be99e5a448f4df91fa5b1df46df20dba20ccb':
                    'distillschedule.program_resume.v1'}


def _source_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def same_training_identity(saved, current):
    if saved == current:
        return True
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    before, after = dict(saved), dict(current)
    old_source, new_source = before.pop('framework', None), after.pop('framework', None)
    if not _source_digest(old_source) or not _source_digest(new_source):
        return False
    previous = before.pop('resume_contract', _LEGACY_TRAINING.get(old_source))
    present = after.pop('resume_contract', None)
    return previous == present == TRAINING_CONTRACT and before == after


def execution_source_compatible(saved, current):
    source = saved.get('source')
    if not _source_digest(source) or not _source_digest(current):
        return False
    contract = saved.get('resume_contract', _LEGACY_EXECUTION.get(source))
    if contract is not None and contract != EXECUTION_CONTRACT:
        return False
    if source == current:
        return True
    return contract == EXECUTION_CONTRACT
