"""Versioned complete-cycle restore contracts, separate from provenance.

Epochs declare deliberate migrations; source-derived semantics also bind actual
mathematics, update clocks, state encoding and selected execution helpers.
"""
TRAINING_CONTRACT = 'distillschedule.training_resume.v1'
EXECUTION_CONTRACT = 'distillschedule.program_resume.v1'

# Provenance only. Pre-descriptor checkpoints can restore under their exact
# original source, not wildcard across releases merely sharing this epoch.
_LEGACY_TRAINING = {'b3a92414c8e360ea9f231962e67a0fdf65ec4abaed12d929b8c5a5c9d692e69a':
                    'distillschedule.training_resume.v1'}
_LEGACY_EXECUTION = {'fe1678279c8af3d8e41a890e898be99e5a448f4df91fa5b1df46df20dba20ccb':
                    'distillschedule.program_resume.v1'}
_MISSING = object()


def _source_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _semantics_match(before, after, *, same_source):
    from .semantics import valid
    # Legacy metadata has no descriptor. Exact-source restore remains available;
    # a descriptor present on either side must not be ignored or malformed.
    if before is _MISSING:
        return same_source and (after is _MISSING or valid(after))
    return valid(before) and valid(after) and before == after


def same_training_identity(saved, current):
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    # Native checkpoint owners have their own exact source/assets contracts;
    # do not reinterpret their opaque identities as public Program metadata.
    if 'framework' not in saved and 'framework' not in current:
        return saved == current
    before, after = dict(saved), dict(current)
    old_source, new_source = before.pop('framework', None), after.pop('framework', None)
    if not _source_digest(old_source) or not _source_digest(new_source):
        return False
    same_source = old_source == new_source
    previous = before.pop('resume_contract', _LEGACY_TRAINING.get(old_source))
    present = after.pop('resume_contract', None)
    if previous is not None and previous != TRAINING_CONTRACT:
        return False
    if present is not None and present != TRAINING_CONTRACT:
        return False
    if not same_source and not previous == present == TRAINING_CONTRACT:
        return False
    if not _semantics_match(before.pop('training_semantics', _MISSING),
                            after.pop('training_semantics', _MISSING), same_source=same_source):
        return False
    return before == after


def execution_source_compatible(saved, current, *, current_semantics=_MISSING):
    if not isinstance(saved, dict):
        return False
    source = saved.get('source')
    if not _source_digest(source) or not _source_digest(current):
        return False
    contract = saved.get('resume_contract', _LEGACY_EXECUTION.get(source))
    if contract is not None and contract != EXECUTION_CONTRACT:
        return False
    if source != current and contract != EXECUTION_CONTRACT:
        return False
    return _semantics_match(saved.get('execution_semantics', _MISSING), current_semantics,
                            same_source=source == current)
