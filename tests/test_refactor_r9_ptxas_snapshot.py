"""Synthetic process/FD snapshots only; never GPU access or PID authority."""
import copy
import errno
import os

import pytest

from test_refactor_r9_transport import _finished_probe_fixture


@pytest.mark.parametrize('command', ['ptxas', 'nvidia-smi'])
def test_ended_recorded_worker_probe_refreshes_full_snapshot_without_adoption(command, monkeypatch):
    from distillschedule.search import r9_snapshot
    r9, helper, tracker, state, owned, controller, current, fresh, calls, absent = _finished_probe_fixture()
    state['device_handle_owners'][0]['comm'] = command
    original_handles = copy.deepcopy(state['device_handle_owners'])
    known_before, record_before = copy.deepcopy(tracker.known), copy.deepcopy(tracker.record)
    side_effects = []
    monkeypatch.setattr(r9_snapshot.os, 'kill', lambda *args: side_effects.append(('kill', args)))
    monkeypatch.setattr(r9_snapshot.os, 'waitpid', lambda *args: side_effects.append(('waitpid', args)))

    updated, live = r9_snapshot.refresh_finished_probe(
        r9, helper, tracker, state, owned, controller, [], proc_stat=absent)

    assert updated is not state and live == owned
    assert calls.count('AUDIT') == calls.count('FULL_FD_SCAN') == calls.count('ORIGINAL_TRACKER') == 1
    assert calls.count('/proc/400/stat') == calls.count('/proc/400') == 2
    assert state['device_handle_owners'] == original_handles
    assert updated['device_handle_owners'] == fresh['device_handle_owners']
    assert tracker.known == known_before and tracker.record == record_before
    assert 400 not in tracker.known and 400 not in live
    assert side_effects == []
    proof = updated['finished_probe_snapshot_rescan']
    assert proof['status'] == 'FRESH_COMPLETE_SCAN_ORIGINAL_EXCLUSIVE_REQUIRED'
    assert proof['full_FD_rescans'] == 1
    assert all(proof[key] is True for key in ('no_signal', 'no_wait', 'no_added_tracker_births',
               'no_NVML_PID_authority', 'no_deleted_or_filtered_FD_rows'))
    assert len(proof['observations']) == 1
    observation = proof['observations'][0]
    assert observation['observed_birth'] == original_handles[0]['birth']
    assert [p['pid'] for p in observation['current_parent_chain']] == [300, 200]
    for phase in ('ENOENT_before', 'ENOENT_after'):
        assert observation[phase] == [{'path': '/proc/400/stat', 'errno': 2},
                                      {'path': '/proc/400', 'errno': 2}]


@pytest.mark.parametrize(('mutation', 'reason'), [
    ('alive', 'STILL_LIVE_OR_REUSED'),
    ('reused', 'STILL_LIVE_OR_REUSED'),
    ('unreadable', 'PermissionError'),
    ('wrong_enoent', 'PRECISE_ENOENT_REQUIRED'),
    ('proc_present', 'PROC_EXISTS_OR_UNREADABLE'),
    ('foreign_parent', 'PARENT_NOT_OWNED'),
    ('untracked_parent', 'PARENT_NOT_TRACKER_PROVEN'),
    ('wrong_uid', 'ONLY_RECORDED_WORKER'),
    ('parent_reused', 'ANCHOR_CHANGED_OR_UNREADABLE'),
    ('parent_unreadable', 'ANCHOR_CHANGED_OR_UNREADABLE'),
    ('parent_foreign_uid', 'ANCHOR_CHANGED_OR_UNREADABLE'),
    ('parent_other_pgid', 'PARENT_NOT_OWNED'),
    ('probe_older_than_parent', 'PARENT_NOT_OWNED'),
    ('no_exec_permit', 'EXEC_PERMIT'),
    ('discovery_unresolved', 'STICKY_TRACKER_FAILURE'),
    ('adoption_unresolved', 'STICKY_TRACKER_FAILURE'),
    ('discovery_failed_once', 'STICKY_TRACKER_FAILURE'),
    ('adoption_error', 'STICKY_TRACKER_FAILURE'),
    ('adopted_reap_error', 'STICKY_TRACKER_FAILURE'),
])
def test_ptxas_eligibility_is_not_ownership_or_permission(mutation, reason):
    from distillschedule.search.r9_snapshot import refresh_finished_probe
    r9, helper, tracker, state, owned, controller, current, fresh, calls, absent = _finished_probe_fixture()
    row = state['device_handle_owners'][0]
    row['comm'] = 'ptxas'
    if mutation == 'alive':
        current[400] = copy.deepcopy(row['birth'])
    elif mutation == 'reused':
        current[400] = {**row['birth'], 'start_ticks': 999}
    elif mutation == 'unreadable':
        def absent(path):
            raise PermissionError(errno.EACCES, 'CPU_FIXTURE_ONLY', path)
    elif mutation == 'wrong_enoent':
        def absent(path):
            raise FileNotFoundError(errno.ENOTDIR, 'CPU_FIXTURE_ONLY', path)
    elif mutation == 'proc_present':
        def absent(path):
            return os.stat_result((0,) * 10)
    elif mutation == 'foreign_parent':
        row['birth']['ppid'] = 999
    elif mutation == 'untracked_parent':
        tracker.known.pop(300)
    elif mutation == 'wrong_uid':
        row['birth']['uid'] += 1
    elif mutation == 'parent_reused':
        current[300] = {**current[300], 'start_ticks': 999}
    elif mutation == 'parent_unreadable':
        current.pop(300)
    elif mutation == 'parent_foreign_uid':
        current[300] = {**current[300], 'uid': controller['uid'] + 1}
    elif mutation == 'parent_other_pgid':
        row['birth']['pgid'] = 999
    elif mutation == 'probe_older_than_parent':
        row['birth']['start_ticks'] = 29
    elif mutation == 'no_exec_permit':
        tracker.record['exec_permit_issued'] = False
    elif mutation in ('discovery_unresolved', 'adoption_unresolved'):
        setattr(tracker, mutation, True)
    else:
        tracker.record[mutation] = 'CPU_FIXTURE_ONLY'
    known_before = copy.deepcopy(tracker.known)
    updated, live = refresh_finished_probe(r9, helper, tracker, state, owned, controller, [], proc_stat=absent)
    assert updated is state and live is owned
    assert reason in state['finished_probe_snapshot_rescan']['reason']
    assert calls.count('FULL_FD_SCAN') == calls.count('ORIGINAL_TRACKER') == 0
    assert tracker.known == known_before and 400 not in tracker.known


@pytest.mark.parametrize('command', ['', 'python', 'ptxas-helper', '/usr/bin/ptxas',
                                    'PTXAS', 'ptxas ', 'nvidia-smiX', None])
def test_only_two_exact_recorded_command_names_are_eligible(command):
    from distillschedule.search.r9_snapshot import refresh_finished_probe
    r9, helper, tracker, state, owned, controller, current, fresh, calls, absent = _finished_probe_fixture()
    state['device_handle_owners'][0]['comm'] = command
    updated, live = refresh_finished_probe(r9, helper, tracker, state, owned, controller, [], proc_stat=absent)
    assert updated is state and live is owned
    assert 'ONLY_RECORDED_WORKER' in state['finished_probe_snapshot_rescan']['reason']
    assert 'FULL_FD_SCAN' not in calls


@pytest.mark.parametrize('field', ['pid', 'ppid', 'pgid', 'start_ticks'])
@pytest.mark.parametrize('value', [0, -1, True, 1.0, '400', None])
def test_ptxas_recorded_identifiers_keep_strict_positive_integer_bounds(field, value):
    from distillschedule.search.r9_snapshot import refresh_finished_probe
    r9, helper, tracker, state, owned, controller, current, fresh, calls, absent = _finished_probe_fixture()
    state['device_handle_owners'][0]['comm'] = 'ptxas'
    state['device_handle_owners'][0]['birth'][field] = value
    updated, live = refresh_finished_probe(r9, helper, tracker, state, owned, controller, [], proc_stat=absent)
    assert updated is state and live is owned
    assert 'ONLY_RECORDED_WORKER' in state['finished_probe_snapshot_rescan']['reason']
    assert 'FULL_FD_SCAN' not in calls


@pytest.mark.parametrize('bound', ['all_rows', 'unmatched_rows', 'duplicate_pid'])
def test_ptxas_refresh_keeps_existing_snapshot_count_and_unique_pid_bounds(bound):
    from distillschedule.search.r9_snapshot import refresh_finished_probe
    r9, helper, tracker, state, owned, controller, current, fresh, calls, absent = _finished_probe_fixture()
    state['device_handle_owners'][0]['comm'] = 'ptxas'
    row = copy.deepcopy(state['device_handle_owners'][0])
    if bound == 'all_rows':
        state['device_handle_owners'] += [copy.deepcopy(fresh['device_handle_owners'][0]) for _ in range(128)]
    elif bound == 'unmatched_rows':
        state['device_handle_owners'] = [{**copy.deepcopy(row), 'birth': {**row['birth'], 'pid': 400 + i}}
                                       for i in range(33)]
    else:
        state['device_handle_owners'].append(copy.deepcopy(row))
    updated, live = refresh_finished_probe(r9, helper, tracker, state, owned, controller, [], proc_stat=absent)
    assert updated is state and live is owned
    assert 'COUNT_BOUND' in state['finished_probe_snapshot_rescan']['reason']
    assert 'FULL_FD_SCAN' not in calls


def test_ptxas_refresh_keeps_existing_five_second_deadline(monkeypatch):
    from distillschedule.search import r9_snapshot
    r9, helper, tracker, state, owned, controller, current, fresh, calls, absent = _finished_probe_fixture()
    state['device_handle_owners'][0]['comm'] = 'ptxas'
    ticks = iter([100, 105])
    monkeypatch.setattr(r9_snapshot.time, 'monotonic', lambda: next(ticks))
    updated, live = r9_snapshot.refresh_finished_probe(
        r9, helper, tracker, state, owned, controller, [], proc_stat=absent)
    assert updated is state and live is owned
    assert '5S_DEADLINE' in state['finished_probe_snapshot_rescan']['reason']
    assert 'FULL_FD_SCAN' not in calls


@pytest.mark.parametrize(('mutation', 'reason'), [
    ('probe_alive', 'STILL_LIVE_OR_REUSED'),
    ('probe_reused', 'STILL_LIVE_OR_REUSED'),
    ('probe_proc_reappeared', 'PROC_EXISTS_OR_UNREADABLE'),
    ('probe_still_in_scan', 'PID_PRESENT_IN_FRESH_SNAPSHOT'),
    ('probe_in_tracker', 'PID_PRESENT_IN_FRESH_SNAPSHOT'),
    ('parent_reused', 'ANCHOR_CHANGED_OR_UNREADABLE'),
    ('parent_zombie', 'ANCHOR_CHANGED_OR_UNREADABLE'),
    ('parent_ppid', 'ANCHOR_CHANGED_OR_UNREADABLE'),
    ('parent_pgid', 'ANCHOR_CHANGED_OR_UNREADABLE'),
    ('controller_reused', 'ANCHOR_CHANGED_OR_UNREADABLE'),
    ('parent_left_tracker', 'PARENT_LEFT_TRACKER'),
    ('discovery_failed_once', 'STICKY_TRACKER_FAILURE'),
    ('adoption_error', 'STICKY_TRACKER_FAILURE'),
    ('adopted_reap_error', 'STICKY_TRACKER_FAILURE'),
])
def test_ptxas_refresh_requires_stable_ancestors_and_absence_after_complete_scan(mutation, reason):
    from distillschedule.search.r9_snapshot import refresh_finished_probe
    r9, helper, tracker, state, owned, controller, current, fresh, calls, absent = _finished_probe_fixture()
    row = state['device_handle_owners'][0]
    row['comm'] = 'ptxas'
    original_audit, original_live = r9.audit, tracker.live
    known_before = copy.deepcopy(tracker.known)
    record_before = copy.deepcopy(tracker.record)

    def audit():
        result = original_audit()
        if mutation == 'probe_alive':
            current[400] = copy.deepcopy(row['birth'])
        elif mutation == 'probe_reused':
            current[400] = {**row['birth'], 'start_ticks': 999}
        elif mutation == 'probe_still_in_scan':
            fresh['device_handle_owners'].append(copy.deepcopy(row))
        elif mutation == 'parent_reused':
            current[300] = {**current[300], 'start_ticks': 999}
        elif mutation == 'parent_zombie':
            current[300] = {**current[300], 'state': 'Z'}
        elif mutation == 'parent_ppid':
            current[300] = {**current[300], 'ppid': 999}
        elif mutation == 'parent_pgid':
            current[300] = {**current[300], 'pgid': 999}
        elif mutation == 'controller_reused':
            current[100] = {**current[100], 'start_ticks': 999}
        return result

    def live():
        result = original_live()
        if mutation == 'parent_left_tracker':
            result.pop(300)
        elif mutation == 'probe_in_tracker':
            result[400] = copy.deepcopy(row['birth'])
        elif mutation in ('discovery_failed_once', 'adoption_error', 'adopted_reap_error'):
            tracker.record[mutation] = 'CPU_FIXTURE_ONLY'
        return result

    original_absent = absent

    def absent(path):
        if mutation == 'probe_proc_reappeared' and 'AUDIT' in calls:
            return os.stat_result((0,) * 10)
        return original_absent(path)

    r9.audit, tracker.live = audit, live
    updated, live_result = refresh_finished_probe(
        r9, helper, tracker, state, owned, controller, [], proc_stat=absent)
    assert updated is state and live_result is owned
    assert reason in state['finished_probe_snapshot_rescan']['reason']
    assert calls.count('AUDIT') == calls.count('FULL_FD_SCAN') == calls.count('ORIGINAL_TRACKER') == 1
    assert tracker.known == known_before and 400 not in tracker.known
    for key, value in record_before.items():
        assert tracker.record[key] == value


def test_ptxas_refresh_preserves_new_foreign_rows_for_unchanged_final_exclusive_guard():
    from distillschedule.search.r9_snapshot import refresh_finished_probe
    r9, helper, tracker, state, owned, controller, current, fresh, calls, absent = _finished_probe_fixture()
    state['device_handle_owners'][0]['comm'] = 'ptxas'
    foreign = {'birth': dict(pid=999, start_ticks=90, uid=controller['uid'] + 1),
               'comm': 'ptxas', 'refs': ['/dev/nvidia0']}
    fresh['device_handle_owners'].append(foreign)
    updated, live = refresh_finished_probe(r9, helper, tracker, state, owned, controller, [], proc_stat=absent)
    assert updated is not state  # Refresh is an observation, NOT an admission decision.
    assert updated['device_handle_owners'] == fresh['device_handle_owners']
    assert foreign in updated['device_handle_owners'] and 999 not in live
    assert not all(r9.same_birth(live.get(row['birth']['pid']), row['birth'])
                   for row in updated['device_handle_owners'])
    assert updated['finished_probe_snapshot_rescan']['status'] == 'FRESH_COMPLETE_SCAN_ORIGINAL_EXCLUSIVE_REQUIRED'
