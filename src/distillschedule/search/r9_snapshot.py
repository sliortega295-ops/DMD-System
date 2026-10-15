"""Refresh an expired worker nvidia-smi/ptxas snapshot without granting PID authority.

An FD scan is not atomic with tracker discovery. A short-lived child may exit
before the tracker ever sees it. Only a same-UID probe whose recorded parent is
still in the admitted worker tree, with precise /proc absence before and after
one fresh full audit, qualifies for a new observation. The caller must still
run the unchanged exclusive_all8 check on the complete fresh result.
"""
import errno
import os
import time


def refresh_finished_probe(r9, helper, tracker, state, owned, controller, ancestors, *, proc_stat=None):
    handles = state['device_handle_owners']
    unmatched = [h for h in handles if not r9.same_birth(owned.get(h['birth']['pid']), h['birth'])]
    if not unmatched:
        return state, owned
    stat = proc_stat or os.stat
    deadline = time.monotonic() + 5
    proof = dict(status='NOT_PROVEN_ORIGINAL_GUARD_MUST_FAIL', attempts=1,
        full_FD_rescans=0, scope='ENDED_WORKER_NVIDIA_SMI_OR_PTXAS_FRESH_OBSERVATION_ONLY',
        no_signal=True, no_wait=True, no_added_tracker_births=True,
        no_NVML_PID_authority=True, no_deleted_or_filtered_FD_rows=True)
    state['finished_probe_snapshot_rescan'] = proof

    def check(ok, reason):
        if time.monotonic() >= deadline:
            raise ValueError('FINISHED_PROBE_5S_DEADLINE')
        if not ok:
            raise ValueError(reason)

    def clean():
        check(not tracker.discovery_unresolved and not tracker.adoption_unresolved
              and not any(tracker.record.get(k) for k in
                  ('discovery_failed_once', 'adoption_error', 'adopted_reap_error')),
              'FINISHED_PROBE_STICKY_TRACKER_FAILURE')

    def active(saved):
        now = helper.pid_identity(saved['pid'])
        check(r9.same_birth(now, saved) and now.get('state') not in ('Z', 'X')
              and now.get('uid') == controller['uid'] == os.getuid()
              and all(now.get(k) == saved.get(k) for k in ('ppid', 'pgid')),
              'FINISHED_PROBE_ANCHOR_CHANGED_OR_UNREADABLE')
        return now

    def absent(pid):
        check(type(pid) is int and pid > 0, 'FINISHED_PROBE_POSITIVE_PID')
        check(helper.pid_identity(pid) is None, 'FINISHED_PROBE_STILL_LIVE_OR_REUSED')
        observations = []
        for path in (f'/proc/{pid}/stat', f'/proc/{pid}'):
            try:
                stat(path)
            except FileNotFoundError as error:
                check(error.errno == errno.ENOENT, 'FINISHED_PROBE_PRECISE_ENOENT_REQUIRED')
                observations.append(dict(path=path, errno=error.errno))
            else:
                raise ValueError('FINISHED_PROBE_PROC_EXISTS_OR_UNREADABLE')
        return observations

    try:
        check(len(handles) <= 128 and len(unmatched) <= 32
              and len({h['birth']['pid'] for h in unmatched}) == len(unmatched),
              'FINISHED_PROBE_COUNT_BOUND')
        clean()
        check(tracker.record.get('exec_permit_issued') is True, 'FINISHED_PROBE_EXEC_PERMIT')
        root = tracker.record.get('child_identity')
        check(isinstance(root, dict) and tracker.child is not None and tracker.child.pid == root['pid']
              and r9.same_birth(owned.get(root['pid']), root), 'FINISHED_PROBE_ADMITTED_ROOT')
        before = {controller['pid']: active(controller), root['pid']: active(root)}
        check(root['ppid'] == controller['pid'], 'FINISHED_PROBE_ROOT_PARENT')
        observations = []
        for handle in unmatched:
            birth = handle['birth']
            check(handle.get('comm') in ('nvidia-smi', 'ptxas') and birth.get('uid') == controller['uid']
                  and all(type(birth.get(k)) is int and birth[k] > 0
                          for k in ('pid', 'ppid', 'pgid', 'start_ticks')),
                  'FINISHED_PROBE_ONLY_RECORDED_WORKER_NVIDIA_SMI_OR_PTXAS')
            parent = owned.get(birth['ppid'])
            check(isinstance(parent, dict) and birth['pgid'] == parent['pgid']
                  and birth['start_ticks'] >= parent['start_ticks'], 'FINISHED_PROBE_PARENT_NOT_OWNED')
            chain, seen = [], set()
            for _ in range(64):
                check(parent['pid'] not in seen and r9.same_birth(tracker.known.get(parent['pid']), parent),
                      'FINISHED_PROBE_PARENT_NOT_TRACKER_PROVEN')
                seen.add(parent['pid'])
                now = active(parent)
                before[parent['pid']] = now
                chain.append(now)
                if parent['pid'] == root['pid']:
                    break
                parent = owned.get(now['ppid'])
                check(isinstance(parent, dict), 'FINISHED_PROBE_PARENT_CHAIN_OUTSIDE_WORKER')
            else:
                raise ValueError('FINISHED_PROBE_64_HOP_BOUND')
            observations.append(dict(observed_birth=birth, current_parent_chain=chain,
                                     ENOENT_before=absent(birth['pid'])))
        fresh = r9.audit()
        fresh['unreadable_verified_ssh_ancestors'] = []
        fresh['device_handle_owners'] = r9.device_handles(helper, ancestors, fresh['unreadable_verified_ssh_ancestors'])
        proof['full_FD_rescans'] = 1
        refreshed = tracker.live()  # Existing tracker algorithm, never insert a historical PID.
        clean()
        pids = {h['birth']['pid'] for h in unmatched}
        check(not pids.intersection(refreshed)
              and not any(h['birth']['pid'] in pids for h in fresh['device_handle_owners']),
              'FINISHED_PROBE_PID_PRESENT_IN_FRESH_SNAPSHOT')
        for observation in observations:
            observation['ENOENT_after'] = absent(observation['observed_birth']['pid'])
        for pid, birth in before.items():
            active(birth)
            if pid != controller['pid']:
                check(r9.same_birth(refreshed.get(pid), birth), 'FINISHED_PROBE_PARENT_LEFT_TRACKER')
        proof.update(status='FRESH_COMPLETE_SCAN_ORIGINAL_EXCLUSIVE_REQUIRED', observations=observations,
                     fresh_FD_count=len(fresh['device_handle_owners']), fresh_live_count=len(refreshed))
        for key in ('FD_birth_reconciliation', 'FD_snapshot_exit_reconciliation', 'verified_zombie_FD_snapshot_rescan'):
            if key in state:
                fresh[key] = state[key]
        fresh['finished_probe_snapshot_rescan'] = proof
        return fresh, refreshed
    except Exception as error:
        proof['reason'] = repr(error)
        return state, owned
