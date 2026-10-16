"""Search-only node telemetry. No Torch import, CUDA context or synchronization."""
from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import subprocess
import threading
import time

from distillschedule.workloads.best_known.registry import repository_root, write_new_json
from .contracts import require

DRIVER_QUERY_TIMEOUT_SECONDS = 30


def observe_source_identity(root: Path) -> dict:
    """Observe Git, never infer a clean tree from a plan or missing metadata."""
    result = dict(status='UNKNOWN', observed_unix_ns=time.time_ns(),
        identity=dict(head=None, dirty=None, diff_sha256=None),
        scope='GIT_BOUNDARY_HEAD_STATUS_AND_TRACKED_BINARY_DIFF_NOT_CONTINUOUS')
    try:
        def git(*args):
            return subprocess.check_output(['git', '-C', str(root), *args],
                timeout=15, stderr=subprocess.PIPE)
        head = git('rev-parse', 'HEAD').decode().strip()
        require(len(head) == 40 and all(c in '0123456789abcdef' for c in head), 'unknown Git HEAD')
        status = git('status', '--porcelain=v1', '--untracked-files=all')
        diff = git('diff', '--no-ext-diff', '--no-textconv', '--binary', 'HEAD', '--')
        require(git('status', '--porcelain=v1', '--untracked-files=all') == status,
                'Git status changed during observation')
        require(git('rev-parse', 'HEAD').decode().strip() == head, 'Git HEAD changed during observation')
        result.update(status='PASS', identity=dict(head=head, dirty=bool(status or diff),
            diff_sha256=hashlib.sha256(diff).hexdigest()),
            git_status_porcelain=status.decode(errors='replace'))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result['error'] = str(error)
    return result


def validate_capacity_policy(policy: dict) -> None:
    require(isinstance(policy, dict) and policy.get('gpu_metric') == 'peak_allocated_bytes',
            'common capacity metric missing/unsupported')
    for key, size in (('memory_limits', 16), ('host_limits', 2)):
        values = policy.get(key, {})
        require(isinstance(values, dict) and set(values) == {str(i) for i in range(size)}
                and all(type(v) is int and v > 0 for v in values.values()),
                'common capacity limits missing/invalid: ' + key)
    for key in ('gpu_reserve_bytes', 'host_reserve_bytes'):
        require(type(policy.get(key)) is int and policy[key] >= 0, 'common reserve missing/invalid: ' + key)


def _capacity_observation() -> tuple[Path, dict]:
    """Resolve the process's actual cgroup-v2 hierarchy, including parent caps."""
    groups = [line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')]
    require(len(groups) == 1 and groups[0].startswith('/'), 'unknown process cgroup-v2 membership')
    def unescape(value):
        for escaped, actual in (('\\040', ' '), ('\\011', '\t'), ('\\012', '\n'), ('\\134', '\\')):
            value = value.replace(escaped, actual)
        return value
    mounts = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        before, separator, after = line.partition(' - ')
        if separator and after.split()[0] == 'cgroup2':
            fields = before.split()
            base, mount = Path(unescape(fields[3])), Path(unescape(fields[4]))
            if Path(groups[0]).is_relative_to(base):
                mounts.append((base, mount))
    require(len(mounts) == 1, 'ambiguous/unknown cgroup-v2 mount')
    base, mount = mounts[0]; current = mount / Path(groups[0]).relative_to(base)
    require(current.is_relative_to(mount), 'cgroup escaped mounted hierarchy')
    limits = []
    for directory in (current, *current.parents):
        raw = (directory / 'memory.max').read_text().strip()
        require(raw == 'max' or raw.isdecimal(), 'unknown cgroup memory.max')
        value = None if raw == 'max' else int(raw)
        require(value is None or value > 0, 'invalid cgroup memory.max')
        limits.append(dict(path=str(directory), memory_max=raw, limit_bytes=value))
        if directory == mount:
            break
    fields = [line.split() for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith('MemTotal:')]
    require(len(fields) == 1 and len(fields[0]) == 3 and fields[0][2] == 'kB'
            and fields[0][1].isdecimal(), 'unknown physical host capacity')
    physical = int(fields[0][1]) * 1024
    require(physical > 0, 'invalid physical host capacity')
    hard_limits = [r['limit_bytes'] for r in limits if r['limit_bytes'] is not None]
    return current, dict(cgroup_path=str(current), cgroup_limits=limits,
        physical_host_bytes=physical,
        cgroup_hard_limit_bytes=min(hard_limits) if hard_limits else None,
        effective_host_limit_bytes=min([physical] + hard_limits),
        scope='OBSERVED_PROCESS_CGROUP_HIERARCHY_AND_NODE_MEMTOTAL')


class ResourceObserver:
    """Two-second driver/cgroup samples spanning construction through teardown.

    Sampled minima are not continuous exact HBM extrema. Native allocator peaks
    remain separate. This identical observer is enabled for every search arm,
    not silently installed in canonical/default timing paths.
    """
    def __init__(self, path: Path, run_id: str, node_rank: int, *,
                 capacity_policy: dict | None = None, source_root: Path | None = None):
        self.path, self.run_id, self.node_rank = path, run_id, node_rank
        self.source_root = source_root if source_root is not None else repository_root()
        self.source_observations = {}
        if capacity_policy is not None:
            validate_capacity_policy(capacity_policy)
            require(type(node_rank) is int and node_rank in (0, 1), 'invalid capacity observer node')
        self.capacity_policy = deepcopy(capacity_policy)
        self.rows, self.errors = [], []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def sample(self) -> None:
        try:
            raw = subprocess.check_output([
                "nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free",
                "--format=csv,noheader,nounits"], text=True, timeout=DRIVER_QUERY_TIMEOUT_SECONDS)
            gpus = [{"device": int(i), "uuid": uuid.strip(),
                     "total_bytes": int(total) * 2**20, "free_bytes": int(free) * 2**20}
                    for i, uuid, total, free in csv.reader(io.StringIO(raw))]
            if len(gpus) != 8 or {r["device"] for r in gpus} != set(range(8)):
                raise ValueError("observer requires all eight allocated GPUs")
            root = Path("/sys/fs/cgroup")
            capacity = None
            if self.capacity_policy is not None:
                root, capacity = _capacity_observation()
                policy = self.capacity_policy
                require(len({gpu['uuid'] for gpu in gpus}) == 8, 'duplicate observed GPU UUIDs')
                require(policy['host_limits'][str(self.node_rank)] <= capacity['effective_host_limit_bytes'],
                        'declared host budget exceeds observed physical/cgroup capacity')
                for gpu in gpus:
                    require(gpu['uuid'] and 0 <= gpu['free_bytes'] <= gpu['total_bytes']
                            and policy['memory_limits'][str(self.node_rank * 8 + gpu['device'])]
                            + policy['gpu_reserve_bytes'] <= gpu['total_bytes'],
                            'declared GPU budget exceeds observed driver capacity')
            sample = {"time_unix_ns": time.time_ns(), "gpus": gpus,
                "cgroup_current_bytes": int((root / "memory.current").read_text()),
                "cgroup_events": {k: int(v) for k, v in
                                  (line.split() for line in (root / "memory.events").read_text().splitlines())}}
            if capacity is not None:
                host_limit = self.capacity_policy['host_limits'][str(self.node_rank)]
                require(0 <= sample['cgroup_current_bytes'] <= host_limit, 'observed cgroup current exceeds common host budget')
                try:
                    peak = int((root / 'memory.peak').read_text())
                except FileNotFoundError:
                    # Older kernels expose a hard limit but no lifetime peak.
                    # Never substitute a sampled current value for that peak.
                    peak = None
                if peak is None:
                    require(self.capacity_policy['host_reserve_bytes'] == 0
                            and capacity['cgroup_hard_limit_bytes'] == host_limit,
                            'unavailable host peak requires zero reserve and the exact finite cgroup hard limit')
                    require(all(type(sample['cgroup_events'].get(key)) is int
                                and sample['cgroup_events'][key] >= 0 for key in ('oom', 'oom_kill')),
                            'unavailable host peak requires observed OOM counters')
                    sample.update(cgroup_peak_status='UNAVAILABLE',
                        cgroup_peak_scope='UNAVAILABLE_CGROUP_V2_MEMORY_PEAK',
                        host_capacity_evidence='FINITE_CGROUP_HARD_LIMIT_AND_UNCHANGED_OOM_NOT_OBSERVED_PEAK')
                else:
                    require(peak >= sample['cgroup_current_bytes'], 'invalid observed cgroup lifetime peak')
                    require(peak + self.capacity_policy['host_reserve_bytes'] <= host_limit,
                            'observed cgroup peak exceeds common host budget')
                    sample.update(cgroup_peak_status='OBSERVED',
                        cgroup_peak_scope='CGROUP_LIFETIME_NOT_RESET_BY_OBSERVER',
                        host_capacity_evidence='OBSERVED_CGROUP_LIFETIME_PEAK')
                sample.update(capacity_observation=capacity, cgroup_peak_bytes=peak)
            self.rows.append(sample)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.errors.append(str(error))

    def _loop(self) -> None:
        while not self.stop_event.wait(2.0):
            self.sample()

    def __enter__(self):
        self.source_observations['before'] = observe_source_identity(self.source_root)
        self._check_source('before')
        self.sample()
        if self.capacity_policy is not None and self.errors:
            self._write()
            raise ValueError('manual-match resource preflight failed: ' + '; '.join(self.errors))
        self.thread.start()
        return self

    def _check_source(self, point):
        source = self.source_observations[point]
        if self.capacity_policy is not None:
            if source['status'] != 'PASS' or source['identity']['dirty'] is not False:
                self.errors.append('manual-match source observation unknown/dirty: ' + point)
            if point == 'after' and source['identity'] != self.source_observations.get('before', {}).get('identity'):
                self.errors.append('manual-match source identity changed during run')

    def _write(self):
        status = 'PASS' if self.rows and not self.errors and not self.thread.is_alive() else 'FAIL'
        extra = dict(source_observations=self.source_observations,
            source_identity=self.source_observations.get('after', self.source_observations.get('before', {})).get(
                'identity', dict(head=None, dirty=None, diff_sha256=None)))
        if self.capacity_policy is not None:
            extra.update(capacity_policy=self.capacity_policy,
                         capacity_policy_scope='DECLARED_COMMON_BUDGET_NOT_OBSERVED_CAPACITY')
        write_new_json(self.path, {"schema_version": "distillschedule.search.resources.v1",
            "status": status, "run_id": self.run_id, "node_rank": self.node_rank, "period_seconds": 2,
            "scope": "SAMPLED_DRIVER_AND_NODE_CGROUP_NOT_CONTINUOUS_EXTREMA",
            "errors": self.errors, "samples": self.rows, **extra})
        return status

    def __exit__(self, exc_type, *_):
        self.stop_event.set()
        self.thread.join(timeout=DRIVER_QUERY_TIMEOUT_SECONDS + 2)
        self.sample()
        self.source_observations['after'] = observe_source_identity(self.source_root)
        self._check_source('after')
        if self.capacity_policy is not None and any(s.get('cgroup_peak_status') == 'UNAVAILABLE' for s in self.rows):
            if len(self.rows) < 2 or any(len({s['cgroup_events'][key] for s in self.rows}) != 1
                                        for key in ('oom', 'oom_kill')):
                self.errors.append('unavailable host peak requires unchanged observed OOM counters across the run')
        status = self._write()
        if self.capacity_policy is not None and status != 'PASS' and exc_type is None:
            raise ValueError('manual-match resource observations failed: ' + '; '.join(self.errors))
