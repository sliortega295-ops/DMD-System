"""Fresh physical/cgroup headroom observation; not an allocation reservation."""
import os
from pathlib import Path


def admit_host_memory(needed, reserve):
    from distillschedule.search.resources import _capacity_observation
    _, observation = _capacity_observation()
    available = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    for row in observation['cgroup_limits']:
        if row['limit_bytes'] is not None:
            current = int((Path(row['path']) / 'memory.current').read_text())
            if not 0 <= current <= row['limit_bytes']:
                raise ValueError('invalid current cgroup memory accounting')
            available = min(available, row['limit_bytes'] - current)
    if needed + reserve > available:
        raise ValueError('host/cgroup headroom cannot hold export/codec plus reserve')
    return dict(required_bytes=needed, reserve_bytes=reserve, available_bytes=available,
                scope='FRESH_PHYSICAL_CGROUP_OBSERVATION_NOT_RESERVATION')
