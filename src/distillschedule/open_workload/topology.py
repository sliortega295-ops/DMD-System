"""Allocated rank topology, not a user-supplied optimization recipe.

Launcher node/local-rank facts determine candidate meshes. Host grouping is a
fallback when no launcher metadata exists. Multiple logical nodes on one CPU
host are permitted for functional tests and explicitly identified as such.
No nodes, ranks, devices or processes are allocated here.
"""
from __future__ import annotations
from collections import defaultdict
import os
import socket
from distillschedule.core.contracts import require

_FIELDS = ('GROUP_RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE')


def from_rank_records(records):
    require(isinstance(records, list) and len(records) >= 2,
            'complete allocated rank records required', path='open.topology')
    world = len(records)
    require(all(isinstance(r, dict) for r in records), 'invalid rank record', path='open.topology')
    require(all(type(r.get('rank')) is int for r in records)
            and {r['rank'] for r in records} == set(range(world)),
            'ranks must cover the allocated world exactly once', path='open.topology')
    records = sorted(records, key=lambda r: r['rank'])
    require(all(isinstance(r.get('host'), str) and r['host'] for r in records),
            'rank host identity required', path='open.topology')
    launcher = [r.get('launcher') for r in records]
    require(all(x is None for x in launcher) or all(isinstance(x, dict) for x in launcher),
            'partial launcher metadata across ranks', path='open.topology')
    groups = defaultdict(list)
    if launcher[0] is not None:
        for record in records:
            item = record['launcher']
            require(set(item) == {'node', 'local_rank', 'local_size'}
                    and all(type(v) is int for v in item.values())
                    and item['node'] >= 0 and 0 <= item['local_rank'] < item['local_size'],
                    'invalid launcher topology', path='open.topology')
            groups[item['node']].append(record)
        for group in groups.values():
            sizes = {r['launcher']['local_size'] for r in group}
            require(sizes == {len(group)} and {r['launcher']['local_rank'] for r in group}
                    == set(range(len(group))) and len({r['host'] for r in group}) == 1,
                    'launcher node is incomplete or spans physical hosts', path='open.topology')
            group.sort(key=lambda r: r['launcher']['local_rank'])
    else:
        for record in records:
            groups[record['host']].append(record)
    ordered = sorted(groups.values(), key=lambda group: min(r['rank'] for r in group))
    nodes = [[r['rank'] for r in group] for group in ordered]
    sizes = {len(group) for group in nodes}
    supported = len(nodes) > 1 and len(sizes) == 1 and next(iter(sizes)) > 1
    return {'world_size': world, 'node_groups': nodes,
            'source': 'launcher' if launcher[0] is not None else 'hostname',
            'hosts': [r['host'] for r in records],
            'logical_nodes_on_one_host': len(nodes) > 1 and len({r['host'] for r in records}) == 1,
            'hsdp_admitted': supported,
            'reason': None if supported else 'HSDP_REQUIRES_TWO_EQUAL_MULTI_RANK_NODES'}


def discover():
    """Collectively validate metadata before any new mesh is constructed."""
    import torch.distributed as dist
    from .replicated import checked, gather
    def local():
        values = {key: os.environ.get(key) for key in _FIELDS}
        require(all(v is None for v in values.values()) or all(v is not None for v in values.values()),
                'incomplete torchrun topology environment', path='open.topology')
        launcher = None
        if values['GROUP_RANK'] is not None:
            launcher = dict(zip(('node', 'local_rank', 'local_size'),
                                (int(values[key]) for key in _FIELDS)))
        return {'rank': dist.get_rank(), 'host': socket.gethostname(), 'launcher': launcher}
    records = gather(checked(local, 'rank topology'))
    return from_rank_records(records)
