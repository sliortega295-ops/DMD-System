"""Topology admission has no model-name or strategy preference."""
import pytest
from distillschedule.open_workload.topology import from_rank_records


def records():
    return [{'rank': i, 'host': 'host'+str(i//2),
             'launcher': {'node': i//2, 'local_rank': i%2, 'local_size': 2}} for i in range(4)]


def test_inferred_node_groups_are_facts_not_a_winning_layout():
    result = from_rank_records(records())
    assert result['node_groups'] == [[0, 1], [2, 3]]
    assert result['hsdp_admitted'] and not result['logical_nodes_on_one_host']


def test_logical_cpu_nodes_do_not_claim_physical_multinode_validation():
    rows = records()
    for row in rows: row['host'] = 'cpu-test-host'
    result = from_rank_records(rows)
    assert result['hsdp_admitted'] and result['logical_nodes_on_one_host']


def test_plain_process_group_uses_hosts_and_does_not_invent_nodes():
    rows = records()
    for row in rows: row.update(host='same-host', launcher=None)
    result = from_rank_records(rows)
    assert result['node_groups'] == [[0, 1, 2, 3]] and not result['hsdp_admitted']


@pytest.mark.parametrize('bad', ['missing_rank', 'partial', 'bad_size', 'duplicate_local', 'split_host', 'bool_rank'])
def test_bad_topology_is_rejected(bad):
    rows = records()
    if bad == 'missing_rank': rows[-1]['rank'] = 0
    if bad == 'partial': rows[-1]['launcher'] = None
    if bad == 'bad_size': rows[-1]['launcher']['local_size'] = 3
    if bad == 'duplicate_local': rows[-1]['launcher']['local_rank'] = 0
    if bad == 'split_host': rows[-1]['host'] = 'other-host'
    if bad == 'bool_rank': rows[0]['rank'] = False
    with pytest.raises(ValueError): from_rank_records(rows)
