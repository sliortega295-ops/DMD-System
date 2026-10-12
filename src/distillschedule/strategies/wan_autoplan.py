"""One command per allocated node: new task -> original-backend plan execution.

No SSH, cluster allocation or implicit restart. Preparation is performed once on
node 0; each node independently validates the same actual input/model bytes.
The existing campaign owns the native launch budget and preserves REVIEW.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

from distillschedule.core.contracts import require
from .wan_prepare import prepare, publish_new_json
from .wan_campaign import run as run_campaign, verify as verify_campaign


def _read(path):
    return json.loads(Path(path).read_text())


def run(task, sources, model_assets, training_config, output, *, node_rank,
        master_addr, master_port, include_fusion=True, max_launches=8,
        execute=False, resume=False, preparation_timeout=7200, launch_timeout=7200):
    require(type(node_rank) is int and node_rank in (0, 1), 'node rank must be 0 or 1', path='wan.auto')
    require(isinstance(master_addr, str) and bool(master_addr)
            and type(master_port) is int and 1024 <= master_port <= 65535,
            'explicit rendezvous address and port required', path='wan.auto')
    require(type(preparation_timeout) in (int, float) and 1 <= preparation_timeout <= 86400,
            'invalid preparation timeout', path='wan.auto')
    require(type(launch_timeout) in (int, float) and 60 <= launch_timeout <= 86400,
            'invalid native launch timeout', path='wan.auto')
    checked = prepare(task, sources, model_assets, training_config, output,
                      include_fusion=include_fusion, max_launches=max_launches)
    key = checked['preparation_sha256']
    root = Path(output).resolve()
    if not execute:
        return {**checked, 'node_rank':node_rank, 'status':'VALIDATED_AUTO_RUN_NOT_EXECUTED',
                'rendezvous':{'master_addr':master_addr, 'master_port':master_port},
                'next':'explicit --execute on both preallocated nodes sharing this output path'}
    failure = root/f'prepare-failure-{key}.json'
    if failure.is_file():
        require(False, 'this preparation failed previously; inspect preserved failure, do not resample: '
                +str(failure), path='wan.auto')
    if node_rank == 0:
        try:
            ready = prepare(task, sources, model_assets, training_config, root,
                            include_fusion=include_fusion, max_launches=max_launches,
                            execute=True, resume=resume)
        except BaseException as exc:
            # Do not overwrite an existing capture, campaign, or another attempt's outcome.
            if root.is_dir() and not failure.exists() and not (root/'prepare.lock').exists() and not (root/'ready.json').exists():
                publish_new_json(failure, {'status':'PREPARATION_FAILED', 'preparation_sha256':key,
                                        'error_type':type(exc).__name__, 'error':str(exc)})
            raise
    else:
        deadline = time.monotonic()+preparation_timeout
        while not (root/'ready.json').is_file():
            if failure.is_file():
                error = _read(failure)
                require(error.get('preparation_sha256') == key, 'peer failure identity mismatch', path='wan.auto')
                raise RuntimeError('node 0 input preparation failed: '+error['error'])
            if time.monotonic() >= deadline:
                raise TimeoutError('node 0 has not published prepared inputs; no GPU launch has occurred')
            time.sleep(.25)
        ready = _read(root/'ready.json')
    require(ready.get('status') == 'READY_FOR_ORIGINAL_WAN_CAMPAIGN'
            and ready.get('preparation_sha256') == key, 'peer prepared a different task or input cohort', path='wan.auto')
    campaign = Path(ready['campaign']).resolve()
    require(campaign == root/'campaign'/'campaign.json', 'prepared campaign escapes selected output', path='wan.auto')
    verified = verify_campaign(campaign)
    require(verified['campaign_sha256'] == ready['campaign_sha256']
            and verified['task_sha256'] == task.digest, 'prepared campaign changed', path='wan.auto')
    # Original controller enforces the finite total launch budget, identity and
    # no-relaunch-on-unknown-outcome contract. Do not replace REVIEW with PASS.
    result = run_campaign(campaign, node_rank=node_rank, master_addr=master_addr,
                          master_port=master_port, execute=True, resume=resume,
                          timeout_seconds=launch_timeout)
    return {**result, 'preparation_sha256':key, 'campaign':str(campaign),
            'input_trace_hash':ready['trace_hash']}
