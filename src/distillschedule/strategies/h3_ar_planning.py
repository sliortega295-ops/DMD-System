"""Source-qualified whole-native-update costs -> common bounded planner.

No historical scalar is imported as a current cost. Each implementation needs
new, full-world same-source two-cycle observations from this native session.
The update is intentionally opaque: no invented G/F split or overlap discount.
"""
import json
from pathlib import Path

from distillschedule.core.contracts import require
from distillschedule.search.cost_model import ComponentContract, CostModel, OperationCost
from distillschedule.search.plan_ir import BindingRule, StateAllocation
from distillschedule.search.service import BoundCandidate, select
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint, git_head, repository_root, write_new_json
from .h3_ar_assets import checked
from .h3_ar_task import H3ARImplementation, native_graph

COLLECTION = 'distillschedule.original_h3_ar_calibration.v1'
PLAN = 'distillschedule.original_h3_ar_plan.v1'
RANKS = tuple(range(16))


def read(path):
    result = json.loads(Path(path).read_text())
    require(isinstance(result, dict), 'expected JSON object', path='h3_ar.plan')
    return result


def source_fingerprint():
    root = repository_root()
    paths = list((root/'src/distillschedule/strategies').glob('h3_ar_*.py'))
    paths += [root/'src/distillschedule'/p for p in ('native/cli.py', 'search/service.py',
        'search/frontier_schedule.py', 'search/plan_ir.py', 'search/cost_model.py', 'search/resources.py',
        'runtime/role_runner.py', 'runtime/version_sync.py', 'core/graph.py', 'core/contracts.py')]
    return fingerprint({str(p.relative_to(root)): file_sha256(p) for p in sorted(paths)})


def source_identity():
    return dict(source_head=git_head(repository_root()), source_fingerprint=source_fingerprint())


def component(task, implementation):
    return ComponentContract(family='original_h3_ar.native_update',
        work=dict(task=task.to_dict(), actual_input_shapes=task.input_shapes, census=task.census,
                  logical_updates=dict(student=1, fake=1, ema=0)),
        precision=dict(autocast='bfloat16', cache_enabled=False, base='native', lora='native_fp32_master_adam'),
        placement=dict(world_size=16, local_world_size=8, zero_stage=3, parameter_optimizer_offload='none',
            checkpoint_boundary_offload=False, shared_native_backbone=True,
            attention_replay_cache_storage='cpu_staged' if implementation.optimized else 'none'),
        implementation=implementation.to_dict(), context=dict(cycle=2, warmup_updates=1, reseed=False,
            checkpoint='native_gradient_checkpointing', cache='previous56_current56',
            concurrent_resources='whole-world-exclusive', cost_scope='cuda_fenced_complete_native_update'))


def _rank_rows(task, rank_paths):
    from .h3_ar_worker import capacity_ok, host_capacity_signature
    require(len(rank_paths) == 16, 'all 16 native rank receipts required', path='h3_ar.calibration')
    rows = [read(p) for p in rank_paths]
    require({r.get('rank') for r in rows} == set(RANKS), 'missing/duplicate native rank', path='h3_ar.calibration')
    require(len({r['input_identity']['sha256'] for r in rows}) == 16,
            '16 distinct original rank samples required', path='h3_ar.calibration')
    rows.sort(key=lambda r: r['rank'])
    first = rows[0]
    implementation = H3ARImplementation.from_dict(first['implementation'])
    identity = first['identity']; current = source_identity()
    require(identity['source'] == current and identity['task_sha256'] == task.digest,
            'stale source/task calibration', path='h3_ar.calibration')
    for row in rows:
        require(row.get('schema_version') == 'distillschedule.original_h3_ar_execution.v1'
                and row.get('status') == 'COMPLETED_ORIGINAL_H3_AR_WORK'
                and row.get('mode') == 'calibrate' and row.get('gpu_execution') == 'NATIVE_CUDA_RUN'
                and row['identity'] == identity and row['task'] == task.to_dict()
                and row['implementation'] == implementation.to_dict()
                and row['profiled'] is False and row['finite'] is True,
                'not a matching native H3 AR calibration', path='h3_ar.calibration')
        require(row.get('capacity_policy') == first['capacity_policy'],
                'different rank capacity policy', path='h3_ar.calibration')
        validate_policy(row['capacity_policy'])
        require(type(row['static_bytes']) is int and row['static_bytes'] > 0,
                'missing actual static allocation', path='h3_ar.calibration')
        require(row['build']['native_implementation_installed'] is True
                and row['build']['implementation'] == implementation.to_dict()
                and row['build']['input_shapes'] == task.input_shapes,
                'requested implementation was not installed', path='h3_ar.calibration')
        require(len(row['cycles']) == 2 and row['initial_versions'] == dict(student=0, fake=0, ema=0)
                and row['final_versions'] == dict(student=2, fake=2, ema=0),
                'not consecutive two-cycle native work', path='h3_ar.calibration')
        for index, cycle in enumerate(row['cycles'], 1):
            require(cycle['cycle'] == index and cycle['census'] == task.census
                    and cycle['updates'] == cycle['adam_updates'] == dict(student=1, critic=1)
                    and cycle['final_versions'] == dict(student=index, fake=index, ema=0)
                    and cycle['prepared']['status'] == 'PASS'
                    and cycle['prepared']['executed_operations'] == ['native_update']
                    and cycle['RNG_reseeded_between_cycles'] is False,
                    'native census/version/dispatch mismatch', path='h3_ar.calibration')
            require(type(cycle['duration_ns']) is int and cycle['duration_ns'] > 0,
                    'invalid native elapsed time', path='h3_ar.calibration')
            require(cycle['capacity_pass'] is True and capacity_ok(cycle['memory'], row['capacity_policy']),
                    'native capacity policy failed', path='h3_ar.calibration')
            require(cycle['memory']['peak_allocated_bytes'] >= row['static_bytes'],
                    'invalid native memory envelope', path='h3_ar.calibration')
            require(bool(cycle['selected_cache']) == implementation.optimized,
                    'actual selected-cache branch differs from implementation', path='h3_ar.calibration')
            effects = cycle['backend_effects']
            require(effects['actual_compute_observed'] is True and effects['implementation'] == implementation.to_dict(),
                    'native implementation compute witness differs', path='h3_ar.calibration')
            if implementation.optimized:
                cache = cycle['selected_cache']
                require(cache.get('cache_epoch') == index-1 and cache.get('source_student_version') == index-1
                        and cache.get('intervening_fake_version') == index
                        and cache.get('same_process_pinned_pool_reuse') is True
                        and cache.get('stale_result_reuse') is False,
                        'native attention cache lifetime/version mismatch', path='h3_ar.calibration')
        require(row['cycles'][0]['state'] and row['cycles'][1]['state'],
                'native optimizer/RNG metadata missing', path='h3_ar.calibration')
        require(host_capacity_signature(row['cycles'][0]['memory']) == host_capacity_signature(row['cycles'][1]['memory']),
                'host capacity/capability changed between native cycles', path='h3_ar.calibration')
    return rows, implementation


def _host_capacity_bindings(rows):
    from .h3_ar_worker import host_capacity_signature
    return [[host_capacity_signature(cycle['memory']) for cycle in row['cycles']] for row in rows]


def _binding(task, identity, rows, implementation):
    return dict(source_head=identity['source']['source_head'], semantic_fingerprint=task.digest,
        hardware_fingerprint=fingerprint(dict(gpus=[r['hardware'] for r in rows],
            host_capacity=_host_capacity_bindings(rows))),
        environment_fingerprint=fingerprint(dict(source=identity['source'], runtimes=[r['runtime'] for r in rows])),
        asset_fingerprint=identity['assets_sha256'], precision_fingerprint=fingerprint(component(task, implementation).precision))


def validate_policy(policy):
    require(isinstance(policy, dict) and set(policy) == {'gpu_limit_bytes', 'gpu_reserve_bytes', 'host_limit_bytes', 'host_reserve_bytes'},
            'explicit common GPU/host capacity policy required', path='h3_ar.capacity')
    for key, value in policy.items():
        require(type(value) is int and value >= (0 if 'reserve' in key else 1), 'invalid capacity policy', path='h3_ar.capacity')


def assemble(task, rank_paths, output):
    rows, implementation = _rank_rows(task, rank_paths)
    first = rows[0]; identity = first['identity']
    binding = _binding(task, identity, rows, implementation)
    contract = component(task, implementation)
    static = max(r['static_bytes'] for r in rows)
    peak = max(max(c['memory']['peak_allocated_bytes'] for c in r['cycles']) for r in rows)
    # A conservative envelope, not cross-rank or cross-stage claimed overlap.
    workspace = max(0, peak - min(r['static_bytes'] for r in rows))
    duration = max(r['cycles'][1]['duration_ns'] for r in rows)
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=False)
    raw_sources = [dict(path=str(Path(p).resolve()), sha256=file_sha256(Path(p))) for p in rank_paths]
    raw = dict(schema_version='distillschedule.operation_calibration.v1', binding=binding,
        key=contract.digest, samples_ns=[duration], workspace_bytes_per_rank=workspace,
        status='PASS', profiled=False, kind='native_operation', operation_contract_sha256=contract.digest,
        component_contract=contract.to_dict(), native_rank_sources=raw_sources,
        aggregation='max16_complete_cycle2_no_component_decomposition', claimable_speedup=False)
    path = root/'native-update-cost.json'; write_new_json(path, raw)
    evidence = {k: raw[k] for k in ('status', 'profiled', 'kind', 'operation_contract_sha256', 'component_contract')}
    evidence.update(source_receipt=str(path), source_receipt_sha256=file_sha256(path))
    costs = CostModel(binding, (OperationCost(contract.digest, (duration,), workspace, evidence),))
    write_new_json(root/'costs.json', costs.to_dict())
    limits = first['capacity_policy']
    manifest = dict(schema_version=COLLECTION, task=task.to_dict(), identity=identity, binding=binding,
        implementation=implementation.to_dict(), static_bytes=static, capacity_policy=limits,
        cost_model=dict(path=str(root/'costs.json'), sha256=file_sha256(root/'costs.json')),
        rank_sources=raw_sources, runtimes=[r['runtime'] for r in rows], hardware=[r['hardware'] for r in rows],
        host_capacity_bindings=_host_capacity_bindings(rows),
        evidence_scope='MEASURED', numerical_status='REVIEW_REQUIRES_MATCHED_REFERENCE',
        historical_scalar_costs_used=False, claimable_speedup=False)
    write_new_json(root/'collection.json', manifest)
    return manifest


def load_collections(task, paths):
    result = []; first = None
    for path in paths:
        doc = read(path)
        require(doc.get('schema_version') == COLLECTION and doc.get('task') == task.to_dict()
                and doc.get('evidence_scope') == 'MEASURED', 'not measured H3 AR costs', path='h3_ar.plan')
        require(doc['identity']['source'] == source_identity(), 'source changed; recalibrate W6', path='h3_ar.plan')
        rank_paths = [checked(ref) for ref in doc['rank_sources']]
        rows, implementation = _rank_rows(task, rank_paths)
        require(doc['identity'] == rows[0]['identity'] and doc['implementation'] == implementation.to_dict(),
                'collection/native binding differs', path='h3_ar.plan')
        require(doc['binding'] == _binding(task, doc['identity'], rows, implementation),
                'collection identity binding changed', path='h3_ar.plan')
        require(doc['runtimes'] == [r['runtime'] for r in rows] and doc['hardware'] == [r['hardware'] for r in rows],
                'collection environment/hardware changed', path='h3_ar.plan')
        require(doc['host_capacity_bindings'] == _host_capacity_bindings(rows),
                'collection host capacity/capability changed', path='h3_ar.plan')
        costs = CostModel.from_dict(read(checked(doc['cost_model']))); costs.verify_receipts()
        costs.bind(doc['binding'], problem_scope='MEASURED')
        contract = component(task, implementation); cost = costs.get_component(contract)
        peak = max(c['memory']['peak_allocated_bytes'] for r in rows for c in r['cycles'])
        workspace = max(0, peak - min(r['static_bytes'] for r in rows))
        require(len(costs.records) == 1 and cost.samples_ns == (max(r['cycles'][1]['duration_ns'] for r in rows),)
                and cost.workspace_bytes_per_rank == workspace
                and doc['static_bytes'] == max(r['static_bytes'] for r in rows)
                and doc['capacity_policy'] == rows[0]['capacity_policy'],
                'cost/capacity differs from actual native measurements', path='h3_ar.plan')
        if first is None:
            first = doc
        else:
            require(all(doc[k] == first[k] for k in ('identity', 'binding', 'runtimes', 'hardware', 'capacity_policy', 'host_capacity_bindings')),
                    'incompatible native calibration cohorts', path='h3_ar.plan')
        result.append((doc, implementation, costs))
    require(result and len({s.id for _, s, _ in result}) == len(result), 'empty/duplicate calibration candidates', path='h3_ar.plan')
    return result


def _candidates(task, collections, memory_limit_bytes):
    graph = native_graph(task); ranks = RANKS; result = []
    for doc, implementation, costs in collections:
        limit = doc['capacity_policy']['gpu_limit_bytes']
        if memory_limit_bytes is not None:
            require(type(memory_limit_bytes) is int and memory_limit_bytes > 0, 'invalid selected memory limit', path='h3_ar.plan')
            limit = min(limit, memory_limit_bytes)
        key = component(task, implementation).digest
        rule = BindingRule('native_update', {}, key, ranks, tuple(f'gpu:{r}' for r in ranks), ('state',), key)
        state = StateAllocation('state', ranks, doc['static_bytes'], ('native_runtime',), False, doc['identity']['assets_sha256'])
        result.append(BoundCandidate(implementation.id, implementation.to_dict(), graph, {'native_update': rule},
            costs, (state,), (), {r: limit for r in ranks}, 'original-h3-ar-world16-native-session'))
    return result


def plan(task, paths, *, memory_limit_bytes=None, max_states=50000, beam_width=0, timeout_seconds=60):
    collections = load_collections(task, paths)
    searched = select(_candidates(task, collections, memory_limit_bytes), max_states=max_states,
                      beam_width=beam_width, timeout_seconds=timeout_seconds)
    require(searched['ranked'], 'no feasible W6 plan within measured capacity/candidates', path='h3_ar.plan')
    result = dict(schema_version=PLAN, task=task.to_dict(), identity=collections[0][0]['identity'],
        collections=[dict(path=str(Path(p).resolve()), sha256=file_sha256(Path(p))) for p in paths],
        selected=searched['ranked'][0], ranked=searched['ranked'], rejected=searched['rejected'],
        candidate_count=searched['candidate_count'],
        uncalibrated=[name for name in ('zero3-native', 'zero3-ap') if name not in {impl.id for _, impl, _ in collections}],
        graph=native_graph(task).to_dict(), evidence_scope='MEASURED',
        capacity_policy=collections[0][0]['capacity_policy'],
        native_lowering='same persistent H3ARSession + original DMD2Trainer.update through PreparedRolePlan',
        candidate_space='only measured zero3-native / accepted zero3-ap; no arbitrary stage rescheduling',
        gpu_execution='NOT_RUN_BY_PLANNER', claimable_speedup=False, numerical_status='REVIEW_REQUIRES_MATCHED_REFERENCE')
    result['plan_sha256'] = fingerprint(result)
    return result


def validate_plan(task, path):
    result = read(path); unsigned = dict(result); sha = unsigned.pop('plan_sha256', None)
    require(result.get('schema_version') == PLAN and sha == fingerprint(unsigned)
            and result.get('task') == task.to_dict(), 'saved W6 plan/task changed', path='h3_ar.run')
    collections = load_collections(task, [checked(ref) for ref in result['collections']])
    require(result['identity'] == collections[0][0]['identity']
            and result['capacity_policy'] == collections[0][0]['capacity_policy']
            and result['graph'] == native_graph(task).to_dict(), 'saved source/graph/capacity changed', path='h3_ar.run')
    selected = result['selected']
    matches = [c for c in _candidates(task, collections, None) if c.id == selected['candidate_id']]
    require(len(matches) == 1, 'saved uncalibrated W6 candidate', path='h3_ar.run')
    # Re-run the common scheduler for that candidate with the saved tighter
    # capacity. Hash-only validation cannot admit a forged lower memory peak.
    candidate = matches[0]
    limits = {r['rank']: r['limit_bytes'] for r in selected['memory']}
    require(set(limits) == set(RANKS) and all(0 < v <= candidate.memory_limits[r] for r, v in limits.items()),
            'saved capacity exceeds calibration', path='h3_ar.run')
    from dataclasses import replace
    expected = select([replace(candidate, memory_limits=limits)])['ranked']
    require(expected, 'saved plan violates capacity', path='h3_ar.run')
    for key in ('implementation', 'order', 'timeline', 'memory', 'predicted_makespan_ns', 'cost_model_sha256', 'graph_sha256', 'executable_backend'):
        require(selected[key] == expected[0][key], 'saved selected '+key+' changed', path='h3_ar.run')
    return result, next(doc for doc, implementation, _ in collections if implementation.id == selected['candidate_id'])
