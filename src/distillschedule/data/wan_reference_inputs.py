"""Original captured W1 traces -> finite prepared CPU1 diagnostic records.

No sampler, source-feature producer, synthetic history, resizing or loss lives
here. Native world16 trace admission is unchanged; actual execution is ONLY
rank0 per-sample, never a world16 numerical-equivalence or quality claim.
"""
from copy import deepcopy
import json
import os
from pathlib import Path

from distillschedule.training.helios_public_assets import checked_pin, read_json, require
from distillschedule.workloads.best_known.registry import fingerprint
from runtime.operation_trace import OperationTrace

SCHEMA = 'distillschedule.wan_cpu_reference_inputs.v1'
SCOPE = 'CPU1_RANK0_PER_SAMPLE_ORIGINAL_W1_NOT_WORLD16'
PRECISION = 'WAN_NATIVE_MIXED_BF16'
CONTROL_BYTES = 2**20


def layout_identity():
    from distillschedule.workloads.best_known.registry import repository_root,file_sha256
    from distillschedule.evidence.source_layout import reviewed_imports
    root = repository_root()
    rows = {row['target']:row for row in reviewed_imports(root)}
    paths = ('runtime/gst_pool_helios.py','runtime/static_fsdp2_p0_helios.py',
             'scripts/distillschedule/probe_static_fsdp2_p0_world16.py')
    require(all(rows[name]['integrated_sha256'] == file_sha256(root/name) for name in paths),
            'unreviewed W1 telemetry source layout overlay')
    return {name:file_sha256(root/name) for name in ('docs/workloads/source_import_map.json',
                                                   'docs/workloads/source_layout.lock.json')}


def input_policy(config):
    keys = ('is_random_drop','random_drop_i2v_ratio','random_drop_v2v_ratio','random_drop_t2v_ratio',
        'corrupt_mode_history','corrupt_mode_prob_history','is_frame_independent_corrupt_history',
        'is_chunk_independent_corrupt_history','noise_corrupt_ratio_history_short',
        'noise_corrupt_ratio_history_mid','noise_corrupt_ratio_history_long','noise_corrupt_clean_prob_history',
        'downsample_min_corrupt_ratio_history','downsample_max_corrupt_ratio_history')
    keys += tuple(role+'_'+key for role in ('generator','critic') for key in
                  ('dynamic_alpha','dynamic_beta','dynamic_sample_type','dynamic_step'))
    return {key:config.get(key,0 if key == 'random_drop_i2v_ratio' else None) for key in keys}


def topology():
    import torch
    require(int(os.environ.get('WORLD_SIZE', '1')) == 1
            and (not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1)
            and torch.get_default_dtype() == torch.float32, 'W1 reference requires actual CPU1/FP32 default topology')


def decode(feeds, task, *, expected=None):
    """Independently verify every operation descriptor before a Program write."""
    import torch
    from runtime.operation_trace import OperationTrace
    from distillschedule.strategies.wan_trace import validate_rank_controls
    topology()
    require(isinstance(feeds, dict) and set(feeds) == {'control', 'control_length', 'tensors'}, 'exact W1 operation tensor tree required')
    encoded = feeds['control']
    require(isinstance(encoded, torch.Tensor) and encoded.dtype == torch.uint8 and encoded.ndim == 1
            and encoded.device.type == 'cpu' and tuple(encoded.shape) == (CONTROL_BYTES,), 'invalid bound W1 controls')
    length = feeds['control_length']
    require(isinstance(length,torch.Tensor) and length.dtype == torch.int64 and tuple(length.shape) == (1,)
            and length.device.type == 'cpu' and 0 < length.item() <= CONTROL_BYTES
            and not encoded[length.item():].count_nonzero().item(), 'invalid W1 control length/padding')
    document = json.loads(bytes(encoded[:length.item()].tolist()))
    require(set(document) == {'schema_version', 'scope', 'task', 'bundle_hash', 'manifests', 'content','layout'}
            and document['schema_version'] == SCHEMA and document['scope'] == SCOPE
            and document['task'] == task.to_dict() and document['layout'] == layout_identity(), 'wrong task/source/reference scope/layout')
    controls = document['manifests']
    require(isinstance(controls, list) and len(controls) == 16, 'complete original rank16 controls required')
    ranks = {rank: OperationTrace(Path('<prepared-native-trace>'), manifest, {})
             for rank, manifest in enumerate(controls)}
    from runtime.operation_trace import _manifest_hash
    require(all(trace.trace_hash == _manifest_hash(trace.manifest) for trace in ranks.values()), 'native rank control hash changed')
    validate_rank_controls(task, ranks)
    tensors = feeds['tensors']
    names = sorted(controls[0]['tensors'])
    require(isinstance(tensors,list) and len(tensors) == len(names), 'native operation tensor inventory changed')
    source = WanReferenceOperationTrace(Path('<prepared-native-trace>'), controls[0], dict(zip(names,tensors,strict=True)))
    source.verify()
    require(all(t.device.type == 'cpu' and not t.requires_grad and bool(t.isfinite().all())
                for t in source.tensors.values()), 'nonfinite/nonresident trace input')
    require(document['content'] == content_identity(source), 'native source tensor/content identity changed')
    if expected is not None:
        require(descriptor(document) == expected, 'prepared source controls/order/content changed')
    # Original entry construction/tensor cloning/operation names remain owned by
    # StaticWorld16Trace. Its disk loader is not called for prepared CPU records.
    trace = WanReferenceTrace.__new__(WanReferenceTrace)
    trace.variant, trace.gan, trace.logical_steps = 'no-gan', False, task.source_steps
    trace.root, trace.bundle_hash = source.root, document['bundle_hash']
    trace._control, trace._loaded = ranks, {0: source}
    trace.binding_hash, trace.reference_document = fingerprint(document), deepcopy(document)
    trace.reference_source = source
    source.reference_owner = trace
    return trace


def content_identity(trace):
    """Not seed/ID metadata: repeated actual source arrays are not fresh data."""
    from runtime.operation_trace import tensor_sha256
    return {name: tensor_sha256(value) for name, value in trace.tensors.items()
            if name.startswith('sample/')}


def descriptor(document):
    return dict(bundle_hash=document['bundle_hash'], control_hash=fingerprint(document),
                content=deepcopy(document['content']))


# Subclass only the entry/storage adapter, never the model/forward or solver.
from runtime.static_world16_trace import StaticWorld16Trace


class WanReferenceOperationTrace(OperationTrace):
    """Storage adapter marker, verified against its complete admitted owner."""


class WanReferenceTrace(StaticWorld16Trace):
    def _load(self, rank):
        require(type(rank) is int and rank == 0, 'CPU reference only admits source rank0')
        return self.reference_source

    def validate_cpu_reference(self):
        topology()
        require(type(self) is WanReferenceTrace and set(self._loaded) == {0}, 'unadmitted CPU trace owner')
        self.reference_source.verify()
        require(self.reference_document['scope'] == SCOPE
                and self.reference_document['layout'] == layout_identity()
                and self.reference_document['manifests'][0] == self.reference_source.manifest
                and self.binding_hash == fingerprint(self.reference_document), 'CPU reference controls mutated')


def cpu_telemetry(device, trace):
    """Telemetry-only opt-in; no allocator/timing zeros or global CUDA patch."""
    import torch
    device = torch.device(device)
    if device.type == 'cuda':
        return False
    require(device.type == 'cpu', 'unsupported non-CUDA reference device')
    # The existing B1 adapter wraps its already-admitted source entry once.
    from distillschedule.runtime.fsdp2_p0 import BatchedOperationTrace
    if type(trace) is BatchedOperationTrace:
        require(len(trace.entries) == 1, 'CPU reference forbids rollout batching')
        trace = trace.entries[0].operation_trace
    # W2 is a separate, closed trace ABI. Never admit an arbitrary object by
    # the presence of validate_cpu_reference (or by a claimed scope string).
    from .helios_nogan_reference_inputs import (HeliosNoGANReferenceTrace,
        HeliosNoGANReferenceOperationTrace, cpu_telemetry as w2_cpu_telemetry)
    from distillschedule.runtime.helios_planner_cycle import _ObservedTrace
    if (type(trace) in (HeliosNoGANReferenceTrace, HeliosNoGANReferenceOperationTrace)
            or (type(trace) is _ObservedTrace and type(trace._trace) is HeliosNoGANReferenceTrace)):
        return w2_cpu_telemetry(device, trace)
    if type(trace) is WanReferenceOperationTrace:
        trace = trace.reference_owner
    if type(trace) is WanReferenceTrace:
        trace.validate_cpu_reference()
    else:
        require(False, 'CPU telemetry requires a fully validated original W1 reference trace')
    return True


def prepare_inputs(source_manifest, output):
    """Copy admitted original rank0 payload, retaining all original rank controls."""
    import torch
    from distillschedule.strategies.wan_task import OriginalWanTask
    from distillschedule.strategies.wan_trace import load_original_trace
    from distillschedule.strategies.wan_planning import source_fingerprint
    from .preparation import _publish
    from .manifest import source_identity
    from .tensors import tensor_tree
    from distillschedule.training.helios_public_assets import local_file
    path = local_file(str(Path(source_manifest).resolve()))
    value = read_json(path)
    require(set(value) == {'schema_version', 'scope', 'evidence_kind', 'task', 'cohorts', 'source','input_config'}
            and value['schema_version'] == SCHEMA and value['scope'] == SCOPE
            and value['source'] == source_fingerprint(), 'W1 source manifest/source changed')
    from distillschedule.training.helios_public_assets import EVIDENCE_KINDS
    require(value['evidence_kind'] in EVIDENCE_KINDS and isinstance(value['cohorts'], list)
            and value['cohorts'], 'finite original cohorts/evidence kind required')
    from distillschedule.strategies.wan_capture import read_training_config
    input_config = checked_pin(value['input_config'])
    read_training_config(input_config)
    base = OriginalWanTask.from_dict(value['task'])
    require(base.first_step == 0 and base.fake_updates == 5, 'W1 cold-zero complete Full5 reference only')
    rows, documents, seen_hashes, seen_contents, seen_sources = [], [source_identity(path),source_identity(input_config)], set(), set(), set()
    def load(cohort, cycle):
        require(set(cohort) == {'root', 'bundle_hash', 'task'}
                and cohort['task'] == {**base.to_dict(), 'first_step': 5*cycle}, 'cohort/source steps reordered')
        task = OriginalWanTask.from_dict(cohort['task'])
        trace = load_original_trace(task, cohort['root'], expected_hash=cohort['bundle_hash'])
        # FULL native input/source validation, not just a rank0 relabel.
        for rank in range(16):
            original = trace._load(rank)
            original.verify()
            source = original.manifest['source']
            require(original.manifest['classification'] == 'NEW_ORIGINAL_WAN_OPERATION_QUALIFIED_INPUTS'
                    and source['source_fingerprint'] == value['source']
                    and source['training_config_sha256'] == value['input_config']['sha256']
                    and source['cohort'] == cycle and source['feature']['rank'] == rank,
                    'original capture/source/config/cohort identity changed')
        native = trace._load(0)
        document = dict(schema_version=SCHEMA, scope=SCOPE, task=task.to_dict(), bundle_hash=trace.bundle_hash,
                        manifests=[trace._control[r].manifest for r in range(16)], content=content_identity(native),layout=layout_identity())
        raw = json.dumps(document,sort_keys=True).encode()
        require(len(raw) <= CONTROL_BYTES, 'native control manifest exceeds bounded reference capacity')
        control = torch.zeros(CONTROL_BYTES,dtype=torch.uint8)
        control[:len(raw)] = torch.tensor(list(raw),dtype=torch.uint8)
        feeds = dict(control=control,control_length=torch.tensor([len(raw)],dtype=torch.int64),
                     tensors=[native.tensors[name] for name in sorted(native.tensors)])
        decode(feeds, task)
        return feeds, document, trace
    for cycle, cohort in enumerate(value['cohorts']):
        feeds, document, trace = load(cohort, cycle)
        content = fingerprint(document['content'])
        raw = trace._control[0].manifest['source']['feature']['sha256']
        require(trace.bundle_hash not in seen_hashes and content not in seen_contents and raw not in seen_sources,
                'reused original trace/source content is not a fresh cohort')
        seen_hashes.add(trace.bundle_hash); seen_contents.add(content); seen_sources.add(raw)
        for rank in range(16):
            root = trace._load(rank).root
            documents.extend(source_identity(root/name) for name in ('trace.json', 'tensors.pt'))
            source = trace._load(rank).manifest['source']['feature']
            identity = source_identity(source['path'])
            require(identity['sha256'] == source['sha256'], 'original source feature bytes changed')
            documents.append(identity)
        documents.append(source_identity(Path(cohort['root'])/'bundle.json'))
        rows.append(dict(id=f'wan-reference-{cycle:06d}', metadata=dict(cycle=cycle, trace=descriptor(document))))
    def produce(row):
        feeds, document, _ = load(value['cohorts'][row['metadata']['cycle']], row['metadata']['cycle'])
        require(descriptor(document) == row['metadata']['trace'], 'native trace changed during preparation')
        return tensor_tree(feeds)
    return _publish(output, rows=rows, provenance=dict(mode='preencoded', documents=documents,
        sources=[dict(id=row['id'], media={'features':source_identity(path)}) for row in rows], encoder=None),
        metadata=dict(schema_version=SCHEMA, scope=SCOPE, precision=PRECISION, source=value['source'],
                      task=base.to_dict(), input_config=value['input_config'],layout=layout_identity(),
                      evidence_kind=value['evidence_kind']), produce=produce)
