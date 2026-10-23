"""Finite W2 CPU1/rank0 records from the original rank16 cohort producer.

Only storage changes here. HeliosCohortTask, all sixteen original controls,
step-qualified conditioning and the native entry constructors remain owners.
"""
from copy import deepcopy
import json
from pathlib import Path

from distillschedule.training.helios_public_assets import checked_pin, read_json, require, EVIDENCE_KINDS
from distillschedule.training.helios_inputs import (HeliosCohortTask, HeliosCohortSources,
    HeliosTrainingTrace, load_stream, load_cohort, validate_controls, validate_rank_payload, source_fingerprint)
from distillschedule.workloads.best_known.registry import fingerprint
from runtime.operation_trace import OperationTrace

SCHEMA = 'distillschedule.helios_nogan_cpu_reference_inputs.v1'
SCOPE = 'CPU1_RANK0_PER_SAMPLE_ORIGINAL_W2_NOT_WORLD16'
PRECISION = 'WAN_NATIVE_MIXED_BF16'
CONTROL_BYTES = 2**20


def topology():
    from distillschedule.training.helios_nogan_bootstrap import _cpu1
    _cpu1()


def layout_identity():
    from distillschedule.workloads.best_known.registry import repository_root, file_sha256
    from distillschedule.evidence.source_layout import reviewed_imports
    root = repository_root()
    rows = {row['target']:row for row in reviewed_imports(root)}
    paths = ('runtime/gst_pool_helios.py', 'runtime/static_fsdp2_p0_helios.py',
             'scripts/distillschedule/probe_static_fsdp2_p0_world16.py')
    require(all(rows[name]['integrated_sha256'] == file_sha256(root/name) for name in paths),
            'unreviewed W2 telemetry source layout overlay')
    return {name:file_sha256(root/name) for name in ('docs/workloads/source_import_map.json',
                                                   'docs/workloads/source_layout.lock.json')}


def content_identity(trace):
    from runtime.operation_trace import tensor_sha256
    # W1's sample/... prefix omits ALL W2 text and masks.
    result = {name:tensor_sha256(value) for name,value in trace.tensors.items()
              if '/sample/' in name and name.startswith('step/')}
    require(len(result) == 10, 'five step-qualified text/mask pairs required')
    return result


def conditioning_fingerprint(content):
    """Compare bytes by within-cohort sample position, not global step names.

    Otherwise copied text under step/15 would appear fresh versus step/10 even
    though the payload hashes are identical. Preserve exact keys in identity.
    """
    steps = sorted({int(name.split('/')[1]) for name in content})
    require(len(steps) == 5, 'five conditioning source steps required')
    return fingerprint({f"sample/{steps.index(int(name.split('/')[1]))}/"+name.split('/sample/',1)[1]:value
                        for name,value in content.items()})


def descriptor(document):
    return dict(bundle_hash=document['bundle_hash'], control_hash=fingerprint(document),
                content=deepcopy(document['content']), sources=deepcopy(document['source_hashes']))


class HeliosNoGANReferenceOperationTrace(OperationTrace):
    """Closed storage marker, usable only with its verified W2 owner."""


class HeliosNoGANReferenceTrace(HeliosTrainingTrace):
    def _load(self, rank):
        require(type(rank) is int and rank == 0, 'W2 CPU execution only admits source rank0')
        return self.reference_source

    def validate_cpu_reference(self):
        topology()
        require(type(self) is HeliosNoGANReferenceTrace and set(self._loaded) == {0}
                and type(self.reference_source) is HeliosNoGANReferenceOperationTrace
                and self.reference_source.reference_owner is self, 'unadmitted W2 CPU trace owner')
        document = self.reference_document
        require(document['scope'] == SCOPE and document['layout'] == layout_identity()
                and document['task'] == self.task.to_dict()
                and self.logical_steps == self.task.source_steps
                and self.binding_hash == fingerprint(document)
                and self._control[0].manifest == self.reference_source.manifest
                and [self._control[r].manifest for r in range(16)] == document['manifests'],
                'W2 CPU trace controls/source changed')
        self.reference_source.verify()
        require(document['content'] == content_identity(self.reference_source), 'W2 conditioning bytes changed')


def cpu_telemetry(device, trace):
    """Closed W2 dispatch; arbitrary objects and unqualified CPU traces refuse."""
    import torch
    from distillschedule.runtime.helios_planner_cycle import _ObservedTrace
    from distillschedule.runtime.fsdp2_p0 import BatchedOperationTrace
    device = torch.device(device)
    if device.type == 'cuda':
        return False
    require(device.type == 'cpu', 'unsupported W2 reference device')
    if type(trace) is _ObservedTrace:
        require(trace._observer.mode == 'cpu-diagnostic', 'W2 observed trace requires explicit CPU diagnostic mode')
        trace = trace._trace
    if type(trace) is BatchedOperationTrace:
        require(len(trace.entries) == 1, 'W2 CPU forbids rollout batching')
        trace = trace.entries[0].operation_trace
    if type(trace) is HeliosNoGANReferenceOperationTrace:
        trace = trace.reference_owner
    require(type(trace) is HeliosNoGANReferenceTrace, 'CPU telemetry requires a validated original W2 reference trace')
    trace.validate_cpu_reference()
    return True


def decode(feeds, task, *, expected=None):
    """Independently admit every source/rank/step/control before Program writes."""
    import torch
    from runtime.operation_trace import _manifest_hash
    topology()
    require(type(task) is HeliosCohortTask and isinstance(feeds,dict)
            and set(feeds) == {'control','control_length','tensors'}, 'exact typed W2 operation tree required')
    encoded, length = feeds['control'], feeds['control_length']
    require(isinstance(encoded,torch.Tensor) and encoded.device.type == 'cpu' and encoded.dtype == torch.uint8
            and tuple(encoded.shape) == (CONTROL_BYTES,), 'invalid bounded W2 controls')
    require(isinstance(length,torch.Tensor) and length.device.type == 'cpu' and length.dtype == torch.int64
            and tuple(length.shape) == (1,) and 0 < length.item() <= CONTROL_BYTES
            and not encoded[length.item():].count_nonzero().item(), 'invalid W2 control length/padding')
    document = json.loads(bytes(encoded[:length.item()].tolist()))
    require(set(document) == {'schema_version','scope','task','bundle_hash','manifests','sources',
        'source_hashes','input_config','source','content','layout'} and document['schema_version'] == SCHEMA
        and document['scope'] == SCOPE and document['task'] == task.to_dict()
        and document['source'] == source_fingerprint() and document['layout'] == layout_identity(),
        'wrong W2 task/source/scope/layout')
    source = HeliosCohortSources.from_dict(document['sources'])
    controls = document['manifests']
    require(isinstance(controls,list) and len(controls) == 16, 'complete W2 rank16 controls required')
    ranks = {rank:OperationTrace(Path('<prepared-W2>'),manifest,{}) for rank,manifest in enumerate(controls)}
    require(all(value.trace_hash == _manifest_hash(value.manifest) for value in ranks.values()), 'W2 rank control hash changed')
    validation = validate_controls(task,ranks)
    hashes = []
    for rank,value in ranks.items():
        provenance = value.manifest['source']
        samples = provenance.get('samples',[])
        require(value.manifest['classification'] == 'NEW_ORIGINAL_HELIOS_STEP_QUALIFIED_INPUTS'
                and provenance == dict(source_fingerprint=document['source'],
                    training_config_sha256=document['input_config']['sha256'],rank=rank,samples=samples,
                    seed=source.seed,cohort=task.index,
                    rng_scope='new_native_CPU_streams_not_prior_training_rng_continuation')
                and len(samples) == 5 and all(row['rank'] == rank and row['step'] == task.source_steps[i]
                    and row['path'] == source.samples[i][rank] for i,row in enumerate(samples)),
                'W2 seed/rank/source/config provenance changed')
        hashes.extend(row['sha256'] for row in samples)
    require(len(set(hashes)) == 80 and document['source_hashes'] == sorted(hashes), 'copied W2 source bytes are not fresh')
    names, tensors = sorted(controls[0]['tensors']), feeds['tensors']
    require(isinstance(tensors,list) and len(tensors) == len(names), 'W2 tensor inventory changed')
    original = HeliosNoGANReferenceOperationTrace(Path('<prepared-W2>'),controls[0],dict(zip(names,tensors,strict=True)))
    original.verify()
    require(all(t.device.type == 'cpu' and not t.requires_grad and bool(t.isfinite().all())
                for t in tensors), 'nonfinite/nonresident W2 trace input')
    require(document['content'] == content_identity(original), 'step-qualified W2 conditioning content changed')
    if expected is not None:
        require(descriptor(document) == expected, 'W2 prepared source controls/order/content changed')
    trace = HeliosNoGANReferenceTrace.__new__(HeliosNoGANReferenceTrace)
    trace.task, trace.root, trace.variant, trace.gan, trace.request = task, original.root, 'no-gan', False, None
    trace.logical_steps, trace.bundle_hash = task.source_steps, document['bundle_hash']
    trace._control, trace._loaded, trace.rank_control_validation = ranks, {0:original}, validation
    trace.binding_hash, trace.reference_document, trace.reference_source = fingerprint(document), document, original
    original.reference_owner = trace
    validate_rank_payload(trace,0,source)
    return trace


def prepare_inputs(source_manifest, output):
    import torch
    from .preparation import _publish
    from .manifest import source_identity
    from .tensors import tensor_tree
    path = Path(source_manifest).resolve()
    value = read_json(path)
    require(set(value) == {'schema_version','scope','evidence_kind','stream','input_config'}
            and value['schema_version'] == SCHEMA and value['scope'] == SCOPE
            and value['evidence_kind'] in EVIDENCE_KINDS, 'exact W2 source envelope required')
    stream_path, config = checked_pin(value['stream']), checked_pin(value['input_config'])
    document = read_json(stream_path)
    count = len(document['cohorts'])
    stream = load_stream(stream_path,cycles=count)
    require(value['input_config']['sha256'] == stream['identity']['training_config_sha256'], 'W2 input config differs')
    documents, rows = [source_identity(p) for p in (path,stream_path,config,stream_path.parent/'inputs.json')], []
    seen_hashes, seen_content, seen_sources = set(), set(), set()

    def load(index):
        task = HeliosCohortTask(index)
        entry = stream['cohorts'][index]
        trace = load_cohort(task,entry['trace_bundle'],expected_hash=entry['trace_hash'])
        native = trace._load(0)
        source_hashes = sorted(row['sha256'] for row in stream['identity']['inventory'][index])
        control = dict(schema_version=SCHEMA,scope=SCOPE,task=task.to_dict(),bundle_hash=trace.bundle_hash,
            manifests=[trace._control[r].manifest for r in range(16)],sources=stream['identity']['sources'][index],
            source_hashes=source_hashes,input_config=value['input_config'],source=stream['identity']['source_fingerprint'],
            content=content_identity(native),layout=layout_identity())
        raw = json.dumps(control,sort_keys=True,allow_nan=False).encode()
        require(len(raw) <= CONTROL_BYTES, 'W2 control manifest exceeds finite capacity')
        encoded = torch.zeros(CONTROL_BYTES,dtype=torch.uint8)
        encoded[:len(raw)] = torch.tensor(list(raw),dtype=torch.uint8)
        feeds = dict(control=encoded,control_length=torch.tensor([len(raw)],dtype=torch.int64),
                     tensors=[native.tensors[name] for name in sorted(native.tensors)])
        decode(feeds,task)
        return feeds,control,trace

    for index in range(count):
        _,control,trace = load(index)
        row = descriptor(control)
        content = conditioning_fingerprint(row['content'])
        require(row['bundle_hash'] not in seen_hashes and content not in seen_content
                and not set(row['sources']) & seen_sources, 'reused W2 trace/source/conditioning bytes')
        seen_hashes.add(row['bundle_hash']); seen_content.add(content); seen_sources.update(row['sources'])
        documents.append(source_identity(trace.root/'bundle.json'))
        for rank in range(16):
            root = trace._control[rank].root
            documents.extend(source_identity(root/name) for name in ('trace.json','tensors.pt'))
        for native in stream['identity']['inventory'][index]:
            identity = source_identity(native['path'])
            require(identity['sha256'] == native['sha256'], 'W2 source feature bytes changed')
            documents.append(identity)
        rows.append(dict(id=f'helios-nogan-reference-{index:06d}',metadata=dict(cycle=index,trace=row)))

    def produce(row):
        feeds,control,_ = load(row['metadata']['cycle'])
        require(descriptor(control) == row['metadata']['trace'], 'W2 stream changed during preparation')
        return tensor_tree(feeds)

    return _publish(output,rows=rows,provenance=dict(mode='preencoded',documents=documents,
        sources=[dict(id=row['id'],media={'features':source_identity(path)}) for row in rows],encoder=None),
        metadata=dict(schema_version=SCHEMA,scope=SCOPE,precision=PRECISION,task=HeliosCohortTask(0).to_dict(),
            source=source_fingerprint(),stream=value['stream'],input_config=value['input_config'],
            evidence_kind=value['evidence_kind'],layout=layout_identity()),produce=produce)
