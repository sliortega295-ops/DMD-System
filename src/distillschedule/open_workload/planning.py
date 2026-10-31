"""Measured common-ABI calibration -> shared search -> selected native execution.

Reference and candidate math come from the same user bindings. No workload name
or fixed update count appears here. Calibration is a paired diagnostic from the
same initial state, not a production training run or a speedup benchmark.
"""
from __future__ import annotations
import json
from contextlib import nullcontext
import os
from pathlib import Path
import platform
import subprocess
import tempfile

from distillschedule.core.contracts import require
from distillschedule.search.cost_model import ComponentContract,CostModel,OperationCost
from distillschedule.search.plan_ir import BindingRule,Lifetime,StateAllocation,graph_from_dict
from distillschedule.search.service import BoundCandidate,select
from distillschedule.workloads.best_known.registry import fingerprint,file_sha256,write_new_json

SCHEMA='distillschedule.open_calibration.v1'
PLAN='distillschedule.open_plan.v1'


def source_revision():
    """Git is optional for installed packages; source hashes remain mandatory."""
    try:
        result=subprocess.run(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parent,
                              text=True,capture_output=True)
    except OSError:
        return '0'*40
    return result.stdout.strip() if result.returncode==0 else '0'*40


def source_identity():
    # The common Program backend is usable from an installed wheel; only the
    # legacy/native model launchers require a source checkout. Keep names stable
    # between editable installs and wheels rather than hashing absolute paths.
    root=Path(__file__).resolve().parents[1]
    paths=[root/'workloads/program.py']
    for d in ('open_workload','search','core','runtime'):
        paths.extend((root/d).glob('*.py'))
    files={str(p.relative_to(root)):file_sha256(p) for p in sorted(set(paths))}
    # Original mathematical helpers are still shipped at their audited runtime/
    # import path. Bind the package Python actually resolves, including wheels
    # and explicit import-path overrides, not an assumed checkout sibling.
    import importlib.util
    spec=importlib.util.find_spec('runtime')
    require(spec is not None and bool(spec.submodule_search_locations),
            'shipped compatibility runtime package is missing', path='open.source')
    for index,location in enumerate(spec.submodule_search_locations):
        compatibility=Path(location).resolve()
        files.update({f'compatibility-runtime-{index}/'+str(p.relative_to(compatibility)):file_sha256(p)
                      for p in sorted(compatibility.rglob('*.py'))})
    return fingerprint(files)


def hardware_identity(device):
    import torch
    value={'platform':platform.platform(),'machine':platform.machine(),'torch':torch.__version__,
           'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
           'device':str(device),'deterministic':torch.are_deterministic_algorithms_enabled(),
           'matmul_precision':torch.get_float32_matmul_precision(),
           'cuda_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32}
    if str(device).startswith('cuda'):
        p=torch.cuda.get_device_properties(device)
        value.update(name=p.name,total_memory=p.total_memory,capability=[p.major,p.minor])
    from .hardware_probe import hardware_facts
    value['accelerator'] = hardware_facts(device)
    return value


def _read(path):
    value=json.loads(Path(path).read_text())
    require(isinstance(value,dict),'expected JSON object',path='open.plan')
    return value


def _resident_budget_identity(trial):
    if trial is None: return None
    return {'schema_version':trial.receipt['schema_version'], 'limits':trial.receipt['budgets']}


def _publish_resident_success(path, value):
    """Install a complete private receipt without overwriting any prior file.

    The same-directory staging file is ours from exclusive creation onward.
    Serialization, flush/fsync and close must all finish before the no-replace
    hard-link commit point. Cleanup after that point cannot reject adoption;
    an unlink failure can leave only a non-authoritative staging file.
    """
    descriptor, temporary=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.pending',dir=path.parent)
    try:
        stream=os.fdopen(descriptor,'w',encoding='utf-8');descriptor=None
        with stream:
            json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)  # Atomic same-filesystem install, never replace.
    finally:
        if descriptor is not None:
            try: os.close(descriptor)
            except BaseException: pass
        try: os.unlink(temporary)
        except BaseException: pass  # Also never mask an earlier writer/install failure.


def _planning_failure(backend, root, progress, error, trial=None):
    # Never touch a preexisting output directory or mask the original failure,
    # in particular the transaction's combined execution/rollback error.
    if not progress.get('output_created'): return
    evidence={k:v for k,v in progress.items() if k!='output_created'}
    if trial is not None:
        # Context exit has already closed the transaction. Keep failed and
        # rollback allocator observations, not just the last exception string.
        evidence['resident_trial']=trial.receipt
        for key,value in trial.receipt.get('candidate_admission',{}).items():
            evidence.setdefault(key,value)
        for key in ('original_error','rollback_error'):
            if hasattr(error,key):
                cause=getattr(error,key)
                evidence[key]=None if cause is None else {
                    'type':type(cause).__module__+'.'+type(cause).__qualname__, 'message':str(cause)}
    try:
        backend.primary(lambda: write_new_json(root/'failure.json',
            dict(evidence,error=str(error),claimable_speedup=False)), 'record planning failure')
    except BaseException as publication_error:
        error.planning_failure_publication_error=repr(publication_error)
        note=getattr(error,'add_note',None)  # Python 3.10 has no exception notes.
        if callable(note): note('failure receipt publication also failed: '+repr(publication_error))


def calibrate(workload,output,*,max_candidates=8,samples=2,max_program_runs=32,
              rtol=1e-5,atol=1e-7,profile_hardware=None,resident_budget=None):
    """Shared calibration; typed budgets opt into private single-owner trials.

    No budget preserves the original CPU/replicated clone provider. Resident
    calibration publishes success only after exact rollback has completed.
    Hardware probes are outside this transaction's capacity envelope.
    """
    from .execution import ExecutionBackend
    backend=ExecutionBackend(workload);root=Path(output).resolve();progress={}
    trial=backend.resident_trial(resident_budget) if resident_budget is not None else None
    try:
        with (trial if trial is not None else nullcontext()):
            manifest=_calibrate(backend,root,progress,max_candidates=max_candidates,samples=samples,
                max_program_runs=max_program_runs,rtol=rtol,atol=atol,
                profile_hardware=profile_hardware,trial=trial)
        if trial is not None: manifest['resident_trial']=trial.receipt
        if trial is None: backend.publish(root/'calibration.json',manifest)
        else: _publish_resident_success(root/'calibration.json',manifest)
        return manifest
    except BaseException as error:
        _planning_failure(backend,root,progress,error,trial)
        raise


def _calibrate(backend,root,progress,*,max_candidates,samples,max_program_runs,
               rtol,atol,profile_hardware,trial):
    workload=backend.workload
    from .checkpoint_policy import native_effect_evidence
    require(type(samples) is int and 1<=samples<=16,'invalid calibration sample count',path='open.calibrate')
    require(type(max_candidates) is int and 1<=max_candidates<=128,'invalid candidate budget',path='open.calibrate')
    inventory=backend.inventory(max_candidates)
    needed=1+samples*len(inventory['candidates'])
    require(type(max_program_runs) is int and needed<=max_program_runs,'budget must include reference and all calibration samples',path='open.calibrate')
    identity=backend.identity();hardware=backend.hardware();source=source_identity()
    # CUDA has an opt-out rather than requiring users to hand-profile A100/H100.
    # Scratch-only H2D and NCCL probes precede paired training measurements.
    # They are diagnostics; their isolated times are never substituted for
    # actual native model/optimizer call observations in the cost model.
    require(profile_hardware is None or type(profile_hardware) is bool,
            'profile_hardware must be a boolean', path='open.calibrate')
    require(trial is None or profile_hardware is not True,
            'resident hardware probes require a separate budget; use profile_hardware=False', path='open.calibrate')
    enabled = (workload.device.type == 'cuda' if profile_hardware is None else profile_hardware) if trial is None else False
    if enabled:
        from .hardware_probe import probe
        observations = backend.consensus(probe(workload.device, distributed=backend.distributed))
        hardware_probe = {'status': 'MEASURED_ISOLATED_GPU_PROBES' if workload.device.type=='cuda'
                          else 'NOT_GPU_PROFILE', 'ranks':observations}
    else:
        hardware_probe = {'status':'NOT_REQUESTED' if trial is None else 'NOT_RUN_RESIDENT_BUDGET', 'ranks':[]}
    backend.mkdir(root); progress['output_created']=True
    # A real git source when available; an archive has an explicit non-Git label.
    head=source_revision()
    binding={'source_head':head,'hardware_fingerprint':fingerprint(hardware),
        'environment_fingerprint':fingerprint({'source':source,'torch':identity['torch']}),
        'asset_fingerprint':fingerprint({'state':identity['initial_state'],'feeds':identity['feeds']}),
        'semantic_fingerprint':workload.program.digest,'precision_fingerprint':fingerprint(hardware)}
    progress['program_runs']=1
    if trial is None:
        reference=backend.session(inventory['candidates'][0]);outputs=reference.run()
        expected=reference.result_state(outputs); del outputs,reference
    else:
        trial.reference(inventory['candidates'][0])
    variants=[];failed=[];actual=1
    for index,implementation in enumerate(inventory['candidates']):
        candidate_graph=backend.validated(lambda:backend.graph(implementation),'candidate graph')
        profiles=[];rank_observations=[];static=0;effects=None;gate=None
        rejected_effect=None
        for _ in range(samples):
            actual+=1; progress['program_runs']=actual
            if trial is None:
                session=backend.session(implementation);out=session.run(measure=True)
                gate=backend.compare(expected,session.result_state(out),rtol=rtol,atol=atol)
            else:
                session,out,gate,native=trial.candidate(implementation,measure=True,
                    rtol=rtol,atol=atol,report_inactive=True)
                if not native[0]['active']:
                    rejected_effect=native[0];break
            if gate['status']!='PASS': break
            backend.observations(session);rank_observations.append(backend.last_observations)
            effects=session.effects
            native_gate=native_effect_evidence(implementation,effects)
            if not native_gate['active']:
                rejected_effect=native_gate
                break
            profiles.append(session.profile);static=max(static,session.static_bytes)
            del out,session
            if trial is not None: trial.restore()
        if rejected_effect is not None:
            from .owner_execution import selected as owner_selected
            reason = ('INAPPLICABLE_NATIVE_OWNER_CAPABILITY' if owner_selected(implementation)
                      else 'INAPPLICABLE_NATIVE_SELECTIVE_CAPABILITY')
            failed.append({'implementation':implementation, 'reason':reason,
                           'native_evidence':rejected_effect});continue
        if gate['status']!='PASS':
            failed.append({'implementation':implementation,'reference_comparison':gate});continue
        directory=root/f'candidate-{index}';backend.mkdir(directory)
        records=[];retained={}
        for i,op in enumerate(workload.program.compile().operations):
            contract=ComponentContract('open.call',{'program':workload.program.digest,'operation':op.operation_id},
                {'fingerprint':binding['precision_fingerprint']},{'backend':backend.key,'device':backend.device,'replicas':backend.world},
                implementation,{'paired_initial_state':identity['initial_state'],'logical_rng':True})
            rows=[p[op.operation_id] for p in profiles]
            raw={'schema_version':'distillschedule.operation_calibration.v1','binding':binding,
                'key':contract.digest,'samples_ns':[p['duration_ns'] for p in rows],
                'workspace_bytes_per_rank':max(p['workspace_bytes'] for p in rows),
                'status':'PASS','profiled':False,'kind':'native_operation',
                'operation_contract_sha256':contract.digest,'component_contract':contract.to_dict(),
                'device':backend.device,'replicas':backend.world,'instrumentation':'call boundary timing and saved-storage accounting',
                'capacity_scope':('CUDA allocator resident phase high-water bound plus tracked tensors; not isolated operation peak'
                                      if trial is not None and workload.device.type=='cuda' else
                                      'CUDA allocator plus tracked tensors' if workload.device.type=='cuda' else 'tracked tensor storage only; not CPU RSS'),
                'claimable_speedup':False,'timing_source':'synchronized_host_wall_ns',
                'gpu_event_diagnostic_ns':[[sample[r]['profile'][op.operation_id].get('cuda_event_ns')
                                           for r in range(backend.world)] for sample in rank_observations],
                'rank_observations':[
                    [sample[r]['profile'][op.operation_id] for r in range(backend.world)]
                    for sample in rank_observations]}
            p=directory/f'op-{i}.json';backend.publish(p,raw)
            evidence={k:raw[k] for k in ('status','profiled','kind','operation_contract_sha256','component_contract')}
            evidence.update(source_receipt=str(p),source_receipt_sha256=file_sha256(p))
            records.append(OperationCost(contract.digest,tuple(raw['samples_ns']),raw['workspace_bytes_per_rank'],evidence))
            retained[op.operation_id]=max(p['retained_bytes'] for p in rows)
        costs=CostModel(binding,tuple(records));costs.verify_receipts()
        cost_path=directory/'costs.json';backend.publish(cost_path,costs.to_dict())
        variants.append({'id':fingerprint(implementation),'implementation':implementation,
                         'cost_model':str(cost_path),'cost_sha256':file_sha256(cost_path),
                         'retained_bytes':retained,'static_bytes':static,'effects':effects,'reference_gate':gate,
                         'native_capability_evidence':native_effect_evidence(implementation,effects),
                         'graph':candidate_graph.to_dict()})
    require(variants,'no numerically admitted implementation',path='open.calibrate')
    require(backend.identity()==identity and source_identity()==source,'workload/source changed during calibration',path='open.calibrate')
    manifest={'schema_version':SCHEMA,'identity':identity,'source':source,'hardware':hardware,
              'hardware_probe':hardware_probe,
              'graph':workload.program.compile().to_dict(),'variants':variants,'failed':failed,
              'inventory':inventory,'budget':{'maximum':max_program_runs,'actual':actual},
              'source_commit_available':head!='0'*40,'reference_order':[c.id for c in workload.program.calls],
              'device':backend.device,'replicas':backend.world,'claimable_speedup':False,
              'gpu_execution':'NOT_RUN' if workload.device.type=='cpu' else 'CALIBRATION_ONLY'}
    if trial is not None:
        manifest['resident_budget']=_resident_budget_identity(trial)
        manifest['measurement_scope']={
            'cost_time':'synchronized native call wall time including saved-tensor instrumentation',
            'transaction_overhead':'snapshot, identity, comparison and rollback not timed or added to operation cost',
            'capacity':('resident phase high-water bound, not isolated kernel/operation peak'
                        if workload.device.type=='cuda' else 'tracked tensor storage only; not CPU RSS'),
            'hardware_probes':'not run; no separate scratch budget'}
    return manifest


def load_calibration(path):
    manifest=_read(path)
    require(manifest.get('schema_version')==SCHEMA and manifest['source']==source_identity(),'stale calibration source',path='open.plan')
    world=manifest.get('replicas',1)
    require(type(world) is int and 1<=world<=1024,'invalid replica count',path='open.plan')
    ranks=tuple(range(world)); resources=tuple(f'gpu:{r}' for r in ranks)
    result=[]
    for row in manifest['variants']:
        graph=graph_from_dict(row.get('graph',manifest['graph']))
        require(file_sha256(Path(row['cost_model']))==row['cost_sha256'],'changed component costs',path='open.plan')
        costs=CostModel.from_dict(_read(row['cost_model']));costs.verify_receipts()
        require(row['id']==fingerprint(row['implementation']) and row['reference_gate']['status']=='PASS',
                'unverified implementation',path='open.plan')
        from .checkpoint_policy import native_effect_evidence
        require(row.get('native_capability_evidence')==native_effect_evidence(row['implementation'],row['effects'])
                and row['native_capability_evidence']['active'],
                'unobserved or changed native capability evidence',path='open.plan')
        rules={}
        for op in graph.operations:
            matches=[r for r in costs.records if r.evidence['component_contract']['work']['operation']==op.operation_id]
            require(len(matches)==1,'missing or duplicate operation cost',path='open.plan')
            c=matches[0]
            require(c.evidence['component_contract']['implementation']==row['implementation'],'wrong implementation cost',path='open.plan')
            rules[op.operation_id]=BindingRule(op.operation_id,{},c.key,ranks,resources,('state',),c.key)
        lives=[]
        for op in graph.operations:
            consumers=tuple(o.operation_id for o in graph.operations if op.operation_id in o.metadata.get('inputs',()))
            if consumers and row['retained_bytes'][op.operation_id]:
                lives.append(Lifetime(op.operation_id,op.operation_id,consumers,ranks,row['retained_bytes'][op.operation_id],'producer_entry'))
        result.append((row,graph,rules,costs,tuple(lives)))
    return manifest,result


def plan(calibration,*,memory_limit_bytes,max_states=50000,beam_width=0,timeout_seconds=60):
    require(type(memory_limit_bytes) is int and memory_limit_bytes>0,'explicit memory budget required',path='open.plan')
    manifest,variants=load_calibration(calibration)
    bound=[]
    ranks=tuple(range(manifest.get('replicas',1)))
    for row,g,rules,costs,lives in variants:
        states=(StateAllocation('state',ranks,row['static_bytes'],('open_runtime',),False,'paired-native-state'),)
        bound.append(BoundCandidate(row['id'],row['implementation'],g,rules,costs,states,lives,{r:memory_limit_bytes for r in ranks},row['implementation']['backend']))
    result=select(bound,max_states=max_states,beam_width=beam_width,timeout_seconds=timeout_seconds)
    require(result['ranked'],'no plan within admitted candidates, capacity or search budget',path='open.plan')
    document={'schema_version':PLAN,'calibration':str(Path(calibration).resolve()),'calibration_sha256':file_sha256(Path(calibration)),
              'identity':manifest['identity'],'source':manifest['source'],'selected':result['ranked'][0],
              'search':result,'memory_limit_bytes':memory_limit_bytes,'device':manifest['device'],
              'enumeration_complete':manifest['inventory']['exhaustive'],'replicas':manifest.get('replicas',1),
              'gpu_execution':'NOT_RUN_BY_SEARCH','claimable_speedup':False}
    chosen=next((v for v in variants if v[0]['id']==document['selected']['candidate_id']),None)
    require(chosen is not None, 'selected candidate cost absent', path='open.plan')
    from .opportunity import report as opportunity_report
    document['opportunity_report']=opportunity_report(document['selected'],*chosen[:4],manifest)
    if 'resident_budget' in manifest: document['resident_budget']=manifest['resident_budget']
    document['plan_sha256']=fingerprint(document)
    return document


def verify_order(selected,graph,rules,costs,lives,static,limit):
    from distillschedule.runtime.role_runner import StaticRoleRunner
    from distillschedule.runtime.version_sync import VersionLedger
    order=tuple(selected['order'])
    runner=StaticRoleRunner('open_runtime',{'call':lambda _:None,'barrier':lambda _:None},versions=VersionLedger(graph.metadata['initial_versions']))
    runner.prepare(graph,order=order)
    ranks=tuple(next(iter(rules.values())).ranks)
    resources=sorted(next(iter(rules.values())).resources)
    require(all(tuple(r.ranks)==ranks and sorted(r.resources)==resources for r in rules.values()),
            'this executor requires full-replica serial calls',path='open.run')
    elapsed=0;peak=static;active={};timeline=[]
    position={key:i for i,key in enumerate(order)}
    last={v.id:max(position[k] for k in v.consumers) for v in lives}
    for i,key in enumerate(order):
        cost=costs.get(rules[key].cost_key)
        for v in lives:
            if v.producer==key: active[v.id]=v.bytes_per_rank
        peak=max(peak,static+sum(active.values())+cost.workspace_bytes_per_rank)
        timeline.append({'operation':key,'start_ns':elapsed,'end_ns':elapsed+cost.duration_ns,'resources':resources})
        elapsed+=cost.duration_ns
        peak=max(peak,static+sum(active.values()))
        active={k:v for k,v in active.items() if last[k]>i}
    require(not active and peak<=limit,'selected order exceeds capacity',path='open.run')
    require(selected['timeline']==timeline and selected['predicted_makespan_ns']==elapsed
            and selected['graph_sha256']==fingerprint(graph.to_dict()) and selected['cost_model_sha256']==costs.digest,
            'selected schedule differs from measured costs/graph',path='open.run')
    require(len(selected['memory'])==len(ranks) and {r['rank'] for r in selected['memory']}==set(ranks)
            and all(r['static_bytes']==static and r['peak_bytes']==peak and r['limit_bytes']==limit
                    for r in selected['memory']),'selected memory changed',path='open.run')
    return order


def paired_execution(backend, implementation, order=None, *, persistent=False,
                     rtol=1e-5, atol=1e-7, execute=None, resident_trial=None):
    """Preupdate admission using clones or an explicit resident transaction.

    Shared by saved plans and explicit stateful physical options, including the
    first continued update after restore. Only the selected session is adopted.
    """
    from .owner_execution import admit, inventory, selected
    if selected(implementation):
        # Reject malformed owner selections before even a cloned reference
        # update. Backend/session admission still owns the complete layout.
        backend.validated(lambda: admit(implementation.get('local', implementation),
                                         inventory(backend.workload.state)), 'owner execution predicates')
    if resident_trial is not None:
        require(resident_trial.backend is backend, 'resident trial backend changed', path='open.run')
        resident_trial.reference()
        return resident_trial.candidate(implementation,order,persistent=persistent,
            rtol=rtol,atol=atol,execute=execute)
    reference = backend.session(backend.inventory(1)['candidates'][0])
    out = reference.run()
    expected = reference.result_state(out)
    del reference, out
    session = backend.session(implementation, persistent=persistent)
    out = execute(session) if execute is not None else session.run(order)
    from .checkpoint_policy import native_effect_evidence
    native = backend.consensus(native_effect_evidence(implementation, session.effects))
    require(all(row['active'] for row in native),
            'selected native capability was not exercised on every rank', path='open.run')
    gate = backend.compare(expected, session.result_state(out), rtol=rtol, atol=atol)
    return session, out, gate, native


def run(workload,plan_path,output,*,rtol=1e-5,atol=1e-7,deep_profile=False,
        persistent=False,resident_budget=None):
    """Run the saved shared plan; resident success adopts exactly one cycle.

    A restored trial cannot return an apparently trained Session. This private
    seam therefore requires persistent ownership, not public CUDA lifecycle
    admission. Profiling allocations require a separate, future budget.
    """
    from .execution import ExecutionBackend
    require(resident_budget is None or persistent is True,
            'resident saved-plan execution requires persistent=True',path='open.run')
    require(resident_budget is None or deep_profile is False,
            'resident deep profiling requires a separate capacity budget',path='open.run')
    backend=ExecutionBackend(workload);root=Path(output).resolve();progress={}
    trial=backend.resident_trial(resident_budget) if resident_budget is not None else None
    try:
        with (trial if trial is not None else nullcontext()):
            result=_run(backend,plan_path,root,progress,rtol=rtol,atol=atol,
                        deep_profile=deep_profile,persistent=persistent,trial=trial)
        return result
    except BaseException as error:
        _planning_failure(backend,root,progress,error,trial)
        raise


def _run(backend,plan_path,root,progress,*,rtol,atol,deep_profile,persistent,trial):
    workload=backend.workload
    def read_plan():
        document=_read(plan_path);unsigned=dict(document);sha=unsigned.pop('plan_sha256',None)
        require(document.get('schema_version')==PLAN and sha==fingerprint(unsigned),'changed plan',path='open.run')
        require(file_sha256(Path(document['calibration']))==document['calibration_sha256'],'changed calibration manifest',path='open.run')
        manifest,variants=load_calibration(document['calibration'])
        return document,sha,manifest,variants
    document,sha,manifest,variants=backend.validated(read_plan,'plan admission')
    require(len(set(backend.consensus(sha)))==1,'rank plan selection differs',path='open.run')
    require(document.get('resident_budget')==manifest.get('resident_budget')==_resident_budget_identity(trial),
            'resident planning budget changed or absent',path='open.run')
    require(manifest['identity']==document['identity']==backend.identity(),'workload parameters, inputs, code or state changed',path='open.run')
    require(backend.hardware()==manifest['hardware'],'runtime environment changed',path='open.run')
    if workload.device.type=='cuda':
        from .hardware_probe import plan_memory_limit
        budget=document.get('hardware_memory_budget',{})
        reserve=budget.get('reserve_bytes',0)
        plan_memory_limit(workload.device,distributed=backend.distributed,
            consensus=backend.consensus,requested=document['memory_limit_bytes'],
            reserve_bytes=reserve)
    # Program/Call metadata may contain native tuples. Admission compares the
    # complete JSON wire contract, not Python tuple-vs-list spelling after save.
    require(json.loads(json.dumps(workload.program.compile().to_dict(),allow_nan=False))==manifest['graph'],
            'wrong callable program',path='open.run')
    selected=document['selected'];matches=[v for v in variants if v[0]['id']==selected['candidate_id']]
    require(len(matches)==1 and selected['implementation']==matches[0][0]['implementation'],'unmeasured implementation',path='open.run')
    row,g,rules,costs,lives=matches[0]
    require(json.loads(json.dumps(backend.graph(row['implementation']).to_dict(),allow_nan=False))==g.to_dict(),
            'execution constraints differ from native backend',path='open.run')
    order=verify_order(selected,g,rules,costs,lives,row['static_bytes'],document['memory_limit_bytes'])
    backend.mkdir(root);progress.update(output_created=True,plan_sha256=sha)
    if trial is not None:
        progress.update(selected_implementation=row['implementation'],executed_order=list(order))
    require(type(deep_profile) is bool, 'invalid deep-profile flag', path='open.run')
    traces=[]
    def execute(session):
        if not deep_profile:
            return session.run(order)
        from .deep_profile import selected_plan_trace
        session.deep_trace_labels=True
        out,trace=selected_plan_trace(lambda:session.run(order),root,
                                      rank=backend.rank,device=workload.device)
        traces.extend(backend.consensus(trace))
        return out
    session,out,gate,actual_native=paired_execution(backend,row['implementation'],order,
        persistent=persistent,rtol=rtol,atol=atol,execute=execute,resident_trial=trial)
    if trial is not None:
        progress.update(reference_gate=gate,native_capability_evidence=actual_native)
    require(trial is None or gate['status']=='PASS',
            'selected resident reference numerical gate failed',path='open.run')
    result={**session.receipt,'plan_sha256':sha,'reference_gate':gate,
            'status':'PASS_SELECTED_PROGRAM' if gate['status']=='PASS' else 'REVIEW_SELECTED_PROGRAM',
            'selected_implementation':row['implementation'],'executed_order':list(order),
            'native_capability_evidence':actual_native,
            'profiler_traces':traces,
            'timing_claim':'PROFILER_TRACE_NOT_BENCHMARK' if deep_profile else 'UNPROFILED_SELECTED_EXECUTION',
            'quality_validation':'NOT_RUN','scope':'common callable ABI; not proof of original DMD protocol parity'}
    if backend.distributed:
        result={**result,'rank_executions':backend.consensus(result),'replicas':backend.world}
        # The public result is shared; local wall clocks live in rank_executions.
        result['wall_ns']=max(r['wall_ns'] for r in result['rank_executions'])
        result['device']=backend.device
        result['effects']={'replica_effects':[r['effects'] for r in result['rank_executions']]}
    if trial is None:
        backend.publish(root/'execution.json',result)
    else:
        def publish(receipt):
            require(source_identity()==manifest['source'], 'source changed during resident execution',path='open.run')
            require(backend.hardware()==manifest['hardware'], 'runtime environment changed',path='open.run')
            result['resident_trial']=receipt
            _publish_resident_success(root/'execution.json',result)
        trial.accept(publish=publish)
    # Session contains actual trained modules/optimizers for the caller, not
    # an advertised resumable production checkpoint.
    return result,session,out
