"""Shared calibrate -> capability search -> bound native execution workflow.

Model code supplies snapshot/restore/signature, not a winning execution strategy.
Calibration executes real bound functions and preserves rejected comparisons.
Declared workspace bounds are NOT mislabeled as measured allocator peaks.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import time
from typing import Callable, Mapping

from distillschedule.core.contracts import require
from distillschedule.search.cost_model import CostModel, OperationCost
from distillschedule.search.plan_ir import validate_binding
from distillschedule.workloads.best_known.registry import fingerprint, file_sha256, git_head, repository_root, write_new_json
from .contracts import json_copy
from .capabilities import call_cost_contract, compile_cases
from .solver import solve_cases
from .runtime import run_serial, validate_serial_schedule


@dataclass
class StateDriver:
    """Native semantic interface; snapshot must include RNG, optimizer and caches.

    signature returns detached/copied evidence; equivalent compares against an
    independently ordered reference. The core cannot prove an arbitrary native
    function's contract, so model bindings and their tests remain necessary.
    """
    snapshot: Callable
    restore: Callable
    signature: Callable
    equivalent: Callable
    identity: Mapping


def binding_identity(matched,driver):
    root=repository_root()
    source={str(p.relative_to(root)):file_sha256(p)
            for p in sorted((root/'src/distillschedule/programs').glob('*.py'))}
    return {"source_head":git_head(root),
        "semantic_fingerprint":matched.program.digest,
        "asset_fingerprint":fingerprint({"program":matched.program.identity,"driver":driver.identity}),
        "precision_fingerprint":fingerprint({k:v.to_dict() for k,v in matched.program.calls.items()}),
        "hardware_fingerprint":fingerprint({"platform":platform.platform(),"machine":platform.machine(),
                                           "declared":driver.identity}),
        "environment_fingerprint":fingerprint({"code":source,"binding":matched.digest,
                                               "python":platform.python_version()})}


def _read(path):
    return json.loads(Path(path).read_text())


def _signed_write(path,value):
    value={**value,"sha256":fingerprint(value)}
    write_new_json(path,value)
    return value


def _signed_read(path):
    raw=_read(path);unsigned=dict(raw);sha=unsigned.pop("sha256",None)
    require(sha==fingerprint(unsigned),"artifact identity changed",path="program.workflow")
    return raw


def calibrate(matched,driver:StateDriver,output,*,repeats=2,max_combinations=64):
    require(type(repeats) is int and 1<=repeats<=10,"invalid calibration repeat budget",path="program.calibrate")
    selections=list(matched.combinations(max_combinations+1))
    require(selections and len(selections)<=max_combinations,"capability budget exceeded or no legal combinations",path="program.calibrate")
    reference={}
    for key,options in matched.options.items():
        defaults=[o for o in options if o.is_reference]
        require(len(defaults)==1,"each call needs exactly one native reference",path="program.calibrate")
        reference[key]=defaults[0]
    require(any(all(s[k].key==v.key for k,v in reference.items()) for s in selections),
            "reference capabilities are mutually incompatible",path="program.calibrate")
    initial=driver.snapshot();binding=binding_identity(matched,driver);root=Path(output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    write_new_json(root/'started.json',{"matched":matched.digest,"binding":binding,
        "candidate_count":len(selections),"repeats":repeats,"gpu_performance":"NOT_CLAIMED"})
    samples={};reports=[]
    try:
        driver.restore(initial)
        run_serial(matched.program,reference)
        expected=driver.signature()
        for selection in selections:
            key=fingerprint({k:v.key for k,v in selection.items()});observed={};ok=True
            for _ in range(repeats):
                driver.restore(initial)
                result=run_serial(matched.program,selection)
                comparison=driver.equivalent(expected,driver.signature())
                require(type(comparison) is bool,"equivalent must return an explicit bool",path="program.calibrate")
                if not comparison:ok=False;break
                for row in result['timings']:
                    observed.setdefault((row['operation'],row['implementation'],key),[]).append(row['duration_ns'])
            reports.append({"selection":{k:v.key for k,v in selection.items()},
                "key":key,"status":"PASS_REFERENCE_COMPARISON" if ok else "REVIEW"})
            if ok:
                for pair,ns in observed.items():samples.setdefault(pair,[]).extend(ns)
        allowed=[row['selection'] for row in reports if row['status']=='PASS_REFERENCE_COMPARISON']
        models=[]
        for (operation,key,context),ns in sorted(samples.items()):
            option=next(v for v in matched.options[operation] if v.key==key)
            selection=next(s for s in selections if fingerprint({k:v.key for k,v in s.items()})==context)
            contract=call_cost_contract(matched,operation,option,selection)
            raw={"schema_version":"distillschedule.operation_calibration.v1","binding":binding,
                 "key":contract.digest,"samples_ns":ns,"workspace_bytes_per_rank":option.workspace_upper_bound,
                 "status":"PASS","profiled":False,"kind":"native_operation",
                 "operation_contract_sha256":contract.digest,"component_contract":contract.to_dict(),
                 "memory_evidence":"DECLARED_BACKEND_UPPER_BOUND_NOT_ALLOCATOR_MEASUREMENT",
                 "device_evidence":json_copy(driver.identity),"claimable_speedup":False}
            receipt=root/(contract.digest+'.json');write_new_json(receipt,raw)
            evidence={k:raw[k] for k in ('status','profiled','kind','operation_contract_sha256','component_contract')}
            evidence.update(source_receipt=str(receipt),source_receipt_sha256=file_sha256(receipt))
            cost=CostModel(binding,(OperationCost(contract.digest,tuple(ns),option.workspace_upper_bound,evidence),))
            path=root/(contract.digest+'.cost.json');write_new_json(path,cost.to_dict())
            models.append({"operation":operation,"implementation":key,"context":context,"path":str(path),"sha256":file_sha256(path)})
        require(matched.digest==_read(root/'started.json')['matched'],"bindings changed during calibration",path="program.calibrate")
        return _signed_write(root/'calibration.json',{"schema_version":"distillschedule.program_calibration.v1",
            "program_sha256":matched.program.digest,"matched_sha256":matched.digest,"binding":binding,
            "models":models,"allowed_selections":allowed,"comparisons":reports,
            "status":"CALIBRATED_BOUND_FUNCTIONS","memory_evidence":"DECLARED_BOUNDS",
            "claimable_speedup":False})
    except BaseException as exc:
        write_new_json(root/'failure.json',{"status":"FAILED","error":str(exc)})
        raise
    finally:
        driver.restore(initial)


def load_calibration(matched,path):
    raw=_signed_read(path)
    require(raw.get('schema_version')=='distillschedule.program_calibration.v1'
            and raw['matched_sha256']==matched.digest and raw['program_sha256']==matched.program.digest,
            "calibration does not match program/capability parameters",path="program.plan")
    models={}
    for row in raw['models']:
        p=Path(row['path']);require(file_sha256(p)==row['sha256'],"cost file changed",path="program.plan")
        model=CostModel.from_dict(_read(p));model.verify_receipts()
        require(dict(model.binding)==raw['binding'],"cost identity differs",path="program.plan")
        pair=(row['operation'],row['implementation'],row['context'])
        require(pair not in models,"duplicate calibration pair",path="program.plan")
        models[pair]=model
    return raw,models


def plan(matched,calibration,output,*,memory_limits,max_combinations=64,**search):
    raw,models=load_calibration(matched,calibration)
    allowed={fingerprint(s) for s in raw['allowed_selections']}
    candidates=[]
    from .capabilities import MatchedProgram
    selections=list(matched.combinations(max_combinations+1))
    require(len(selections)<=max_combinations,"binding budget too small",path="program.plan")
    for selection in selections:
        ids={k:v.key for k,v in selection.items()}
        if fingerprint(ids) not in allowed:continue
        one=MatchedProgram(matched.program,{k:(v,) for k,v in selection.items()})
        candidates.extend(compile_cases(one,models,memory_limits,max_combinations=1))
    result=solve_cases(candidates,max_cases=max_combinations,**search)
    require(result['ranked'],"no feasible schedule among validated bindings",path="program.plan")
    chosen=result['ranked'][0]
    from .events import validate_event_schedule
    case=next(c for c in candidates if c.key==chosen['case_key'])
    serial=all('executor:serial' in b.resources for b in case.bindings.values())
    if serial:validate_serial_schedule(case,chosen['schedule'])
    else:validate_event_schedule(case,chosen['schedule'])
    return _signed_write(output,{"schema_version":"distillschedule.program_plan.v1",
        "program_sha256":matched.program.digest,"matched_sha256":matched.digest,
        "calibration":str(Path(calibration).resolve()),"calibration_sha256":file_sha256(Path(calibration)),
        "memory_limits":dict(memory_limits),"selected":result['ranked'][0],
        "search":result,"executor":"prepared_serial" if serial else "completion_events","claimable_speedup":False})


def execute(matched,driver,plan_path,*,check_reference=True):
    raw=_signed_read(plan_path)
    require(raw.get('schema_version')=='distillschedule.program_plan.v1'
            and raw['program_sha256']==matched.program.digest and raw['matched_sha256']==matched.digest,
            "plan does not match live program/bindings",path="program.run")
    require(file_sha256(Path(raw['calibration']))==raw['calibration_sha256'],"calibration changed",path="program.run")
    calibration,models=load_calibration(matched,raw['calibration'])
    require(calibration['binding']==binding_identity(matched,driver),"native environment/source/asset identity changed",path="program.run")
    ids=raw['selected']['selection']
    require(ids in calibration['allowed_selections'],"selection failed native reference comparison",path="program.run")
    selected={k:next(v for v in matched.options[k] if v.key==key) for k,key in ids.items()}
    from .capabilities import MatchedProgram
    case=next(compile_cases(MatchedProgram(matched.program,{k:(v,) for k,v in selected.items()}),
        models,{int(k):v for k,v in raw['memory_limits'].items()},max_combinations=1))
    require(case.digest==raw['selected']['case_sha256'],"resolved case identity differs",path="program.run")
    require(raw['executor'] in {'prepared_serial','completion_events'},"unregistered executor",path="program.run")
    if raw['executor']=='prepared_serial':validate_serial_schedule(case,raw['selected']['schedule'])
    else:
        from .events import validate_event_schedule
        validate_event_schedule(case,raw['selected']['schedule'])
    initial=driver.snapshot()
    try:
        expected=None
        if check_reference:
            reference={k:next(v for v in options if v.is_reference) for k,options in matched.options.items()}
            run_serial(matched.program,reference);expected=driver.signature();driver.restore(initial)
        if raw['executor']=='prepared_serial':
            result=run_serial(matched.program,selected,order=raw['selected']['schedule']['order'])
        else:
            from .events import run_events
            result=run_events(matched.program,selected,case,raw['selected']['schedule'])
        passed=driver.equivalent(expected,driver.signature()) if check_reference else None
        require(not check_reference or passed is True,"selected execution is REVIEW, not PASS",path="program.run")
        return {"status":"EXECUTED_SELECTED_BINDINGS","receipt":result['receipt'],"timings":result['timings'],
                "selection":ids,"reference_comparison":"PASS" if check_reference else "NOT_RUN",
                "claimable_speedup":False,"outputs":result['outputs']}
    except BaseException:
        driver.restore(initial)
        raise
