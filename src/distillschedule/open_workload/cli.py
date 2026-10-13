"""Public factory entry, with no model or workload name in the dispatcher.

The explicitly supplied Python factory describes training math/state. It never
needs to import the planner or construct optimization candidates. Factories are
trusted user code, executed only with --execute (including model construction).
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import file_sha256,write_new_json


def load_factory(name,parameters):
    from .torch_backend import Workload
    path,sep,symbol=name.rpartition(':')
    require(sep and symbol.isidentifier(),'factory must be a local.py:callable',path='open.cli')
    source=Path(path).expanduser().resolve()
    require(source.is_file() and source.suffix=='.py','local Python factory required',path='open.cli')
    sha=file_sha256(source)
    module_name='distillschedule_user_'+sha
    spec=importlib.util.spec_from_file_location(module_name,source)
    require(spec is not None and spec.loader is not None,'factory cannot be loaded',path='open.cli')
    module=importlib.util.module_from_spec(spec);sys.modules[module_name]=module
    spec.loader.exec_module(module)
    factory=getattr(module,symbol,None)
    require(callable(factory),'factory entry not callable',path='open.cli')
    workload=factory(parameters)
    require(isinstance(workload,Workload),'factory must return Workload, not a schedule/recipe',path='open.cli')
    require(file_sha256(source)==sha,'factory source changed during construction',path='open.cli')
    workload.identity_extra={**workload.identity_extra,'factory_sha256':sha,'factory_entry':symbol,'factory_parameters':parameters}
    return workload.validate()


def main(argv=None):
    p=argparse.ArgumentParser(prog='distillctl workloads native program')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('describe','calibrate','plan','run','auto'):
        q=sub.add_parser(name)
        q.add_argument('--output',type=Path,required=name!='describe')
        if name!='plan':
            q.add_argument('--factory',required=True)
            q.add_argument('--parameters',type=Path,required=True)
            q.add_argument('--execute',action='store_true')
        if name in ('calibrate','auto'):
            q.add_argument('--max-candidates',type=int,default=8)
            q.add_argument('--samples',type=int,default=2)
            q.add_argument('--max-program-runs',type=int,default=32)
        if name in ('plan','auto'):
            q.add_argument('--memory-limit-bytes',type=int,required=True)
            q.add_argument('--max-states',type=int,default=50000)
            q.add_argument('--beam-width',type=int,default=0)
        if name=='plan': q.add_argument('--calibration',type=Path,required=True)
        if name=='run': q.add_argument('--plan',type=Path,required=True)
    args=p.parse_args(argv)
    try:
        if args.command!='plan' and not args.execute:
            print(json.dumps({'status':'NOT_EXECUTED','reason':'factory is user code; --execute required','model_constructed':False}))
            return 0
        from .planning import calibrate,plan,run
        if args.command=='plan':
            result=plan(args.calibration,memory_limit_bytes=args.memory_limit_bytes,max_states=args.max_states,beam_width=args.beam_width)
            write_new_json(args.output,result)
        else:
            parameters=json.loads(args.parameters.read_text())
            require(isinstance(parameters,dict),'parameter object required',path='open.cli')
            workload=load_factory(args.factory,parameters)
            if args.command=='describe':
                from .torch_backend import candidates
                result={'program':workload.program.to_dict(),'graph':workload.program.compile().to_dict(),
                        'capabilities':candidates(workload),'gpu_execution':'NOT_RUN'}
            elif args.command=='calibrate':
                result=calibrate(workload,args.output,max_candidates=args.max_candidates,samples=args.samples,max_program_runs=args.max_program_runs)
            elif args.command=='run':
                result,_,_=run(workload,args.plan,args.output)
            else:
                # Two additional executions are the independent reference and
                # selected-plan confirmation, not free out-of-budget trials.
                from .torch_backend import candidates
                required=3+args.samples*len(candidates(workload,limit=args.max_candidates)['candidates'])
                require(required<=args.max_program_runs,'auto budget includes calibration and two confirmation runs',path='open.cli')
                root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
                calibration=calibrate(workload,root/'calibration',max_candidates=args.max_candidates,
                            samples=args.samples,max_program_runs=args.max_program_runs-2)
                selected=plan(root/'calibration/calibration.json',memory_limit_bytes=args.memory_limit_bytes,
                              max_states=args.max_states,beam_width=args.beam_width)
                write_new_json(root/'plan.json',selected)
                result,_,_=run(workload,root/'plan.json',root/'execution')
                result['program_runs']=calibration['budget']['actual']+2
                write_new_json(root/'result.json',result)
        print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
        return 0 if result.get('status')!='REVIEW_SELECTED_PROGRAM' else 3
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);return 2


if __name__=='__main__': raise SystemExit(main())
