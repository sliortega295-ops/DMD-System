"""Lock ALREADY LOCAL native Wan2.2 component directories; never downloads.

Operator-declared source pin and file hashes are not pretrained authenticity or
a full-model capacity grant. Runtime validates all native configs/budgets first.
"""
import argparse
import json
from pathlib import Path

from distillschedule.training.catalog import resolve
from distillschedule.workloads.best_known.registry import file_sha256


def lock_bundle(bundle, *, max_parameters, dtype_policy='strict_fp32', student='transformer', fake='transformer', teacher='transformer'):
    root=Path(bundle).expanduser().resolve(strict=True)
    if type(max_parameters) is not int or max_parameters<=0 or dtype_policy not in ('strict_fp32','bf16_or_fp32_to_fp32'):
        raise ValueError('positive CPU component parameter budget and explicit supported dtype policy required')
    def component(relative):
        folder=(root/relative).resolve(strict=True)
        if not folder.is_dir() or not folder.is_relative_to(root):
            raise ValueError('every component must be inside the explicit local bundle')
        files={str(p.relative_to(folder)):file_sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        if not files:raise ValueError('empty local component')
        return dict(path=str(folder.relative_to(root)),files=files,max_parameters=max_parameters,dtype_policy=dtype_policy)
    source=resolve('model','wan22-ti2v5b-t2v')['source']
    value=dict(schema_version='distillschedule.wan22_t2v_assets.v1',family='wan22-ti2v5b-t2v',
        source={key:source[key] for key in ('model_id','revision')},evidence_kind='LOCAL_PRETRAINED_UNVERIFIED',
        pipeline=dict(_class_name='WanPipeline',expand_timesteps=True,transformer_2=None,boundary_ratio=None,task='t2v'),
        roles=dict(student=component(student),fake=component(fake),teacher=component(teacher)),
        media={name:component(name) for name in ('vae','text_encoder','tokenizer')},text_max_length=226)
    target=root/'assets.json'
    with target.open('x') as output:json.dump(value,output,indent=2);output.write('\n')
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('bundle')
    parser.add_argument('--cpu-max-parameters',required=True,type=int)
    parser.add_argument('--dtype-policy',choices=['strict_fp32','bf16_or_fp32_to_fp32'],default='strict_fp32')
    for role in ('student','fake','teacher'):parser.add_argument('--'+role,default='transformer')
    args=parser.parse_args()
    print(lock_bundle(args.bundle,max_parameters=args.cpu_max_parameters,dtype_policy=args.dtype_policy,
        student=args.student,fake=args.fake,teacher=args.teacher))
