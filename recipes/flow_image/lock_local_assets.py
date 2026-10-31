"""Inventory already-present native Diffusers/Transformers folders; no downloads.

This records operator-supplied file identities, not remote/pretrained authenticity
or a CPU memory-capacity grant. No model or Torch import is needed for locking.
"""
import argparse
import json
from pathlib import Path

from distillschedule.training.catalog import resolve
from distillschedule.workloads.best_known.registry import file_sha256


def lock_bundle(bundle, family, *, max_parameters, text_max_length, dtype_policy='strict_fp32',
                student='transformer', fake='transformer', teacher='transformer'):
    root=Path(bundle).expanduser().resolve(strict=True)
    if type(max_parameters) is not int or max_parameters<=0 or type(text_max_length) is not int or text_max_length<=0:
        raise ValueError('explicit positive CPU parameter budget and text length required')
    if family not in ('flux2-klein-base','qwen-image') or dtype_policy not in ('strict_fp32','bf16_or_fp32_to_fp32'):
        raise ValueError('unsupported pinned family or storage dtype policy')
    def component(relative):
        folder=(root/relative).resolve(strict=True)
        if not folder.is_dir() or not folder.is_relative_to(root):
            raise ValueError('all native components must be inside the explicit local bundle')
        files={str(p.relative_to(folder)):file_sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        if not files: raise ValueError('empty local component')
        return dict(path=str(folder.relative_to(root)),files=files,max_parameters=max_parameters,dtype_policy=dtype_policy)
    source=resolve('model',family)['source']
    manifest=dict(schema_version='distillschedule.flow_image_assets.v1',family=family,
        source={key:source[key] for key in ('model_id','revision')},evidence_kind='LOCAL_PRETRAINED_UNVERIFIED',
        roles=dict(student=component(student),fake=component(fake),teacher=component(teacher)),
        media={name:component(name) for name in ('vae','text_encoder','tokenizer')},text_max_length=text_max_length)
    target=root/'assets.json'
    with target.open('x') as output: json.dump(manifest,output,indent=2);output.write('\n')
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle')
    parser.add_argument('--family',required=True,choices=['flux2-klein-base','qwen-image'])
    parser.add_argument('--cpu-max-parameters',required=True,type=int)
    parser.add_argument('--text-max-length',required=True,type=int)
    parser.add_argument('--dtype-policy',choices=['strict_fp32','bf16_or_fp32_to_fp32'],default='strict_fp32')
    for name in ('student','fake','teacher'):parser.add_argument('--'+name,default='transformer')
    args=parser.parse_args()
    print(lock_bundle(args.bundle,args.family,max_parameters=args.cpu_max_parameters,text_max_length=args.text_max_length,
        dtype_policy=args.dtype_policy,student=args.student,fake=args.fake,teacher=args.teacher))
