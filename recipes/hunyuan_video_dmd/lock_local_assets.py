"""Inventory ALREADY LOCAL HunyuanVideo1.5 T2V assets; no downloads or weights load.

Model assets use Tencent Hunyuan terms, not the framework/dependency Apache grant.
This operator declaration is not pretrained authenticity or a capacity approval.
"""
import argparse
import json
from pathlib import Path

from distillschedule.training.catalog import resolve
from distillschedule.workloads.best_known.registry import file_sha256


def lock_bundle(bundle, *, max_parameters, dtype_policy='strict_fp32', student='transformer', fake='transformer', teacher='transformer'):
    root = Path(bundle).expanduser().resolve(strict=True)
    if type(max_parameters) is not int or max_parameters <= 0 or dtype_policy not in ('strict_fp32','bf16_or_fp32_to_fp32'):
        raise ValueError('positive CPU component parameter budget and explicit supported storage dtype policy required')
    def component(relative):
        folder = (root/relative).resolve(strict=True)
        if not folder.is_dir() or not folder.is_relative_to(root): raise ValueError('every component must be inside explicit local bundle')
        files = {str(p.relative_to(folder)):file_sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        if not files: raise ValueError('empty local native component')
        return dict(path=str(folder.relative_to(root)),files=files,max_parameters=max_parameters,dtype_policy=dtype_policy)
    source = resolve('model','hunyuan15-t2v')['source']
    value = dict(schema_version='distillschedule.hunyuan15_t2v_assets.v1',family='hunyuan15-t2v',
        source={key:source[key] for key in ('model_id','revision')},evidence_kind='LOCAL_PRETRAINED_UNVERIFIED',
        pipeline=dict(_class_name='HunyuanVideo15Pipeline',task='t2v',use_meanflow=False,distilled=False),
        text_policy=dict(primary_max_length=1000,crop_start=108,hidden_layer=-3,secondary_max_length=256,image_tokens=729),
        roles=dict(student=component(student),fake=component(fake),teacher=component(teacher)),
        media={name:component(name) for name in ('vae','text_encoder','text_encoder_2','tokenizer','tokenizer_2')})
    target = root/'assets.json'
    with target.open('x') as output: json.dump(value,output,indent=2); output.write('\n')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('bundle')
    parser.add_argument('--cpu-max-parameters',required=True,type=int)
    parser.add_argument('--dtype-policy',choices=['strict_fp32','bf16_or_fp32_to_fp32'],default='strict_fp32')
    for role in ('student','fake','teacher'): parser.add_argument('--'+role,default='transformer')
    args = parser.parse_args()
    print(lock_bundle(args.bundle,max_parameters=args.cpu_max_parameters,dtype_policy=args.dtype_policy,
        student=args.student,fake=args.fake,teacher=args.teacher))
