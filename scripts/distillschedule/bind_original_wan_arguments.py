#!/usr/bin/env python3
"""One-time, hash-bound source/provenance publication for original Wan binding.

Only the two listed helper files and their source-import-map rows can change.
Used by the isolated PR branch's checkpoint workflow; not a runtime patcher.
"""
from pathlib import Path
import hashlib
import json
import subprocess

FILES={
 'runtime/gst_pool_helios.py':('006a5a6a99277e2dc348eb44c65e00c3c3a3b4fb9e4d9bb170d12ef1f93133ca','73c174a4ff7865226b03492e9da07b23d2cf4b7fbd360ed4684d772148db4aa9'),
 'runtime/static_fsdp2_p0_helios.py':('f44579ab6959eff7a835f3771cb1f3f4f08f9ffd1b5d8da2ebb4549208b4d3a8','4c5e820841b129fe01d2766bc949aa06d46c7f7b4c59cb9b60e9900218969e6c')}

def git(*args):return subprocess.check_output(['git',*args],text=True).strip()
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    start=git('rev-parse','HEAD')
    values={}
    for name,(before,after) in FILES.items():
        if digest(name)==after:continue
        if digest(name)!=before:raise ValueError('unexpected source preimage: '+name)
        text=Path(name).read_text()
        limit=text.index('\ndef prepare_dmd_score_batch') if 'gst_pool' in name else len(text)
        body,tail=text[:limit],text[limit:]
        target='entry' if 'gst_pool' in name else 'loaded'
        old=f'    tensors = {target}.tensors'
        if body.count(old)!=1:raise ValueError('ambiguous native entry')
        body=body.replace(old,'    from distillschedule.strategies.wan import rollout_parameters\n'
                          f'    parameters = rollout_parameters({target}.operation_trace)\n'+old)
        for a,b in [('base_height=48','base_height=parameters["base_height"]'),
                    ('base_width=80','base_width=parameters["base_width"]'),
                    ('stage2_num_inference_steps_list=[2, 2, 2]','stage2_num_inference_steps_list=parameters["stage2_num_inference_steps_list"]'),
                    ('denoising_step_list=[1000, 750, 500, 250]','denoising_step_list=parameters["denoising_step_list"]'),
                    ('num_critic_input_frames=9','num_critic_input_frames=parameters["num_critic_input_frames"]')]:
            if a not in body:raise ValueError('missing original invocation constant: '+a)
            body=body.replace(a,b)
        if target=='entry':
            body=body.replace('if list(generated_x0.shape) != [1, 16, 9, 48, 80]:',
                              'if tuple(generated_x0.shape) != parameters["latent_shape"]:')
        else:
            body=body.replace('if list(generated_x0.shape) != [batch_size, 16, 9, 48, 80]:',
                 'expected_shape = (batch_size, *parameters["latent_shape"][1:])\n    if tuple(generated_x0.shape) != expected_shape:')
            body=body.replace('{[batch_size, 16, 9, 48, 80]}','{expected_shape}')
        result=body+tail
        if hashlib.sha256(result.encode()).hexdigest()!=after:raise ValueError('unexpected generated source')
        values[name]=result
    if values:
        for name,text in values.items():Path(name).write_text(text)
        git('add','--',*values);git('diff','--cached','--check')
        git('commit','-m','feat(wan): bind original rollout geometry and stage budgets to validated task')
    source_commit=git('rev-parse','HEAD')
    path=Path('docs/workloads/source_import_map.json');doc=json.loads(path.read_text())
    changed=False
    for name,(before,after) in FILES.items():
        row=next(r for r in doc['files'] if r['target']==name)
        if row['integrated_sha256']==after:continue
        if row['integrated_sha256']!=before:raise ValueError('unexpected previous provenance')
        row.setdefault('adaptations',[]).append(dict(base_integration_head=start,
            previous_integrated_sha256=before,new_integrated_sha256=after,source_commit=source_commit,
            reason='Use the original sampler with task-bound geometry and stage budgets; preserve exact legacy defaults.'))
        row['integrated_sha256']=after;changed=True
    if changed:
        path.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
        git('add',str(path));git('commit','-m','fix(provenance): record original Wan argument-binding source')
    print('Published source head:',source_commit)
    print('Validated provenance head:',git('rev-parse','HEAD'))

if __name__=='__main__':main()
