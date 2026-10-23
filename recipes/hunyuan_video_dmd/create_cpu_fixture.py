"""SYNTHETIC_TEST_ONLY actual tiny native Hunyuan1.5/QwenText/ByT5/temporal4 VAE.

Only this explicit test fixture creates random weights. No production fallback,
GPU, remote fetch or pretrained/quality/license claim. Tiny constructor choices
derive from the pinned Diffusers0.37.1 ABI, not a redistributed full model config.
"""
import json
from pathlib import Path


def build_fixture(root, *, cycles=3):
    import numpy as np
    from PIL import Image
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import Qwen2TokenizerFast, Qwen2_5_VLTextConfig, Qwen2_5_VLTextModel, T5Config, T5EncoderModel, ByT5Tokenizer
    from distillschedule.training import hunyuan_video as api
    from distillschedule.data.hunyuan_video_inputs import SCHEMA, prepare_inputs
    from distillschedule.data.manifest import source_identity
    from distillschedule.workloads.best_known.registry import file_sha256
    root = Path(root).resolve(); root.mkdir(parents=True,exist_ok=False)
    def save(name, model):
        model.float().eval().save_pretrained(root/name,safe_serialization=True)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(433)
        native = api.classes()[0](in_channels=9,out_channels=4,num_layers=1,num_refiner_layers=1,
            num_attention_heads=2,attention_head_dim=16,mlp_ratio=2,patch_size=1,patch_size_t=1,
            text_embed_dim=16,text_embed_2_dim=16,image_embed_dim=16,rope_axes_dim=[4,6,6],
            target_size=16,task_type='t2v',use_meanflow=False)
        save('base',native)
        with torch.no_grad(): native.proj_out.weight.add_(.02)
        save('fake',native)
        save('vae',api.classes()[1](in_channels=3,out_channels=3,latent_channels=4,
            block_out_channels=[32,32,32],layers_per_block=1,spatial_compression_ratio=4,
            temporal_compression_ratio=4,scaling_factor=1.03682))
        special = ['[PAD]','[UNK]','<|im_start|>','<|im_end|>']
        vocab = {word:index for index,word in enumerate(special)}
        for byte in sorted(pre_tokenizers.ByteLevel.alphabet()): vocab.setdefault(byte,len(vocab))
        backend = Tokenizer(models.BPE(vocab=vocab,merges=[],unk_token=special[1]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        tokenizer = Qwen2TokenizerFast(tokenizer_object=backend,pad_token=special[0],unk_token=special[1],
            bos_token=special[2],eos_token=special[3],additional_special_tokens=special[2:],
            chat_template="{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
        tokenizer.save_pretrained(root/'tokenizer')
        text = Qwen2_5_VLTextModel(Qwen2_5_VLTextConfig(vocab_size=len(tokenizer),hidden_size=16,
            intermediate_size=32,num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=2,
            bos_token_id=2,eos_token_id=3,pad_token_id=0,
            rope_parameters=dict(rope_type='default',mrope_section=[1,1,2])))
        save('text_encoder',text)
        # Exercise the actual public model's historical native RoPE spelling.
        # Raw bytes are inventoried and never rewritten by production loading.
        path = root/'text_encoder/config.json'; config = json.loads(path.read_text())
        rope = config.pop('rope_parameters'); config['rope_theta'] = rope['rope_theta']
        config['rope_scaling'] = dict(type='mrope',mrope_section=rope['mrope_section'])
        path.write_text(json.dumps(config,indent=2)+'\n')
        tokenizer2 = ByT5Tokenizer(model_max_length=256)
        tokenizer2.save_pretrained(root/'tokenizer_2')
        save('text_encoder_2',T5EncoderModel(T5Config(vocab_size=len(tokenizer2),d_model=16,d_kv=8,d_ff=32,
            num_layers=1,num_heads=2,dropout_rate=0.,pad_token_id=0,eos_token_id=1)))
    def spec(name):
        folder = root/name
        return dict(path=name,files={str(p.relative_to(folder)):file_sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()},max_parameters=10_000_000)
    manifest = dict(schema_version=api.ASSET_SCHEMA,family=api.FAMILY,source=dict(model_id=api.MODEL_ID,revision=api.REVISION),
        evidence_kind='SYNTHETIC_TEST_ONLY',pipeline=api.PIPELINE,text_policy=api.TEXT_POLICY,
        roles=dict(student=spec('base'),fake=spec('fake'),teacher=spec('base')),
        media={name:spec(name) for name in ('vae','text_encoder','text_encoder_2','tokenizer','tokenizer_2')})
    (root/'assets.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (root/'frames').mkdir()
    for i in range(5):
        pixels = ((np.arange(16*16*3).reshape(16,16,3)*7+i*17)%256).astype('uint8')
        Image.fromarray(pixels).save(root/'frames'/f'{i:04d}.png')
    raw = dict(schema_version=SCHEMA,family=api.FAMILY,assets='assets.json',mode='raw',records=[
        dict(id=f'native-video-{i}',prompt=('a bird','a sign says "hello"','a blue boat on a lake')[i%3],
            video=dict(path='frames',sha256=source_identity(root/'frames',frames=True)['sha256'])) for i in range(cycles)])
    (root/'raw.json').write_text(json.dumps(raw,indent=2)+'\n')
    prepared = prepare_inputs(root/'raw.json',root/'prepared')
    parameters = dict(assets=str(root/'assets.json'),trainable=dict(mode='lora',rank=2,alpha=2.,targets=['proj_out']),
        student_lr=.003,fake_lr=.004,scheduler_gamma=.95,ema_decay=.9,generation_sigmas=[1.,.75,.5,.25,0.])
    policy = dict(fake_updates=5,fake_sample_policy='reuse_student',rollout_sigmas=[1.,.75,.5,.25,0.],gradient_policy='all_steps',
        sigma_min=.1,sigma_max=.9,teacher_cfg=1.,fake_cfg=1.,student_loss='flow_x0_surrogate',fake_loss='flow_velocity_mse',
        normalizer='per_modality_mean_abs',normalizer_epsilon=1e-5,modality_weights=dict(video=1.),max_grad_norm=1.)
    config = dict(schema_version='distillschedule.training.v1',model=dict(name=api.FAMILY,parameters=parameters),
        data=dict(prepared=str(prepared)),algorithm=dict(name='dmd',parameters=policy),execution=dict(mode='reference'),
        trainer=dict(cycles=cycles,seed=101,device='cpu',output_dir=str(root/'training'),checkpoint_every=1))
    (root/'train.json').write_text(json.dumps(config,indent=2)+'\n')
    return dict(root=root,config=root/'train.json',assets=root/'assets.json',prepared=prepared)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('output'); parser.add_argument('--cycles',type=int,default=3)
    args = parser.parse_args()
    print(json.dumps({k:str(v) for k,v in build_fixture(args.output,cycles=args.cycles).items()},indent=2))
