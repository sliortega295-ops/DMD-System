"""SYNTHETIC_TEST_ONLY: real tiny native classes, never pretrained quality evidence.

This is the only random-weight creator in this recipe. Production adapters have
no random fallback. No remote asset fetch, alternate architecture or GPU use.
"""
import json
from pathlib import Path


def build_fixture(root, family, *, cycles=3):
    import numpy as np
    from PIL import Image
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import Qwen2TokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    from distillschedule.data.flow_image_inputs import family_api, SCHEMA
    from distillschedule.training.flow_image import ASSET_SCHEMA
    from distillschedule.workloads.best_known.registry import file_sha256
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False)
    api = family_api(family)
    native_cls, vae_cls, _, _ = api.classes()
    def save(name, model):
        model.float().eval().save_pretrained(root / name, safe_serialization=True)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(432)
        if family == 'flux2-klein-base':
            config = dict(patch_size=1, in_channels=16, out_channels=16, num_layers=1, num_single_layers=1,
                num_attention_heads=2, attention_head_dim=16, joint_attention_dim=48,
                timestep_guidance_channels=16, mlp_ratio=2, axes_dims_rope=[4,4,4,4], guidance_embeds=False)
            vae_config = dict(in_channels=3,out_channels=3,block_out_channels=[8,8],norm_num_groups=4,
                down_block_types=['DownEncoderBlock2D']*2,up_block_types=['UpDecoderBlock2D']*2,
                latent_channels=4,layers_per_block=1,sample_size=8,patch_size=[2,2])
        else:
            config = dict(patch_size=2,in_channels=16,out_channels=4,num_layers=1,num_attention_heads=2,
                attention_head_dim=16,joint_attention_dim=16,axes_dims_rope=[4,6,6],guidance_embeds=False,
                zero_cond_t=False,use_additional_t_cond=False,use_layer3d_rope=False)
            vae_config = dict(base_dim=4,z_dim=4,dim_mult=[1,2],num_res_blocks=1,temperal_downsample=[False],
                latents_mean=[.1,-.1,.2,-.2],latents_std=[.8,.9,1.1,1.2])
        native = native_cls(**config).float().eval()
        save('base', native)
        with torch.no_grad():
            native.proj_out.weight.add_(.02)
        save('fake', native)
        vae = vae_cls(**vae_config).float().eval()
        if family == 'flux2-klein-base':
            with torch.no_grad():
                vae.bn.running_mean.copy_(torch.linspace(-.1,.1,16))
                vae.bn.running_var.copy_(torch.linspace(.8,1.2,16))
        save('vae', vae)
        special = ['[PAD]', '[UNK]', '<|im_start|>', '<|im_end|>', '<|image_pad|>', '<|video_pad|>', '<|vision_start|>', '<|vision_end|>']
        vocab = {name: i for i,name in enumerate(special)}
        for byte in sorted(pre_tokenizers.ByteLevel.alphabet()):
            vocab.setdefault(byte, len(vocab))
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        tokenizer = Qwen2TokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]',
            eos_token='<|im_end|>', bos_token='<|im_start|>', model_max_length=2048,
            additional_special_tokens=special[2:],
            chat_template="{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
        tokenizer.save_pretrained(root / 'tokenizer')
        if family == 'flux2-klein-base':
            text_config = Qwen3Config(vocab_size=len(tokenizer),hidden_size=16,intermediate_size=32,
                num_hidden_layers=28,num_attention_heads=2,num_key_value_heads=2,head_dim=8,max_position_embeddings=512,
                pad_token_id=0,bos_token_id=2,eos_token_id=3,use_cache=False)
            text = Qwen3ForCausalLM(text_config).float().eval()
            length = 48
        else:
            text_config = Qwen2_5_VLConfig(text_config=dict(vocab_size=len(tokenizer),hidden_size=16,intermediate_size=32,
                num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=512,
                pad_token_id=0,bos_token_id=2,eos_token_id=3,use_cache=False,
                rope_parameters=dict(rope_type='default',rope_theta=10000,mrope_section=[1,1,2])),
                vision_config=dict(depth=1,hidden_size=16,intermediate_size=32,num_heads=2,out_hidden_size=16,
                    patch_size=2,spatial_merge_size=1,temporal_patch_size=1,fullatt_block_indexes=[0],window_size=4),
                image_token_id=4,video_token_id=5,vision_start_token_id=6,vision_end_token_id=7)
            text = Qwen2_5_VLForConditionalGeneration(text_config).float().eval()
            length = 256
        save('text_encoder', text)
    def spec(name):
        folder = root / name
        return dict(path=name, files={str(p.relative_to(folder)):file_sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()},
                    max_parameters=10_000_000)
    manifest = dict(schema_version=ASSET_SCHEMA,family=family,source=dict(model_id=api.MODEL_ID,revision=api.REVISION),
        evidence_kind='SYNTHETIC_TEST_ONLY',roles=dict(student=spec('base'),fake=spec('fake'),teacher=spec('base')),
        media=dict(vae=spec('vae'),text_encoder=spec('text_encoder'),tokenizer=spec('tokenizer')),text_max_length=length)
    (root / 'assets.json').write_text(json.dumps(manifest,indent=2))
    rgb = (np.arange(8*8*3).reshape(8,8,3) * 7 % 256).astype('uint8')
    Image.fromarray(rgb).save(root / 'input.png')
    raw = dict(schema_version=SCHEMA,family=family,assets=str(root/'assets.json'),mode='raw',records=[
        dict(id=f'native-{i}',prompt=('a bird','a red bird','a bird by a blue lake')[i % 3],image=dict(path='input.png',sha256=file_sha256(root/'input.png')))
        for i in range(cycles)])
    (root/'raw.json').write_text(json.dumps(raw,indent=2))
    # The standard preparer plugin is also exercised by public CLI tests.
    from distillschedule.data.flow_image_inputs import prepare_inputs
    prepared = prepare_inputs(root/'raw.json',root/'prepared')
    parameters = dict(assets=str(root/'assets.json'),trainable=dict(mode='lora',rank=2,alpha=2.,targets=['proj_out']),
        student_lr=.003,fake_lr=.004,scheduler_gamma=.95,ema_decay=.9,generation_sigmas=[1.,.75,.5,.25,0.])
    policy = dict(fake_updates=5,fake_sample_policy='reuse_student',rollout_sigmas=[1.,.75,.5,.25,0.],gradient_policy='all_steps',
        sigma_min=.1,sigma_max=.9,teacher_cfg=1.,fake_cfg=1.,student_loss='flow_x0_surrogate',fake_loss='flow_velocity_mse',
        normalizer='per_modality_mean_abs',normalizer_epsilon=1e-5,modality_weights=dict(image=1.),max_grad_norm=1.)
    config = dict(schema_version='distillschedule.training.v1',model=dict(name=family,parameters=parameters),
        data=dict(prepared=str(prepared)),algorithm=dict(name='dmd',parameters=policy),execution=dict(mode='reference'),
        trainer=dict(cycles=cycles,seed=101,device='cpu',output_dir=str(root/'training'),checkpoint_every=1))
    (root/'train.json').write_text(json.dumps(config,indent=2))
    return dict(root=root,config=root/'train.json',assets=root/'assets.json',prepared=prepared,family=family)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('--family',required=True,choices=['flux2-klein-base','qwen-image'])
    parser.add_argument('--cycles',type=int,default=3)
    args=parser.parse_args()
    print(json.dumps({key:str(value) for key,value in build_fixture(args.output,args.family,cycles=args.cycles).items()},indent=2))
