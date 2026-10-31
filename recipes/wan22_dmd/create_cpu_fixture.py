"""SYNTHETIC_TEST_ONLY: actual tiny native Wan/UMT5/residual VAE, never quality.

This recipe alone creates random weights. Production has no random fallback.
No remote fetching, causal substitutes, experts, GPU or model downloads.
"""
import json
from pathlib import Path


def build_fixture(root, *, cycles=3, samples=None):
    samples = cycles if samples is None else samples
    if type(cycles) is not int or cycles < 1 or type(samples) is not int or samples < 1:
        raise ValueError('positive explicit synthetic cycle/sample counts required')
    import numpy as np
    from PIL import Image
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
    from transformers import T5TokenizerFast, UMT5Config, UMT5EncoderModel
    from distillschedule.training import wan22 as api
    from distillschedule.data.wan22_inputs import SCHEMA, prepare_inputs
    from distillschedule.data.manifest import source_identity
    from distillschedule.workloads.best_known.registry import file_sha256
    root = Path(root).resolve(); root.mkdir(parents=True,exist_ok=False)
    def save(name,model):
        model.float().eval().save_pretrained(root/name,safe_serialization=True)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(432)
        native = api.classes()[0](patch_size=(1,2,2),in_channels=4,out_channels=4,num_layers=1,
            num_attention_heads=2,attention_head_dim=16,text_dim=16,freq_dim=16,ffn_dim=64,
            image_dim=None,added_kv_proj_dim=None,rope_max_seq_len=16)
        save('base',native)
        with torch.no_grad(): native.proj_out.weight.add_(.02)
        save('fake',native)
        vae = api.classes()[1](base_dim=4,decoder_base_dim=4,z_dim=4,dim_mult=[1,2,2,2],num_res_blocks=1,
            temperal_downsample=[False,True,True],latents_mean=[.1,-.1,.2,-.2],latents_std=[.8,.9,1.1,1.2],
            is_residual=True,in_channels=12,out_channels=12,patch_size=2,scale_factor_temporal=4,scale_factor_spatial=16)
        save('vae',vae)
        # Actual pinned upstream JSON contains this historical false-only no-op.
        path = root/'vae/config.json'; config = json.loads(path.read_text()); config['clip_output'] = False
        path.write_text(json.dumps(config,indent=2)+'\n')
        special = ['[PAD]','[UNK]','</s>']
        vocab = {token:i for i,token in enumerate(special)}
        for byte in sorted(pre_tokenizers.ByteLevel.alphabet()): vocab.setdefault(byte,len(vocab))
        # Single-byte tokens match the source probe; native T5 save/restore
        # requires an actual Unigram vocabulary (not a Qwen/BPE checkpoint).
        backend = Tokenizer(models.Unigram([(token,0.) for token in vocab],unk_id=1))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        backend.post_processor = processors.TemplateProcessing(single='$A </s>',special_tokens=[('</s>',2)])
        tokenizer = T5TokenizerFast(tokenizer_object=backend,extra_ids=0,pad_token='[PAD]',unk_token='[UNK]',eos_token='</s>',model_max_length=512)
        tokenizer.save_pretrained(root/'tokenizer')
        text = UMT5EncoderModel(UMT5Config(vocab_size=len(tokenizer),d_model=16,d_kv=8,d_ff=32,num_layers=1,
            num_heads=2,dropout_rate=0.,pad_token_id=0,eos_token_id=2))
        save('text_encoder',text)
    def spec(name):
        folder = root/name
        return dict(path=name,files={str(p.relative_to(folder)):file_sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()},max_parameters=10_000_000)
    manifest = dict(schema_version=api.ASSET_SCHEMA,family=api.FAMILY,source=dict(model_id=api.MODEL_ID,revision=api.REVISION),
        evidence_kind='SYNTHETIC_TEST_ONLY',pipeline=api.PIPELINE,roles=dict(student=spec('base'),fake=spec('fake'),teacher=spec('base')),
        media={name:spec(name) for name in ('vae','text_encoder','tokenizer')},text_max_length=226)
    (root/'assets.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (root/'frames').mkdir()
    for i in range(5):
        pixels = ((np.arange(32*32*3).reshape(32,32,3)*7+i*17)%256).astype('uint8')
        Image.fromarray(pixels).save(root/'frames'/f'{i:04d}.png')
    raw = dict(schema_version=SCHEMA,family=api.FAMILY,assets='assets.json',mode='raw',records=[
        dict(id=f'native-video-{i}',prompt=('a bird','a red bird','a blue boat on a lake')[i%3],
            video=dict(path='frames',sha256=source_identity(root/'frames',frames=True)['sha256'])) for i in range(samples)])
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
    parser.add_argument('--samples',type=int)
    args = parser.parse_args()
    print(json.dumps({k:str(v) for k,v in build_fixture(args.output,cycles=args.cycles,samples=args.samples).items()},indent=2))
