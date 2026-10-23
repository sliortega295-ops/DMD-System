"""RANDOM tiny ACTUAL native Wan/CausalWan/VAE assets, never pretrained quality.

Uses 21 latent frames, the original minimum; never shrinks temporal semantics.
No network, CUDA, synthetic replacement model or shared environment mutation.
"""
import argparse
import json
from pathlib import Path


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def build_fixture(root, source_root, *, cycles=3, ratio=5):
    import torch
    from distillschedule.training.self_forcing import original, ASSET_SCHEMA
    from distillschedule.workloads.best_known.registry import file_sha256
    from distillschedule.data.self_forcing_inputs import prepare_inputs, SCHEMA
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False)
    assets = root / 'RANDOM_TINY_ACTUAL_NATIVE_NOT_PRETRAINED'; assets.mkdir()
    api = original(source_root)
    config = dict(model_type='t2v', patch_size=[1, 2, 2], text_len=8, in_dim=16, out_dim=16,
        dim=32, ffn_dim=64, freq_dim=16, text_dim=16, num_heads=2, num_layers=1,
        qk_norm=True, cross_attn_norm=True)
    causal_config = {**config, 'local_attn_size': -1, 'sink_size': 0}
    vae_config = dict(dim=4, z_dim=16, dim_mult=[1, 2, 2, 2], num_res_blocks=1,
                      attn_scales=[], temperal_downsample=[True, True, False], dropout=0.)
    manifest = dict(schema_version=ASSET_SCHEMA, evidence_kind='RANDOM_TINY_ACTUAL_NATIVE_CLASSES_NOT_PRETRAINED')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        for role, cls, architecture in [('generator', api.causal.CausalWanModel, causal_config),
                ('real_score', api.wan.WanModel, config), ('fake_score', api.wan.WanModel, config),
                ('vae', api.vae.WanVAE_, vae_config)]:
            model = cls(**architecture)
            if role != 'vae':
                # Native zero-head initialization is appropriate for pretraining,
                # but supplies no score signal at step0 in an entirely random
                # no-pretraining fixture. Explicit random ASSET weights provide
                # nonzero native losses without changing a single model method.
                torch.nn.init.normal_(model.head.head.weight, std=.02)
            target = assets / (role + '.pt'); torch.save(model.state_dict(), target)
            manifest[role] = dict(config=architecture,
                weights=dict(path=target.name, sha256=file_sha256(target)))
        negative = assets / 'negative_prompt_embeds.pt'
        torch.save(torch.randn(1, 8, 16), negative)
        manifest['negative_prompt_embeds'] = dict(path=negative.name, sha256=file_sha256(negative))
        inputs = root / 'native-conditions'; inputs.mkdir()
        stream_records = {'generator': [], 'critic': []}
        for cycle in range(cycles):
            for draw in range(ratio + 1):
                index = cycle * (ratio + 1) + draw
                role = 'generator' if draw == 0 else 'critic'
                tensor = inputs / f'prompt-{index:04d}.pt'
                torch.save(torch.randn(1, 8, 16), tensor)
                stream_records[role].append(dict(id=f'native-prompt-{index:04d}',
                    embedding=dict(path=str(tensor.relative_to(root)), sha256=file_sha256(tensor))))
    _json(assets / 'assets.json', manifest)
    prepared = {}
    for role, records in stream_records.items():
        source = root / (role + '-inputs.json')
        _json(source, dict(schema_version=SCHEMA, mode='preencoded', records=records))
        prepared[role] = str(prepare_inputs(source, root / 'prepared' / role))
    settings = dict(image_or_video_shape=[1, 21, 16, 4, 4], lr=1e-4, lr_critic=2e-4,
        dfake_gen_update_ratio=ratio, train_iters=max(100, cycles * ratio), ema_start_step=5, ema_weight=.9,
        negative_prompt='RANDOM_EXPLICIT_NATIVE_EMBEDDING_NOT_TEXT_ENCODER_OUTPUT')
    training = dict(schema_version='distillschedule.training.v1',
        model=dict(factory='distillschedule.training.self_forcing:build_model', parameters=dict(
            source_root=str(Path(source_root).resolve()), assets=str(assets / 'assets.json'), settings=settings)),
        algorithm=dict(factory='distillschedule.training.self_forcing:build_self_forcing'),
        data=dict(streams={role: dict(prepared=path, samples_per_cycle=1 if role == 'generator' else ratio)
                           for role, path in prepared.items()}),
        execution=dict(adapter='program', mode='reference'),
        trainer=dict(cycles=cycles, seed=17, device='cpu', output_dir=str(root / 'run'), checkpoint_every=1))
    _json(root / 'training.json', training)
    _json(root / 'EVIDENCE.json', dict(evidence_kind=manifest['evidence_kind'], pretrained_weights='NONE',
        quality='NOT_RUN', gpu='NOT_RUN', conditioning='EXPLICIT_RANDOM_NATIVE_EMBEDDING_ASSETS_NOT_TEXT_ENCODER_OUTPUT',
        source_root=str(Path(source_root).resolve()), temporal_contract='NATIVE_MIN21_FRAMES_BLOCK3'))
    return dict(root=str(root), assets=str(assets), prepared=prepared, config=str(root / 'training.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output'); parser.add_argument('--source-root', required=True)
    parser.add_argument('--cycles', type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(build_fixture(args.output, args.source_root, cycles=args.cycles), indent=2))
