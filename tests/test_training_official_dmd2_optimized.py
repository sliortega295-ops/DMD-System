"""Source-bound portfolio admission; CPU checks are not GPU qualification."""
from copy import deepcopy
from types import SimpleNamespace

import pytest


def test_optimized_recipe_requires_real_cuda_ownership_before_assets(tmp_path):
    from distillschedule.training.sd import _arguments
    with pytest.raises(ValueError, match='CUDA.*world16|world16.*CUDA'):
        _arguments(dict(source_root=str(tmp_path), model_id=str(tmp_path),
            optimization_profile='dmd2-sd15-resident-tree-v1', device='cpu',
            settings=dict(generator_lr=1e-6, guidance_lr=1e-6)), sdxl=False)


def test_profile_does_not_silently_enable_gan_or_change_native_math():
    from distillschedule.training.official_optimizations.dmd2 import validate_settings
    valid = dict(sdxl=False, batch_size=1, resolution=512, latent_resolution=64,
        latent_channel=4, dfake_gen_update_ratio=10, num_denoising_step=1,
        gradient_checkpointing=True, cls_on_clean_image=True, gen_cls_loss=True,
        gen_cls_loss_weight=.001, guidance_cls_loss_weight=.01, use_fp16=True,
        denoising=False, backward_simulation=False, generator_lora=False,
        gan_alone=False, diffusion_gan=True, diffusion_gan_max_timestep=1000)
    validate_settings(SimpleNamespace(**valid))
    for key, value in [('batch_size', 2), ('gradient_checkpointing', False),
                       ('use_fp16', False), ('cls_on_clean_image', False),
                       ('sdxl', True)]:
        bad = deepcopy(valid); bad[key] = value
        with pytest.raises(ValueError, match='profile'):
            validate_settings(SimpleNamespace(**bad))


def test_profile_does_not_mutate_collective_environment(monkeypatch):
    from distillschedule.training.official_optimizations.dmd2 import validate_environment
    monkeypatch.setenv('NCCL_ALGO', 'Ring')
    monkeypatch.setenv('NCCL_MIN_CTAS', '16')
    with pytest.raises(ValueError, match='NCCL_ALGO'):
        validate_environment()
    import os
    assert os.environ['NCCL_ALGO'] == 'Ring'
    monkeypatch.setenv('NCCL_ALGO', 'allreduce:Tree')
    validate_environment()


def test_reference_stays_fp32_and_does_not_enable_optimized_owner():
    from distillschedule.training.dmd2_owner import DMD2Owner
    from distillschedule.training.official_optimizations.dmd2 import DMD2OptimizedOwner
    assert DMD2Owner._optimizer_options(SimpleNamespace()) == {}
    assert DMD2OptimizedOwner._optimizer_options(SimpleNamespace()) == {'fused': True}


def test_completed_dmd2_census_reaches_public_execution_receipt(monkeypatch):
    from distillschedule.training.dmd2_collective import DMD2CollectiveOwner
    from distillschedule.training.official_optimizations.dmd2 import DMD2OptimizedOwner
    monkeypatch.setattr(DMD2CollectiveOwner, 'validate_collective_execution', lambda _: {'roles': {}})
    owner = DMD2OptimizedOwner.__new__(DMD2OptimizedOwner)
    owner._last_optimization_receipt = {'profile': 'fixture', 'cast': {'entered': 10}}
    result = DMD2OptimizedOwner.validate_collective_execution(owner)
    assert result['optimization_effects'] == owner._last_optimization_receipt
    result['optimization_effects']['cast']['entered'] = 0
    assert owner._last_optimization_receipt['cast']['entered'] == 10
