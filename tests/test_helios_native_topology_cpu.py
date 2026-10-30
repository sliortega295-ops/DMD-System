"""Original native 40-block topology in its own CPU process and namespace.

This imports the native helios.* class, independently of the public reference
helper namespace. It does not qualify attention math, CUDA or model quality.
"""
import pytest


def test_target_actual_helios_block_class_checkpoint_and_cpu_fsdp_ownership(tmp_path, monkeypatch):
    """Small actual class/topology test for the pinned native environment.

    No full-size model, GPU, numerical qualification or pretrained assets.
    The import-only capability probe raises the original unsupported-GPU
    exception instead of initializing CUDA; no GPU capability is fabricated.
    """
    from types import SimpleNamespace as NS
    from test_training_helios_native_history import task, _cpu_import_capability_probe
    pytest.importorskip('ftfy');pytest.importorskip('omegaconf')
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, FSDPModule
    from distillschedule.training.helios_worker import _backend
    from distillschedule.runtime.fsdp2_static import configure_fsdp2_checkpointing
    from distillschedule.runtime.helios_native_history import admit_engine,generator_phase
    _backend()  # original native namespace/dependency owner, no model or download
    from runtime.critic_pool_helios import configure_offline_helios
    import kernels
    original_probe = torch.cuda.get_device_capability
    assert not torch.cuda.is_initialized()
    with monkeypatch.context() as imports:
        # CPU Torch raises AssertionError for a CUDA query, whereas the original
        # optional backend catches RuntimeError/ImportError on unsupported GPUs.
        # Exercise that exact no-Flash branch without changing the pinned source.
        imports.setattr(torch.cuda, 'get_device_capability',
                        _cpu_import_capability_probe(original_probe, cuda_version=torch.version.cuda))
        imports.setattr(kernels, 'get_kernel', kernels.get_kernel)
        configure_offline_helios()  # Original offline admission, no downloads.
        from helios.modules.transformer_helios import HeliosTransformer3DModel, HeliosTransformerBlock
    assert torch.cuda.get_device_capability is original_probe
    assert not dist.is_initialized() and not torch.cuda.is_initialized()
    dist.init_process_group('gloo',init_method='file://'+str(tmp_path/'actual-topology'),rank=0,world_size=1)
    try:
        model=HeliosTransformer3DModel(num_attention_heads=1,attention_head_dim=12,ffn_dim=24,
            freq_dim=12,text_dim=4096,num_layers=40,rope_dim=(4,4,4),cross_attn_norm=True)
        model.enable_gradient_checkpointing();configure_fsdp2_checkpointing(model)
        mesh=init_device_mesh('cpu',(1,));mp=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32)
        for block in model.blocks:
            assert isinstance(block,HeliosTransformerBlock)
            fully_shard(block,mesh=mesh,mp_policy=mp,reshard_after_forward=True)
        fully_shard(model,mesh=mesh,mp_policy=mp,reshard_after_forward=True)
        assert isinstance(model,FSDPModule)
        engine=NS(module=model,optimizer=NS(param_groups=[dict(params=list(model.parameters()))]))
        actual=admit_engine(engine)
        assert actual['checkpoint_blocks']==40 and actual['fsdp_units']==41
        with generator_phase(engine,task=task(),with_grad=False,action_id='r0') as (_,receipt):
            assert receipt['unshard_count']==41
        assert receipt['reshard_count']==41 and receipt['restored']
        original=model._gradient_checkpointing_func
        model._gradient_checkpointing_func=lambda *a,**k:None
        with pytest.raises(ValueError,match='checkpoint function'): admit_engine(engine)
        model._gradient_checkpointing_func=original
    finally:
        dist.destroy_process_group()
    assert not dist.is_initialized() and not torch.cuda.is_initialized()
