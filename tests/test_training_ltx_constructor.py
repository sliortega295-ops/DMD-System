"""Actual native connector construction must not select a machine's GPU."""
import runpy
from pathlib import Path

import torch

from test_training_ltx_cpu import source_root


def test_native_cpu_media_never_probes_visible_cuda(source_root, monkeypatch):
    from distillschedule.training.ltx_source import original
    from distillschedule.training.ltx_media import NativeMedia
    recipe = Path(__file__).resolve().parents[1] / 'recipes/ltx25/create_cpu_fixture.py'
    _, architecture = runpy.run_path(str(recipe))['tiny_architectures']()
    api = original(source_root)
    connector = api.source.import_module('ltx_core.text_encoders.gemma.embeddings_connector')
    original_attention = connector.Attention
    original_init = connector.Embeddings1DConnector.__init__
    # On the unmodified parent this clears AUTOMATIC's process cache, so the
    # regression cannot accidentally pass due to an earlier CPU-only build.
    if hasattr(api.attention.automatic_attention, 'cache_clear'):
        api.attention.automatic_attention.cache_clear()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU codec construction probed a visible CUDA device')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', forbidden)
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        media = NativeMedia(api.source, architecture)
    assert not torch.cuda.is_initialized()
    assert all(t.device.type == 'cpu' for t in (*media.parameters(), *media.buffers()))
    assert connector.Attention is original_attention
    assert connector.Embeddings1DConnector.__init__ is original_init
    assert api.attention.automatic_attention is api.attention._sdpa_full_priority
    assert api.attention.automatic_masked_attention is api.attention._sdpa_full_priority
    for name in ('video_connector', 'audio_connector'):
        owned = getattr(media.text_processor, name)
        assert type(owned) is connector.Embeddings1DConnector
        for block in owned.transformer_1d_blocks:
            assert type(block) is connector._BasicTransformerBlock1D
            assert type(block.attn1) is api.attention.Attention
