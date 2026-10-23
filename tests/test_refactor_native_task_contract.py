"""Dependency-light shape/protocol/capability admission tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from distillschedule.native.task import NativeTask
from distillschedule.native.planning import capabilities


@pytest.mark.parametrize("kwargs",[
    {"fake_updates":0},{"fake_updates":65},{"fake_updates":True},{"denoising_steps":0},
    {"video_height":31},{"video_frames":4},{"compression":(0,8,8)},
    {"dtype":"float8"},{"device":"cuda:banana"},{"seed":-1},
    {"ema_after":(3,)},{"ema_after":(1,0)},{"target_modules":()},
    {"sigma_shift":float("nan")},{"normalization_floor":0},
    {"score_sigma_min":0.99},{"teacher_guidance":-1},{"lora_alpha":True},
    {"adam_betas":(0.9,1.0)},{"protocol":"diffsynth-autoregressive-h3"},
    {"fixture":False},{"backend":"minimax-h3-grid1000"},
])
def test_invalid_task_fails_closed(kwargs):
    with pytest.raises(ValueError):NativeTask(**kwargs)


def test_unknown_protocol_fields_not_ignored():
    for key in ("gan","previous_latent","rank_override","world_size","loss_override"):
        with pytest.raises(ValueError,match="unknown task field"):
            NativeTask.from_dict({key:1})


def test_roundtrip_and_shape_inference():
    t=NativeTask(video_height=384,video_width=640,video_frames=81,channels=16,fake_updates=1,
                 denoising_steps=8,lora_rank=32,lora_alpha=32)
    assert t.latent_shape==(1,16,21,48,80)
    assert NativeTask.from_dict(t.to_dict()).digest==t.digest
    assert capabilities(t)["denoising_steps"]==8


def test_validate_does_not_import_torch():
    root=Path(__file__).resolve().parents[1]
    script="from distillschedule.native.task import NativeTask; from distillschedule.native.planning import capabilities; import sys; capabilities(NativeTask()); assert 'torch' not in sys.modules"
    r=subprocess.run([sys.executable,"-c",script],cwd=root,env={**os.environ,"PYTHONPATH":str(root/"src")},capture_output=True,text=True)
    assert r.returncode==0,r.stderr
