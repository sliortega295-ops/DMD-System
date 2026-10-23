"""Actual CPU autograd on tiny audiovisual records, not a quality benchmark.

The fixed statistics encoder consumes decoded RGB frames, PCM audio and UTF-8
text. Three small, randomly initialized networks implement a detached Fake-
minus-Teacher clean-prediction gradient through the existing DMD builder. Teacher is frozen,
not pretrained. This is deliberately NOT W1-W6 or a production AV model.

Run ``python -m distillschedule.examples.tiny_av --output DIR --samples 6``
to create real PPM frames, WAV audio, text, a raw manifest and a training request.
Nothing is downloaded, and the fixture only admits CPU execution.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
from typing import Mapping
import wave

import torch
from torch import nn
import torch.nn.functional as F

from distillschedule.open_workload.dmd import (
    DMDHook, DMDHooks, build_dmd_workload, dmd_inputs,
)
from distillschedule.training.adapters import ModelComponents
from runtime.dmd_boundary import (
    add_noise, critic_denoising_loss, detached_dmd_surrogate_loss,
    dmd_input_gradient, shifted_sigma_schedule,
)


QUALITY_STATUS = "TINY_MEDIA_TRAINING_FIXTURE_NOT_PRETRAINED_QUALITY"
_CHANNELS = 4


def _positive_integer(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


def _positive_float(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def encode(record, parameters):
    """Encode actual media deterministically; this is not a pretrained encoder.

    ``record`` is the raw-data adapter's decoded THWC uint8 RGB, TxC float PCM,
    sample rate and literal text. The fourth latent channel contains temporal
    audio RMS. Conditioning combines a four-bin UTF-8 histogram and audio
    statistics. The leading batch dimension is intentionally part of this
    example's own ABI, not imposed by the preparation layer.
    """
    frames = _positive_integer(parameters.get("frames", 2), "frames", 8)
    size = _positive_integer(parameters.get("size", 4), "size", 32)
    video, audio, text = record.get("video"), record.get("audio"), record.get("text")
    if (not isinstance(video, torch.Tensor) or video.dtype != torch.uint8
            or video.device.type != "cpu" or video.ndim != 4
            or video.shape[-1] != 3 or min(video.shape) < 1):
        raise ValueError("encoder needs nonempty CPU THWC uint8 RGB video")
    if not isinstance(audio, Mapping):
        raise ValueError("encoder needs decoded audio samples and sample_rate")
    samples, sample_rate = audio.get("samples"), audio.get("sample_rate")
    if (not isinstance(samples, torch.Tensor) or samples.device.type != "cpu"
            or samples.dtype != torch.float32 or samples.ndim != 2
            or min(samples.shape) < 1 or not torch.isfinite(samples).all()
            or type(sample_rate) is not int or sample_rate < 1):
        raise ValueError("encoder needs finite nonempty TxC float32 PCM and a positive sample rate")
    if not isinstance(text, str):
        raise ValueError("encoder needs decoded UTF-8 text")
    with torch.no_grad():
        rgb = video.permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1
        rgb = F.adaptive_avg_pool3d(rgb, (frames, size, size))
        # Pooling accepts audio shorter than the chosen temporal latent extent.
        power = samples.square().mean(dim=1).reshape(1, 1, -1)
        audio_energy = F.adaptive_avg_pool1d(power, frames).sqrt().clamp(0, 1)
        audio_latent = audio_energy.reshape(1, 1, frames, 1, 1).expand(-1, -1, -1, size, size)
        encoded_text = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
        histogram = torch.bincount(encoded_text.remainder(_CHANNELS), minlength=_CHANNELS).float()
        histogram = histogram / max(1, encoded_text.numel())
        mono = samples.mean(dim=1)
        crossings = ((mono[1:] * mono[:-1]) < 0).float().mean() if mono.numel() > 1 else mono.new_zeros(())
        statistics = torch.stack((samples.mean(), samples.square().mean().sqrt(), crossings,
                                  samples.new_tensor(min(samples.shape[0] / sample_rate, 1.0))))
        condition = (histogram * 2 - .5 + statistics).reshape(1, _CHANNELS, 1, 1, 1)
        return {"latent": torch.cat((rgb, audio_latent), dim=1).contiguous(),
                "condition": condition.contiguous()}


class TinyAVNetwork(nn.Module):
    """A pointwise channel MLP with conditioning and normalized diffusion time."""

    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(_CHANNELS * 2 + 1, width), nn.Tanh(),
                                    nn.Linear(width, _CHANNELS))

    def forward(self, value, condition, timestep=None):
        batch = value.shape[0]
        time = (value.new_zeros(batch) if timestep is None else timestep.to(value))
        time = time.reshape(batch, 1, 1, 1, 1).expand(batch, 1, *value.shape[2:])
        inputs = torch.cat((value, condition.expand_as(value), time), dim=1)
        return self.layers(inputs.movedim(1, -1)).movedim(-1, 1).contiguous()


def build_model(parameters):
    """Factory: explicit model/optimizer state, no registry or solver selection."""
    if parameters.get("device", "cpu") != "cpu":
        raise ValueError("tiny_av is a CPU-only training fixture, not a GPU recipe")
    seed = parameters.get("seed", 7)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    widths = tuple(_positive_integer(parameters.get(role + "_width", default), role + "_width", 256)
                   for role, default in (("student", 12), ("fake", 16), ("teacher", 24)))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        student, fake, teacher = (TinyAVNetwork(width) for width in widths)
    teacher.requires_grad_(False).eval()
    state = {"student": student, "fake": fake, "teacher": teacher,
             "student_optimizer": torch.optim.AdamW(student.parameters(), lr=_positive_float(
                 parameters.get("student_lr", .002), "student_lr")),
             "fake_optimizer": torch.optim.AdamW(fake.parameters(), lr=_positive_float(
                 parameters.get("fake_lr", .003), "fake_lr"))}
    return ModelComponents(state=state, immutable_modules=("teacher",), metadata={
        "quality_status": QUALITY_STATUS, "pretrained": False,
        "teacher": "frozen_random_initialization", "encoder": "fixed_actual_media_statistics",
        "latent_channels": _CHANNELS, "ema": False,
    }, generate=generate, decode=decode).validate()


def generate(state, feeds, parameters):
    """Return a tiny AV feature latent; the lifecycle owns inference grad mode."""
    return state["student"](feeds["latent"], feeds["condition"])


def decode(value, parameters):
    """Render RGB diagnostic frames and audio features, NOT an audio waveform."""
    if not isinstance(value, torch.Tensor) or value.ndim != 5 or value.shape[:2] != (1, _CHANNELS):
        raise ValueError("tiny_av decode requires one BCTHW-style 4-channel feature sample")
    value = value.detach().cpu()
    video = ((value[0, :3].permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return {"video": video, "audio_features": value[0, 3].mean(dim=(1, 2)),
            "quality_status": QUALITY_STATUS}


def _student_forward(ctx):
    feeds = dmd_inputs(ctx)["feeds"]
    return ctx.state["student"](feeds["latent"], feeds["condition"])


def _noise(ctx):
    values = dmd_inputs(ctx)
    feeds = values["feeds"]
    epsilon = (feeds["student_noise"] if ctx.attributes["phase"] == "student"
               else feeds["fake_noise"][ctx.attributes["fake_index"]])
    return {"noisy": add_noise(values["generated"], epsilon, feeds["timestep"],
                               feeds["sigmas"], feeds["timesteps"]),
            "epsilon": epsilon, "timestep": feeds["timestep"]}


def _teacher_forward(ctx):
    values = dmd_inputs(ctx)
    feeds, noise = values["feeds"], values["noise"]
    model, condition = ctx.state["teacher"], feeds["condition"]
    time = noise["timestep"].float() / feeds["schedule_length"]
    return {"cond": model(noise["noisy"], condition, time),
            "uncond": model(noise["noisy"], torch.zeros_like(condition), time)}


def _fake_forward(ctx):
    values = dmd_inputs(ctx)
    feeds, noise = values["feeds"], values["noise"]
    return ctx.state["fake"](noise["noisy"], feeds["condition"],
                             noise["timestep"].float() / feeds["schedule_length"])


def _student_loss(ctx):
    values = dmd_inputs(ctx)
    feeds, noise = values["feeds"], values["noise"]
    # The score paths are detached; only the Student output keeps its graph.
    # This is actual source-semantic DMD math, not a random scalar loss.
    with torch.no_grad():
        gradient, _ = dmd_input_gradient(
            estimated_clean=values["generated"].detach(), noisy_sample=noise["noisy"],
            timestep=noise["timestep"], sigmas=feeds["sigmas"], timesteps=feeds["timesteps"],
            fake_flow_cond=values["fake"], real_flow_cond=values["teacher"]["cond"],
            real_flow_uncond=values["teacher"]["uncond"],
            real_guidance_scale=float(feeds["guidance_scale"]),
        )
    return detached_dmd_surrogate_loss(values["generated"], gradient)


def _fake_loss(ctx):
    values = dmd_inputs(ctx)
    return critic_denoising_loss(values["fake"], values["noise"]["epsilon"], values["generated"])


def build_algorithm(components, feeds, parameters, cycle, seed):
    """Build ONE Student1/FakeN cycle with user-defined math and explicit noise.

    ``fake_sample_policy`` is algorithm semantics, never an optimization winner.
    The existing runtime/planner alone supplies execution choices.
    """
    fake_updates = _positive_integer(parameters.get("fake_updates", 2), "fake_updates", 32)
    steps = _positive_integer(parameters.get("schedule_steps", 16), "schedule_steps", 10000)
    policy = parameters.get("fake_sample_policy", "reuse_student")
    if policy not in ("reuse_student", "fresh_student"):
        raise ValueError("fake_sample_policy must be reuse_student or fresh_student")
    guidance = _positive_float(parameters.get("guidance_scale", 1.5), "guidance_scale")
    if type(cycle) is not int or cycle < 0 or type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("nonnegative cycle and valid integer seed required")
    latent, condition = feeds.get("latent"), feeds.get("condition")
    if (not isinstance(latent, torch.Tensor) or latent.device.type != "cpu"
            or latent.dtype != torch.float32 or latent.ndim != 5 or latent.shape[1] != _CHANNELS
            or min(latent.shape) < 1 or not torch.isfinite(latent).all()
            or not isinstance(condition, torch.Tensor) or condition.device != latent.device
            or condition.dtype != latent.dtype or condition.shape != (latent.shape[0], _CHANNELS, 1, 1, 1)
            or not torch.isfinite(condition).all()):
        raise ValueError("tiny_av needs finite CPU float32 latent Bx4xTxHxW and condition Bx4x1x1x1")
    reserved = {"student_noise", "fake_noise", "timestep", "sigmas", "timesteps", "guidance_scale", "schedule_length"}
    if reserved.intersection(feeds):
        raise ValueError("prepared feeds conflict with algorithm-owned noise/schedule names")
    generator = torch.Generator(device="cpu").manual_seed((seed + 1009 * cycle) % (2**63))
    sigmas, timesteps = shifted_sigma_schedule(num_train_timesteps=steps, timestep_shift=1, device="cpu")
    expanded = {**feeds,
                "student_noise": torch.randn(latent.shape, generator=generator),
                "fake_noise": tuple(torch.randn(latent.shape, generator=generator) for _ in range(fake_updates)),
                "timestep": torch.randint(1, steps + 1, (latent.shape[0],), generator=generator),
                "sigmas": sigmas, "timesteps": timesteps,
                "guidance_scale": torch.tensor(guidance), "schedule_length": torch.tensor(float(steps))}
    hooks = DMDHooks(student_forward=DMDHook(_student_forward, replay_safe=True),
                     fake_forward=DMDHook(_fake_forward, replay_safe=True),
                     teacher_forward=_teacher_forward, noise=_noise,
                     student_loss=_student_loss, fake_loss=_fake_loss)
    state = components.state
    required = {"student", "fake", "teacher", "student_optimizer", "fake_optimizer"}
    return build_dmd_workload(
        **{key: state[key] for key in required}, hooks=hooks, feeds=expanded,
        cycles=1, fake_updates=fake_updates, fake_sample_policy=policy, seed=seed,
        extra_state={key: value for key, value in state.items()
                     if key not in required | {"student_grad", "fake_grad"}},
        immutable_modules=components.immutable_modules,
        identity_extra={"quality_status": QUALITY_STATUS, "training_cycle": cycle, "ema": False},
        replica_batch_size=latent.shape[0],
    )


def build_fixture(output, *, samples=6):
    """Write actual, tiny media and relative-path manifests; never overwrite."""
    samples = _positive_integer(samples, "samples", 1000)
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"fixture output is not empty: {output}")
    raw = output / "raw"
    raw.mkdir(parents=True)
    records = []
    for index in range(samples):
        name = f"sample-{index:06d}"
        sample = raw / name
        frames = sample / "frames"
        frames.mkdir(parents=True)
        for frame in range(3):
            pixels = bytes(channel for y in range(8) for x in range(8) for channel in (
                (x * 31 + index * 19 + frame * 29) % 256,
                (y * 31 + index * 23 + frame * 17) % 256,
                ((x + y) * 15 + index * 11 + frame * 37) % 256))
            (frames / f"{frame:03d}.ppm").write_bytes(b"P6\n8 8\n255\n" + pixels)
        with wave.open(str(sample / "tone.wav"), "wb") as wav:
            wav.setparams((1, 2, 8000, 800, "NONE", "not compressed"))
            tone = (int(12000 * math.sin(2 * math.pi * (220 + index * 35) * tick / 8000))
                    for tick in range(800))
            wav.writeframes(struct.pack("<800h", *tone))
        (sample / "caption.txt").write_text(
            f"Sample {index}: a moving RGB pattern accompanied by a {220 + index * 35} Hz tone.\n",
            encoding="utf-8")
        records.append({"id": name, "video": f"{name}/frames", "audio": f"{name}/tone.wav",
                        "text": {"path": f"{name}/caption.txt"},
                        "metadata": {"quality_status": QUALITY_STATUS, "pretrained": False}})
    manifest = {"schema_version": "distillschedule.data.raw.v1",
                "encoder": {"factory": "distillschedule.examples.tiny_av:encode",
                            "parameters": {"frames": 2, "size": 4}}, "records": records}
    raw_manifest = raw / "manifest.json"
    raw_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    config = {"schema_version": "distillschedule.training.v1",
              "model": {"factory": "distillschedule.examples.tiny_av:build_model",
                        "parameters": {"student_width": 12, "fake_width": 16, "teacher_width": 24,
                                       "student_lr": .002, "fake_lr": .003}},
              "data": {"prepared": "prepared/manifest.json"},
              "algorithm": {"factory": "distillschedule.examples.tiny_av:build_algorithm",
                            "parameters": {"fake_updates": 2, "fake_sample_policy": "reuse_student",
                                           "schedule_steps": 16, "guidance_scale": 1.5}},
              "execution": {"mode": "reference"},
              "trainer": {"output_dir": "run", "cycles": 4, "seed": 7,
                          "checkpoint_every": 2, "device": "cpu"}}
    training_config = output / "train.json"
    training_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return {"quality_status": QUALITY_STATUS, "raw_manifest": str(raw_manifest),
            "training_config": str(training_config)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args(argv)
    print(json.dumps(build_fixture(args.output, samples=args.samples), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
