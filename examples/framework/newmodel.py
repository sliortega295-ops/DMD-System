"""Copyable external model adapter, not a model registry or a new solver.

This runnable template deliberately retains the tiny encoder/latent contract.
Replace its networks, encoder, generation/decode and DMD hooks for your own
model. Never describe these random tiny weights as a pretrained AV model.
"""
from __future__ import annotations

import torch
from torch import nn

from distillschedule.examples.tiny_av import (
    QUALITY_STATUS, build_algorithm, decode, encode, generate,
)
from distillschedule.training.adapters import ModelComponents

__all__ = ["build_model", "build_algorithm", "encode"]


class NewModel(nn.Module):
    """Example replacement architecture with the same four-channel I/O ABI."""

    def __init__(self, width):
        super().__init__()
        self.input = nn.Linear(9, width)
        self.output = nn.Linear(width, 4)

    def forward(self, value, condition, timestep=None):
        time = value.new_zeros(value.shape[0]) if timestep is None else timestep.to(value)
        time = time.reshape(-1, 1, 1, 1, 1).expand(value.shape[0], 1, *value.shape[2:])
        features = torch.cat((value, condition.expand_as(value), time), dim=1).movedim(1, -1)
        return self.output(torch.nn.functional.silu(self.input(features))).movedim(-1, 1).contiguous()


def build_model(parameters):
    """Own model construction and optimizer math; placement lives in config."""
    if parameters.get("device", "cpu") != "cpu":
        raise ValueError("this copyable tiny template is CPU-only until you adapt it")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(parameters.get("seed", 7))
        student = NewModel(parameters.get("student_width", 10))
        fake = NewModel(parameters.get("fake_width", 14))
        teacher = NewModel(parameters.get("teacher_width", 20))
    teacher.requires_grad_(False).eval()
    return ModelComponents(
        state={"student": student, "fake": fake, "teacher": teacher,
               "student_optimizer": torch.optim.AdamW(student.parameters(), lr=parameters.get("student_lr", .002)),
               "fake_optimizer": torch.optim.AdamW(fake.parameters(), lr=parameters.get("fake_lr", .003))},
        immutable_modules=("teacher",),
        metadata={"quality_status": QUALITY_STATUS, "pretrained": False,
                  "external_template": True, "ema": False},
        generate=generate, decode=decode,
    ).validate()


# build_algorithm(components, feeds, parameters, cycle, seed) is imported above.
# It returns the existing open-workload Workload for ONE Student1/FakeN cycle.
# Keep it only while this exact noise, detached score and optimizer contract fits
# your algorithm. Otherwise supply your own DMDHooks or Program/Call graph.
# Neither factory chooses checkpointing, layouts, planners or solver modes.
