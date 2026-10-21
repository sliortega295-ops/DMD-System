"""External CPU workflow: previous-cycle features, post-cycle EMA and schedulers.

This is actual tiny Student1/FakeN DMD with a RANDOM frozen Teacher, not W1-W6,
pretrained quality, GPU validation or performance evidence. No solver changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from distillschedule.examples import tiny_av
from distillschedule.open_workload.dmd import dmd_inputs
from distillschedule.workloads.program import Call

encode = tiny_av.encode


def build_model(parameters):
    allowed = {"device", "seed", "student_width", "fake_width", "teacher_width",
               "student_lr", "fake_lr", "scheduler_gamma"}
    if set(parameters) - allowed:
        raise ValueError("unknown newworkflow model parameters")
    gamma = _bounded(parameters.get("scheduler_gamma", .95), "scheduler_gamma", 0, 1)
    components = tiny_av.build_model({key: value for key, value in parameters.items() if key != "scheduler_gamma"})
    state = components.state
    state.update(student_ema={name: parameter.detach().clone()
                              for name, parameter in state["student"].named_parameters()},
                 history=torch.zeros(1, 4, 1, 1, 1),
                 student_scheduler=torch.optim.lr_scheduler.ExponentialLR(state["student_optimizer"], gamma=gamma),
                 fake_scheduler=torch.optim.lr_scheduler.ExponentialLR(state["fake_optimizer"], gamma=gamma))
    components.metadata = {**components.metadata, "external_workflow": "previous_feature_post_cycle_commit_v1",
                           "ema": "post_cycle_student_parameter_ema", "history": "previous_preupdate_student_channel_mean",
                           "scheduler": "post_cycle_exponential_per_role", "scope": "CPU_SINGLE_PROCESS_ONLY",
                           "generation": "CURRENT_STUDENT_WITH_ZERO_HISTORY_NOT_EMA",
                           "tiny_helper_sha256": hashlib.sha256(Path(tiny_av.__file__).read_bytes()).hexdigest()}
    # Portable generation deliberately uses current Student with the prepared
    # condition and ZERO history, not unexported initial history/EMA. The default
    # exporter includes the named Student/Fake/Teacher modules, not these trees.
    components.validate_batch = validate_batch
    return components.validate()


def _bounded(value, name, lower, upper):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower < value <= upper:
        raise ValueError(f"{name} must be finite in ({lower}, {upper}]")
    return float(value)


def _condition(ctx):
    feeds = dmd_inputs(ctx)["feeds"]
    return feeds["condition"] + ctx.attributes["history_scale"] * ctx.state["history"]


def _student_forward(ctx):
    feeds = dmd_inputs(ctx)["feeds"]
    return ctx.state["student"](feeds["latent"], _condition(ctx))


def _teacher_forward(ctx):
    values = dmd_inputs(ctx)
    feeds, noise = values["feeds"], values["noise"]
    condition = _condition(ctx)
    time = noise["timestep"].float() / feeds["schedule_length"]
    return {"cond": ctx.state["teacher"](noise["noisy"], condition, time),
            "uncond": ctx.state["teacher"](noise["noisy"], torch.zeros_like(condition), time)}


def _fake_forward(ctx):
    values = dmd_inputs(ctx)
    return ctx.state["fake"](values["noise"]["noisy"], _condition(ctx),
                              values["noise"]["timestep"].float() / values["feeds"]["schedule_length"])


def _commit_cycle(ctx):
    state = ctx.state
    for name, parameter in state["student"].named_parameters():
        state["student_ema"][name].lerp_(parameter, 1 - ctx.attributes["ema_decay"])
    # This PRE-update sample is the same detached sample used in this cycle's
    # DMD loss. Commit it only after every real Student/Fake update completes.
    sample = ctx.inputs["dmd.c0.detach"]
    state["history"].copy_(sample.mean(dim=(0, 2, 3, 4), keepdim=True))
    state["student_scheduler"].step()
    state["fake_scheduler"].step()


def build_algorithm(components, feeds, parameters, cycle, seed):
    from distillschedule.training.distributed import rank_world
    if rank_world()[1] != 1:
        raise ValueError("newworkflow has rank-local feature history and is single-process only")
    allowed = {"fake_updates", "fake_sample_policy", "schedule_steps", "guidance_scale", "history_scale", "ema_decay"}
    if set(parameters) - allowed:
        raise ValueError("unknown newworkflow algorithm parameters")
    scale = _bounded(parameters.get("history_scale", .25), "history_scale", 0, 1)
    decay = _bounded(parameters.get("ema_decay", .9), "ema_decay", 0, 1)
    workload = tiny_av.build_algorithm(components, feeds, {key: value for key, value in parameters.items()
                                                          if key not in ("history_scale", "ema_decay")}, cycle, seed)
    # Reuse existing genuine noise/surrogate/critic losses and the established
    # Student1/FakeN graph. Only these three model-owned forwards read history.
    replacements = {"student_forward": _student_forward, "fake_forward": _fake_forward,
                    "teacher_forward": _teacher_forward}
    calls = tuple(replace(call, reads=(*call.reads, "history"),
                          attributes={**call.attributes, "history_scale": scale})
                  if call.binding in replacements else call for call in workload.program.calls)
    writes = ("student_ema", "history", "student_scheduler", "fake_scheduler",
              "student_optimizer", "fake_optimizer")
    commit = Call("workflow.commit_cycle", "commit_cycle", ("dmd.c0.detach",), ("student", *writes), writes,
                  after=tuple(call.id for call in calls), attributes={"ema_decay": decay})
    workload.program = replace(workload.program, calls=(*calls, commit),
                               parameters={**workload.program.parameters, "workflow": "previous_feature_post_cycle_commit_v1",
                                           "history_scale": scale, "ema_decay": decay})
    workload.bindings = {**workload.bindings, **replacements, "commit_cycle": _commit_cycle}
    # Explicit local-only scope, not an equal-rank replicated-history assertion.
    workload.replica_semantics = None
    workload.identity_extra = {**workload.identity_extra, "stateful_workflow": True}
    return workload.validate()


def validate_batch(state, feeds, parameters):
    """Held-out flow agreement with the frozen RANDOM Teacher, not media quality.

    Compare both models on the same held-out feature/condition at t=0. This
    diagnostic is deterministic and independent of training noise/loss samples.
    Use zero history as the explicit portable deployment convention.
    """
    if parameters:
        raise ValueError("newworkflow validate_batch does not accept parameters")
    student = state["student"](feeds["latent"], feeds["condition"])
    teacher = state["teacher"](feeds["latent"], feeds["condition"])
    return {"student_teacher_flow_mse": (student - teacher).square().mean()}


def build_fixture(output, *, samples=6, validation_samples=2):
    """Select disjoint raw records using the existing tiny media fixture/parser."""
    for name, count in (("samples", samples), ("validation_samples", validation_samples)):
        if type(count) is not int or not 1 <= count <= 500:
            raise ValueError(f"{name} must be in [1, 500]")
    output = Path(output).expanduser().resolve()
    tiny_av.build_fixture(output, samples=samples + validation_samples)
    manifest = output / "raw/manifest.json"
    raw = json.loads(manifest.read_text())
    validation_raw = {**raw, "records": raw["records"][samples:]}
    raw["records"] = raw["records"][:samples]
    manifest.write_text(json.dumps(raw, indent=2) + "\n")
    validation_manifest = manifest.with_name("validation-manifest.json")
    validation_manifest.write_text(json.dumps(validation_raw, indent=2) + "\n")
    path = output / "train.json"
    config = json.loads(path.read_text())
    reference = str(Path(__file__).resolve())
    config["model"]["factory"] = reference + ":build_model"
    config["model"]["parameters"]["scheduler_gamma"] = .95
    config["algorithm"]["factory"] = reference + ":build_algorithm"
    config["algorithm"]["parameters"].update(history_scale=.25, ema_decay=.9)
    config["data"]["validation_prepared"] = "validation-prepared/manifest.json"
    config["trainer"].update(cycles=min(4, samples), validation={"every_cycles": 2, "samples": validation_samples})
    path.write_text(json.dumps(config, indent=2) + "\n")
    return {"training_config": str(path), "raw_manifest": str(manifest),
            "validation_raw_manifest": str(validation_manifest), "quality_status": tiny_av.QUALITY_STATUS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--validation-samples", type=int, default=2)
    args = parser.parse_args(argv)
    print(json.dumps(build_fixture(args.output, samples=args.samples, validation_samples=args.validation_samples), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
