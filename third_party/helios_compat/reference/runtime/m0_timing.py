from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterable


def load_phase_events(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def summarize_five_step_timing(
    events: Iterable[dict[str, Any]], *, expected_start: int, expected_steps: int = 10
) -> dict[str, Any]:
    if expected_steps <= 0 or expected_steps % 5:
        raise ValueError("expected_steps must be a positive multiple of five")
    ends = [event for event in events if event.get("phase") == "training_step_end"]
    by_step: dict[int, dict[str, Any]] = {}
    for event in ends:
        step = int(event["global_step"])
        if step in by_step:
            raise ValueError(f"duplicate training_step_end for step {step}")
        by_step[step] = event
    expected = list(range(expected_start, expected_start + expected_steps))
    if sorted(by_step) != expected:
        raise ValueError(f"step mismatch: expected {expected}, got {sorted(by_step)}")

    rows: list[dict[str, Any]] = []
    for step in expected:
        event = by_step[step]
        elapsed = float(event["elapsed_seconds"])
        data_wait = float(event["data_wait_seconds"])
        if elapsed < 0 or data_wait < 0:
            raise ValueError(f"negative duration at step {step}")
        train_generator = bool(event["train_generator"])
        expected_generator = (step - expected_start) % 5 == 0
        if train_generator != expected_generator:
            raise ValueError(
                f"update schedule mismatch at step {step}: "
                f"train_generator={train_generator}, expected={expected_generator}"
            )
        rows.append(
            {
                "global_step": step,
                "train_generator": train_generator,
                "device_synchronized_seconds": elapsed,
                "data_wait_seconds": data_wait,
                "end_to_end_seconds": elapsed + data_wait,
            }
        )

    generator = [row["end_to_end_seconds"] for row in rows if row["train_generator"]]
    critic = [row["end_to_end_seconds"] for row in rows if not row["train_generator"]]
    cycles = []
    for offset in range(0, expected_steps, 5):
        group = rows[offset : offset + 5]
        cycles.append(
            {
                "start_step": group[0]["global_step"],
                "end_step": group[-1]["global_step"],
                "seconds": sum(row["end_to_end_seconds"] for row in group),
            }
        )
    g_mean = statistics.fmean(generator)
    c_mean = statistics.fmean(critic)
    weighted = (g_mean + 4.0 * c_mean) / 5.0
    return {
        "expected_start": expected_start,
        "expected_steps": expected_steps,
        "per_step": rows,
        "generator_update_step_seconds_mean": g_mean,
        "critic_only_step_seconds_mean": c_mean,
        "official_weighted_step_seconds": weighted,
        "samples_per_second": 1.0 / weighted,
        "gpu_seconds_per_sample": weighted,
        "cycle_seconds": cycles,
        "cycle_seconds_mean": statistics.fmean(item["seconds"] for item in cycles),
        "data_wait_seconds_total": sum(row["data_wait_seconds"] for row in rows),
        "device_synchronized_seconds_total": sum(row["device_synchronized_seconds"] for row in rows),
        "end_to_end_seconds_total": sum(row["end_to_end_seconds"] for row in rows),
    }
