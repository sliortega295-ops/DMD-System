"""The finite-stream recipe uses the public CLI grammar, without dispatch."""
import argparse
from pathlib import Path
import shlex

import pytest

from distillschedule.cli import main


def test_incumbent_stream_recipe_commands_parse_without_execution(monkeypatch):
    recipe = Path(__file__).resolve().parents[1] / "recipes/wan_ar/INCUMBENT_STREAM.md"
    commands = [shlex.split(line)[1:] for line in recipe.read_text().splitlines()
                if line.startswith("distillctl ")]
    config = Path("/path/to/wan-stream.json")
    expected = [
        {"command": "config", "action": "validate", "config": config},
        {"command": "doctor", "config": config},
        {"command": "train", "config": config, "dry_run": True, "resume": None},
        {"command": "train", "config": config, "dry_run": False, "resume": None},
        {"command": "train", "config": config, "dry_run": False,
         "resume": Path("/path/to/run/checkpoints/cycle-000001"),
         "output": Path("/path/to/resumed-run")},
    ]
    assert len(commands) == len(expected), "Keep all five public lifecycle examples executable."

    class ParsedOnly(Exception):
        pass

    parse_args = argparse.ArgumentParser.parse_args
    parsed = []

    def parse_without_dispatch(parser, args=None, namespace=None):
        parsed.append(parse_args(parser, args, namespace))
        # Stop before config loading, model imports, native admission or execution.
        raise ParsedOnly

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_without_dispatch)
    for command, fields in zip(commands, expected):
        with pytest.raises(ParsedOnly):
            main(command)
        assert {key: getattr(parsed[-1], key) for key in fields} == fields
    assert len(parsed) == len(expected)
