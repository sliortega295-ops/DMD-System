"""Real pre-model transport CLI accepts child and preserves all C51 subgates."""
from pathlib import Path
import ast,importlib.util,sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'scripts/distillschedule/probe_livewan_crossscale_transport.py'
CONTROL='c51-continuous-b16-generator-fused-rms'
CHILD='c52-c51-critic-ffn-down-retention-v1'

def test_actual_production_transport_parser_accepts_both_arms_without_cuda(tmp_path):
 spec=importlib.util.spec_from_file_location('actual_transport_probe',PATH);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 for variant in (CONTROL,CHILD):
  args=m.build_parser().parse_args(['--output-dir',str(tmp_path),'--pipeline-variant',variant])
  assert args.pipeline_variant==variant and args.timeout_seconds==180
 with pytest.raises(SystemExit):m.build_parser().parse_args(['--output-dir',str(tmp_path),'--pipeline-variant','unknown-unsafe-candidate'])
 assert not tmp_path.exists() or not list(tmp_path.iterdir())

def test_child_executes_every_parent_transport_subgate_not_just_argparse():
 tree=ast.parse(PATH.read_text());groups=[]
 for node in ast.walk(tree):
  if isinstance(node,(ast.Set,ast.Tuple,ast.List)):
   values=[x.value for x in node.elts if isinstance(x,ast.Constant)and isinstance(x.value,str)]
   if CONTROL in values:
    assert CHILD in values
    groups.append(values)
 assert len(groups)>=6  # parser, Gloo, paired shadow, AR, P2P ordering, trace pairing


def test_complete_launcher_has_no_unmapped_c51_variant_list_for_new_child():
 paths=['run_livewan_crossscale_world16.py','launch_livewan_crossscale_world16_node.sh','summarize_livewan_trajectory_world16.py','summarize_livewan_two_cycle_timing_world16.py']
 for name in paths:
  text=(ROOT/'scripts/distillschedule'/name).read_text();assert CONTROL in text and CHILD in text
