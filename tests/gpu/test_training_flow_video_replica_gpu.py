"""Opt-in actual tiny native video FP32 NCCL2; SYNTHETIC_TEST_ONLY, no quality/timing claim."""
import os
import pytest
import torch

from test_training_flow_video_replicated import native_video_replica_case


@pytest.mark.skipif(os.environ.get('DMD_NATIVE_FLOW_REPLICATED') != '1' or torch.cuda.device_count() < 2,
    reason='native video NCCL2 NOT_RUN: requires opt-in DMD_NATIVE_FLOW_REPLICATED=1 and two owned GPUs')
@pytest.mark.parametrize('module',['wan22','hunyuan_video'])
def test_actual_native_video_nccl2_independent_oracle_and_fresh_cli(tmp_path,module):
    native_video_replica_case(tmp_path,module,'cuda')
