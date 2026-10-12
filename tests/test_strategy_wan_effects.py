from copy import deepcopy
import pytest
from distillschedule.strategies.wan_strategy import WanStrategy
from distillschedule.strategies.wan_evidence import validate_build


def build(strategy):
    sync='WORLD16_GRAD_READY_ASYNC_SUM_THEN_DIVIDE_ALL_REDUCE' if strategy.placement=='resident' else 'FSDP2_SUM_THEN_DIVIDE_REDUCE_SCATTER'
    return {role:{'engine':{'gradient_sync':sync},'placement':{
        'param_compute_dtype_policy':'PRESERVE_ORIGINAL_PER_PARAMETER','trainable_parameter_dtypes':['torch.float32'],
        'frozen_parameter_dtypes':['torch.bfloat16']},'metadata':{
        'modulation_chunk_runtime':{'effective_tokens':[strategy.chunk_tokens]},
        'generator_nograd_lora_fused_epilogue':{'requested':strategy.lora_epilogue,
         'enabled':strategy.lora_epilogue and role=='generator','fused_invocation_count':10,'fused_element_count':100}}}
        for role in ('generator','critic')}


@pytest.mark.parametrize('strategy',[WanStrategy('fsdp',512,False),WanStrategy('resident',2048,True)])
def test_declared_strategy_must_match_actual_backend_receipt(strategy):
    receipt=build(strategy)
    assert validate_build(strategy,receipt,executed=True)['status']=='PASS_BACKEND_EFFECTS'
    bad=deepcopy(receipt);bad['generator']['engine']['gradient_sync']='ordinary synchronous'
    with pytest.raises(ValueError,match='communication'):validate_build(strategy,bad)
    bad=deepcopy(receipt);bad['critic']['metadata']['modulation_chunk_runtime']['effective_tokens']=[17]
    with pytest.raises(ValueError,match='chunk'):validate_build(strategy,bad)


def test_fusion_flag_without_actual_fused_work_rejected():
    s=WanStrategy('resident',2048,True);b=build(s)
    b['generator']['metadata']['generator_nograd_lora_fused_epilogue']['fused_invocation_count']=0
    with pytest.raises(ValueError,match='no elements'):validate_build(s,b,executed=True)
