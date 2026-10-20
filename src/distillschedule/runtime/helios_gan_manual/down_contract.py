"""Stdlib-only full native C-scope contract; never a prefix guard relaxation."""
from __future__ import annotations
from dataclasses import dataclass

HEAD='ba52a2446b7f090cd31af9504679e68770d32603'
FA3='_flash_attn3_cuda_373a45c._flash_attn_forward.default'
GRAPHS=('B1_DENOISE','B3_GAN_R1')
MIN_DRIVER_FREE=2<<30

def require(ok,why):
    if not ok: raise ValueError(why)

@dataclass(frozen=True)
class ScopeKey:
    rank:int
    cycle:int
    index:int
    logical_step:int
    version:int
    graph:str
    block:int
    role:str='critic'
    stage:str='CRITIC'

    def validate(self):
        require(all(type(getattr(self,n))is int for n in('rank','cycle','index','logical_step','version','block')),'EXACT_SCOPE_INTEGER_TYPES')
        require(0<=self.rank<16 and self.role=='critic'and self.stage=='CRITIC','NATIVE_C_TRAINING_ONLY')
        require(self.cycle in(1,2)and 0<=self.index<5 and self.logical_step==1000+self.index,'EXACT_CYCLE_INDEX_SOURCE_STEP')
        require(self.version==10+5*(self.cycle-1)+self.index,'EXACT_NATIVE_C_VERSION_10_TO_19')
        require(self.graph in GRAPHS and 0<=self.block<40,'EXACT_GRAPH_BLOCK')
        return self

@dataclass(frozen=True)
class TensorMeta:
    shape:tuple
    stride:tuple
    dtype:str
    device:str
    requires_grad:bool
    storage_id:int
    storage_offset:int

@dataclass(frozen=True)
class Leaf:
    module:str
    input:TensorMeta
    weight:TensorMeta
    bias:TensorMeta

def select_down(key,leaf,args,kwargs,*,rows):
    key.validate()
    require(type(rows)is int and rows>0,'SOURCE_ATTESTED_ROWS_REQUIRED')
    require(type(leaf)is Leaf and leaf.module=='critic.blocks.'+str(key.block)+'.ffn.net.2.base_layer','ACTUAL_NAMED_DOWN_OWNERSHIP')
    require(len(args)==3 and set(kwargs)<={'alpha','beta','out'}and not kwargs.get('out'),'FUNCTIONAL_ADDMM_ABI')
    require(kwargs.get('alpha',1)==kwargs.get('beta',1)==1,'ORIGINAL_ADDMM_SCALING')
    require(leaf.weight.shape==(5120,13824)and leaf.bias.shape==(5120,),'NAMED_DOWN_BASE_GEOMETRY')
    require(not leaf.weight.requires_grad and not leaf.bias.requires_grad,'FROZEN_BASE_ONLY')
    require(leaf.weight.device==f'cuda:{key.rank%8}','EXACT_LOCAL_RANK_CUDA_DEVICE')
    bias,value,mat2=args
    require(value.shape==(rows,13824)and mat2.shape==(13824,5120),'SOURCE_DOWN_OPERAND_ROWS_GEOMETRY')
    require(bias==leaf.bias,'ACTUAL_BIAS_BINDING')
    require(mat2.storage_id==leaf.weight.storage_id and mat2.storage_offset==leaf.weight.storage_offset
        and mat2.stride==tuple(reversed(leaf.weight.stride)),'FROZEN_TRANSPOSE_VIEW_BINDING')
    require(value.storage_id==leaf.input.storage_id and value.storage_offset==leaf.input.storage_offset
        and value.shape[-1]==leaf.input.shape[-1],'ACTUAL_INPUT_STORAGE_BINDING')
    require(all(m.dtype=='torch.bfloat16'and m.device==leaf.weight.device for m in(*args,leaf.input,leaf.weight,leaf.bias)),'ONE_DEVICE_BF16_REQUIRED')
    require(not mat2.requires_grad,'NO_TRAINABLE_WEIGHT_CACHE')
    return True

def plan():
    return dict(status='PLAN_SOURCE_ONLY_DEFAULT_OFF_NOT_ADMITTED',source_head=HEAD,default_enabled=False,
        versions=list(range(10,20)),cycles=2,C_updates=10,graphs=20,combined_contexts=800,
        original_even_training=dict(contexts=400,save=800,reuse=800),extra_FA3=0,
        down=dict(original=800,replay=800),C_raw=dict(FA2_FW=0,FA2_BW=0,FA3_FW=2400,FA3_BW=1600),
        expected_physical_versions=[dict(generator=2,critic=10,ema=800),dict(generator=3,critic=15,ema=801),dict(generator=4,critic=20,ema=802)],
        counts='SOURCE_EXPECTED_NOT_NATIVE_MEASURED',GPU='NOT_RUN',SSH='NOT_RUN',formal_eligible=False)
