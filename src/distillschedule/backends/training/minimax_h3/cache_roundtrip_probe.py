"""Default-off full-payload cache fingerprints; never a numerical admission."""
from __future__ import annotations
import hashlib
from pathlib import Path

CONTRACT=dict(schema='h3.fa3_cache_roundtrip_fingerprint.v1',cycles=2,graphs_per_cycle=6,
    blocks=50,tensors_per_block=2,pairs_per_rank=1200,
    scope='EVERY_CYCLE_GRAD_GRAPH_OUT_AND_LSE_ORIGINAL_VS_REPLAY',
    training_values_and_gradients='ORIGINAL_UNMODIFIED',RNG_consumption=0,
    host_value_reads='ONLY_AFTER_EXISTING_CYCLE_STOP_FENCE',timing_eligible=False,
    fingerprint='TWO_INDEX_MIXED_UINT32_REDUCTIONS_FULL_LOGICAL_PAYLOAD',
    claim_boundary='MATCHING_FINGERPRINTS_NOT_CRYPTOGRAPHIC_OR_FULL_BITWISE_EQUALITY_PROOF')


def source_contract():
    root=Path(__file__).parent
    return dict(CONTRACT,source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ('cache_roundtrip_probe.py','cache_roundtrip_kernels.py')})


def _mix(x):
    x&=0xffffffff;x^=x>>16;x=(x*0x7feb352d)&0xffffffff
    x^=x>>15;x=(x*0x846ca68b)&0xffffffff
    return (x^(x>>16))&0xffffffff


def reference_words(bits):
    a,b=0,0
    for i,value in enumerate(bits):
        if type(value)is not int or not 0<=value<=0xffffffff:
            raise ValueError('ROUNDTRIP_REFERENCE_EXPECTS_INTEGER_BITS')
        a^=_mix(value^_mix(i+0x9e3779b9))
        b=(b+_mix(value+_mix(i^0x85ebca6b)))&0xffffffff
    return [a,b]


def cpu_fingerprint(torch,value):
    if value.device.type!='cpu' or value.dtype not in (torch.bfloat16,torch.float32):
        raise ValueError('ROUNDTRIP_REFERENCE_CPU_ONLY')
    bits=value.detach().contiguous().view(torch.int16 if value.dtype==torch.bfloat16 else torch.int32)
    mask=0xffff if value.dtype==torch.bfloat16 else 0xffffffff
    return reference_words([int(x)&mask for x in bits.reshape(-1).tolist()])


def gpu_selftest(torch,device):
    from .cache_roundtrip_kernels import fingerprint
    before=(torch.get_rng_state().clone(),torch.cuda.get_rng_state(device).clone())
    cases=[]
    for dtype in (torch.bfloat16,torch.float32):
        source=(torch.arange(3*17*128,dtype=torch.float32).view(1,3,17,128)/257).to(dtype)
        source.view(-1)[:6]=torch.tensor([0.,-0.,1.,-1.,float('inf'),float('nan')],dtype=dtype)
        for value in (source,source.transpose(1,2),source[0]):
            expected=cpu_fingerprint(torch,value)
            cuda=value.to(device);actual=fingerprint(torch,cuda).cpu().tolist()
            if actual!=expected:raise ValueError('ROUNDTRIP_GPU_KNOWN_BITS_SELFTEST_FAILED')
            changed=value.clone();changed[(0,)*changed.ndim]=3.25
            mutated=fingerprint(torch,changed.to(device)).cpu().tolist()
            if mutated!=cpu_fingerprint(torch,changed) or mutated==actual:
                raise ValueError('ROUNDTRIP_GPU_MUTATION_SELFTEST_FAILED')
            cases.append(dict(dtype=str(dtype),shape=list(value.shape),stride=list(value.stride()),
                              expected=expected,actual=actual,mutation_detected=True))
    if not torch.equal(before[0],torch.get_rng_state()) or not torch.equal(before[1],torch.cuda.get_rng_state(device)):
        raise ValueError('ROUNDTRIP_SELFTEST_CONSUMED_TRAINING_RNG')
    return dict(status='PASS_GPU_KNOWN_BITS_AND_STRIDES',cases=cases,RNG_unchanged=True)


class RoundTripProbe:
    def __init__(self,torch,*,digest=None):
        self.torch=torch;self.cycle=0;self.groups=[];self.completed=[];self.selftest=None
        self.digest=digest

    def begin(self,cycle):
        if cycle!=len(self.completed)+1 or cycle not in (1,2) or self.groups:
            raise ValueError('ROUNDTRIP_CYCLE_ORDER_CHANGED')
        self.cycle=cycle

    def capture(self,stage,block,values):
        if stage=='backward':return
        if stage not in ('original','replay') or self.cycle not in (1,2) or len(values)!=2:
            raise ValueError('ROUNDTRIP_STAGE_OR_ARITY_CHANGED')
        if stage=='original'and block==0:
            if len(self.groups)>=6 or (self.groups and self.groups[-1]['replay_next']!=-1):
                raise ValueError('ROUNDTRIP_OVERLAPPING_OR_EXTRA_GRAPH')
            self.groups.append(dict(original_next=0,replay_next=49,items={}))
        if not self.groups:raise ValueError('ROUNDTRIP_MISSING_GRAPH')
        group=self.groups[-1];key=stage+'_next'
        if group[key]!=block:raise ValueError('ROUNDTRIP_BLOCK_ORDER_CHANGED')
        group[key]+=1 if stage=='original'else -1
        if stage=='replay'and group['original_next']!=50:
            raise ValueError('ROUNDTRIP_REPLAY_BEFORE_FORWARD_COMPLETION')
        if self.digest is None:
            from .cache_roundtrip_kernels import fingerprint
            self.digest=lambda value:fingerprint(self.torch,value)
        for index,value in enumerate(values):
            identity=(stage,block,index)
            if identity in group['items']:raise ValueError('ROUNDTRIP_DUPLICATE_TENSOR')
            group['items'][identity]=dict(shape=list(value.shape),stride=list(value.stride()),
                dtype=str(value.dtype),numel=value.numel(),digest=self.digest(value))

    def finish_cycle(self,*,outside_cycle):
        if not outside_cycle or len(self.groups)!=6 or any(g['original_next']!=50 or g['replay_next']!=-1 for g in self.groups):
            raise ValueError('ROUNDTRIP_INCOMPLETE_CYCLE_OR_UNSAFE_HOST_READ')
        pending=[];metadata=[]
        for graph,group in enumerate(self.groups):
            if len(group['items'])!=200:raise ValueError('ROUNDTRIP_INCOMPLETE_TENSOR_CENSUS')
            for block in range(50):
                for tensor in range(2):
                    original=group['items']['original',block,tensor]
                    replay=group['items']['replay',block,tensor]
                    if any(original[k]!=replay[k]for k in ('shape','dtype','numel')):
                        raise ValueError('ROUNDTRIP_PAYLOAD_METADATA_CHANGED')
                    pending.extend((original['digest'],replay['digest']))
                    metadata.append(dict(graph=graph,block=block,tensor=('out','lse')[tensor],
                        shape=original['shape'],dtype=original['dtype'],numel=original['numel'],
                        original_stride=original['stride'],replay_stride=replay['stride']))
        # One small host copy after the existing fence, never a full payload copy.
        values=self.torch.stack(pending).cpu().tolist()
        for i,row in enumerate(metadata):
            row.update(original=values[2*i],replay=values[2*i+1],fingerprint_match=values[2*i]==values[2*i+1])
        result=dict(cycle=self.cycle,pairs=len(metadata),mismatches=sum(not r['fingerprint_match']for r in metadata),rows=metadata)
        self.completed.append(result);self.groups=[]
        return result

    def receipt(self):
        if len(self.completed)!=2 or self.groups or any(r['pairs']!=600 for r in self.completed):
            raise ValueError('ROUNDTRIP_FINAL_CENSUS_CHANGED')
        return dict(contract=source_contract(),status='MATCHED_FINGERPRINTS_NOT_NUMERICAL_ADMISSION'
            if not any(r['mismatches']for r in self.completed)else 'MISMATCH',
            pairs=1200,cycles=self.completed,selftest=self.selftest,numerical_admission=False,formal_timing_eligible=False)
