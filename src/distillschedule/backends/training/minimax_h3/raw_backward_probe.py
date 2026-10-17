"""Default-off diagnostic; repeat one actual FA3 backward, return original grads.

No Torch import/global hook/RNG consumption. All scalar tensor reads happen only
at the original diagnostic finish fence. This cannot authorize native timing.
"""
from __future__ import annotations

CONTRACT = dict(schema='h3.same_inputs_raw_backward.v1', rank=0, block=49,
    nondeterministic_extra_calls=3, deterministic_calls=2,
    training_gradients='ORIGINAL_UNMODIFIED', full_tensor_metrics=True,
    copies='NO_EXTRA_HOST_TENSOR_VALUE_READS_INSIDE_CYCLE', timing_eligible=False)


def metrics(torch, reference, value, *, chunk=1048576):
    if reference.shape != value.shape or reference.dtype != value.dtype or reference.device != value.device:
        raise RuntimeError('RAW_REPEAT_OUTPUT_METADATA_CHANGED')
    # Advanced gather is bounded even for strided native Q/K/V gradients;
    # reshape(-1) must not allocate an unexpected full contiguous copy.
    def part(tensor,start,stop):
        if tensor.is_contiguous():return tensor.detach().view(-1)[start:stop]
        index=torch.arange(start,stop,device=tensor.device,dtype=torch.int64)
        coordinates=[]
        for axis,size in enumerate(tensor.shape):
            stride=1
            for n in tensor.shape[axis+1:]:stride*=n
            coordinates.append((index//stride)%size)
        return tensor.detach()[tuple(coordinates)]
    total=torch.zeros(4,device=reference.device,dtype=torch.float64)
    for i in range(0,reference.numel(),chunk):
        raw_a=part(reference,i,i+min(chunk,reference.numel()-i))
        raw_b=part(value,i,i+min(chunk,reference.numel()-i))
        a=raw_a.float();b=raw_b.float();d=b-a
        total[0].add_((d*d).sum(dtype=torch.float64))
        total[1].add_((a*a).sum(dtype=torch.float64))
        total[2].add_(torch.count_nonzero(raw_b!=raw_a))
        total[3].add_(torch.count_nonzero(~torch.isfinite(b)))
    return dict(shape=list(reference.shape),dtype=str(reference.dtype),numel=reference.numel(),device_totals=total)


def run(torch, raw_backward, inputs, primary, scale):
    if len(inputs)!=6 or len(primary)!=3:
        raise RuntimeError('RAW_REPEAT_EXPECT_REAL_QKV_DOUT_OUT_LSE')
    dout,q,k,v,out,lse=inputs
    identities=[dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),
                     device=str(t.device),pointer=int(t.data_ptr()),version=int(t._version)) for t in inputs]
    rows=[]
    def call(deterministic):
        gradients=[torch.empty_like(t) for t in (q,k,v)]
        raw_backward(dout,q,k,v,out,lse,None,None,None,None,None,None,*gradients,
                     scale,False,-1,-1,0.0,deterministic,0)
        return gradients
    with torch.no_grad():
        for index in range(CONTRACT['nondeterministic_extra_calls']):
            trial=call(False)
            rows.append(dict(kind='same_input_nondeterministic_vs_original',index=index,
                gradients=[metrics(torch,a,b) for a,b in zip(primary,trial)]))
            del trial
        first=call(True)
        rows.append(dict(kind='deterministic_vs_original',index=0,
            gradients=[metrics(torch,a,b) for a,b in zip(primary,first)]))
        second=call(True)
        rows.append(dict(kind='same_input_deterministic_repeat',index=1,
            gradients=[metrics(torch,a,b) for a,b in zip(first,second)]))
        del first,second
    if any(t._version!=identity['version'] for t,identity in zip(inputs,identities)):
        raise RuntimeError('RAW_REPEAT_INPUT_VERSION_MUTATED')
    return dict(contract=dict(CONTRACT),inputs=identities,rows=rows,
                outputs_are_training_gradients=False,native_timing='NOT_RUN_DIAGNOSTIC_EXTRA_KERNELS')


def finalize(result):
    import math
    value=dict(result,rows=[])
    for row in result['rows']:
        item=dict(row,gradients=[])
        for gradient in row['gradients']:
            squared_error,squared_norm,different,nonfinite=gradient['device_totals'].cpu().tolist()
            item['gradients'].append({**{k:v for k,v in gradient.items() if k!='device_totals'},
                'squared_error':squared_error,'squared_reference_norm':squared_norm,
                'normalized_l2':math.sqrt(squared_error/max(squared_norm,1e-300)),
                'unequal_elements':int(different),'nonfinite_elements':int(nonfinite),
                'bitwise_equal':different==0})
        value['rows'].append(item)
    return value
