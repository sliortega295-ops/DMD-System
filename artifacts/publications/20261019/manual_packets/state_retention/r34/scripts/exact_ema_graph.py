"""Manual opt-in. Original FP32 EMA recurrence/kernels, ten known steps only."""
from contextlib import contextmanager

def schedule(ema,count=10):
    if type(count)is not int or count!=10 or type(ema.optimization_step)is not int or ema.optimization_step!=10:
        raise ValueError('exact original two-cycle EMA10..20 contract required')
    if getattr(ema,'implementation',None)!='reference':raise ValueError('original reference EMA required')
    return {step:float(ema.get_decay(step))for step in range(11,21)}

def identities(ema):
    return tuple((p.data_ptr(),tuple(p.shape),str(p.dtype),str(p.device),s.data_ptr(),tuple(s.shape),str(s.dtype),str(s.device))for p,s in zip(ema._parameters,ema._shadows,strict=True))

class ExactEMAStep:
    """Pure host version/identity driver, separately CPU-testable with real math."""
    def __init__(self,ema,graphs,receipt):
        self.ema,self.graphs,self.receipt=ema,graphs,receipt
        self.decays=schedule(ema);self.expected=11;self.addresses=identities(ema)
        if set(graphs)!=set(self.decays):raise ValueError('incomplete exact decay graph set')

    def step(self):
        ema=self.ema;step=ema.optimization_step+1
        if step!=self.expected or step not in self.decays:raise ValueError('EMA graph skipped/repeated/extra update')
        if identities(ema)!=self.addresses:raise ValueError('EMA/source storage dtype/shape/device changed')
        decay=float(ema.get_decay(step))
        if decay!=self.decays[step]:raise ValueError('EMA decay schedule changed after capture')
        ema.optimization_step=step;ema.cur_decay_value=decay;ema.last_transfer_bytes=0
        self.graphs[step].replay()
        self.expected+=1;self.receipt['graph_replays']+=1;self.receipt['observed_steps'].append(step)
        return decay

def capture(ema,torch,receipt):
    from distillschedule.runtime.fsdp2_static import ema_update_reference
    decays=schedule(ema)
    if not ema._parameters or len(ema._parameters)!=len(ema._shadows):raise ValueError('incomplete original EMA state')
    for p,s in zip(ema._parameters,ema._shadows,strict=True):
        if type(p)is not torch.nn.Parameter or type(s)is not torch.Tensor or not p.is_cuda or not s.is_cuda or p.dtype!=torch.float32 or s.dtype!=torch.float32 or p.shape!=s.shape:
            raise ValueError('complete original Resident FP32 EMA, no DTensor/offload/fused path')
    # Warm up only pointwise dispatch on tiny dummy state. Never perform an
    # extra update of the real shadow or consume training RNG.
    dummy=torch.zeros(1,device=ema._shadows[0].device,dtype=torch.float32)
    ema_update_reference(dummy,dummy,decays[11])
    torch.cuda.synchronize(dummy.device)
    pool=torch.cuda.graph_pool_handle();stream=torch.cuda.Stream(device=dummy.device);graphs={}
    stream.wait_stream(torch.cuda.current_stream(dummy.device))
    for step,decay in decays.items():
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,pool=pool,stream=stream):
            for shadow,parameter in zip(ema._shadows,ema._parameters,strict=True):
                ema_update_reference(shadow,parameter.detach().float(),decay)
        if graph.pool()!=pool:raise ValueError('EMA did not use its own sequential shared pool')
        graphs[step]=graph
    torch.cuda.current_stream(dummy.device).wait_stream(stream)
    torch.cuda.synchronize(dummy.device)
    receipt.update(captured_graphs=len(graphs),tensor_count=len(ema._parameters),capture_steps=list(graphs),decays={str(k):v for k,v in decays.items()},capture_replay_order='SAME_ASCENDING11_TO20_SERIAL_ORIGINAL_CONSUMER_STREAM',math='ORIGINAL_EMA_UPDATE_REFERENCE_SUB_MUL_SUB_FP32_NOT_FOREACH',weights_merged=False)
    driver=ExactEMAStep(ema,graphs,receipt)
    driver.stream_identity=(str(torch.cuda.current_stream(dummy.device).device),torch.cuda.current_stream(dummy.device).cuda_stream)
    return driver

@contextmanager
def explicit_worker(worker,*,enabled,source_sha256,packet_sha256):
    original=worker.OriginalWanSession
    class Session(original):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            import torch
            ema=self.ema
            self._ema_graph_cycle=0
            receipt=dict(schema_version='dmd.wan_exact_ema_graph.v1',enabled=enabled,source_sha256=source_sha256,packet_sha256=packet_sha256,graph_replays=0,reference_steps=0,captured_graphs=0,observed_steps=[],contract_failures=0,planner=False,weights_merged=False,optimizer_changed=False,math_changed=False)
            self._ema_graph_receipt=receipt;old_step=ema.step
            driver=capture(ema,torch,receipt)if enabled else None
            def step():
                try:
                    if driver is not None:
                        current=torch.cuda.current_stream(self.device)
                        if (str(current.device),current.cuda_stream)!=driver.stream_identity:raise ValueError('shared EMA graph pool changed consumer stream')
                        return driver.step()
                    value=old_step();receipt['reference_steps']+=1;receipt['observed_steps'].append(ema.optimization_step);return value
                except BaseException:receipt['contract_failures']+=1;raise
            ema.step=step
            for role in ('generator','critic'):self.build[role]['metadata']['exact_ema_graph_experiment']=receipt

        def prepare(self,**kwargs):
            self._ema_graph_cycle+=1
            if self._ema_graph_cycle not in (1,2):raise ValueError('exact EMA graph only admits two original cycles')
            prepared=super().prepare(**kwargs);cycle=self._ema_graph_cycle;owner=self
            class Prepared:
                def run(self):
                    value=prepared.run();r=owner._ema_graph_receipt
                    expected=list(range(11,11+5*cycle))
                    if r['observed_steps']!=expected or owner.ema.optimization_step!=10+5*cycle or r['contract_failures']:
                        raise ValueError('actual EMA execution/update census failed')
                    if r['graph_replays']!=(5*cycle if enabled else 0)or r['reference_steps']!=(0 if enabled else 5*cycle):
                        raise ValueError('wrong real EMA implementation executed')
                    return value
                def receipt(self):return prepared.receipt()
            return Prepared()
    worker.OriginalWanSession=Session
    try:yield
    finally:worker.OriginalWanSession=original
