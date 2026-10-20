"""Manual opt-in only. Original host action order and mathematics are unchanged."""
from contextlib import contextmanager,nullcontext

def lookahead_order():
    return ('g','r0','r1','c0','e0','r2','c1','e1','r3','c2','e2','r4','c3','e3','c4','e4')


class StreamExecution:
    def __init__(self,torch,device,receipt):
        self.torch,self.device,self.receipt=torch,device,receipt
        self.consumer=torch.cuda.current_stream(device)
        self.producer=torch.cuda.Stream(device=device)
        self.events={};self.next_r=0;self.next_c=0
        receipt['streams_created']+=1

    def record_output(self,value):
        # Scope's rollout event has already been recorded; keep all exchanged
        # storages alive on their actual consumer, without host tensor reads.
        t=self.torch
        if t.is_tensor(value):
            if value.is_cuda:
                value.record_stream(self.consumer)
                self.receipt['consumer_storage_records']+=1
        elif isinstance(value,dict):
            for x in value.values():self.record_output(x)
        elif isinstance(value,(list,tuple)):
            for x in value:self.record_output(x)
        elif value is not None and not isinstance(value,(str,int,float,bool)):
            raise ValueError('unsupported exchanged tensor-tree object')

    @contextmanager
    def scope(self,op):
        name=op.operation_id
        self.receipt['host_launch_order'].append(name)
        if self.torch.cuda.current_stream(self.device)!=self.consumer:
            raise ValueError('original consumer stream changed')
        if name==f'r{self.next_r}' and 0<=self.next_r<5:
            index=self.next_r
            if index==0:
                self.producer.wait_stream(self.consumer)
                self.receipt['G_commit_stream_fences']+=1
            with self.torch.cuda.stream(self.producer):
                yield self
                event=self.torch.cuda.Event(enable_timing=False)
                event.record(self.producer)
            self.events[index]=event;self.next_r+=1
            self.receipt['rollout_ready_events']+=1
        elif name==f'c{self.next_c}' and 0<=self.next_c<5:
            index=self.next_c
            if index not in self.events:raise ValueError('missing own rollout-ready event')
            self.consumer.wait_event(self.events[index])
            self.receipt['critic_event_waits']+=1
            yield None
            self.next_c+=1
        elif name=='g' or name in {f'e{i}'for i in range(5)}:
            yield None
        else:
            raise ValueError('unexpected/duplicate/out-of-order original operation '+name)

    def finish(self):
        if self.next_r!=5 or self.next_c!=5 or len(self.events)!=5:
            raise ValueError('incomplete G5/C5 stream execution')
        if self.receipt['consumer_storage_records']<=0:
            raise ValueError('consumer lifetime records absent')
        self.receipt['completed_overlap_cycles']+=1

def graph_sizes(session):
    g=getattr(session.g.module,'_distillschedule_rollout_cuda_graph',None)
    c=getattr(session.c.module,'_distillschedule_critic_grad_cuda_graph',None)
    if g is None or c is None or not g._cache or not c.cache:
        raise ValueError('original G/C graphs must already be warm')
    return (len(g._cache),len(c.cache))

def require_private_parameters(session):
    g={p.untyped_storage().data_ptr()for p in session.g.module.parameters()}
    c={p.untyped_storage().data_ptr()for p in session.c.module.parameters()}
    if not g or not c or g.intersection(c):
        raise ValueError('overlap requires physically independent original G/C parameters')

@contextmanager
def explicit_worker(worker,*,enabled,entry_sha256,packet_sha256):
    original=worker.OriginalWanSession
    class Session(original):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            self._overlap_prepare_count=0
            self._overlap_receipt=dict(schema_version='dmd.wan_post_g_stream_overlap.v2',enabled=enabled,
                entry_sha256=entry_sha256,packet_sha256=packet_sha256,planner=False,
                weights_merged=False,precision_changed=False,host_action_order_changed=enabled,
                requested_C2_order=list(lookahead_order()) if enabled else None,
                cold_cycle_original_stream=True,streams_created=0,G_commit_stream_fences=0,
                rollout_ready_events=0,critic_event_waits=0,consumer_storage_records=0,
                completed_overlap_cycles=0,host_launch_order=[],contract_failures=0,
                graph_cache_entries_before=None,graph_cache_entries_after=None)
            for role in ('generator','critic'):
                self.build[role]['metadata']['post_g_stream_overlap_experiment']=self._overlap_receipt

        def prepare(self,*,operation_scope=None,operation_order=None):
            if operation_scope is not None:raise ValueError('no profiler/calibration scope in native stream experiment')
            self._overlap_prepare_count+=1
            if self._overlap_prepare_count not in (1,2):raise ValueError('exact two-cycle scope required')
            if not enabled or self._overlap_prepare_count==1:
                return super().prepare(operation_scope=operation_scope,operation_order=operation_order)
            import torch
            require_private_parameters(self)
            before=graph_sizes(self);self._overlap_receipt['graph_cache_entries_before']=list(before)
            execution=StreamExecution(torch,self.device,self._overlap_receipt)
            if operation_order is not None:raise ValueError('explicit manual lookahead does not accept another order')
            prepared=super().prepare(operation_scope=execution.scope,operation_order=lookahead_order())
            owner=self
            class Prepared:
                def run(self):
                    try:
                        result=prepared.run();execution.finish()
                        after=graph_sizes(owner);owner._overlap_receipt['graph_cache_entries_after']=list(after)
                        if before!=after:raise ValueError('new graph capture after cold cycle')
                        return result
                    except BaseException:
                        owner._overlap_receipt['contract_failures']+=1;raise
                def receipt(self):return prepared.receipt()
            return Prepared()
    worker.OriginalWanSession=Session
    try:yield
    finally:worker.OriginalWanSession=original
