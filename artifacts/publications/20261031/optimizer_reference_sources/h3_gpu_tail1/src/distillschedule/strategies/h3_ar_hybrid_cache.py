"""Opt-in GPU-tail1 storage for the original exact native selected-output pool.

No changed predictions, kernels, autograd, input identity or epoch/version rules.
Only block49's native return trees stay on GPU instead of pinned CPU. All other
blocks use the SHA-pinned parent. Global parent factory is restored immediately.
"""
import contextlib
from distillschedule.core.contracts import require

GPU_BLOCKS=(49,)


def pool_class(parent):
    torch=parent.torch
    from torch.utils._pytree import tree_flatten,tree_map
    class HybridPool(parent.PrefetchPool):
        @contextlib.contextmanager
        def capture_group(self,namespace):
            if namespace[1] not in GPU_BLOCKS:
                with super().capture_group(namespace):yield
                return
            require(self.group is None,'GPU cache source group nested',path='h3_ar.hybrid')
            self.group=dict(namespace=namespace,device=None,last=None,count=0,gpu=True)
            completed=False
            try:
                yield;completed=True
            finally:
                group=self.group;self.group=None
                if completed:require(group['count']==2,'exact long/tail GPU source pair required',path='h3_ar.hybrid')
                self.counters['GPU_source_block_groups' if completed else 'GPU_partial_failed_groups']+=1

        def store_uses(self,namespace,ordinal,args,kwargs,result,uses):
            if namespace[1] not in GPU_BLOCKS:return super().store_uses(namespace,ordinal,args,kwargs,result,uses)
            group=self.group
            require(self.enabled and group is not None and group.get('gpu') is True and group['namespace']==namespace
                    and uses in (1,2) and ordinal in (0,1),'bound GPU source/lifetime required',path='h3_ar.hybrid')
            leaves,spec=tree_flatten(result);values=[];versions=[];payload=0;device=None;storages={}
            for index,value in enumerate(leaves):
                if isinstance(value,torch.Tensor):
                    require(self.cpu_test or value.is_cuda or value.numel()<=16,'large host return in native op',path='h3_ar.hybrid')
                    if value.is_cuda or self.cpu_test:
                        require(device is None or value.device==device,'single GPU cache owner required',path='h3_ar.hybrid');device=value.device
                        values.append(value.detach());versions.append((index,int(value._version)))
                        payload+=value.numel()*value.element_size()
                        storage=value.untyped_storage();key=(str(value.device),int(storage.data_ptr()))
                        if storage.nbytes():storages[key]=int(storage.nbytes())
                    else:values.append(value.detach().clone())
                else:values.append(value)
            require(device is not None,'native GPU return tensor absent',path='h3_ar.hybrid')
            ready=None if self.cpu_test else torch.cuda.Event()
            if not self.cpu_test:ready.record(torch.cuda.current_stream(device))
            group['device']=device;group['count']+=1
            if not hasattr(self,'gpu_storage_owners'):self.gpu_storage_owners={}
            for key,nbytes in storages.items():
                old=self.gpu_storage_owners.get(key)
                if old is None:
                    self.gpu_storage_owners[key]=(nbytes,1);self.counters['GPU_live_unique_storage_bytes']+=nbytes
                else:
                    require(old[0]==nbytes,'retained storage resized',path='h3_ar.hybrid')
                    self.gpu_storage_owners[key]=(nbytes,old[1]+1)
            self.counters['GPU_peak_unique_storage_bytes']=max(self.counters['GPU_peak_unique_storage_bytes'],self.counters['GPU_live_unique_storage_bytes'])
            entry=dict(buffers=[],values=values,spec=spec,device=device,ready=ready,
                signature=parent.signature((args,kwargs)),remaining=uses,consumer_limit=uses,
                prefetch_namespace=namespace,prefetch_ordinal=ordinal,gpu_storage=True,
                gpu_versions=versions,gpu_payload=payload,gpu_storages=storages)
            self.immutable_entries.append(entry)
            self.counters['source_calls']+=1;self.counters['cached_original_attention_calls']+=1
            self.counters['GPU_source_calls']+=1;self.counters['GPU_saved_bytes']+=payload
            self.counters['GPU_live_bytes']+=payload
            self.counters['GPU_peak_live_bytes']=max(self.counters['GPU_peak_live_bytes'],self.counters['GPU_live_bytes'])
            return entry

        def restore(self,entry,args,kwargs):
            gpu=entry.get('gpu_storage',False)
            if gpu:
                require(entry['remaining']>0 and all(entry['values'][i]._version==version for i,version in entry['gpu_versions']),
                        'retained original native output mutated/expired',path='h3_ar.hybrid')
            result=super().restore(entry,args,kwargs)
            if gpu:
                # Distinct TensorImpls avoid attaching a new native autograd node
                # to the same cache-owned Tensor object. Physical values/layout
                # are unchanged, and all reads follow the original event waits.
                result=tree_map(lambda t:t.detach() if isinstance(t,torch.Tensor) else t,result)
                self.counters['GPU_consumer_calls']+=1;self.counters['GPU_avoided_H2D_bytes']+=entry['gpu_payload']
                if entry['remaining']==0:
                    for i,_ in entry['gpu_versions']:entry['values'][i]=None
                    self.counters['GPU_live_bytes']-=entry['gpu_payload'];self.counters['GPU_release_calls']+=1
                    for key,nbytes in entry['gpu_storages'].items():
                        stored,count=self.gpu_storage_owners[key]
                        require(stored==nbytes and count>0,'GPU storage owner mismatch',path='h3_ar.hybrid')
                        if count==1:
                            del self.gpu_storage_owners[key];self.counters['GPU_live_unique_storage_bytes']-=nbytes
                        else:self.gpu_storage_owners[key]=(stored,count-1)
            return result

        def finish_selected(self):
            require(not self.prefetched and not self.prefetch_groups,'pending native lookahead',path='h3_ar.hybrid')
            n=len(self.blocks)
            require(n==50 and self.prefetch_stats['groups']==3*(n-1)
                    and self.prefetch_stats['entries']==self.prefetch_stats['actual_prefetched_consumers']==6*(n-1)
                    and self.prefetch_stats['on_demand_head_entries']==6
                    and self.last_consumed=={'STUDENT_GRAD':(n-1,1),'STUDENT_REPLAY':(0,1),'FAKE_REPLAY':(0,1)},
                    'unchanged full three native traversals required',path='h3_ar.hybrid')
            require(self.counters['source_calls']==2*(self.counters['D2H_block_groups']+self.counters['GPU_source_block_groups'])
                    and self.counters['GPU_source_block_groups']==2 and self.counters['GPU_source_calls']==4
                    and self.counters['GPU_consumer_calls']==6 and self.counters['GPU_release_calls']==4
                    and self.counters['GPU_live_bytes']==0 and self.counters['GPU_live_unique_storage_bytes']==0
                    and not self.gpu_storage_owners,'GPU source/consumer/release census failed',path='h3_ar.hybrid')
            # Keep the base original queue/consumer/pinned-buffer checks. The old
            # CPU-only FIFO assertion is replaced by the strict combined proof
            # above; no fabricated D2H group count enters the receipt.
            result=parent.ad.OriginalPool.finish_selected(self)
            result.update(storage='CPU_TEST_HYBRID' if self.cpu_test else 'hybrid_gpu_tail1_pinned_cpu',
                GPU_retained_blocks=list(GPU_BLOCKS),GPU_cache_no_value_conversion=True,
                GPU_cache_consumed_and_released=True,
                GPU_release_scope='CACHE_OWNED_REFS_ONLY; NATIVE_AUTOGRAD_AND_STREAM_LIFETIMES_REMAIN',
                GPU_saved_bytes_scope='CUMULATIVE_LOGICAL_OUTPUT_BYTES_NOT_PEAK',
                GPU_storage_peak_scope='CACHE_OWNED_PHYSICAL_STORAGE_DEDUPED_BY_DEVICE_POINTER',H2D_prefetch=dict(self.prefetch_stats),
                H2D_prefetch_window_main_blocks=1,strict_native_consumer_order=True,
                native_backward_unchanged=True,native_attention_kernels_unchanged=True,
                extra_host_sync_or_collective=False,max_D2H_inflight='one CPU long+tail block group',
                only_cache_storage_and_transfer_lifetime_changed=True)
            return result
    return HybridPool


def install(epoch,assets,native,module,adapter,trainer):
    parent=assets.helper('native_selected_h2d_prefetch_r1')
    original=parent.PrefetchPool
    require(original.__name__=='PrefetchPool' and original.__module__==parent.__name__,
            'nested/unknown parent cache factory',path='h3_ar.hybrid')
    child=pool_class(parent)
    parent.PrefetchPool=child
    try:pool,view=epoch.install(native,module,adapter,trainer)
    finally:parent.PrefetchPool=original
    require(isinstance(pool,child) and pool.enabled is True and parent.PrefetchPool is original,'versioned GPU cache factory not installed/restored',path='h3_ar.hybrid')
    return epoch,pool,view


def validate_receipt(cache):
    c=cache.get('counters',{})
    require(cache.get('storage')=='hybrid_gpu_tail1_pinned_cpu' and cache.get('GPU_retained_blocks')==list(GPU_BLOCKS)
            and cache.get('GPU_cache_no_value_conversion') is True and cache.get('GPU_cache_consumed_and_released') is True
            and c.get('GPU_source_calls')==4 and c.get('GPU_consumer_calls')==6 and c.get('GPU_release_calls')==4
            and c.get('GPU_live_bytes')==0 and c.get('GPU_live_unique_storage_bytes')==0
            and c.get('GPU_peak_unique_storage_bytes',0)>0 and c.get('GPU_saved_bytes',0)>0 and c.get('GPU_avoided_H2D_bytes',0)>0,
            'requested hybrid cache not observed in actual native cycle',path='h3_ar.hybrid')
