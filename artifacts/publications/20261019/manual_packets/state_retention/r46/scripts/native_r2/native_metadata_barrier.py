"""Manual-only native metadata barrier restoration; absolutely no tensor math."""
from collections import Counter

POLICY='restore-native-cu-metadata-barrier.v1'


class NativeMetadataBarrier:
    def __init__(self,bounds,*,enabled=False):
        if type(enabled)is not bool or len(bounds)!=3 or not(bounds[0]==0<bounds[1]<bounds[2]):
            raise ValueError('EXACT_NATIVE_CU_BOUNDARIES_REQUIRED')
        self.bounds=list(bounds)
        self.enabled=enabled
        self.counts=Counter()
        self.previous=Counter()

    def wrap(self,original):
        if not self.enabled:
            return original
        def call(q,k,v,cu_q,cu_k,*args,**kwargs):
            if cu_q is not cu_k:
                self.counts['failures']+=1
                raise ValueError('ORIGINAL_SELF_ATTENTION_CU_IDENTITY_CHANGED')
            # Deliberately the same native host-visible metadata read, not a
            # value sketch, random draw, extra forward or tensor transformation.
            observed=cu_q.tolist()
            self.counts['actual_native_metadata_reads']+=1
            if observed!=self.bounds:
                self.counts['failures']+=1
                raise ValueError('ORIGINAL_NATIVE_CU_VALUE_CHANGED')
            return original(q,k,v,cu_q,cu_k,*args,**kwargs)
        return call

    def finish_cycle(self,cycle):
        delta=self.counts-self.previous
        wanted={'actual_native_metadata_reads':350} if self.enabled else {}
        if dict(delta)!=wanted:
            raise ValueError('ACTUAL_NATIVE_METADATA_BARRIER_CENSUS:'+repr(dict(delta)))
        self.previous=self.counts.copy()
        return dict(policy=POLICY,enabled=self.enabled,cycle=cycle,
            actual_native_metadata_reads=delta.get('actual_native_metadata_reads',0),
            contract_failures=delta.get('failures',0),
            scope='ONLY_ALREADY_ELIGIBLE_R43_NOGRAD_STUDENT_FA3_CALLS',
            tensor_math_changed=False,original_FA3_backend_unchanged=True,
            original_parent_dispatch_value_reads=0,
            effective_backend_host_value_reads=350 if self.enabled else 0)


def install_on_session(session,*,enabled):
    policy=session.manual_attention_policy
    if policy is None or session.manual_state_policy is not None:
        raise ValueError('ORIGINAL_R43_ATTENTION_WITHOUT_BANK_REQUIRED')
    barrier=NativeMetadataBarrier(policy.bounds,enabled=enabled)
    backend=policy.backend
    backend.flash_attn_varlen_func=barrier.wrap(backend.flash_attn_varlen_func)
    return barrier


def annotate_final(row,binding,enabled):
    if row.get('status')!='COMPLETED_ORIGINAL_H3_AR_WORK':
        return row
    if (row['profiled']is not False or row['schema_version']!='distillschedule.manual_h3_ar_execution.v1'
            or row['manual_policy_id']!='uncached-rollout-fa3-varlen-v1'):
        raise ValueError('ORIGINAL_FA3_MANUAL_PRODUCER_REQUIRED')
    return dict(row,schema_version='distillschedule.manual_h3_ar_metadata_barrier_execution.v1',
        manual_external_packet=binding,restore_native_barrier=enabled,timer_unchanged=True)
