"""One common Program macrocycle over the original stateful DMD2 owner."""
from .dmd2_owner import DMD2Owner
from .dmd2_collective import DMD2CollectiveOwner
from .sd import require


def native_cycle(ctx):
    return ctx.state['native_cycle'].run(ctx.inputs, cycle=ctx.attributes['cycle'])


def build_dmd2(components, feeds, parameters, cycle, seed):
    from .official_optimizations.dmd2 import DMD2OptimizedOwner
    from .dmd2_fsdp import BACKEND as FSDP2_BACKEND, DMD2FSDP2Owner
    from distillschedule.open_workload.torch_backend import Workload
    from distillschedule.workloads.program import Program, Call
    require(not parameters and set(components.state) == {'native_cycle'}
            and type(components.state['native_cycle']) in (DMD2Owner, DMD2CollectiveOwner, DMD2OptimizedOwner, DMD2FSDP2Owner),
            'actual SDUniModel DMD2 owner without algorithm overrides required')
    owner = components.state['native_cycle']
    owner._validate_live()
    owner.validate_feeds(feeds)
    # Construction also occurs before checkpoint restore, so cycle equality is
    # checked by the native execution owner, not the pre-restore builder.
    require(type(cycle) is int and cycle >= 0, 'invalid Program cycle')
    call = Call(id='dmd2.native_cycle', binding='official_dmd2_cycle', inputs=tuple(sorted(feeds)),
        reads=('native_cycle',), writes=('native_cycle',), grad='external_grad', replay_safe=False,
        attributes={'cycle': cycle})
    if 'optimization_profile' in owner.identity:
        scope = 'OFFICIAL_DMD2_NATIVE_MIXED_OPTIMIZED'
    elif owner.identity.get('native_collective', {}).get('backend') == FSDP2_BACKEND:
        scope = 'OFFICIAL_DMD2_NATIVE_FSDP2_REFERENCE'
    elif 'native_collective' in owner.identity:
        scope = 'OFFICIAL_DMD2_NATIVE_FP32_COLLECTIVE'
    else:
        scope = 'OFFICIAL_DMD2_NATIVE_FP32_SINGLE_DEVICE'
    return Workload(program=Program((call,), initial_versions={'native_cycle': cycle},
        feeds=tuple(sorted(feeds)), outputs=(call.id,)), bindings={'official_dmd2_cycle': native_cycle},
        state=components.state, feeds=feeds, seed=seed,
        identity_extra={'scope': scope, 'ratio': owner.ratio,
                        'upstream': owner.source.identity})
