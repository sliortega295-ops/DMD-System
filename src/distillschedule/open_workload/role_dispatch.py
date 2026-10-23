"""Checked producer call orders lowered by the existing role-pool executor.

Only the interleaving of monotone forwards and FIFO backwards changes. No
optimizer commit can move, and no Score receive, collective, offload or model
semantics are inferred from a sequence of strings.
"""
from collections.abc import Mapping
from distillschedule.core.contracts import require


def validate_dispatch(task, layout, dispatch):
    require(layout.generator_parallelism==layout.score_parallelism=='replicated' and
            layout.activation_policy=='resident',
            'planned producer orders require replicated resident pools',path='role.dispatch')
    require(isinstance(dispatch,Mapping) and set(dispatch)=={str(r) for r in layout.generator},
            'one producer sequence per Generator rank required',path='role.dispatch')
    result={}
    for rank in layout.generator:
        indices=[i for i in range(task.microbatches) if layout.owners(i)[0]==rank]
        actions=dispatch[str(rank)]
        require(isinstance(actions,(list,tuple)) and len(actions)==2*len(indices) and
                all(isinstance(a,str) for a in actions),
                'invalid producer sequence',path='role.dispatch')
        forward=[];backward=[];pending=[]
        for action in actions:
            if action.startswith('g'):
                require(len(forward)<len(indices) and action==f'g{indices[len(forward)]}',
                        'producer forward order or owner changed',path='role.dispatch')
                require(len(pending)<layout.window,'producer window exceeded',path='role.dispatch')
                value=indices[len(forward)];forward.append(value);pending.append(value)
            elif action.startswith('b'):
                require(pending and action==f'b{pending[0]}',
                        'backward needs its original FIFO forward',path='role.dispatch')
                backward.append(pending.pop(0))
            else:
                raise ValueError('role.dispatch: unknown producer action')
        require(forward==backward==indices and not pending,
                'producer sequence does not drain all gradients',path='role.dispatch')
        result[str(rank)]=list(actions)
    return result


def from_order(task,layout,order):
    """Project a common graph order; never invent per-rank timing or actions."""
    require(isinstance(order,(list,tuple)) and len(order)==len(set(order)),
            'invalid common operation order',path='role.dispatch')
    dispatch={}
    for rank in layout.generator:
        keys={f'{p}{i}' for i in range(task.microbatches) for p in ('g','b')
              if layout.owners(i)[0]==rank}
        dispatch[str(rank)]=[key for key in order if key in keys]
    return validate_dispatch(task,layout,dispatch)
