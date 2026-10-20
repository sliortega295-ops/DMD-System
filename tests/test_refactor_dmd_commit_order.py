"""Literal Student1/FakeN commit ordering survives optimizer search.

Forward/backward prefetch of independent work may still be reordered; the
caller-declared optimizer commit boundary is not an optimization candidate.
"""
import pytest
from distillschedule.open_workload.dmd import DMDHooks,build_dmd_program


def hook(ctx):
    return None


def reaches(graph, source, target):
    todo=[source];seen=set()
    while todo:
        item=todo.pop()
        if item==target:
            return True
        if item not in seen:
            seen.add(item)
            todo.extend(e.target for e in graph.dependencies if e.source==item)
    return False


@pytest.mark.parametrize('policy',['reuse_student','fresh_student'])
@pytest.mark.parametrize('count',[1,5])
def test_student_commit_precedes_each_fake_commit(policy,count):
    hooks=DMDHooks(student_forward=hook,fake_forward=hook,teacher_forward=hook,
                   noise=hook,student_loss=hook,fake_loss=hook)
    program=build_dmd_program(hooks=hooks,feed_names=('batch',),cycles=2,
                              fake_updates=count,fake_sample_policy=policy)
    graph=program.compile()
    for cycle in range(2):
        for index in range(count):
            assert reaches(graph,f'dmd.c{cycle}.student_update',f'dmd.c{cycle}.f{index}.update')
    assert reaches(graph,f'dmd.c0.f{count-1}.update','dmd.c1.student_update')
    if policy=='reuse_student':
        # Do not serialize independent no-grad preparation/gradient work merely
        # to order the two actual optimizer commits.
        assert not reaches(graph,'dmd.c0.student_update','dmd.c0.f0.forward')
