"""Real user-facing lifecycle acceptance; no pretrained-quality claim."""
import pytest
import torch


def test_persistent_session_keeps_optimizer_parameter_ownership():
    from distillschedule.open_workload.torch_backend import Session
    from distillschedule.workloads.program import Call, Program
    from distillschedule.open_workload.torch_backend import Workload
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    state = {"model": model, "optimizer": optimizer}
    workload = Workload(Program((Call("read", "read", ("x",), ("model",)),),
                                {"model": 0, "optimizer": 0}, ("x",), ("read",)),
                        {"read": predict}, state, {"x": torch.ones(1, 2)})
    session = Session(workload, {"backend": "torch-callable-resident-v1",
                                "share_immutable": False, "checkpoint_bindings": []},
                      copy_state=False)
    assert session.state["model"] is model
    assert session.state["optimizer"] is optimizer
    session.run()


def predict(ctx):
    return ctx.state["model"](ctx.inputs["x"])


def test_checkpoint_complete_boundary_roundtrip_and_rejects_tampering(tmp_path):
    from distillschedule.training.checkpoint import save_checkpoint, restore_checkpoint
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state = {"model": model, "optimizer": optimizer, "history": torch.ones(2)}
    expected = {key: val.clone() for key, val in model.state_dict().items()}
    checkpoint = save_checkpoint(tmp_path / "checkpoint", state, identity={"config": "abc"},
                                 cycle=2, cursor=2, versions={"model": 2, "optimizer": 2})
    with torch.no_grad():
        model.weight.zero_()
    restored = restore_checkpoint(checkpoint, state, identity={"config": "abc"})
    assert restored["cycle"] == restored["cursor"] == 2
    assert all(torch.equal(model.state_dict()[key], value) for key, value in expected.items())
    with pytest.raises(ValueError, match="identity"):
        restore_checkpoint(checkpoint, state, identity={"config": "different"})
    (checkpoint / "_SUCCESS").unlink()
    with pytest.raises(ValueError, match="complete"):
        restore_checkpoint(checkpoint, state, identity={"config": "abc"})


class ProgressOwner:
    def __init__(self, value=0):
        self.value = value
        self.loaded = False

    def state_dict(self):
        return {'value':self.value}

    def validate_state_dict(self, value):
        if set(value)!={'value'} or type(value['value']) is not int:
            raise ValueError('bad progress owner state')

    def validate_checkpoint_progress(self, value, progress):
        if progress is None or value['value']!=progress['cycle'] or progress['cursor']!=progress['cycle']:
            raise ValueError('inner/outer progress mismatch')

    def load_state_dict(self, value):
        self.value=value['value']
        self.loaded=True


def test_optional_progress_hook_stages_every_owner_before_mutation(tmp_path):
    from distillschedule.training.checkpoint import pack_state,unpack_state,save_checkpoint
    original = {'first':torch.tensor([3.]),'owner':ProgressOwner(2)}
    packed = pack_state(original)
    live = {'first':torch.tensor([9.]),'owner':ProgressOwner(0)}
    with pytest.raises(ValueError,match='progress mismatch'):
        unpack_state(packed,live,progress=dict(cycle=3,cursor=3,versions={}))
    assert live['first'].item()==9 and live['owner'].value==0 and not live['owner'].loaded
    with pytest.raises(ValueError,match='progress mismatch'):
        save_checkpoint(tmp_path/'bad',original,identity={},cycle=3,cursor=3,versions={})
    assert not (tmp_path/'bad').exists()
    unpack_state(packed,live,progress=dict(cycle=2,cursor=2,versions={}))
    assert live['first'].item()==3 and live['owner'].value==2 and live['owner'].loaded


def test_optional_training_admission_runs_before_output_and_update(tmp_path,monkeypatch):
    from distillschedule.training import session
    from distillschedule.training.adapters import ModelComponents
    from distillschedule.training.config import TrainingConfig
    from distillschedule.examples.tiny_av import build_fixture
    from distillschedule.data import prepare
    raw = build_fixture(tmp_path/'fixture')
    # Existing fixture returns its raw manifest as part of a report.
    if isinstance(raw,dict): raw = raw['raw_manifest']
    prepared = prepare(raw,tmp_path/'prepared')
    config = TrainingConfig.from_dict(dict(schema_version='distillschedule.training.v1',
        model=dict(factory='distillschedule.examples.tiny_av:build_model'),
        algorithm=dict(factory='distillschedule.examples.tiny_av:build_algorithm'),
        data=dict(prepared=str(prepared)),execution=dict(mode='reference'),
        trainer=dict(cycles=1,device='cpu',output_dir=str(tmp_path/'output'))),source=tmp_path/'config.json')
    called=[]
    def reject(**kwargs):
        assert set(kwargs)=={'state','request','prepared_identity','prepared_manifest'}
        called.append(True)
        raise ValueError('explicit admission refusal')
    monkeypatch.setattr(session,'_construct_model',lambda config:(
        ModelComponents(state={'value':torch.tensor(0.)},admit_training=reject),{}))
    with pytest.raises(ValueError,match='explicit admission refusal'):
        session.train(config)
    assert called==[True] and not (tmp_path/'output').exists()


def test_optional_training_admission_must_be_callable():
    from distillschedule.training.adapters import ModelComponents
    with pytest.raises(ValueError,match='admit_training must be callable'):
        ModelComponents(state={'value':0},admit_training=True).validate()
    assert ModelComponents(state={'value':0}).validate().admit_training is None
