"""SYNTHETIC_TEST_ONLY producer/runtime hooks, real CPU EMA recurrence."""
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib,importlib.util,json,sys
import pytest,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import exact_ema_graph as module
s=importlib.util.spec_from_file_location('ema_unit_fixture',R/'tests/test_exact_ema_graph.py');fixture=importlib.util.module_from_spec(s);s.loader.exec_module(fixture)
s=importlib.util.spec_from_file_location('ema_manual',R/'scripts/manual_entry.py');entry=importlib.util.module_from_spec(s);s.loader.exec_module(entry)

@pytest.mark.parametrize('enabled',[False,True])
def test_actual_worker_hook_counts_math_versions_and_restoration(monkeypatch,enabled):
    def fake_capture(ema,torch,receipt):
        d,r=fixture.driver(ema);d.receipt=receipt;d.stream_identity=('SYNTHETIC_CPU_STREAM',7)
        receipt['captured_graphs']=10;return d
    monkeypatch.setattr(module,'capture',fake_capture)
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device:NS(device='SYNTHETIC_CPU_STREAM',cuda_stream=7))
    class Base:
        def __init__(self,**kwargs):self.ema=fixture.ema();self.device='SYNTHETIC_CPU_STREAM';self.build={role:dict(metadata={})for role in('generator','critic')}
        def prepare(self,**kwargs):
            assert kwargs==dict(operation_scope='SYNTHETIC_UNCHANGED_SCOPE',operation_order=('SYNTHETIC',))
            owner=self
            class Prepared:
                def run(self):
                    for _ in range(5):owner.ema.step()
                    return dict(SYNTHETIC_TEST_ONLY=True)
                def receipt(self):return dict(SYNTHETIC_TEST_ONLY=True)
            return Prepared()
    worker=NS(OriginalWanSession=Base);reference=fixture.ema();rng=torch.get_rng_state().clone()
    with module.explicit_worker(worker,enabled=enabled,source_sha256='SYNTHETIC_SOURCE',packet_sha256='SYNTHETIC_PACKET'):
        session=worker.OriginalWanSession()
        for _ in range(2):
            prepared=session.prepare(operation_scope='SYNTHETIC_UNCHANGED_SCOPE',operation_order=('SYNTHETIC',));assert prepared.run()==dict(SYNTHETIC_TEST_ONLY=True)
            for _ in range(5):fixture.FreshTrainableEMA.step(reference)
            assert all(torch.equal(a,b)for a,b in zip(reference._shadows,session.ema._shadows,strict=True))
        r=session.build['critic']['metadata']['exact_ema_graph_experiment']
        assert r['observed_steps']==list(range(11,21))and r['graph_replays']==(10 if enabled else 0)and r['reference_steps']==(0 if enabled else 10)
        assert session.ema.optimization_step==20 and r['contract_failures']==0
        with pytest.raises(ValueError,match='two original cycles'):session.prepare()
    assert worker.OriginalWanSession is Base and torch.equal(rng,torch.get_rng_state())

@pytest.mark.parametrize('damage',['wrong_arm','wrong_head','parent_wrong_arm','entry_hash'])
def test_real_entry_rejects_misbound_parent_or_source(tmp_path,damage):
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    parent=tmp_path/'SYNTHETIC-parent.py';parent.write_text('# SYNTHETIC_TEST_ONLY\n')
    sp=tmp_path/'SYNTHETIC-stream.json';sp.write_text(json.dumps(dict(schema_version='dmd.wan_post_g_stream_manual_experiment.v2',arm='control'if damage=='parent_wrong_arm'else'overlap',source_head='H')))
    p=dict(schema='dmd.wan_exact_ema_manual.v1',source_head='H',arm='control',run_id='SYNTHETIC_TEST_ONLY',entry_files={n:sha(R/'scripts'/n)for n in ('manual_entry.py','exact_ema_graph.py')},parent_entry=str(parent),parent_entry_sha256=sha(parent),stream_packet=str(sp),stream_packet_sha256=sha(sp))
    if damage=='wrong_arm':p['arm']='merged'
    if damage=='wrong_head':p['source_head']='WRONG'
    if damage=='entry_hash':p['entry_files']['exact_ema_graph.py']='WRONG'
    with pytest.raises(AssertionError):entry.validate(p,head='H',remaining=['--stream-packet',str(sp),'--stream-packet-sha256',sha(sp)])
