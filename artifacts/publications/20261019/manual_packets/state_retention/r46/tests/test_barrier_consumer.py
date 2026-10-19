"""SYNTHETIC_TEST_ONLY actual counter -> final writer -> actual16-rank consumer."""
from pathlib import Path
import importlib.util,json
import pytest

R=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
c=load('barrier_controller',R/'scripts/native_r2/h3_ar_match_controller.py')
m=load('barrier_producer',R/'scripts/native_metadata_barrier.py')


def fixture(tmp_path,monkeypatch,phase):
    old=load('original_r43_fixture',R.parent/'r43/tests/test_manual_owner_contract.py')
    monkeypatch.setattr(old,'c',c)
    cfg=old.fixture(tmp_path,monkeypatch,'manual-child')
    enabled=phase=='manual-child'
    if not enabled:(tmp_path/'manual-child').rename(tmp_path/'manual-control')
    cfg['manual_packet']=dict(path='/SYNTHETIC_TEST_ONLY/packet.json',sha256='a'*64)
    class Cu:
        def tolist(self):return [0,34876,34880]
    cu=Cu()
    for rank in range(16):
        path=tmp_path/phase/f'rank_{rank:03d}.json'
        row=json.loads(path.read_text())
        barrier=m.NativeMetadataBarrier([0,34876,34880],enabled=enabled)
        wrapped=barrier.wrap(lambda *args,**kwargs:None)
        for i,cycle in enumerate(row['cycles'],1):
            if enabled:
                for j in range(350):wrapped(None,None,None,cu,cu)
            cycle['backend_effects']['manual_native_metadata_barrier']=barrier.finish_cycle(i)
        row=m.annotate_final(row,cfg['manual_packet'],enabled)
        path.write_text(json.dumps(row))
    return cfg


@pytest.mark.parametrize('phase',['manual-control','manual-child'])
def test_real_producer_to_consumer_each_arm(tmp_path,monkeypatch,phase):
    cfg=fixture(tmp_path,monkeypatch,phase)
    assert len(c.validate_ranks(cfg,phase))==16
    assert all(r[1]['restore_native_barrier']==(phase=='manual-child')for r in c.validate_ranks(cfg,phase))


@pytest.mark.parametrize('failure',['wrong_arm','wrong_packet','wrong_read_count','timer_change',
    'wrong_schema','profiled','missing_cycle','wrong_calls','bad_FA3','merged'])
def test_all_original_and_new_bindings_reject(tmp_path,monkeypatch,failure):
    cfg=fixture(tmp_path,monkeypatch,'manual-child');path=tmp_path/'manual-child/rank_000.json'
    row=json.loads(path.read_text())
    if failure=='wrong_arm':row['restore_native_barrier']=False
    if failure=='wrong_packet':row['manual_external_packet']['sha256']='b'*64
    if failure=='wrong_read_count':row['cycles'][1]['backend_effects']['manual_native_metadata_barrier']['actual_native_metadata_reads']=349
    if failure=='timer_change':row['timer_unchanged']=False
    if failure=='wrong_schema':row['schema_version']='distillschedule.manual_h3_ar_execution.v1'
    if failure=='profiled':row['profiled']=True
    if failure=='missing_cycle':row['cycles'].pop()
    if failure=='wrong_calls':row['cycles'][0]['census']['student']=8
    if failure=='bad_FA3':row['cycles'][0]['backend_effects']['manual_uncached_rollout_attention']['per_cycle_counts']['actual_FA3_varlen_calls']=349
    if failure=='merged':row['weights_merged']=True
    path.write_text(json.dumps(row))
    with pytest.raises((ValueError,RuntimeError,KeyError)):c.validate_ranks(cfg,'manual-child')
