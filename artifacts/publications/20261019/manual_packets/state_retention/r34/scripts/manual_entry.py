"""Versioned EMA-only manual ablation around the frozen real R30 entry."""
from pathlib import Path
import argparse,hashlib,json,runpy,subprocess,sys
from exact_ema_graph import explicit_worker

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate(packet,*,head,remaining):
    assert set(packet)=={'schema','source_head','arm','run_id','entry_files','parent_entry','parent_entry_sha256','stream_packet','stream_packet_sha256'}
    assert packet['schema']=='dmd.wan_exact_ema_manual.v1'and packet['source_head']==head and packet['arm']in ('control','graph')
    assert len(packet['run_id'])>8 and Path(packet['parent_entry']).is_absolute()and sha(packet['parent_entry'])==packet['parent_entry_sha256']
    assert set(packet['entry_files'])=={'manual_entry.py','exact_ema_graph.py'}
    for name,digest in packet['entry_files'].items():assert sha(Path(__file__).parent/name)==digest
    assert sha(packet['stream_packet'])==packet['stream_packet_sha256']
    assert remaining[remaining.index('--stream-packet')+1]==packet['stream_packet']
    assert remaining[remaining.index('--stream-packet-sha256')+1]==packet['stream_packet_sha256']
    parent=json.loads(Path(packet['stream_packet']).read_text())
    assert parent['schema_version']=='dmd.wan_post_g_stream_manual_experiment.v2'and parent['arm']=='overlap'and parent['source_head']==head

def main():
    p=argparse.ArgumentParser();p.add_argument('--ema-packet',required=True);p.add_argument('--ema-packet-sha256',required=True);a,rest=p.parse_known_args();assert sha(a.ema_packet)==a.ema_packet_sha256
    packet=json.loads(Path(a.ema_packet).read_text());from distillschedule.strategies import wan_worker as worker
    head=subprocess.check_output(['git','-C',str(worker.repository_root()),'rev-parse','HEAD'],text=True).strip();validate(packet,head=head,remaining=rest)
    oldargv=sys.argv;sys.path.insert(0,str(Path(packet['parent_entry']).parent))
    try:
        with explicit_worker(worker,enabled=packet['arm']=='graph',source_sha256=packet['entry_files']['exact_ema_graph.py'],packet_sha256=a.ema_packet_sha256):
            sys.argv=[packet['parent_entry'],*rest];runpy.run_path(packet['parent_entry'],run_name='__main__')
    finally:sys.argv=oldargv

if __name__=='__main__':main()
