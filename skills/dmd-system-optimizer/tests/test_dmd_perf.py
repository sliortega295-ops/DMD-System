"""CPU unit/contract tests; fixtures are synthetic and make no DMD performance claim."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('dmd_perf', ROOT/'scripts/dmd_perf.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def fixtures(directory):
    p=directory/'receipt.txt'; p.write_text('synthetic unit-test receipt, NOT a real validation')
    ev={'path':'receipt.txt','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'locator':'unit-test fixture only'}
    ident={'id':'control-1','seconds':100.,'source_identity':'fixture-source','evidence_grade':'accepted_unprofiled','evidence':[ev], 'timer_scope':'full-cycle','semantic_contract':'fixed','hardware_id':'same','claim_class':'strict'}
    c={'schema_version':1,'analysis_id':'a1','workload_id':'unit-test','search_mode':'target-seeking','timer_scope':'full-cycle','semantic_contract':'fixed','hardware_id':'same','claim_class':'strict','baseline':copy.deepcopy(ident),'control':copy.deepcopy(ident),'target':{'speedup':1.5},'survey':{k:{'status':'checked','evidence_note':'fixture'} for k in 'MLQK'},'budget':{'max_gpu_hours':8.,'used_gpu_hours':0.,'max_full_runs':2,'used_full_runs':0},'policy':{'noise_floor_seconds':.1,'small_gain_fraction_of_gap':.1}}
    def op(i,layer,values):
        return {'id':i,'layer':layer,'kind':'ordinary','analysis_id':'a1','control_id':'control-1','mechanism':'fixture','semantic_risks':'fixture','selection_reason':'fixture','evidence':[copy.deepcopy(ev)],'estimate':{'kind':'e2e_counterfactual','basis':'fixture-only','gross_saving_seconds':dict(zip(['low','central','high'],values)),'added_exposed_cost_seconds':{'low':0.,'central':.1,'high':.2}},'cost':{'engineer_hours':1.,'gpu_hours':2.,'full_runs':1},'probe':{'question':'fixture','stop_rule':'fixture'},'enables':[]}
    return c, {'schema_version':1,'opportunities':[op('residency','L',[10.,15.,20.]),op('norm','K',[.5,1.,1.5])]}


class ArithmeticTests(unittest.TestCase):
    def test_15x(self):
        r=m.target(100,90,1.5)
        self.assertAlmostEqual(r['target_seconds'],200/3)
        self.assertAlmostEqual(r['remaining_saving_seconds'],70/3)
    def test_baseline_not_reset(self):
        r=m.target(120,90,1.5)
        self.assertEqual(r['target_seconds'],80)
        self.assertEqual(r['remaining_saving_seconds'],10)
    def test_target_met(self):
        self.assertEqual(m.target(100,60,1.5)['remaining_saving_seconds'],0)
    def test_invalid_values(self):
        for v in [True,False,None,0,-1,math.inf,math.nan]:
            with self.assertRaises(m.InputError): m.target(v,100,1.5)
    def test_amdahl(self):
        op={'estimate':{'kind':'serial_amdahl','basis':'fixture','path_assumptions':'exclusive unchanged critical path','critical_component_seconds':4.,'local_speedup':{'low':2.,'central':2.,'high':2.},'added_exposed_cost_seconds':{'low':0.,'central':0.,'high':0.}}}
        r=m.estimate(op,100)
        self.assertEqual(r['net_saving_seconds']['central'],2)
        self.assertEqual(r['predicted_runtime_seconds']['central'],98)
    def test_net_interval(self):
        op={'estimate':{'kind':'e2e_counterfactual','basis':'fixture','gross_saving_seconds':{'low':5.,'central':8.,'high':10.},'added_exposed_cost_seconds':{'low':1.,'central':2.,'high':3.}}}
        self.assertEqual(m.estimate(op,100)['net_saving_seconds'],{'low':2.,'central':6.,'high':9.})
    def test_bad_interval(self):
        with self.assertRaises(m.InputError): m.interval({'low':3,'central':2,'high':4},'x')
    def test_cap_violation(self):
        op={'estimate':{'kind':'e2e_counterfactual','basis':'fixture','conditional_saving_cap_seconds':1,'gross_saving_seconds':{'low':1,'central':2,'high':3},'added_exposed_cost_seconds':{'low':0,'central':0,'high':0}}}
        with self.assertRaises(m.InputError): m.estimate(op,100)
    def test_zero_runtime_rejected(self):
        op={'estimate':{'kind':'e2e_counterfactual','basis':'fixture','gross_saving_seconds':{'low':100,'central':100,'high':100},'added_exposed_cost_seconds':{'low':0,'central':0,'high':0}}}
        with self.assertRaises(m.InputError): m.estimate(op,100)
    def test_branch_switch(self):
        r=m.dag_model(m.load(ROOT/'examples/synthetic-dag.json'))
        self.assertEqual(r['base_model_seconds'],11)
        self.assertEqual(r['candidate_model_seconds'],10)
        self.assertEqual(r['model_saving_seconds'],1)
        self.assertEqual(r['candidate_critical_path'],['B','join'])
    def test_cyclic_graph(self):
        with self.assertRaises(m.InputError): m.dag_model({'nodes':[{'id':'a','seconds':1,'depends_on':['b']},{'id':'b','seconds':1,'depends_on':['a']}]})
    def test_missing_dependency(self):
        with self.assertRaises(m.InputError): m.dag_model({'nodes':[{'id':'a','seconds':1,'depends_on':['missing']}]})
    def test_compare_descriptive_only(self):
        r=m.compare({'timer_scope':'f','semantic_contract':'s','hardware_id':'h','unprofiled':True,'paired':True,'control_seconds':[100,102,98],'candidate_seconds':[90,91,89]})
        self.assertEqual(r['formal_acceptance'],'NOT_DECIDED')
        self.assertEqual(r['paired_saving_seconds'],[10.,11.,9.])
    def test_profiled_compare_rejected(self):
        with self.assertRaises(m.InputError): m.compare({'timer_scope':'f','semantic_contract':'s','hardware_id':'h','unprofiled':False})


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.d=Path(self.tmp.name)
        self.c,self.o=fixtures(self.d)
    def tearDown(self): self.tmp.cleanup()
    def run_gate(self,i='residency'):
        return m.gate(self.c,self.o,i,self.d,self.d)
    def test_admit(self): self.assertEqual(self.run_gate()['status'],'ADMIT_BOUNDED_EXPERIMENT')
    def test_small_kernel_deferred(self): self.assertEqual(self.run_gate('norm')['status'],'DEFER')
    def test_unknown_estimate(self):
        self.o['opportunities'][0]['estimate']=None
        self.assertEqual(self.run_gate()['status'],'PROBE_ONLY')
    def test_stale_control(self):
        self.o['opportunities'][0]['control_id']='old'
        self.assertEqual(self.run_gate()['status'],'INVALID')
    def test_stale_analysis(self):
        self.o['opportunities'][0]['analysis_id']='old'
        self.assertEqual(self.run_gate()['status'],'INVALID')
    def test_historical_not_current(self):
        self.c['control']['evidence_grade']='historical'
        self.assertEqual(self.run_gate()['status'],'INVALID')
    def test_scope_mismatch(self):
        self.c['baseline']['timer_scope']='critic-only'
        self.assertEqual(self.run_gate()['status'],'INVALID')
    def test_hash_mismatch(self):
        (self.d/'receipt.txt').write_text('changed')
        self.assertEqual(self.run_gate()['status'],'INVALID')
    def test_missing_evidence(self):
        self.o['opportunities'][0]['evidence']=[]
        self.assertEqual(self.run_gate()['status'],'INVALID')
    def test_budget_stop(self):
        self.c['budget']['used_gpu_hours']=7
        self.assertEqual(self.run_gate()['status'],'DEFER')
    def test_budget_unset(self):
        self.c['budget']['max_gpu_hours']=None
        self.assertEqual(self.run_gate()['status'],'PROBE_ONLY')
    def test_incomplete_survey(self):
        self.c['survey']['M']['status']='unknown'
        self.assertEqual(self.run_gate('norm')['status'],'PROBE_ONLY')
    def test_nonpositive_upside(self):
        self.o['opportunities'][0]['estimate']['added_exposed_cost_seconds']={'low':25.,'central':26.,'high':27.}
        self.assertEqual(self.run_gate()['status'],'DEFER')
    def test_enabler_probe(self):
        op=self.o['opportunities'][1];op.update(kind='enabler',enables=['residency'],capacity_argument='free tested workspace to fit model')
        self.assertEqual(self.run_gate('norm')['status'],'PROBE_ONLY')
    def test_enabler_after_probe(self):
        op=self.o['opportunities'][1];op.update(kind='enabler',enables=['residency'],capacity_argument='fixture',enabler_probe_passed=True,enabler_probe_evidence=copy.deepcopy(op['evidence']))
        op['estimate']['scope']='enabled_portfolio'
        self.assertEqual(self.run_gate('norm')['status'],'ADMIT_BOUNDED_EXPERIMENT')
    def test_enabler_cycle(self):
        self.o['opportunities'][0]['enables']=['norm'];self.o['opportunities'][1]['enables']=['residency']
        with self.assertRaises(m.InputError): self.run_gate()
    def test_duplicate_ids(self):
        self.o['opportunities'].append(copy.deepcopy(self.o['opportunities'][0]))
        with self.assertRaises(m.InputError): self.run_gate()
    def test_synthetic_never_admitted(self):
        c=m.load(ROOT/'examples/synthetic-campaign.json');o=m.load(ROOT/'examples/synthetic-opportunities.json')
        self.assertEqual(m.gate(c,o,'model-residency',self.d,self.d)['status'],'INVALID')
    def test_no_fake_zero_cost(self):
        self.o['opportunities'][0]['cost']['engineer_hours']=0
        with self.assertRaises(m.InputError): self.run_gate()

    def test_enabler_requires_joint_forecast(self):
        op=self.o['opportunities'][1]
        op.update(kind='enabler',enables=['residency'],capacity_argument='fixture',enabler_probe_passed=True,enabler_probe_evidence=copy.deepcopy(op['evidence']))
        self.assertEqual(self.run_gate('norm')['status'],'PROBE_ONLY')

    def test_missing_baseline_source(self):
        self.c['baseline']=[]
        with self.assertRaises(m.InputError): self.run_gate()

    def test_missing_baseline_source_identity_rejected(self):
        del self.c['baseline']['source_identity']
        self.assertEqual(self.run_gate()['status'], 'INVALID')

    def test_unknown_alternative_does_not_admit_small_gain(self):
        self.o['opportunities'][0]['estimate'] = None
        self.assertEqual(self.run_gate('norm')['status'], 'PROBE_ONLY')

    def test_missing_enabler_receipt(self):
        op=self.o['opportunities'][1];op.update(kind='enabler',enables=['residency'],capacity_argument='fixture',enabler_probe_passed=True)
        self.assertEqual(self.run_gate('norm')['status'],'INVALID')
    def test_claim_class_mismatch(self):
        self.c['control']['claim_class']='approximate'
        self.assertEqual(self.run_gate()['status'],'INVALID')
    def test_invalid_op_object(self):
        self.o['opportunities'].append('not an object')
        with self.assertRaises(m.InputError): self.run_gate()
    def test_invalid_enables_type(self):
        self.o['opportunities'][0]['enables']=[{}]
        with self.assertRaises(m.InputError): self.run_gate()


class InspectTests(unittest.TestCase):
    def test_inspect_synthetic_git_checkout(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'src/distillschedule').mkdir(parents=True)
            folder=root/'configs/distillschedule/best_known';folder.mkdir(parents=True)
            value={'semantics':'synthetic','contract':{'updates':{'generator':1}},'runtime':{'launcher':'none'},'memory':{'model_swap':False},'validation':{'server_revalidation':'NOT_RUN'}}
            (folder/'test_profile.json').write_text(json.dumps(value))
            (folder/'registry.lock.json').write_text(json.dumps({'profiles':{'test_profile':m.digest(value)}}))
            subprocess.run(['git','init','-q',str(root)],check=True)
            subprocess.run(['git','-C',str(root),'add','configs'],check=True)
            subprocess.run(['git','-C',str(root),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture'],check=True)
            r=m.inspect(root)
            self.assertTrue(r['profiles'][0]['lock_matches'])
            self.assertFalse(r['gpu_validation_performed'])
            value['memory']['model_swap']=True
            (folder/'test_profile.json').write_text(json.dumps(value))
            r=m.inspect(root)
            self.assertFalse(r['profiles'][0]['lock_matches'])
            self.assertFalse(r['clean_status'])
    def test_inspect_wrong_root(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(m.InputError): m.inspect(Path(d))

    def test_external_training_repository_supported(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'train.py').write_text('# external workload fixture')
            subprocess.run(['git','init','-q',str(root)],check=True)
            subprocess.run(['git','-C',str(root),'add','train.py'],check=True)
            subprocess.run(['git','-C',str(root),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture'],check=True)
            r=m.inspect(root)
            self.assertEqual(r['mode'], 'external-source')
            self.assertEqual(r['profiles'], [])


class InputTests(unittest.TestCase):
    def test_bad_json(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.json'
            for content in ['{"x":1,"x":2}','{"x":NaN}','{"x":1e999}']:
                p.write_text(content)
                with self.assertRaises(m.InputError): m.load(p)
    def test_cli_target(self):
        r=subprocess.run([sys.executable,str(ROOT/'scripts/dmd_perf.py'),'target','--baseline-seconds','100','--latency-reduction','.5'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0); self.assertEqual(json.loads(r.stdout)['target_seconds'],50)
    def test_cli_error(self):
        r=subprocess.run([sys.executable,str(ROOT/'scripts/dmd_perf.py'),'target','--baseline-seconds','0','--speedup','1.5'],capture_output=True,text=True)
        self.assertEqual(r.returncode,2)


if __name__=='__main__': unittest.main()
