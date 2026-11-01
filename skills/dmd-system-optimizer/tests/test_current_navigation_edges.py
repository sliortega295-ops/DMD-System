"""Incomplete local checkouts and exact user identities must not derail recall."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[1]
ROOT = SKILL.parents[1]
sys.path.insert(0,str(SKILL/'scripts'))
import dmd_current as current


class NavigationEdges(unittest.TestCase):
    def minimal(self,root):
        for relative in (current.REGISTRY,current.PROFILES,current.WORKLOADS,current.STRATEGIES):
            dst=root/relative;dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/relative,dst)
        folder=root/'recipes/dmd2_sd15';folder.mkdir(parents=True)
        for name in ('training.json','README.md'):
            shutil.copyfile(ROOT/'recipes/dmd2_sd15'/name,folder/name)
        return folder

    def test_bad_recipes_warn_and_do_not_hide_healthy_recipe(self):
        for value in ([],None,{'schema_version':'distillschedule.training.v1','model':[]},
                      {'schema_version':'distillschedule.training.v1','model':{'parameters':None}}):
            with self.subTest(value=value),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);folder=self.minimal(root)
                (folder/'training.in-progress.json').write_text(json.dumps(value))
                result=current.lookup('SD1.5 DMD2',repo=root)
                self.assertEqual(result['status'],'CHECKOUT_METADATA_NOT_RUNTIME_VALIDATED')
                self.assertEqual(result['matches'][0]['id'],'dmd2_sd15')
                self.assertTrue(any(w['action']=='RECIPE_SKIPPED' for w in result['warnings']))
                self.assertFalse(result['automatic_application'])

    def test_invalid_json_warns_and_preserves_healthy_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=self.minimal(root)
            (folder/'training.wip.json').write_text('{')
            result=current.lookup('SD1.5',repo=root)
            self.assertTrue(result['warnings'])
            self.assertEqual(result['matches'][0]['id'],'dmd2_sd15')

    def test_empty_readme_uses_folder_name_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=self.minimal(root);(folder/'README.md').write_text('')
            result=current.lookup('SD1.5',repo=root)
            self.assertEqual(result['matches'][0]['title'],'dmd2_sd15')
            self.assertTrue(any(w['action']=='FOLDER_NAME_USED' for w in result['warnings']))

    def test_exact_short_workload_ids_have_no_unrelated_results(self):
        for query,wanted in (('W4','self_forcing'),('w6','h3_ar')):
            result=current.lookup(query)
            self.assertEqual([r['id'] for r in result['matches']],[wanted])
            self.assertGreater(result['matches'][0]['score'],0)
        self.assertEqual(current.lookup('W999')['matches'],[])

    def test_chinese_official_self_forcing_is_first_and_not_dmd2(self):
        result=current.lookup('官方Self-Forcing')
        self.assertEqual(result['matches'][0]['id'],'self_forcing_official')
        self.assertFalse(any(r['id'].startswith('dmd2') for r in result['matches']))
        self.assertFalse(any('dmd2' in c['id'] for c in result['related_capabilities_require_recipe_check']))

    def test_h3_query_does_not_suggest_dmd2_only_capabilities(self):
        result=current.lookup('H3 AR FA3')
        self.assertEqual(result['matches'][0]['id'],'h3_ar')
        self.assertFalse(any('dmd2' in c['implementation'] for c in result['related_capabilities_require_recipe_check']))
        self.assertFalse(any(r['id']=='wan_ar' for r in result['matches']))

    def test_previous_model_queries_still_resolve(self):
        for query,wanted in [('SD1.5 DMD2 resident tree fused Adam','dmd2_sd15'),
                             ('Official Self-Forcing FSDP2 R6','self_forcing_official'),
                             ('HunyuanVideo-1.5','hunyuan_video_dmd')]:
            self.assertEqual(current.lookup(query)['matches'][0]['id'],wanted)
        result=current.lookup('FLUX Qwen Image frozen base sharing')
        self.assertEqual({r['id'] for r in result['matches'][:2]},{'flux_dmd','qwen_image_dmd'})
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
