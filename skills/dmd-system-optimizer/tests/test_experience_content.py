"""Curated semantic regressions: source direction, scope and advice, never GPU proof."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

SKILL = Path(__file__).resolve().parents[1]
REPO = SKILL.parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
import dmd_knowledge as knowledge
import dmd_navigation as navigation


class ExperienceContentTests(unittest.TestCase):
    def case(self, ident, **context):
        return knowledge.read_case(ident, context)

    def recalled(self, query, ident, context=None):
        result = knowledge.brief_recall(knowledge.recall(query, context or {}, limit=20))
        return next(c for p in result["matches"] for group in p["cases"].values()
                    for c in group if c["id"] == ident)

    def test_r46_retrieval_preserves_source_intervention_direction(self):
        source = json.loads((REPO / "artifacts/publications/20261019/evidence/state_retention/r46/conclusion.json").read_text())
        c = self.recalled("r46", "helios-w6-metadata-r46")
        self.assertEqual(c["facts"]["metadata_reads_per_rank_cycle"], source["actual_native_metadata_reads_per_rank_cycle"])
        self.assertGreater(c["facts"]["metadata_reads_per_rank_cycle"]["candidate"], c["facts"]["metadata_reads_per_rank_cycle"]["control"])
        self.assertEqual(c["facts"]["intervention"], "restore_native_metadata_barrier")
        self.assertEqual(c["facts"]["candidate_C2_s"], source["candidate_C2_s"])
        self.assertEqual(c["interpretation"]["status"], "MECHANISM_UNRESOLVED")
        self.assertEqual(c["evidence_status"], "HISTORICAL_REVIEW")

    def test_three_retained_objects_have_different_cost_conditions(self):
        parameter = self.case("helios-w2-retention-r36")
        activation = self.case("helios-w1-tail8-oom")
        result = self.case("helios-w3-retention-r1")
        self.assertEqual(len({x["pattern"] for x in (parameter, activation, result)}), 3)
        self.assertIn("exposed_gather", parameter["performance_conditions"])
        self.assertNotIn("exposed_gather", activation["performance_conditions"])
        self.assertNotIn("exposed_gather", result["performance_conditions"])
        self.assertIn("recomputation_on_critical_path", activation["performance_conditions"])
        self.assertIn("replay_reuses_saved_output", result["performance_conditions"])
        self.assertEqual(result["facts"]["role"], "Critic")
        self.assertEqual(result["facts"]["blocks"], [38, 39])
        self.assertEqual(result["navigation"]["implementations"][0]["id"], "w3-frozen-ffn-up-replay")

    def test_r43_neighbor_cannot_reverse_r46_direction(self):
        brief = self.recalled("r43", "helios-w6-fa3-r43")
        c = self.case(brief["id"])
        self.assertIn("恢复原生 metadata barrier", c["lesson"])
        self.assertNotIn("后续去同步", c["lesson"])
        self.assertEqual(c["outcome"], "negative_observation")

    def test_w3_historical_runtime_rejects_h100_eight_gpu_as_is(self):
        context = {"workload": "W3", "gpu_family": "H100", "nodes": 1,
                   "gpus_per_node": 8, "physical_gpus": 8}
        c = knowledge.locate_case("helios-w3-retention-r1", context)
        impl = c["implementations"][0]
        self.assertEqual(impl["next_action"], "DO_NOT_APPLY_AS_IS")
        self.assertTrue({"gpu_family", "physical_gpus"}.issubset(
            impl["correctness_comparison"]["conflicting"]))
        # A guard on this historical adapter does not prohibit the mechanism.
        self.assertNotIn("gpu_family", c["correctness_comparison"]["conflicting"])
        self.assertFalse(impl["direct_reuse_verified"])

    def test_historical_summary_discovery_carries_content_erratum(self):
        result = knowledge.source_search("SIX_WORKLOAD_RESULTS_CORRECTION", limit=100)
        source = next(s for s in result["matches"] if s["path"].endswith("SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md"))
        correction = source["content_corrections"][0]
        self.assertEqual(correction["correct_intervention"], "restore_native_metadata_barrier")
        self.assertEqual(correction["reads_per_rank_cycle"], {"control": 0, "candidate": 350})
        self.assertFalse(source["automatic_application"])

    def test_w6_tail_cache_is_output_storage_not_parameter_residency(self):
        c = self.case("helios-w6-retention-r1")
        self.assertEqual(c["facts"]["storage_change"], "pinned_cpu_to_gpu")
        self.assertEqual(c["facts"]["block"], 49)
        self.assertNotIn("exposed_gather", c["performance_conditions"])
        self.assertEqual(c["navigation"]["implementations"][0]["id"], "h3-block49-hybrid-output-cache")

    def test_w6_local_commit_has_public_byte_identical_source_archive(self):
        c = self.case("helios-w6-retention-r1")
        ref = next(r for r in c["evidence"] if "historical_origin" in r)
        self.assertFalse(ref["historical_origin"]["commit_publicly_fetchable"])
        self.assertNotEqual(ref["revision"], ref["historical_origin"]["local_commit"])
        self.assertIn("artifacts/publications/20261031/optimizer_reference_sources/", ref["path"])
        self.assertEqual(hashlib.sha256((REPO / ref["path"]).read_bytes()).hexdigest(), ref["sha256"])
        impl = navigation.validate_catalog()["h3-block49-hybrid-output-cache"]
        for code in navigation.references(impl):
            self.assertEqual(code["revision"], ref["revision"])
            self.assertEqual(code["historical_origin"]["copy_policy"], "BYTE_IDENTICAL_SOURCE_SNAPSHOT")

    def test_g8_failure_does_not_inherit_later_critic_failure(self):
        c = self.case("helios-full5-pr8-oom")
        self.assertEqual(c["failure"]["role"], "Generator")
        self.assertEqual(c["facts"]["temporary_allocation_mib"], 216)
        self.assertEqual(c["facts"]["performance_status"], "NOT_RUN")
        self.assertIn("Step10 Generator checkpoint backward", c["retry"]["minimum_probe"])

    def test_profile_failure_ranks_are_controllers_on_two_nodes(self):
        c = self.recalled("profile teardown", "helios-profile-teardown-incomplete")
        self.assertEqual(c["facts"]["blocked_ranks"], [0, 8])
        self.assertEqual(c["facts"]["completed_rank_count"], 14)
        self.assertEqual(c["facts"]["profile_status"], "INCOMPLETE")

    def test_r3_denominator_compares_reward_optimization_contribution(self):
        c = self.case("rb-end-to-end-denominator")
        self.assertTrue(c["facts"]["r3_reward_service_in_timed_step"])
        self.assertEqual(c["facts"]["r3_with_reward_optimization_speedup"], 2.08)
        self.assertEqual(c["facts"]["r3_optimized_reward_both_arms_speedup"], 1.88)
        self.assertEqual(c["facts"]["r3_comparison_axis"], "optimization_contribution")
        self.assertIn("docs/results/multimodal.md", [r["path"] for r in c["evidence"]])

    def test_disabling_offload_requires_fit_and_conflicts_with_training(self):
        c = self.case("rb-inference-offload-vram-dependent", training_mode="training")
        self.assertEqual(c["next_action"], "DO_NOT_APPLY_AS_IS")
        self.assertTrue(c["performance_conditions"]["full_model_and_workspace_fit"])
        self.assertNotIn("capacity_limits_candidate", c["performance_conditions"])

    def test_small_differences_remain_unresolved_in_brief(self):
        for ident in ("helios-crosskv-nogan-negligible", "helios-static-context-cache-negative"):
            with self.subTest(ident=ident):
                c = self.recalled("缓存", ident)
                self.assertEqual(c["outcome"], "inconclusive")
                self.assertEqual(c["interpretation"]["status"], "HYPOTHESIS_NOT_ISOLATED")

    def test_mixed_workloads_expose_independent_scope_comparisons(self):
        for ident, query in (("helios-fa2-selective-retention", "SAC"), ("helios-modulation-chunk-neutral", "modulation")):
            with self.subTest(ident=ident):
                c = self.recalled(query, ident, {"workload": "W3", "gan_enabled": True, "update_ratio": "G1/C5/EMA1"})
                rows = {o["id"]: o for o in c["observation_comparisons"]}
                self.assertIn("workload", rows["w2"]["context_comparison"]["conflicting"])
                self.assertFalse(rows["w3"]["context_comparison"]["conflicting"])
                self.assertNotIn("workload", self.case(ident)["conditions"])

    def test_timing_branch_is_not_qualified_by_old_checker_branch(self):
        for ident in ("helios-scalar-log-elision", "helios-free-memory-gc"):
            with self.subTest(ident=ident):
                c = self.case(ident, timing_path_consumers_checked=False)
                self.assertFalse(c["facts"]["same_path_validation"])
                self.assertEqual(c["next_action"], "DO_NOT_APPLY_AS_IS")

    def test_concept_and_example_navigation_never_claim_direct_reuse(self):
        for ident, relation, action in (
            ("rb-qkv-layout-state-mapping", "conceptual_reference", "CONCEPT_ONLY"),
            ("rb-fsdp-stable-multiforward", "adaptation_example", "ADAPT_BEFORE_USE")):
            with self.subTest(ident=ident):
                c = knowledge.locate_case(ident, {})["implementations"][0]
                self.assertEqual(c["relationship"], relation)
                self.assertEqual(c["relationship_action"], action)
                self.assertFalse(c["direct_reuse_verified"])

    def test_h100_profile_parser_exposes_topology_conflict(self):
        c = knowledge.locate_case("h100-partial-profile-rank-coverage", {"nodes": 1, "gpus_per_node": 8, "physical_gpus": 8})
        impl = c["implementations"][0]
        self.assertEqual(impl["relationship"], "diagnostic_tool")
        self.assertEqual(impl["next_action"], "DO_NOT_APPLY_AS_IS")
        self.assertIn("physical_gpus", impl["correctness_comparison"]["conflicting"])

    def test_dispatch_navigation_points_to_executor_not_cost_model(self):
        c = knowledge.locate_case("planner-dispatch-realized", {})
        self.assertEqual(c["implementations"][0]["entrypoints"][0]["symbol"], "RolePoolExecutor._run_cycle")

    def test_finite_ema_example_rejects_unbounded_schedule(self):
        c = knowledge.locate_case("helios-w1-ema-r34", {"ema_coefficients": "unbounded_training"})
        self.assertEqual(c["implementations"][0]["relationship"], "historical_implementation")
        self.assertEqual(c["implementations"][0]["next_action"], "DO_NOT_APPLY_AS_IS")
        self.assertEqual(c["evidence_status"], "HISTORICAL_REVIEW")

    def test_analysis_source_cannot_be_relabelled_reusable_runtime(self):
        c = copy.deepcopy(self.case("h100-partial-profile-rank-coverage"))
        c["navigation"]["implementations"][0]["relationship"] = "reusable_candidate"
        with self.assertRaisesRegex(ValueError, "not a runtime"):
            navigation.validate_case(c, navigation.validate_catalog())

    def test_policy_failure_and_canary_do_not_become_full_run_failures(self):
        c = self.case("helios-capacity-policy-rejection-not-oom")
        self.assertTrue(c["facts"]["calibration_completed"])
        self.assertFalse(c["facts"]["cuda_oom"])
        self.assertEqual(c["facts"]["native_timing_status"], "NOT_RUN")
        c = self.recalled("QKV", "helios-frozen-qkv-canary-no-benefit")
        self.assertEqual(c["facts"]["full_workload_performance_status"], "NOT_RUN")
        c = self.recalled("B2", "helios-b2-rollout-numerical-rejection")
        self.assertEqual(c["facts"]["performance_status"], "NOT_RUN")
        self.assertEqual(c["facts"]["preserved_failed_rank_receipts"], 15)

    def test_trainable_timestep_is_not_routed_to_nograd_implementation(self):
        c = knowledge.locate_case("helios-trainable-phase-local-timestep", {"grad_mode": "no_grad"})
        self.assertEqual(c["next_action"], "DO_NOT_APPLY_AS_IS")
        self.assertEqual(c["implementations"][0]["entrypoints"][0]["symbol"], "fused_gated_residual_trainable_temb_backward_chunk")

    def test_logical_clock_mapping_preserves_native_optimizer(self):
        c = self.recalled("logical clock", "helios-logical-native-clock-mapping")
        self.assertFalse(c["facts"]["optimizer_state_rewritten"])
        self.assertEqual(c["facts"]["initial_physical"], {"G": 2, "C": 10, "EMA": 800})
        self.assertEqual(c["facts"]["initial_logical"], {"G": 200, "C": 1000, "EMA": 800})


if __name__ == "__main__":
    unittest.main()
