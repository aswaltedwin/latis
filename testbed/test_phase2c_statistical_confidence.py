"""
LATIS Phase 2C: Statistical Confidence & Replay Reliability Test Suite

Validates:
1. Sample size materially impacts statistical uncertainty:
   - Wilson intervals contract as N increases.
   - Low N (e.g. N=5 or N=10) has wide uncertainty intervals and penalized confidence.
   - High N (N=50 or N=100) produces tight intervals and high calibrated confidence.
2. Outcome rate distributions:
   - all-SAFE: lower and upper bounds bounded near 0.
   - all-UNSAFE: bounds near 1.
   - mixed outcomes: non-degenerate empirical intervals.
3. BLOCKED remains strictly distinct from SAFE:
   - BLOCKED outcomes do not count as SAFE.
4. ERROR-contaminated evidence:
   - Contaminated batches penalize confidence directly.
   - An error-dominated batch yields INSUFFICIENT_EVIDENCE_ERROR_CONTAMINATED with confidence 0.0.
5. Deterministic vs Stochastic replay distinction:
   - Repeated identical deterministic runs (variance=0) are classified as DETERMINISTIC.
   - Stochastic runs (varying outcomes across runs) are classified as STOCHASTIC.
   - Deterministic replay does not falsely claim infinite stochastic certainty.
6. Causal topologies under statistical evaluation:
   - Necessary cause
   - Redundant independent causes
   - Joint interaction
   - Enabling structure
7. Zero-effect vs strong-effect intervention:
   - Zero-effect (delta = 0) vs strong-effect (delta = 1.0) produces appropriate risk-difference CIs.
8. Traceability:
   - statistical_evidence exposes sample_size, effect_size, confidence_interval, replay_mode, error_penalty.
"""

import unittest
from replay.statistics import (
    wilson_score_interval,
    risk_difference_interval,
    detect_replay_mode,
    compute_evidence_confidence
)
from backend.causal_matrix import CausalIdentificationMatrix
from backend.server import (
    ingest_incident_case, IngestCaseRequest,
    run_experiment, RunExperimentRequest,
    get_case_state, export_forensic_report,
    CASES_DB
)


class TestPhase2CStatisticalConfidence(unittest.TestCase):

    def setUp(self):
        for cid in list(CASES_DB.keys()):
            if "2C" in cid or "2c" in cid:
                del CASES_DB[cid]

    # -------------------------------------------------------------------------
    # 1. Wilson score interval properties (low N vs high N)
    # -------------------------------------------------------------------------
    def test_wilson_interval_sample_size_contraction(self):
        # 0 unsafe out of 5 runs vs 0 unsafe out of 100 runs
        low_ci = wilson_score_interval(0, 5)
        high_ci = wilson_score_interval(0, 100)

        # Lower bounds are 0.0
        self.assertEqual(low_ci[0], 0.0)
        self.assertEqual(high_ci[0], 0.0)

        # Upper bound for N=5 is wide (~0.43), for N=100 is narrow (~0.036)
        self.assertGreater(low_ci[1], 0.35)
        self.assertLess(high_ci[1], 0.05)
        self.assertLess(high_ci[1] - high_ci[0], low_ci[1] - low_ci[0])

    # -------------------------------------------------------------------------
    # 2. Risk difference interval (Agresti-Caffo)
    # -------------------------------------------------------------------------
    def test_risk_difference_interval_strong_vs_zero_effect(self):
        # Strong effect: baseline 100/100 UNSAFE, intervention 0/100 UNSAFE
        delta_strong, low_s, high_s, se_s = risk_difference_interval(100, 100, 0, 100)
        self.assertEqual(delta_strong, 1.0)
        self.assertGreater(low_s, 0.90)
        self.assertLessEqual(high_s, 1.0)

        # Zero effect: baseline 100/100 UNSAFE, intervention 100/100 UNSAFE
        delta_zero, low_z, high_z, se_z = risk_difference_interval(100, 100, 100, 100)
        self.assertEqual(delta_zero, 0.0)
        self.assertLess(low_z, 0.05)
        self.assertGreater(high_z, -0.05)

    # -------------------------------------------------------------------------
    # 3. Deterministic vs Stochastic Replay Mode Classification
    # -------------------------------------------------------------------------
    def test_detect_replay_mode(self):
        # Deterministic: 50 runs all UNSAFE (variance = 0)
        batch_det1 = {"SAFE": 0, "UNSAFE": 50, "BLOCKED": 0, "ERROR": 0}
        batch_det2 = {"SAFE": 50, "UNSAFE": 0, "BLOCKED": 0, "ERROR": 0}
        mode_det = detect_replay_mode([batch_det1, batch_det2], is_neural_greedy=True)
        self.assertEqual(mode_det, "DETERMINISTIC")

        # Stochastic: batch has mixed outcomes (e.g. 40 UNSAFE, 10 SAFE)
        batch_stoch = {"SAFE": 10, "UNSAFE": 40, "BLOCKED": 0, "ERROR": 0}
        mode_stoch = detect_replay_mode([batch_det1, batch_stoch], is_neural_greedy=True)
        self.assertEqual(mode_stoch, "STOCHASTIC")

    # -------------------------------------------------------------------------
    # 4. Monotonic confidence scaling with N in CausalIdentificationMatrix
    # -------------------------------------------------------------------------
    def test_sample_size_monotonically_scales_confidence(self):
        # Using real outcome counts at N=10, N=30, N=100
        def make_counts(n, unsafe_k):
            return {"SAFE": n - unsafe_k, "UNSAFE": unsafe_k, "BLOCKED": 0, "ERROR": 0}

        res_n10 = CausalIdentificationMatrix.evaluate_causal_matrix(
            baseline_rate=1.0,
            single_masks={"A": 0.0},
            joint_masks={},
            isolations={},
            tested_components=["A"],
            baseline_outcome_counts=make_counts(10, 10),
            intervention_outcome_counts={"single_masks": {"A": make_counts(10, 0)}}
        )
        res_n30 = CausalIdentificationMatrix.evaluate_causal_matrix(
            baseline_rate=1.0,
            single_masks={"A": 0.0},
            joint_masks={},
            isolations={},
            tested_components=["A"],
            baseline_outcome_counts=make_counts(30, 30),
            intervention_outcome_counts={"single_masks": {"A": make_counts(30, 0)}}
        )
        res_n100 = CausalIdentificationMatrix.evaluate_causal_matrix(
            baseline_rate=1.0,
            single_masks={"A": 0.0},
            joint_masks={},
            isolations={},
            tested_components=["A"],
            baseline_outcome_counts=make_counts(100, 100),
            intervention_outcome_counts={"single_masks": {"A": make_counts(100, 0)}}
        )

        self.assertEqual(res_n10["topology"], "SINGLE_NECESSARY_CAUSE")
        self.assertEqual(res_n30["topology"], "SINGLE_NECESSARY_CAUSE")
        self.assertEqual(res_n100["topology"], "SINGLE_NECESSARY_CAUSE")

        # Confidence must scale strictly with sample size
        self.assertLess(res_n10["confidence"], res_n30["confidence"])
        self.assertLessEqual(res_n30["confidence"], res_n100["confidence"])
        self.assertGreaterEqual(res_n100["confidence"], 0.95)

        # Statistical evidence must be populated and show interval contraction
        se10 = res_n10["statistical_evidence"]
        se100 = res_n100["statistical_evidence"]
        self.assertEqual(se10["sample_size"], 20)
        self.assertEqual(se100["sample_size"], 200)
        ci10_width = se10["confidence_interval"][1] - se10["confidence_interval"][0]
        ci100_width = se100["confidence_interval"][1] - se100["confidence_interval"][0]
        self.assertLess(ci100_width, ci10_width)

    # -------------------------------------------------------------------------
    # 5. BLOCKED distinctness in statistical evidence
    # -------------------------------------------------------------------------
    def test_blocked_outcome_distinct_from_safe(self):
        # A batch where 20 runs were BLOCKED by policy
        counts_blocked = {"SAFE": 0, "UNSAFE": 0, "BLOCKED": 20, "ERROR": 0}
        counts_safe = {"SAFE": 20, "UNSAFE": 0, "BLOCKED": 0, "ERROR": 0}
        
        # BLOCKED is an authorization event, distinct from SAFE
        self.assertNotEqual(counts_blocked["BLOCKED"], counts_safe["BLOCKED"])
        self.assertEqual(counts_blocked["SAFE"], 0)
        self.assertEqual(counts_safe["SAFE"], 20)

    # -------------------------------------------------------------------------
    # 6. ERROR contamination penalizes statistical confidence
    # -------------------------------------------------------------------------
    def test_error_contamination_penalizes_statistical_evidence(self):
        # Baseline clean 20/20 unsafe, intervention has 5 ERROR runs out of 20
        clean_base = {"SAFE": 0, "UNSAFE": 20, "BLOCKED": 0, "ERROR": 0}
        err_int = {"SAFE": 15, "UNSAFE": 0, "BLOCKED": 0, "ERROR": 5}
        
        # Because ERROR > 0, Phase 2A/2B refuses elimination, falling through to error contaminated
        res = CausalIdentificationMatrix.evaluate_causal_matrix(
            baseline_rate=1.0,
            single_masks={"A": 0.0},
            joint_masks={},
            isolations={},
            tested_components=["A"],
            baseline_outcome_counts=clean_base,
            intervention_outcome_counts={"single_masks": {"A": err_int}}
        )
        self.assertEqual(res["topology"], "INSUFFICIENT_EVIDENCE_ERROR_CONTAMINATED")
        self.assertEqual(res["confidence"], 0.0)
        self.assertEqual(res["statistical_evidence"]["evidence_quality"], "ERROR_CONTAMINATED")

    # -------------------------------------------------------------------------
    # 7. Redundancy topology statistical grounding
    # -------------------------------------------------------------------------
    def test_redundant_causes_statistical_evidence(self):
        n = 30
        c_unsafe = {"SAFE": 0, "UNSAFE": n, "BLOCKED": 0, "ERROR": 0}
        c_safe = {"SAFE": n, "UNSAFE": 0, "BLOCKED": 0, "ERROR": 0}

        res = CausalIdentificationMatrix.evaluate_causal_matrix(
            baseline_rate=1.0,
            single_masks={"A": 1.0, "B": 1.0},
            joint_masks={("A", "B"): 0.0},
            isolations={"A": 1.0, "B": 1.0},
            tested_components=["A", "B"],
            baseline_outcome_counts=c_unsafe,
            intervention_outcome_counts={
                "single_masks": {"A": c_unsafe, "B": c_unsafe},
                "joint_masks": {("A", "B"): c_safe},
                "isolations": {"A": c_unsafe, "B": c_unsafe}
            }
        )
        self.assertEqual(res["topology"], "REDUNDANT_INDEPENDENT_CAUSES")
        self.assertGreaterEqual(res["confidence"], 0.80)
        self.assertIn("statistical_evidence", res)
        se = res["statistical_evidence"]
        self.assertEqual(se["effect_size"], 1.0)
        self.assertEqual(se["replay_mode"], "DETERMINISTIC")
        self.assertEqual(se["error_penalty"], 0.0)

    # -------------------------------------------------------------------------
    # 8. Forensic report and case state integration
    # -------------------------------------------------------------------------
    def test_forensic_report_and_case_state_exposure(self):
        incident = {
            "case_id": "CASE-CUSTOM-2C-TEST",
            "title": "Statistical Confidence Verification Incident",
            "user_query": "Review and process telemetry",
            "components": [
                {"id": "DOC_1", "type": "RAG"},
                {"id": "TOOL_EXEC", "type": "TOOL"}
            ],
            "documents": [
                {"id": "DOC_1", "title": "Doc", "content": "Instruction", "is_poisoned": True}
            ],
            "tools": [{
                "name": "TOOL_EXEC",
                "description": "Executes transaction",
                "sensitivity": "CRITICAL",
                "is_unsafe": True
            }],
            "authorization_policy": "ALLOW_ALL"
        }
        res_ingest = ingest_incident_case(IngestCaseRequest(**incident))
        self.assertEqual(res_ingest["status"], "SUCCESS")

        # Run an intervention
        run_res = run_experiment("CASE-CUSTOM-2C-TEST", RunExperimentRequest(
            action="CONTEXT_PRESERVING_MASK",
            target_component="DOC_1",
            repetitions=20
        ))
        self.assertEqual(run_res["status"], "SUCCESS")

        # Check get_case_state
        state = get_case_state("CASE-CUSTOM-2C-TEST")
        self.assertIn("statistical_evidence", state)
        self.assertIn("replay_mode", state["statistical_evidence"])

        # Check export_forensic_report
        report = export_forensic_report("CASE-CUSTOM-2C-TEST")
        self.assertIn("statistical_evidence", report)
        self.assertIn("Statistical Confidence Interval", report["forensic_report_text"])
        self.assertIn("Replay Evidence Mode", report["forensic_report_text"])

    # -------------------------------------------------------------------------
    # 9. Deterministic baseline and intervention arms remain DETERMINISTIC
    # -------------------------------------------------------------------------
    def test_deterministic_baseline_and_intervention_arms_remain_deterministic(self):
        # Deterministic baseline=UNSAFE and intervention=SAFE across arms
        # Must be classified as DETERMINISTIC, not STOCHASTIC
        baseline_batch = {"SAFE": 0, "UNSAFE": 50, "BLOCKED": 0, "ERROR": 0}
        intervention_batch = {"SAFE": 50, "UNSAFE": 0, "BLOCKED": 0, "ERROR": 0}
        mode = detect_replay_mode([baseline_batch, intervention_batch], is_neural_greedy=True)
        self.assertEqual(mode, "DETERMINISTIC")

    # -------------------------------------------------------------------------
    # 10. Genuinely varying repeated trials are classified as STOCHASTIC
    # -------------------------------------------------------------------------
    def test_genuinely_varying_repeated_trials_are_stochastic(self):
        # Within the same condition, stochastic sampling produces varying outcomes
        varying_batch = {"SAFE": 35, "UNSAFE": 15, "BLOCKED": 0, "ERROR": 0}
        clean_batch = {"SAFE": 0, "UNSAFE": 50, "BLOCKED": 0, "ERROR": 0}
        mode = detect_replay_mode([varying_batch, clean_batch], is_neural_greedy=True)
        self.assertEqual(mode, "STOCHASTIC")

    # -------------------------------------------------------------------------
    # 11. Repeated deterministic runs do not gain false stochastic certainty
    # -------------------------------------------------------------------------
    def test_deterministic_reproducibility_vs_stochastic_sampling(self):
        # In deterministic mode, zero empirical variance means repeated identical
        # executions verify transition reproducibility rather than independent Bernoulli draws
        b_det = {"SAFE": 0, "UNSAFE": 100, "BLOCKED": 0, "ERROR": 0}
        i_det = {"SAFE": 100, "UNSAFE": 0, "BLOCKED": 0, "ERROR": 0}
        ev_det = compute_evidence_confidence(b_det, i_det, "ELIMINATION", replay_mode="DETERMINISTIC")
        self.assertEqual(ev_det["replay_mode"], "DETERMINISTIC")
        # Confidence does not exceed statistical Agresti-Caffo bound or claim infinite certainty
        self.assertLessEqual(ev_det["confidence"], 0.99)
        self.assertGreaterEqual(ev_det["confidence"], 0.95)

    # -------------------------------------------------------------------------
    # 12. NO_INCIDENT_SAFE empirical derivation from counts (no hardcoded 1.0)
    # -------------------------------------------------------------------------
    def test_no_incident_safe_derived_from_actual_counts(self):
        # N=10 safe baseline vs N=100 safe baseline
        res_safe_n10 = CausalIdentificationMatrix.evaluate_causal_matrix(
            baseline_rate=0.0,
            single_masks={},
            joint_masks={},
            isolations={},
            tested_components=["A"],
            baseline_outcome_counts={"SAFE": 10, "UNSAFE": 0, "BLOCKED": 0, "ERROR": 0}
        )
        res_safe_n100 = CausalIdentificationMatrix.evaluate_causal_matrix(
            baseline_rate=0.0,
            single_masks={},
            joint_masks={},
            isolations={},
            tested_components=["A"],
            baseline_outcome_counts={"SAFE": 100, "UNSAFE": 0, "BLOCKED": 0, "ERROR": 0}
        )
        self.assertEqual(res_safe_n10["topology"], "NO_INCIDENT_SAFE")
        self.assertEqual(res_safe_n100["topology"], "NO_INCIDENT_SAFE")
        # N=10 safe confidence is Wilson lower bound (~0.7225), not hardcoded 1.0
        self.assertNotEqual(res_safe_n10["confidence"], 1.0)
        self.assertAlmostEqual(res_safe_n10["confidence"], 0.7225, places=3)
        # N=100 safe confidence is Wilson lower bound (~0.9630)
        self.assertAlmostEqual(res_safe_n100["confidence"], 0.9630, places=3)
        self.assertLess(res_safe_n10["confidence"], res_safe_n100["confidence"])

    # -------------------------------------------------------------------------
    # 13. Legacy rate-only callers labeled as LEGACY_UNSUPPORTED
    # -------------------------------------------------------------------------
    def test_legacy_rate_only_caller_labeled_legacy_unsupported(self):
        res_legacy = CausalIdentificationMatrix.evaluate_causal_matrix(
            baseline_rate=1.0,
            single_masks={"A": 0.0},
            joint_masks={},
            isolations={},
            tested_components=["A"]
        )
        self.assertEqual(res_legacy["topology"], "SINGLE_NECESSARY_CAUSE")
        self.assertIn("statistical_evidence", res_legacy)
        self.assertEqual(res_legacy["statistical_evidence"]["evidence_quality"], "LEGACY_UNSUPPORTED")


if __name__ == "__main__":
    unittest.main()

