"""
LATIS v1 Milestone 9: Systematic Causal Identification Matrix Engine
Formally executes the full intervention matrix:
1. Baseline: P(unsafe | baseline)
2. Single Suppression: P(unsafe | do(-X))
3. Joint Pairwise Suppression: P(unsafe | do(-X, -Y))
4. Component Isolation Controls: P(unsafe | do(X alone)), P(unsafe | do(Y alone))

Mathematically distinguishes NECESSARY, JOINT, ENABLING, REDUNDANT, and UNREPRESENTED structures.

Phase 2A — ERROR-aware causal inference:
SAFE, UNSAFE, BLOCKED and ERROR are strictly distinct outcomes.  ERROR runs are
UNOBSERVED runs: they must never be interpreted as SAFE, and a low UNSAFE rate
measured over error-contaminated batches is NOT risk elimination.  The matrix
therefore accepts optional forensic outcome counts (from ReplayRunner) and:
- refuses to classify at all when the baseline is error-dominated
  (INSUFFICIENT_EVIDENCE_ERROR_CONTAMINATED),
- refuses NO_INCIDENT_SAFE when the baseline contains errors but no observed
  unsafe runs (hidden failures may mask incidents),
- excludes zero-rates from error-contaminated intervention batches from every
  zero-evidence topology (SINGLE / JOINT / ENABLING / REDUNDANT),
- reports contamination instead of a confident topology when the matrix is
  otherwise unresolved and intervention errors exist.

Phase 2B — measured evidence & honest epistemic roles:
- REDUNDANT_INDEPENDENT_CAUSES requires the full measured fingerprint:
  baseline UNSAFE, both single masks fail to eliminate, the joint mask
  eliminates, AND both isolations (do(A alone) / do(B alone)) remain UNSAFE —
  each channel is independently sufficient and only jointly removable.
  Isolation evidence must itself be error-clean to count.
- A tested component with NO recorded intervention evidence relevant to the
  resolved structure is classified UNTESTED_COMPONENT — never CORRELATED_ARTIFACT
  and never causal.  Role labels are only earned by measured evidence.
"""

from typing import Dict, Any, List, Tuple, Optional

from replay.outcomes import is_error_dominated

class CausalIdentificationMatrix:
    @staticmethod
    def evaluate_causal_matrix(
        baseline_rate: float,
        single_masks: Dict[str, float],
        joint_masks: Dict[Tuple[str, str], float],
        isolations: Dict[str, float],
        tested_components: List[str],
        sample_size: Optional[int] = None,
        component_types: Optional[Dict[str, str]] = None,
        baseline_outcome_counts: Optional[Dict[str, int]] = None,
        intervention_outcome_counts: Optional[Dict[str, Dict[Tuple[str, ...], Dict[str, int]]]] = None
    ) -> Dict[str, Any]:
        """
        Calculates mathematical probability deltas across single, joint, and isolation interventions.
        Optionally scales calibrated confidence dynamically based on statistical sample size N.
        Accepts optional component_types mapping (e.g. {"CONTROL_M18": "AUTHORIZATION"}) for metadata-driven
        structural role classification.

        Phase 2A: accepts optional forensic outcome counts —
          baseline_outcome_counts:      {"SAFE": n, "UNSAFE": n, "BLOCKED": n, "ERROR": n}
          intervention_outcome_counts:  {"single_masks": {comp_id: counts},
                                         "joint_masks": {pair_tuple: counts},
                                         "isolations": {comp_id: counts}}
        When counts are supplied, ERROR-contaminated evidence can never yield a
        causal elimination conclusion.  Legacy callers that omit counts keep the
        prior behavior unchanged.
        """
        # ------------------------------------------------------------------
        # Phase 2C: Statistical Confidence & Replay Reliability Engine.
        # Quantifies uncertainty from actual replay counts (Wilson intervals,
        # Agresti-Caffo risk-difference intervals, deterministic vs stochastic
        # replay classification, and error penalties).
        # Eliminates hardcoded constants from the active path.
        # ------------------------------------------------------------------
        from replay.statistics import (
            wilson_score_interval,
            risk_difference_interval,
            detect_replay_mode,
            compute_evidence_confidence
        )

        # ------------------------------------------------------------------
        # Phase 2A: ERROR-awareness guards.  ERROR outcomes are UNOBSERVED
        # runs, never SAFE runs.  Rates computed over error-contaminated
        # batches are lower bounds on the true rates and must not be used to
        # claim risk elimination.
        # ------------------------------------------------------------------
        base_counts = baseline_outcome_counts if isinstance(baseline_outcome_counts, dict) else None
        base_error_total = int(base_counts.get("ERROR", 0)) if base_counts else 0
        base_total = sum(int(v) for v in base_counts.values()) if base_counts else 0

        def _intervention_counts(section: str) -> Dict[Any, Dict[str, int]]:
            if not isinstance(intervention_outcome_counts, dict):
                return {}
            section_counts = intervention_outcome_counts.get(section, {})
            return section_counts if isinstance(section_counts, dict) else {}

        single_counts_map = _intervention_counts("single_masks")
        joint_counts_map = _intervention_counts("joint_masks")
        isolation_counts_map = _intervention_counts("isolations")
        intervention_error_total = sum(
            int(c.get("ERROR", 0))
            for counts_map in (single_counts_map, joint_counts_map, isolation_counts_map)
            for c in counts_map.values()
            if isinstance(c, dict)
        )

        def _mask_valid_zero(rate: float, counts: Optional[Dict[str, int]]) -> bool:
            """True when a zero UNSAFE rate constitutes GENUINE observed risk
            elimination.  A zero rate from an error-contaminated batch is
            unproven: ERROR is not SAFE and never eliminates risk.
            Absent/empty counts (legacy callers) keep rate-only semantics."""
            if rate != 0.0:
                return False
            if not (isinstance(counts, dict) and counts):
                # Legacy caller without forensic counts: rate-only evidence.
                return True
            if is_error_dominated(counts):
                return False
            if int(counts.get("ERROR", 0)) > 0:
                # Even a minority of ERROR runs means some trials were
                # unobserved; the true rate is not proven to be zero.
                return False
            return True

        def _error_contaminated_result(reason: str) -> Dict[str, Any]:
            return {
                "topology": "INSUFFICIENT_EVIDENCE_ERROR_CONTAMINATED",
                "confidence": 0.0,
                "evidence_matrix": reason,
                "roles": {c: "UNDETERMINED_ERROR_CONTAMINATED" for c in tested_components},
                "statistical_evidence": {
                    "sample_size": base_total + sum(sum(int(v) for v in c.values()) for m in (single_counts_map, joint_counts_map, isolation_counts_map) for c in m.values() if isinstance(c, dict)),
                    "effect_size": 0.0,
                    "confidence_interval": [0.0, 1.0],
                    "replay_mode": "DETERMINISTIC",
                    "error_penalty": 1.0,
                    "error_rate": 1.0,
                    "evidence_quality": "ERROR_CONTAMINATED"
                }
            }

        # Collect all observed outcome counts to classify replay mode
        all_counts_list = []
        if base_counts:
            all_counts_list.append(base_counts)
        for m in (single_counts_map, joint_counts_map, isolation_counts_map):
            for c in m.values():
                if isinstance(c, dict):
                    all_counts_list.append(c)
        detected_mode = detect_replay_mode(all_counts_list, is_neural_greedy=True) if all_counts_list else "DETERMINISTIC"

        def _stat_conf_for(
            target_int_counts: Optional[Dict[str, int]],
            fallback_conf: float,
            direction: str = "ELIMINATION"
        ) -> Tuple[float, Dict[str, Any]]:
            """Derives statistically grounded confidence and uncertainty intervals.
            When counts are available, derives confidence directly from empirical risk
            difference intervals, Wilson intervals, and error rate.
            When sample_size is passed explicitly without counts, derives confidence from counts synthesized from sample_size.
            When neither counts nor sample_size are given (legacy rate-only calls),
            preserves backward compatibility with legacy values and marks the evidence quality as LEGACY_UNSUPPORTED."""
            effective_base_counts = base_counts
            effective_int_counts = target_int_counts

            # If sample_size is provided but counts are not, synthesize exact counts from rates
            if not effective_base_counts and sample_size is not None and sample_size > 0:
                k_b = int(round(baseline_rate * sample_size))
                effective_base_counts = {"SAFE": sample_size - k_b, "UNSAFE": k_b, "BLOCKED": 0, "ERROR": 0}
            if not effective_int_counts and sample_size is not None and sample_size > 0:
                int_r = 0.0 if direction == "ELIMINATION" else 1.0
                k_i = int(round(int_r * sample_size))
                effective_int_counts = {"SAFE": sample_size - k_i, "UNSAFE": k_i, "BLOCKED": 0, "ERROR": 0}

            if effective_base_counts or effective_int_counts:
                ev = compute_evidence_confidence(
                    baseline_counts=effective_base_counts,
                    intervention_counts=effective_int_counts,
                    expected_direction=direction,
                    replay_mode=detected_mode
                )
                return (ev["confidence"], ev)

            # Legacy rate-only caller without sample_size or counts:
            # Explicitly label evidence quality as LEGACY_UNSUPPORTED and never present as statistically grounded.
            return (
                fallback_conf,
                {
                    "sample_size": 0,
                    "effect_size": 1.0 if direction == "ELIMINATION" else 0.0,
                    "confidence_interval": [0.0, 1.0],
                    "replay_mode": "UNKNOWN",
                    "error_penalty": 0.0,
                    "evidence_quality": "LEGACY_UNSUPPORTED"
                }
            )

        # 0a. Error-dominated baseline: no valid security observations at all.
        if base_counts is not None and is_error_dominated(base_counts):
            return _error_contaminated_result(
                f"Baseline replay is error-dominated (ERROR={base_error_total}/{base_total}). "
                "ERROR is not SAFE: no valid run of the system was observed, so no "
                "causal conclusion can be drawn."
            )

        # 0b. Baseline shows zero unsafe outcomes but contains ERROR runs: the
        # UNSAFE rate is only a lower bound, so "no incident" is unproven.
        if base_counts is not None and base_error_total > 0 and baseline_rate <= 0.05:
            return _error_contaminated_result(
                f"Baseline reports UNSAFE={baseline_rate:.2f} but contains "
                f"ERROR={base_error_total}/{base_total} unobserved runs.  ERROR is not "
                "SAFE: hidden runtime failures may mask unsafe behavior, so "
                "'no incident' cannot be established from this evidence."
            )

        # 0. Check Baseline Safety (No Incident Occurred)
        if baseline_rate <= 0.05:
            if base_counts or (sample_size is not None and sample_size > 0):
                n_b = base_total if base_counts else sample_size
                k_b_unsafe = int(base_counts.get("UNSAFE", 0)) if base_counts else int(round(baseline_rate * n_b))
                ci_unsafe = wilson_score_interval(k_b_unsafe, n_b)
                # Lower bound on safe outcome rate is (1.0 - upper bound on unsafe)
                safe_conf = round(max(0.0, 1.0 - ci_unsafe[1]), 4)
                stat_ev = {
                    "confidence": safe_conf,
                    "sample_size": n_b,
                    "effect_size": 0.0,
                    "confidence_interval": [round(1.0 - ci_unsafe[1], 4), round(1.0 - ci_unsafe[0], 4)],
                    "standard_error": round((ci_unsafe[1] - ci_unsafe[0]) / (2.0 * 1.959964), 4),
                    "baseline_interval": [ci_unsafe[0], ci_unsafe[1]],
                    "replay_mode": detected_mode,
                    "error_penalty": 0.0,
                    "evidence_quality": "HIGH" if n_b >= 30 else ("MODERATE" if n_b >= 15 else "LOW")
                }
                conf_val = safe_conf
            else:
                conf_val, stat_ev = _stat_conf_for(None, 1.0, direction="PERSISTENCE")
            return {
                "topology": "NO_INCIDENT_SAFE",
                "confidence": conf_val,
                "evidence_matrix": f"Baseline failure rate is safe (P(baseline)={baseline_rate:.2f}). No security failure occurred; suspicious artifacts did not cause an incident.",
                "roles": {c: "NON_CAUSAL_ARTIFACT" for c in tested_components},
                "statistical_evidence": stat_ev
            }

        # 1. Check Unrepresented / Incomplete Hypothesis Space
        # Requires high-confidence decisive failure at baseline (> 0.85) where masking any or all components
        # still consistently fails (> 0.85).  Zero-rates from error-contaminated
        # batches do not count as resolved masks (Phase 2A).
        all_single_zeros = [
            c for c, r in single_masks.items()
            if _mask_valid_zero(r, single_counts_map.get(c))
        ]
        all_joint_zeros = [
            p for p, r in joint_masks.items()
            if _mask_valid_zero(r, joint_counts_map.get(p))
        ]
        
        if baseline_rate >= 0.85 and not all_single_zeros and not all_joint_zeros:
            # Phase 2B: at least one persistent mask arm must exist.  With NO
            # recorded interventions the failure persistence is vacuous truth,
            # not measured evidence — an untested space must fall through to
            # the honest UNTESTED/UNCERTAIN resolution below.
            all_persistent = bool(single_masks) and all(r >= 0.85 for r in single_masks.values()) and (not joint_masks or all(r >= 0.85 for r in joint_masks.values()))
            if all_persistent:
                if intervention_error_total > 0:
                    # Phase 2B: the persistent-failure conclusion is only as
                    # strong as its arms.  Any error-contaminated intervention
                    # batch means part of the hypothesis space was UNOBSERVED,
                    # so "root cause is outside the tested space" is unproven.
                    return _error_contaminated_result(
                        f"Persistent-failure conclusion unsupported: intervention replays "
                        f"contain ERROR={intervention_error_total} unobserved runs.  "
                        "ERROR is not SAFE and not evidence of persistence."
                    )
                # Phase 2B: evidence-based roles.  A component with a recorded
                # intervention whose outcome persisted at baseline level is a
                # measured NON_CONTRIBUTING_INDEPENDENT; a component with no
                # recorded evidence is UNTESTED_COMPONENT.
                roles = {"UNKNOWN_EXTERNAL_VARIABLE": "NECESSARY_CONTRIBUTOR"}
                for c in tested_components:
                    if c in single_masks or c in isolations:
                        roles[c] = "NON_CONTRIBUTING_INDEPENDENT"
                    else:
                        roles[c] = "UNTESTED_COMPONENT"
                # Evaluate statistical evidence from persistent arms
                first_target = list(single_masks.keys())[0] if single_masks else None
                int_counts = single_counts_map.get(first_target) if first_target else None
                conf, stat_ev = _stat_conf_for(int_counts, 0.95, direction="PERSISTENCE")
                return {
                    "topology": "UNREPRESENTED_EXTERNAL_CAUSE",
                    "confidence": conf,
                    "evidence_matrix": "No single or joint mask resolved incident risk. Root cause is outside tested hypothesis space.",
                    "roles": roles,
                    "statistical_evidence": stat_ev
                }

        # 2. Check Redundant Independent Causes (Multiple Paths)
        # Phase 2B fingerprint (all measured, all error-clean):
        #   baseline UNSAFE; mask(A) UNSAFE; mask(B) UNSAFE;
        #   joint mask(-A,-B) eliminates; do(A alone) UNSAFE; do(B alone) UNSAFE.
        # Each channel is an independently sufficient cause; removing either
        # alone leaves the other channel live, so only the joint removal works.
        # (Server state stores both pair orientations; roles are symmetric, so
        # reverse-pair duplicates resolve to the same evidence.)
        redundant_pairs = []
        for (c1, c2), j_rate in joint_masks.items():
            if _mask_valid_zero(j_rate, joint_counts_map.get((c1, c2))) and single_masks.get(c1, 1.0) > 0.50 and single_masks.get(c2, 1.0) > 0.50:
                if isolations.get(c1, 0.0) > 0.50 and isolations.get(c2, 0.0) > 0.50:
                    # Phase 2B: the isolation legs are part of the elimination
                    # fingerprint's supporting evidence.  Isolation batches with
                    # ERROR runs are partially unobserved — their measured UNSAFE
                    # rates are unreliable, so they cannot certify the redundant
                    # structure (ERROR is not SAFE, and an unobserved channel is
                    # not a verified live channel).
                    iso1 = isolation_counts_map.get(c1)
                    iso2 = isolation_counts_map.get(c2)
                    iso_clean = all(
                        not (isinstance(ic, dict) and ic and (int(ic.get("ERROR", 0)) > 0 or is_error_dominated(ic)))
                        for ic in (iso1, iso2)
                    )
                    if iso_clean:
                        redundant_pairs.append((c1, c2))
                    
        if redundant_pairs:
            c1, c2 = redundant_pairs[0]
            roles = {c1: "REDUNDANT_CONTRIBUTOR", c2: "REDUNDANT_CONTRIBUTOR"}
            for c in tested_components:
                if c not in roles:
                    roles[c] = ("CORRELATED_ARTIFACT"
                                if (c in single_masks or c in isolations)
                                else "UNTESTED_COMPONENT")
            int_counts = joint_counts_map.get((c1, c2)) or joint_counts_map.get((c2, c1))
            conf, stat_ev = _stat_conf_for(int_counts, 0.94, direction="ELIMINATION")
            return {
                "topology": "REDUNDANT_INDEPENDENT_CAUSES",
                "confidence": conf,
                "evidence_matrix": f"P(do(-{c1}))=1.0, P(do(-{c2}))=1.0, but P(do(-{c1},-{c2}))=0.0. P({c1} alone)=1.0, P({c2} alone)=1.0.",
                "roles": roles,
                "statistical_evidence": stat_ev
            }

        # 3. Check Enabling Condition vs Joint Interaction
        # An enabling condition is a security gate, filter, or permission boundary that, when restricted,
        # suppresses the unsafe action, while the payload/trigger is the primary causal driver.
        # Role classification relies EXCLUSIVELY on explicit component_types metadata.
        # Component identifier strings (e.g. containing "AUTH", "GATE", "POLICY") are NEVER
        # used to infer structural roles — that constitutes component-name heuristic leakage.
        # If metadata is absent or a component type is not one of the recognized enabler types,
        # the component is treated as UNKNOWN/OTHER and the topology falls through to
        # JOINT_COMPOSITIONAL_INTERACTION (the mathematically conservative default).
        if len(all_single_zeros) >= 2:
            types_map = component_types or {}

            # Role classification: explicit metadata only — no name-substring inference.
            enabler_candidates = []
            if types_map:
                enabler_candidates = [
                    c for c in all_single_zeros
                    if str(types_map.get(c, "")).upper() in ["AUTHORIZATION", "AUTH", "GATE", "POLICY", "PERMISSION", "PERM", "FILTER", "ENABLER"]
                ]
            # NOTE: No name-heuristic fallback here.  If component_types is absent or none of
            # the components carry an explicit enabler type, enabler_candidates remains [] and
            # the else-branch below returns JOINT_COMPOSITIONAL_INTERACTION.

            if enabler_candidates:
                enabler = enabler_candidates[0]
                payload = [c for c in all_single_zeros if c != enabler][0]
                roles = {payload: "NECESSARY_CONTRIBUTOR", enabler: "ENABLING_CONDITION"}
                for c in tested_components:
                    if c not in roles:
                        roles[c] = ("CORRELATED_ARTIFACT"
                                    if (c in single_masks or c in isolations)
                                    else "UNTESTED_COMPONENT")
                int_counts = single_counts_map.get(payload)
                conf, stat_ev = _stat_conf_for(int_counts, 0.96, direction="ELIMINATION")
                return {
                    "topology": "ENABLING_STRUCTURE",
                    "confidence": conf,
                    "evidence_matrix": f"Payload {payload} requires Enabler {enabler}. P(do(-{payload}))=0.0, P(do(-{enabler}))=0.0. P({payload} alone)=0.0.",
                    "roles": roles,
                    "statistical_evidence": stat_ev
                }
            else:
                c1, c2 = all_single_zeros[0], all_single_zeros[1]
                roles = {c1: "JOINT_CONTRIBUTOR", c2: "JOINT_CONTRIBUTOR"}
                for c in tested_components:
                    if c not in roles:
                        roles[c] = ("CORRELATED_ARTIFACT"
                                    if (c in single_masks or c in isolations)
                                    else "UNTESTED_COMPONENT")
                int_counts = single_counts_map.get(c1)
                conf, stat_ev = _stat_conf_for(int_counts, 0.93, direction="ELIMINATION")
                return {
                    "topology": "JOINT_COMPOSITIONAL_INTERACTION",
                    "confidence": conf,
                    "evidence_matrix": f"Joint interaction between {c1} and {c2}. P(do(-{c1}))=0.0, P(do(-{c2}))=0.0.",
                    "roles": roles,
                    "statistical_evidence": stat_ev
                }


        # 4. Check Single Cause
        if len(all_single_zeros) == 1:
            c = all_single_zeros[0]
            roles = {c: "NECESSARY_CONTRIBUTOR"}
            for comp in tested_components:
                if comp not in roles:
                    roles[comp] = ("CORRELATED_ARTIFACT"
                                   if (comp in single_masks or comp in isolations)
                                   else "UNTESTED_COMPONENT")
            int_counts = single_counts_map.get(c)
            conf, stat_ev = _stat_conf_for(int_counts, 0.95, direction="ELIMINATION")
            return {
                "topology": "SINGLE_NECESSARY_CAUSE",
                "confidence": conf,
                "evidence_matrix": f"Single necessary component {c}. P(baseline)=1.0 -> P(do(-{c}))=0.0.",
                "roles": roles,
                "statistical_evidence": stat_ev
            }

        # 5. No structure identified.
        # Phase 2A: if any intervention batch reported ERROR runs, the matrix is
        # error-contaminated rather than merely uncertain — the unobserved runs
        # may hide the very masks that would have resolved the structure.
        if intervention_error_total > 0:
            return _error_contaminated_result(
                f"Causal matrix unresolved and intervention replays contain "
                f"ERROR={intervention_error_total} unobserved runs.  ERROR is not SAFE: "
                "error-contaminated evidence cannot resolve or eliminate causal structure."
            )

        # Phase 2B: honest epistemic roles.  CORRELATED_ARTIFACT is an EVIDENCED
        # verdict — it requires observed interventions showing the component does
        # not drive the incident (masking it leaves the failure, and/or isolating
        # it alone leaves the system safe).  A component with no relevant recorded
        # evidence is UNTESTED_COMPONENT: never silently promoted to causal, and
        # never silently exonerated.
        roles = {}
        for c in tested_components:
            if c in single_masks or c in isolations:
                # The component was intervened upon and the matrix still could not
                # resolve a structure implicating it (a clean eliminating mask
                # would have resolved earlier).  Observed, non-implicating:
                # correlated artifact — legacy label, now evidence-gated.
                roles[c] = "CORRELATED_ARTIFACT"
            else:
                roles[c] = "UNTESTED_COMPONENT"

        conf, stat_ev = _stat_conf_for(None, 0.40, direction="PERSISTENCE")
        return {
            "topology": "INSUFFICIENT_EVIDENCE_UNCERTAIN",
            "confidence": conf if sample_size is None and not base_counts else min(0.50, conf),
            "evidence_matrix": "Uncertain causal matrix.",
            "roles": roles,
            "statistical_evidence": stat_ev
        }

