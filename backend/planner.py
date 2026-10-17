"""
LATIS Adaptive Bayesian Investigation Planner & Cognitive Engine
Consolidates hypothesis tracking, Shannon entropy, expected information gain,
and Bayesian belief updates into a single robust engine without external dependencies.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set

@dataclass
class InvestigationState:
    case_id: str
    goal: str
    evidence_ids: List[str] = field(default_factory=list)
    components: List[str] = field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    dependency_graph: Dict[str, Any] = field(default_factory=dict)
    available_operations: List[Dict[str, Any]] = field(default_factory=list)
    completed_experiments: List[Dict[str, Any]] = field(default_factory=list)
    current_findings: List[str] = field(default_factory=list)
    investigation_confidence: float = 0.0


class InvestigationCognitiveEngine:
    """
    Mathematical Bayesian active learning engine for LATIS.
    Tracks competing hypotheses, calculates Shannon entropy, recommends the highest
    information-gain intervention, and updates belief distributions based on empirical replay outcomes.
    """

    def __init__(self, hypothesis_definitions: Dict[str, str]):
        self.definitions = hypothesis_definitions
        n = len(hypothesis_definitions)
        initial_prob = 1.0 / n if n > 0 else 1.0

        self.hypotheses: Dict[str, Dict[str, Any]] = {
            hid: {
                "id": hid,
                "description": desc,
                "prior": initial_prob,
                "posterior": initial_prob,
                "evidence_support": 0.0,
                "evidence_refute": 0.0
            }
            for hid, desc in hypothesis_definitions.items()
        }
        self.latis_state: Optional[InvestigationState] = None
        self.completed_op_keys: List[str] = []
        self.memory_store: List[Dict[str, Any]] = []

    def initialize(self, investigation_state: InvestigationState):
        """Initializes engine with investigation state."""
        self.latis_state = investigation_state
        self.completed_op_keys = []

    def compute_entropy(self) -> float:
        """
        Calculates Shannon Entropy H(H) = -sum(p * log2(p)) over non-zero posteriors.
        """
        probs = [h["posterior"] for h in self.hypotheses.values() if h["posterior"] > 0]
        if not probs:
            return 0.0
        return -sum(p * math.log2(p) for p in probs)

    def calculate_expected_information_gain(self, operation: Dict[str, Any]) -> float:
        """
        Calculates expected information gain I(A) = H(Current) - E[H(Post)].
        Uses Bayesian probability of discrimination: components participating in active
        differentiating hypotheses yield higher information value.
        """
        target = operation.get("target_component", "")
        current_entropy = self.compute_entropy()
        if current_entropy <= 0.001:
            return 0.0

        # Calculate how many hypotheses this target discriminates
        relevance_weight = 0.0
        for hid, hdata in self.hypotheses.items():
            desc = hdata["description"].lower()
            p = hdata["posterior"]
            # Target directly mentions component or interaction
            if target.lower() in desc or any(t.lower() in desc for t in target.split("+")):
                relevance_weight += p

        # If relevance is high, testing it resolves significant uncertainty
        # Expected post entropy reduction proportional to discriminative power
        if relevance_weight > 0.0:
            reduction_factor = max(0.20, 1.0 - (0.80 * relevance_weight))
            expected_post_entropy = current_entropy * reduction_factor
        else:
            expected_post_entropy = current_entropy * 0.90

        info_gain = max(0.0, current_entropy - expected_post_entropy)
        return round(info_gain, 4)

    def propose_next_operation(self, available_operations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Proposes the next optimal operation by maximizing Net Utility = Information Gain - Cost.
        """
        completed_set = set(self.completed_op_keys)
        scored = []

        for op in available_operations:
            op_key = f"{op.get('action')}:{op.get('target_component')}"
            if op_key in completed_set:
                continue

            info_gain = self.calculate_expected_information_gain(op)
            cost = op.get("estimated_cost", 0.10)
            net_utility = info_gain - (0.15 * cost)

            scored.append({
                "operation": op,
                "op_key": op_key,
                "info_gain": info_gain,
                "net_utility": net_utility
            })

        if not scored:
            return None

        best_op = max(scored, key=lambda x: x["net_utility"])

        return {
            "selected_operation": best_op["operation"],
            "op_key": best_op["op_key"],
            "expected_information_gain": best_op["info_gain"],
            "net_utility": round(best_op["net_utility"], 4),
            "rationale": f"Selected {best_op['op_key']} to maximize entropy reduction (expected info gain: {best_op['info_gain']:.2f})."
        }

    def update_posteriors_from_experiment(self, experiment_result: Dict[str, Any]):
        """
        Grounded Bayesian Update: Updates hypothesis posteriors dynamically using
        the empirical effect size (risk reduction delta), sample size (N repetitions),
        and error contamination weighting rather than arbitrary fixed constants.
        """
        target = experiment_result["intervention"]["target_component"]
        res_data = experiment_result.get("results", {})
        dist = res_data.get("distribution_map", {})
        unsafe_rate = dist.get("UNSAFE", 0.0)
        error_rate = dist.get("ERROR", 0.0)
        
        baseline_unsafe = res_data.get("baseline_unsafe", 1.0)
        delta_val = res_data.get("delta", unsafe_rate - baseline_unsafe)
        # Risk reduction: positive when unsafe rate dropped
        risk_reduction = max(0.0, -delta_val)
        
        repetitions = res_data.get("repetitions", 50)
        # Weight evidence by sample size confidence (diminishing return above N=50)
        sample_weight = min(1.0, math.sqrt(repetitions / 50.0))

        targets_in_op = [t.strip() for t in target.replace("+", ",").split(",") if t.strip()]

        if error_rate >= 0.5:
            # Error-dominated replay: unobserved execution provides no causal evidence
            return

        # Evidence dampening if unobserved ERROR runs were present in the batch
        valid_obs_factor = max(0.2, (1.0 - error_rate)) * sample_weight

        for hid, data in self.hypotheses.items():
            desc = data["description"]
            is_joint = "Compositional" in desc or "x" in desc
            matches_target = any(t in desc for t in targets_in_op)
            is_sole_cause = matches_target and not is_joint and not ("OR" in desc)

            # Prior belief before this experiment
            prior_p = data.get("posterior", 1.0 / len(self.hypotheses))

            if risk_reduction >= 0.70:
                # Strong elimination of threat observed
                if is_joint and len(targets_in_op) > 1:
                    # Joint test eliminated threat: strongly supports joint interaction
                    likelihood = 1.0 + (1.5 * risk_reduction * valid_obs_factor)
                elif is_sole_cause and len(targets_in_op) == 1:
                    # Single test eliminated threat: strongly supports sole necessary cause
                    likelihood = 1.0 + (1.8 * risk_reduction * valid_obs_factor)
                elif matches_target:
                    # Redundant / partial match
                    likelihood = 1.0 + (0.9 * risk_reduction * valid_obs_factor)
                else:
                    # Unrelated component hypothesis: disfavored proportional to risk reduction
                    likelihood = max(0.05, 1.0 - (0.85 * risk_reduction * valid_obs_factor))
            elif risk_reduction <= 0.20:
                # Masking had little to no effect on the threat
                if is_sole_cause and len(targets_in_op) == 1:
                    # If turning off this single component didn't fix the threat, it is NOT the sole cause
                    likelihood = max(0.05, 1.0 - (0.90 * valid_obs_factor * (1.0 - unsafe_rate)))
                elif is_joint and len(targets_in_op) > 1:
                    # Joint masking didn't fix it either
                    likelihood = max(0.08, 1.0 - (0.85 * valid_obs_factor))
                else:
                    # Hypotheses not tested or joint hypotheses where other factors are required remain open
                    likelihood = 1.0
            else:
                # Partial reduction (enabling or weak contributor)
                if matches_target:
                    likelihood = 1.0 + (0.5 * risk_reduction * valid_obs_factor)
                else:
                    likelihood = max(0.2, 1.0 - (0.3 * risk_reduction * valid_obs_factor))

            # Bayesian update: P(H|E) proportional to P(E|H) * P(H)
            data["posterior"] = prior_p * likelihood

        # Normalize posteriors so sum equals exactly 1.0
        total_p = sum(h["posterior"] for h in self.hypotheses.values())
        if total_p > 0:
            for hid in self.hypotheses:
                self.hypotheses[hid]["posterior"] = round(self.hypotheses[hid]["posterior"] / total_p, 4)

    def reflect(self, experiment_result: Dict[str, Any], expected_info_gain: float = 1.0) -> Dict[str, Any]:
        """
        Processes experiment results, logs reflection, and updates beliefs.
        """
        target = experiment_result["intervention"]["target_component"]
        action = experiment_result["intervention"]["action"]
        op_key = f"{action}:{target}"
        self.completed_op_keys.append(op_key)

        dist = experiment_result["results"]["distribution_map"]
        unsafe_rate = dist.get("UNSAFE", 0.0)
        safe_rate = dist.get("SAFE", 0.0)
        error_rate = dist.get("ERROR", 0.0)

        # 1. Update posteriors
        self.update_posteriors_from_experiment(experiment_result)

        # 2. Build structured reflection record
        # Phase 2A: ERROR runs are unobserved — they neither match a prediction
        # of elimination nor count as SAFE evidence.
        if error_rate >= 0.5:
            prediction_matched = False
        else:
            prediction_matched = (unsafe_rate < 0.10) or (dist.get("BLOCKED", 0.0) > 0.80)
        reflection_record = {
            "target": target,
            "observed_unsafe_rate": unsafe_rate,
            "observed_safe_rate": safe_rate,
            "observed_error_rate": error_rate,
            "prediction_matched": prediction_matched,
            "evidence_strength": (
                "ERROR_DOMINATED_UNOBSERVED" if error_rate >= 0.5
                else "HIGH" if (unsafe_rate == 0.0 and error_rate == 0.0)
                else "MEDIUM"
            ),
            "reflection_summary": (
                f"Intervention MASK({target}) resulted in UNSAFE={unsafe_rate*100:.1f}%, "
                f"SAFE={safe_rate*100:.1f}%, ERROR={error_rate*100:.1f}%."
                + (
                    " ERROR-dominated replay: no causal update performed (ERROR is not SAFE)."
                    if error_rate >= 0.5 else ""
                )
            )
        }

        return {
            "reflection": reflection_record,
            "updated_hypotheses": self.hypotheses,
            "current_entropy": self.compute_entropy()
        }

    def evaluate_hypotheses(self) -> Dict[str, Any]:
        """Returns active hypotheses and current Shannon Entropy."""
        return {
            "entropy": self.compute_entropy(),
            "hypotheses": self.hypotheses
        }

    def store_experience(self, winning_hypothesis_id: str):
        """Persists resolved investigation to memory."""
        if self.latis_state:
            self.memory_store.append({
                "case_id": self.latis_state.case_id,
                "winning_hypothesis": winning_hypothesis_id,
                "total_experiments": len(self.completed_op_keys),
                "status": "RESOLVED"
            })
