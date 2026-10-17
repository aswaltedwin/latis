"""
LATIS v1 Milestone 12: LATIS Causal-Sufficiency Governance Engine
Enforces governance admissibility & closure rules:
- Is closure justified?
- Are candidate joint interaction dependencies unresolved?
- Does information gain drop prematurely before interaction testing is complete?

Phase 2A: SAFE, UNSAFE, BLOCKED and ERROR are strictly distinct outcomes.
ERROR runs are unobserved runs — never SAFE, never risk elimination.  Evidence
whose UNSAFE rate is computed over error-contaminated replays is inadmissible
as closure evidence.
"""

from typing import Dict, Any, List

from replay.outcomes import is_error_dominated

class LATISCausalSufficiencyGovernance:
    @staticmethod
    def evaluate_closure_admissibility(
        single_masks: Dict[str, float],
        joint_masks: Dict[tuple, float],
        proposed_action: Dict[str, Any],
        hypotheses_posteriors: Dict[str, Any],
        single_masks_outcome_counts: Dict[str, Dict[str, int]] = None,
        joint_masks_outcome_counts: Dict[tuple, Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Determines whether the proposed investigation action or closure request is admissible.
        If entropy drops prematurely but joint interaction hypotheses remain unresolved,
        governance REJECTS premature closure and forces candidate joint interaction checks.

        Phase 2A: optional forensic outcome counts may be supplied per mask
        ({component_id: counts} / {pair_tuple: counts}).  A "zero unsafe rate"
        measured over an error-dominated batch is NOT elimination evidence and
        cannot justify closure.
        """
        top_hyp = max(hypotheses_posteriors.values(), key=lambda h: h["posterior"])
        
        # Check if single masks eliminated risk or if all single masks failed (P=1.0)
        # Phase 2A: a zero rate backed only by ERROR-contaminated replay is not
        # elimination evidence.  Absent/empty forensic counts (legacy callers)
        # keep the original rate-only semantics.
        single_counts = single_masks_outcome_counts or {}
        joint_counts = joint_masks_outcome_counts or {}

        def _counts_genuine(counts: Any) -> bool:
            """True when the counts provide no ERROR-contamination signal."""
            if not (isinstance(counts, dict) and counts):
                return True  # legacy / no forensic counts supplied
            return not is_error_dominated(counts)

        zero_singles = [
            c for c, r in single_masks.items()
            if r == 0.0 and _counts_genuine(single_counts.get(c))
        ]
        one_singles = [c for c, r in single_masks.items() if r > 0.50]
        
        # Rule 1: If top hypothesis is compositional or if single masks returned 1.0 (redundant path),
        # closure is INADMISSIBLE until candidate joint mask is executed.
        if (len(zero_singles) >= 2 or len(one_singles) >= 2) and not any(
            _counts_genuine(joint_counts.get(p)) for p in joint_masks
        ):
            return {
                "admissible": False,
                "override_action": "EXECUTE_CANDIDATE_JOINT_MASK",
                "reason": "Governance rejected premature closure: Candidate joint interaction hypotheses remain unresolved."
            }
            
        return {
            "admissible": True,
            "override_action": None,
            "reason": "Closure admissible under LATIS evidence sufficiency rules."
        }
