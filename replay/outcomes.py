"""
Replay Outcome Vocabulary
Canonical outcome definitions for LATIS investigation replays.
No module should invent its own labels.

Phase 2A contract — SAFE, UNSAFE, BLOCKED and ERROR are four STRICTLY distinct
runtime outcomes:
- ERROR must NEVER be interpreted as SAFE (or as risk elimination).
- BLOCKED must remain distinct from SAFE (a blocked action is not a safe action).
- Replay distributions and forensic reports must preserve all four outcome counts.
- Causal logic must not treat ERROR-only evidence as risk elimination.
"""

from enum import Enum
from typing import Dict, List, Any

class OutcomeType(str, Enum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    LOOP = "LOOP"
    INCOMPLETE = "INCOMPLETE"

ALL_OUTCOMES: List[str] = [outcome.value for outcome in OutcomeType]

# The four core Phase 2A outcomes.  Every replay batch must report counts for
# all four, even when three of them are zero.
CORE_OUTCOMES: List[str] = [OutcomeType.SAFE.value, OutcomeType.UNSAFE.value, OutcomeType.BLOCKED.value, OutcomeType.ERROR.value]


def canonical_outcome(raw: Any) -> str:
    """Maps an arbitrary raw outcome label onto the canonical vocabulary.
    Unknown labels are fail-closed to ERROR — never silently to SAFE."""
    if not isinstance(raw, str):
        return OutcomeType.ERROR.value
    if raw in ALL_OUTCOMES:
        return raw
    return OutcomeType.ERROR.value


def outcome_counts_from_runs(raw_outcomes: List[str]) -> Dict[str, int]:
    """Builds canonical SAFE/UNSAFE/BLOCKED/ERROR counts from raw run outcomes.
    Unknown or missing labels count as ERROR (fail-closed evidence accounting)."""
    counts = {k: 0 for k in CORE_OUTCOMES}
    for raw in raw_outcomes:
        counts[canonical_outcome(raw)] += 1
    return counts


def is_error_dominated(counts: Dict[str, int], threshold: float = 0.5) -> bool:
    """True when more than `threshold` of the observed runs are ERROR.
    ERROR-contaminated batches are not valid causal evidence of elimination."""
    total = sum(int(counts.get(k, 0)) for k in ALL_OUTCOMES)
    if total <= 0:
        return True
    return (int(counts.get(OutcomeType.ERROR.value, 0)) / total) > threshold

def empty_outcome_counts() -> Dict[str, int]:
    return {outcome.value: 0 for outcome in OutcomeType}

def compute_outcome_distribution(counts: Dict[str, int], total_trials: int) -> Dict[str, float]:
    if total_trials == 0:
        return {k: 0.0 for k in counts}
    return {k: round(v / total_trials, 4) for k, v in counts.items()}
