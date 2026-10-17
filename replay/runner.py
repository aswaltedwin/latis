"""
Replay Batch Runner
Executes N repetitions of pipeline runs under given intervention conditions and aggregates statistical distributions.

Phase 2A: batches preserve the full SAFE/UNSAFE/BLOCKED/ERROR outcome space.
`counts` and `distribution` carry all four core outcomes (plus extended labels);
`outcome_counts` exposes the forensic four-way tally and `error_rate` the
fraction of ERROR runs.  ERROR runs are NEVER folded into SAFE and never
silently discarded — the causal layer treats error-dominant batches as
insufficient evidence (see backend/causal_matrix.py).
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    # Import cycle guard (Phase 2A): real_sandbox.pipeline imports
    # replay.outcomes, so this module must not import it at runtime.
    # It is only needed here as a constructor type annotation.
    from real_sandbox.pipeline import RealSandboxPipeline

from replay.outcomes import (
    empty_outcome_counts,
    compute_outcome_distribution,
    outcome_counts_from_runs,
    OutcomeType,
)

class ReplayRunner:
    def __init__(self, pipeline: RealSandboxPipeline):
        self.pipeline = pipeline

    def run_batch(
        self,
        user_query: str,
        repetitions: int = 50,
        masked_chunks: Optional[Set[str]] = None,
        tool_auth_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes N trials under exact condition and aggregates counts and distribution.
        """
        counts = empty_outcome_counts()
        detailed_runs = []

        for i in range(repetitions):
            try:
                run_res = self.pipeline.execute_run(
                    user_query=user_query,
                    masked_chunks=masked_chunks,
                    tool_auth_override=tool_auth_override
                )
                
                raw_outcome = run_res.get("outcome", OutcomeType.ERROR.value)
                if raw_outcome in counts:
                    counts[raw_outcome] += 1
                else:
                    counts[OutcomeType.ERROR.value] += 1

                detailed_runs.append({
                    "run_idx": i + 1,
                    "outcome": raw_outcome,
                    "tool_call": run_res.get("tool_call")
                })
            except Exception as e:
                counts[OutcomeType.ERROR.value] += 1
                detailed_runs.append({
                    "run_idx": i + 1,
                    "outcome": OutcomeType.ERROR.value,
                    "error": str(e)
                })

        distribution = compute_outcome_distribution(counts, repetitions)

        # Phase 2A forensic outcome accounting: exact four-way tally preserved.
        outcome_counts = outcome_counts_from_runs(
            [r["outcome"] for r in detailed_runs]
        )
        error_rate = round(outcome_counts["ERROR"] / repetitions, 4) if repetitions > 0 else 0.0

        return {
            "repetitions": repetitions,
            "counts": counts,
            "distribution": distribution,
            "outcome_counts": outcome_counts,
            "error_rate": error_rate,
            "sample_runs": detailed_runs[:5]  # first 5 runs for inspection
        }
