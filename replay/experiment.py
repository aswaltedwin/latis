"""
LATIS Investigation Experiment Evidence Record
Stores complete intervention setup, baseline evidence, and intervention evidence.
Separates raw evidence collection from causal interpretation.
"""

import time
import uuid
import hashlib
import json
from typing import Dict, Any, Optional, Set
from replay.runner import ReplayRunner

class LATISExperiment:
    def __init__(
        self,
        case_id: str,
        target_component: str,
        intervention_action: str,  # e.g. "CONTEXT_PRESERVING_MASK", "TOOL_AUTH_RESTRICT"
        user_query: str,
        repetitions: int = 50,
        masked_chunks: Optional[Set[str]] = None,
        tool_auth_override: Optional[str] = None
    ):
        self.experiment_id = f"EXP-{uuid.uuid4().hex[:8]}"
        self.case_id = case_id
        self.target_component = target_component
        self.intervention_action = intervention_action
        self.user_query = user_query
        self.repetitions = repetitions
        self.masked_chunks = masked_chunks or set()
        self.tool_auth_override = tool_auth_override
        
        self.created_at = time.time()
        self.baseline_results: Optional[Dict[str, Any]] = None
        self.intervention_results: Optional[Dict[str, Any]] = None

    def _hash_condition(self, condition: Dict[str, Any]) -> str:
        serialized = json.dumps(condition, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]

    def execute(self, runner: ReplayRunner, include_baseline: bool = True) -> Dict[str, Any]:
        """
        Executes baseline condition (if needed) and intervention condition.
        """
        # Baseline execution
        if include_baseline and self.baseline_results is None:
            self.baseline_results = runner.run_batch(
                user_query=self.user_query,
                repetitions=self.repetitions,
                masked_chunks=set(),
                tool_auth_override=None
            )

        # Intervention execution
        self.intervention_results = runner.run_batch(
            user_query=self.user_query,
            repetitions=self.repetitions,
            masked_chunks=self.masked_chunks,
            tool_auth_override=self.tool_auth_override
        )

        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        baseline_cond = {
            "masked_chunks": [],
            "tool_auth": "DEFAULT"
        }
        intervention_cond = {
            "masked_chunks": sorted(list(self.masked_chunks)),
            "tool_auth": self.tool_auth_override or "DEFAULT"
        }

        return {
            "experiment_id": self.experiment_id,
            "case_id": self.case_id,
            "target_component": self.target_component,
            "intervention_action": self.intervention_action,
            "user_query": self.user_query,
            "repetitions": self.repetitions,
            "conditions": {
                "baseline": baseline_cond,
                "baseline_hash": self._hash_condition(baseline_cond),
                "intervention": intervention_cond,
                "intervention_hash": self._hash_condition(intervention_cond)
            },
            "baseline_results": self.baseline_results,
            "intervention_results": self.intervention_results,
            "timestamp": self.created_at
        }
