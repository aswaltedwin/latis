"""
LATIS v1 Milestone 14: Adaptive Joint-Candidate Expansion Policy Engine
Implements adaptive expansion of candidate joint interaction pairs:
- Avoids testing all N*(N-1)/2 pairs (which explodes compute).
- Dynamically selects candidate pairs based on posterior uncertainty and evidence sufficiency.
"""

from typing import Dict, Any, List, Tuple

class AdaptiveJointExpansionPolicy:
    @staticmethod
    def select_joint_candidates(
        single_masks: Dict[str, float],
        hypotheses_posteriors: Dict[str, Any],
        mode: str = "ADAPTIVE"
    ) -> List[Tuple[str, str]]:
        comps = list(single_masks.keys())
        
        if mode == "FIXED_TOP_2":
            if len(comps) >= 2:
                return [(comps[0], comps[1])]
            return []
            
        elif mode == "TOP_K_3":
            pairs = []
            top3 = comps[:3]
            for i in range(len(top3)):
                for j in range(i+1, len(top3)):
                    pairs.append((top3[i], top3[j]))
            return pairs
            
        elif mode == "POSTERIOR_THRESHOLD":
            # Select all components where single mask yielded P=0.0 OR P=1.0 under redundant path
            candidates = [c for c, r in single_masks.items() if r == 0.0 or r >= 0.90][:4]
            pairs = []
            for i in range(len(candidates)):
                for j in range(i+1, len(candidates)):
                    pairs.append((candidates[i], candidates[j]))
            return pairs
            
        elif mode == "ADAPTIVE":
            # Dynamic expansion: start with top 2, expand to top 4 if top hypothesis remains compositional
            top_h = max(hypotheses_posteriors.values(), key=lambda h: h["posterior"])
            candidates = comps[:2]
            
            # If evidence suggests compositional/redundant interaction, expand to top 4 candidates
            if "Compositional" in top_h["description"] or "redundant" in top_h["description"].lower() or len([c for c, r in single_masks.items() if r == 0.0]) >= 2:
                candidates = comps[:4]
                
            pairs = []
            for i in range(len(candidates)):
                for j in range(i+1, len(candidates)):
                    pairs.append((candidates[i], candidates[j]))
            return pairs

        return []
