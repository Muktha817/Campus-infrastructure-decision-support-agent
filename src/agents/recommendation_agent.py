"""
Recommendation Agent: Determines actionable repair procedures, safety protocols,
estimated cost, repair duration, and urgency classification.
"""

from typing import Dict, Any, List, Optional
import json
import numpy as np
from .llm_provider import LLMProvider

class RecommendationAgent:
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm = llm_provider or LLMProvider()

    def recommend(
        self,
        reported_issue: str,
        diagnosed_cause: str,
        retrieved_cases: List[Dict[str, Any]],
        equipment_type: str,
        location: str = ""
    ) -> Dict[str, Any]:
        """
        Formulates fix steps, safety precautions, time/cost estimates, and urgency.
        """
        # Determine average cost and time from historical records
        times = [c.get("resolution_time_hrs", 3.0) for c in retrieved_cases if "resolution_time_hrs" in c]
        costs = [c.get("cost_estimate_usd", 350) for c in retrieved_cases if "cost_estimate_usd" in c]

        avg_time = round(float(np.mean(times)), 1) if times else 3.5
        avg_cost = int(np.mean(costs)) if costs else 400

        # Determine urgency
        urgency = self._determine_urgency(reported_issue, diagnosed_cause, retrieved_cases, location)

        # Check if LLM is active
        if self.llm.mode in ["gemini", "openai"]:
            llm_res = self._recommend_with_llm(reported_issue, diagnosed_cause, retrieved_cases, equipment_type, location, avg_time, avg_cost, urgency)
            if llm_res:
                return llm_res

        # Local deterministic expert fallback
        return self._recommend_with_expert_rules(retrieved_cases, avg_time, avg_cost, urgency)

    def _determine_urgency(
        self,
        reported_issue: str,
        cause: str,
        retrieved_cases: List[Dict[str, Any]],
        location: str
    ) -> str:
        text = f"{reported_issue} {cause} {location}".lower()
        if any(w in text for w in ["trapped", "smoke", "fire", "flood", "substation", "server", "hospital", "outage", "ground fault"]):
            return "Critical"
        if any(w in text for w in ["leak", "chiller", "hot spot", "generator", "overheat", "pressure"]):
            return "High"
        
        # Check historical cases majority
        if retrieved_cases:
            urgencies = [c.get("urgency", "Medium") for c in retrieved_cases]
            return max(set(urgencies), key=urgencies.count)
        return "Medium"

    def _recommend_with_llm(
        self,
        reported_issue: str,
        diagnosed_cause: str,
        retrieved_cases: List[Dict[str, Any]],
        equipment_type: str,
        location: str,
        est_time: float,
        est_cost: int,
        default_urgency: str
    ) -> Optional[Dict[str, Any]]:
        top_past_fixes = "\n".join([f"- Case {c.get('id')}: Fix: {c.get('fix_action')}" for c in retrieved_cases])

        system_prompt = (
            "You are a Senior Facility Maintenance Director. "
            "Given the diagnosed cause and historical fixes, formulate an actionable repair plan. "
            "Output strictly valid JSON with:\n"
            "{\n"
            '  "fix_steps": ["Step 1...", "Step 2...", "Step 3..."],\n'
            '  "safety_precautions": ["Precautions..."],\n'
            '  "estimated_time_hrs": 3.5,\n'
            '  "estimated_cost_usd": 450,\n'
            '  "urgency": "High",\n'
            '  "urgency_rationale": "Explanation of urgency level"\n'
            "}"
        )

        user_prompt = (
            f"Equipment: {equipment_type} at {location}\n"
            f"Diagnosed Cause: {diagnosed_cause}\n"
            f"Historical Fixes:\n{top_past_fixes}\n"
            f"Baseline Time: {est_time} hrs, Baseline Cost: ${est_cost}, Baseline Urgency: {default_urgency}\n"
            "Provide the detailed plan in JSON."
        )

        try:
            raw_text = self.llm.generate_completion(user_prompt, system_prompt)
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()
            return json.loads(raw_text)
        except Exception:
            return None

    def _recommend_with_expert_rules(
        self,
        retrieved_cases: List[Dict[str, Any]],
        avg_time: float,
        avg_cost: int,
        urgency: str
    ) -> Dict[str, Any]:
        top_case = retrieved_cases[0] if retrieved_cases else {}
        top_fix = top_case.get("fix_action", "Isolate equipment, inspect assembly, replace worn parts and verify operation.")
        
        # Break down fix into steps
        raw_steps = [s.strip() for s in top_fix.replace(";", ".").split(".") if len(s.strip()) > 5]
        if len(raw_steps) < 2:
            raw_steps = [
                "Isolate power feed and perform Lockout/Tagout (LOTO).",
                f"Perform primary remediation: {top_fix}",
                "Conduct mechanical and electrical calibration tests.",
                "Return equipment to BMS supervisory control and document resolution."
            ]
        else:
            raw_steps.insert(0, "Isolate power supply and follow standard safety isolation.")
            raw_steps.append("Perform full load validation test and clear supervisory alarms.")

        safety = [
            "Mandatory Lockout/Tagout (LOTO) on primary power source.",
            "Wear appropriate PPE (Arc flash visor/gloves for switchgear, safety goggles & insulated footwear).",
            "Ensure backup system or emergency bypass is online if servicing critical systems."
        ]

        rationales = {
            "Critical": "Immediate life safety hazard, risk of server/hospital downtime, or total facility outage.",
            "High": "Equipment degradation actively impacting building operations or risking permanent hardware failure.",
            "Medium": "Operational efficiency or comfort reduced; requires resolution within 24-48 hours.",
            "Low": "Minor cosmetic, nuisance or scheduled maintenance item with no immediate disruption risk."
        }

        return {
            "fix_steps": raw_steps,
            "safety_precautions": safety,
            "estimated_time_hrs": avg_time,
            "estimated_cost_usd": avg_cost,
            "urgency": urgency,
            "urgency_rationale": rationales.get(urgency, "Assigned based on historical maintenance impact.")
        }
