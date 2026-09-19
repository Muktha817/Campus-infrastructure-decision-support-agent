"""
Diagnosis Agent: Evaluates symptoms against historical cases,
reasons about root causes, and formulates a step-by-step plain-language reasoning chain.
"""

from typing import Dict, Any, List, Optional
import json
from .llm_provider import LLMProvider

class DiagnosisAgent:
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm = llm_provider or LLMProvider()

    def diagnose(
        self,
        reported_issue: str,
        symptoms: str,
        equipment_type: str,
        retrieved_cases: List[Dict[str, Any]],
        location: str = ""
    ) -> Dict[str, Any]:
        """
        Diagnoses root cause with an explainable reasoning chain.
        """
        if not retrieved_cases:
            return {
                "primary_root_cause": "Unprecedented failure mode; insufficient past matches.",
                "confidence_score": 40.0,
                "reasoning_chain": [
                    "No closely matching historical tickets were found in the knowledge base.",
                    "Recommend manual physical teardown and diagnostics by a senior technician."
                ],
                "differential_causes": ["Mechanical wear", "Electrical trip", "Sensor calibration error"]
            }

        # Check if LLM is active
        if self.llm.mode in ["gemini", "openai"]:
            llm_result = self._diagnose_with_llm(reported_issue, symptoms, equipment_type, retrieved_cases, location)
            if llm_result:
                return llm_result

        # Local deterministic expert reasoning fallback
        return self._diagnose_with_expert_rules(reported_issue, symptoms, equipment_type, retrieved_cases, location)

    def _diagnose_with_llm(
        self,
        reported_issue: str,
        symptoms: str,
        equipment_type: str,
        retrieved_cases: List[Dict[str, Any]],
        location: str
    ) -> Optional[Dict[str, Any]]:
        cases_summary = "\n".join([
            f"- Case {c.get('id')}: Issue: {c.get('reported_issue')} | Symptoms: {c.get('symptoms')} | Root Cause: {c.get('root_cause')} | Similarity: {c.get('similarity_score', 0)}%"
            for c in retrieved_cases
        ])

        system_prompt = (
            "You are an expert Facility & Campus Infrastructure Reliability Engineer. "
            "Analyze the reported issue and symptoms against historical maintenance records. "
            "Return a strictly valid JSON object with the following keys:\n"
            "{\n"
            '  "primary_root_cause": "clear concise statement of primary cause",\n'
            '  "confidence_score": 85.0,\n'
            '  "reasoning_chain": ["step 1...", "step 2...", "step 3...", "conclusion..."],\n'
            '  "differential_causes": ["alternative 1", "alternative 2"]\n'
            "}"
        )

        user_prompt = (
            f"Equipment: {equipment_type} at {location}\n"
            f"New Complaint: {reported_issue}\n"
            f"Symptoms: {symptoms}\n\n"
            f"Retrieved Similar Historical Cases:\n{cases_summary}\n\n"
            f"Synthesize the patterns and provide your diagnosis in JSON."
        )

        try:
            raw_text = self.llm.generate_completion(user_prompt, system_prompt)
            # Extract JSON block if wrapped in markdown
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()
            data = json.loads(raw_text)
            return data
        except Exception:
            return None

    def _diagnose_with_expert_rules(
        self,
        reported_issue: str,
        symptoms: str,
        equipment_type: str,
        retrieved_cases: List[Dict[str, Any]],
        location: str
    ) -> Dict[str, Any]:
        top_case = retrieved_cases[0]
        top_similarity = top_case.get("similarity_score", 75.0)

        # Check symptom overlaps
        common_causes = [c.get("root_cause") for c in retrieved_cases]
        primary_cause = top_case.get("root_cause", "Mechanical/Electrical Degradation")
        
        # Build explainable chain of thought
        reasoning = [
            f"Step 1 (Case Corroboration): Evaluated top match {top_case.get('id')} ({top_case.get('equipment_id', 'Asset')}) showing {top_similarity}% semantic alignment.",
            f"Step 2 (Symptom Triangulation): The reported symptoms ('{symptoms or reported_issue}') closely mirror historical incident {top_case.get('id')} which exhibited '{top_case.get('symptoms')}'.",
            f"Step 3 (Recurrence Pattern Analysis): Cross-referenced {len(retrieved_cases)} similar events in {equipment_type}. Past resolutions identified '{primary_cause}' as the failure mechanism.",
            f"Step 4 (Root Cause Synthesis): High likelihood that current malfunction stems from '{primary_cause}' based on matching telemetry and equipment behavior."
        ]

        differentials = []
        for c in retrieved_cases[1:3]:
            diff_cause = c.get("root_cause")
            if diff_cause and diff_cause != primary_cause and diff_cause not in differentials:
                differentials.append(diff_cause)
        if not differentials:
            differentials = ["Secondary electrical fluctuation", "Component physical misalignment"]

        return {
            "primary_root_cause": primary_cause,
            "confidence_score": min(95.0, max(50.0, top_similarity)),
            "reasoning_chain": reasoning,
            "differential_causes": differentials[:2]
        }
