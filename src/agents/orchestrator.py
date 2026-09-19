"""
LangGraph Multi-Agent Orchestrator:
Retrieval Agent -> Diagnosis Agent -> Recommendation Agent -> Decision Synthesizer
"""

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

from .retrieval_agent import RetrievalAgent
from .diagnosis_agent import DiagnosisAgent
from .recommendation_agent import RecommendationAgent
from .llm_provider import LLMProvider
from ..retrieval.vector_store import MaintenanceVectorStore

class IncidentState(TypedDict):
    reported_issue: str
    symptoms: str
    equipment_type: str
    location: str
    top_k: int
    retrieved_cases: List[Dict[str, Any]]
    diagnosis: Dict[str, Any]
    recommendation: Dict[str, Any]
    execution_logs: List[str]

class InfrastructureDecisionOrchestrator:
    def __init__(self, vector_store: Optional[MaintenanceVectorStore] = None):
        self.vector_store = vector_store or MaintenanceVectorStore()
        self.llm_provider = LLMProvider()
        self.retrieval_agent = RetrievalAgent(self.vector_store)
        self.diagnosis_agent = DiagnosisAgent(self.llm_provider)
        self.recommendation_agent = RecommendationAgent(self.llm_provider)
        
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(IncidentState)

        # 1. Retrieval Node
        def retrieve_node(state: IncidentState) -> Dict[str, Any]:
            logs = list(state.get("execution_logs", []))
            logs.append(f"🔍 [Retrieval Agent] Scanning FAISS index for '{state['reported_issue']}' (Scope: {state['equipment_type']})")
            
            cases = self.retrieval_agent.retrieve(
                reported_issue=state["reported_issue"],
                symptoms=state.get("symptoms", ""),
                equipment_type=state.get("equipment_type", "All"),
                location=state.get("location", ""),
                top_k=state.get("top_k", 4)
            )
            logs.append(f"✅ [Retrieval Agent] Retrieved {len(cases)} historical matches. Top match score: {cases[0]['similarity_score'] if cases else 0}%")
            return {"retrieved_cases": cases, "execution_logs": logs}

        # 2. Diagnosis Node
        def diagnose_node(state: IncidentState) -> Dict[str, Any]:
            logs = list(state.get("execution_logs", []))
            logs.append("🧠 [Diagnosis Agent] Correlating telemetry symptoms with historical failure patterns...")

            diagnosis = self.diagnosis_agent.diagnose(
                reported_issue=state["reported_issue"],
                symptoms=state.get("symptoms", ""),
                equipment_type=state.get("equipment_type", "All"),
                retrieved_cases=state["retrieved_cases"],
                location=state.get("location", "")
            )
            logs.append(f"✅ [Diagnosis Agent] Primary Cause identified: '{diagnosis.get('primary_root_cause')}' ({diagnosis.get('confidence_score')}% confidence)")
            return {"diagnosis": diagnosis, "execution_logs": logs}

        # 3. Recommendation Node
        def recommend_node(state: IncidentState) -> Dict[str, Any]:
            logs = list(state.get("execution_logs", []))
            logs.append("🛠️ [Recommendation Agent] Formulating repair procedure, calculating cost/time estimates and urgency...")

            recommendation = self.recommendation_agent.recommend(
                reported_issue=state["reported_issue"],
                diagnosed_cause=state["diagnosis"].get("primary_root_cause", ""),
                retrieved_cases=state["retrieved_cases"],
                equipment_type=state.get("equipment_type", "All"),
                location=state.get("location", "")
            )
            logs.append(f"✅ [Recommendation Agent] Assigned Urgency: {recommendation.get('urgency')} | Est. Time: {recommendation.get('estimated_time_hrs')}h | Est. Cost: ${recommendation.get('estimated_cost_usd')}")
            return {"recommendation": recommendation, "execution_logs": logs}

        # Add Nodes & Edges
        builder.add_node("retrieve_cases", retrieve_node)
        builder.add_node("diagnose_root_cause", diagnose_node)
        builder.add_node("recommend_action", recommend_node)

        builder.add_edge(START, "retrieve_cases")
        builder.add_edge("retrieve_cases", "diagnose_root_cause")
        builder.add_edge("diagnose_root_cause", "recommend_action")
        builder.add_edge("recommend_action", END)

        return builder.compile()

    def process_incident(
        self,
        reported_issue: str,
        symptoms: str = "",
        equipment_type: str = "All",
        location: str = "",
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Executes the LangGraph multi-agent decision support pipeline.
        """
        initial_state: IncidentState = {
            "reported_issue": reported_issue,
            "symptoms": symptoms,
            "equipment_type": equipment_type,
            "location": location,
            "top_k": top_k,
            "retrieved_cases": [],
            "diagnosis": {},
            "recommendation": {},
            "execution_logs": []
        }

        final_state = self.graph.invoke(initial_state)
        return final_state
