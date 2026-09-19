"""
Automated End-to-End System Tests:
Verifies FAISS Vector Store, Multi-Agent LangGraph Pipeline,
and Technician Feedback Loop.
"""

import pytest
import shutil
from pathlib import Path
from src.retrieval.vector_store import MaintenanceVectorStore
from src.agents.orchestrator import InfrastructureDecisionOrchestrator
from src.feedback.feedback_handler import FeedbackHandler

def test_knowledge_base_loading():
    store = MaintenanceVectorStore()
    assert store.index.ntotal >= 200, f"Expected >=200 records, got {store.index.ntotal}"
    records = store.get_all_records()
    assert len(records) >= 200
    types = store.get_equipment_types()
    assert "HVAC" in types
    assert "Diesel Generator" in types
    assert "Elevator" in types

def test_semantic_retrieval():
    store = MaintenanceVectorStore()
    query = "Server room cooling failed, temperature spiking and high head pressure"
    results = store.search(query=query, top_k=3, equipment_filter="HVAC")
    
    assert len(results) > 0
    top_result = results[0]
    assert "similarity_score" in top_result
    assert top_result["similarity_score"] > 50.0
    assert "server" in top_result["reported_issue"].lower() or "cooling" in top_result["reported_issue"].lower() or "crac" in top_result["symptoms"].lower()

def test_orchestrator_pipeline():
    orchestrator = InfrastructureDecisionOrchestrator()
    result = orchestrator.process_incident(
        reported_issue="Elevator car stuck between floors with trapped passengers",
        symptoms="Safety circuit tripped, car stopped abruptly, door lock open alarm",
        equipment_type="Elevator",
        location="Science Tower"
    )

    assert "retrieved_cases" in result
    assert len(result["retrieved_cases"]) > 0
    
    diagnosis = result["diagnosis"]
    assert "primary_root_cause" in diagnosis
    assert "confidence_score" in diagnosis
    assert "reasoning_chain" in diagnosis
    assert len(diagnosis["reasoning_chain"]) >= 3
    
    recommendation = result["recommendation"]
    assert "fix_steps" in recommendation
    assert "estimated_cost_usd" in recommendation
    assert "urgency" in recommendation
    assert recommendation["urgency"] == "Critical"  # Because trapped passengers is a critical event

def test_technician_feedback_loop(tmp_path):
    # Use temporary test files
    temp_json = tmp_path / "test_records.json"
    temp_stats = tmp_path / "test_stats.json"
    
    # Copy main dataset to temp
    base_json = Path(__file__).resolve().parent.parent / "data" / "maintenance_records.json"
    shutil.copy(base_json, temp_json)

    store = MaintenanceVectorStore(data_path=str(temp_json))
    initial_count = store.index.ntotal

    feedback_handler = FeedbackHandler(vector_store=store, stats_path=str(temp_stats))
    
    # Add new unique incident
    unique_issue = "Solar inverter arc fault on Building 4 solar array"
    unique_symptoms = "Inverter fault F104, thermal trip on DC isolator switch"
    unique_fix = "Replaced charred DC connector pair and recalibrated arc detection threshold."

    success, msg, details = feedback_handler.record_feedback(
        equipment_type="Electrical Switchgear",
        reported_issue=unique_issue,
        symptoms=unique_symptoms,
        root_cause="Degraded MC4 solar connector high resistance contact",
        fix_action=unique_fix,
        resolution_time_hrs=2.5,
        cost_estimate_usd=180,
        urgency="High",
        is_accurate=True,
        technician_name="Tech Dave"
    )

    assert success is True
    assert store.index.ntotal == initial_count + 1

    # Verify search for this exact issue now retrieves our newly confirmed case with highest similarity
    search_res = store.search(query=f"{unique_issue} {unique_symptoms}", top_k=1)
    assert len(search_res) > 0
    assert search_res[0]["reported_issue"] == unique_issue
    assert search_res[0]["similarity_score"] >= 80.0
