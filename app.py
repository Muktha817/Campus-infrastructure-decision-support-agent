"""
Facility Incident Command Center
Campus/Facility Infrastructure Decision-Support Platform
Track B - B3: Multi-Agent Orchestration & Decision Support
Enterprise Facility Management Architecture: LangGraph + FAISS + SentenceTransformers
"""

import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path

# Configure page with professional industrial layout
st.set_page_config(
    page_title="Facility Incident Command Center | Infrastructure Decision Support",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Industrial CSS System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #17212B;
    }

    /* Overall background styling */
    .stApp {
        background-color: #F8FAFC;
    }

    /* Command Center Top Header */
    .command-header-container {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        padding-bottom: 1rem;
        margin-bottom: 1.2rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .command-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #17212B;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .command-subtitle {
        font-size: 0.92rem;
        color: #64748B;
        margin-top: 0.25rem;
    }
    .header-status-strip {
        display: flex;
        gap: 0.75rem;
        align-items: center;
        flex-wrap: wrap;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.75rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #E2E8F0;
        background: #FFFFFF;
        color: #334155;
    }
    .status-dot-green {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #16A34A;
        display: inline-block;
    }

    /* 4-Card Industrial KPI Summary Row */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #17212B;
        line-height: 1.2;
    }
    .kpi-meta {
        font-size: 0.75rem;
        color: #16A34A;
        margin-top: 0.35rem;
        font-weight: 500;
    }

    /* Multi-Agent Workflow Process Ribbon */
    .process-ribbon {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.75rem 1.25rem;
        margin: 1.2rem 0;
        font-size: 0.82rem;
        color: #475569;
        font-weight: 500;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .process-step {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }
    .step-num {
        background: #F1F5F9;
        color: #1E293B;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid #CBD5E1;
    }
    .step-num-active {
        background: #2563EB;
        color: #FFFFFF;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .process-arrow {
        color: #94A3B8;
        font-weight: 600;
    }

    /* Section Containers */
    .section-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #17212B;
        letter-spacing: -0.01em;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }
    .section-desc {
        font-size: 0.85rem;
        color: #64748B;
        margin-bottom: 1rem;
    }

    /* Scenario Card Styles */
    .scenario-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 0.75rem;
        margin-bottom: 1.2rem;
    }
    .scenario-btn-box {
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.85rem;
        background: #FFFFFF;
        transition: border-color 0.15s ease;
    }
    .scenario-btn-box:hover {
        border-color: #2563EB;
        background: #F8FAFC;
    }
    .scenario-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #17212B;
        margin-bottom: 0.2rem;
    }
    .scenario-sub {
        font-size: 0.75rem;
        color: #64748B;
        line-height: 1.3;
    }

    /* Status & Urgency Badges */
    .badge-critical {
        background: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FCA5A5;
        padding: 3px 9px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.78rem;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }
    .badge-high {
        background: #FFFBEB;
        color: #D97706;
        border: 1px solid #FCD34D;
        padding: 3px 9px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.78rem;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }
    .badge-medium {
        background: #FEFCE8;
        color: #CA8A04;
        border: 1px solid #FEF08A;
        padding: 3px 9px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.78rem;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }
    .badge-low {
        background: #F0FDF4;
        color: #16A34A;
        border: 1px solid #86EFAC;
        padding: 3px 9px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.78rem;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }

    /* Reasoning Step Card */
    .reasoning-row {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #2563EB;
        border-radius: 0 4px 4px 0;
        padding: 0.65rem 0.9rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: #1E293B;
        line-height: 1.45;
    }

    /* Historical Case Card */
    .case-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.9rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .case-id {
        font-size: 0.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .case-match {
        font-size: 0.78rem;
        font-weight: 700;
        color: #16A34A;
    }

    /* Technician Validation Box */
    .validation-box {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 1.25rem;
        margin-top: 1rem;
    }

    /* Clean Streamlit form styling */
    div[data-testid="stForm"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 6px !important;
        padding: 1.25rem !important;
        background: #FFFFFF !important;
    }

    /* Professional Button Styling */
    .stButton > button {
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.88rem;
        transition: all 0.15s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# Imports of backend components
from src.retrieval.vector_store import MaintenanceVectorStore
from src.agents.orchestrator import InfrastructureDecisionOrchestrator
from src.feedback.feedback_handler import FeedbackHandler

# Initialize cached instances
@st.cache_resource(show_spinner="Initializing Knowledge Base & AI Agents...")
def get_system():
    vector_store = MaintenanceVectorStore()
    orchestrator = InfrastructureDecisionOrchestrator(vector_store=vector_store)
    feedback_handler = FeedbackHandler(vector_store=vector_store)
    return vector_store, orchestrator, feedback_handler

vector_store, orchestrator, feedback_handler = get_system()

# Helper for taxonomy mapping
EQUIPMENT_DISPLAY_MAP = {
    "All": "All",
    "AC Unit / HVAC": "HVAC",
    "Diesel Generator": "Diesel Generator",
    "Elevator": "Elevator",
    "Pump / Water Supply": "Water Supply & Pumps",
    "Electrical Switchgear": "Electrical Switchgear",
    "Other": "All"
}
DISPLAY_LIST = list(EQUIPMENT_DISPLAY_MAP.keys())

# ==========================================
# SIDEBAR DESIGN
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 1rem;">
        <div style="font-size: 1.15rem; font-weight: 800; color: #17212B; letter-spacing: -0.01em;">FACILITY SUPPORT</div>
        <div style="font-size: 0.8rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em;">Decision Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; margin-bottom:0.4rem;'>Operations</div>", unsafe_allow_html=True)
    nav_choice = st.radio(
        "Navigation",
        [
            "Incident Triage",
            "Multi-Incident Analysis",
            "Maintenance Knowledge Base",
            "System Analytics",
            "Architecture & Workflow"
        ],
        label_visibility="collapsed"
    )

    metrics = feedback_handler.get_metrics()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; margin-bottom:0.6rem;">System Status</div>
    <div style="font-size:0.82rem; color:#334155; line-height:1.8;">
        <div><span class="status-dot-green"></span> AI Engine Online</div>
        <div><span class="status-dot-green"></span> Knowledge Base Indexed</div>
        <div><span class="status-dot-green"></span> Retrieval Engine Ready</div>
    </div>
    <div style="margin-top:1rem; padding:0.6rem 0.8rem; background:#F1F5F9; border-radius:4px; font-size:0.75rem; color:#475569;">
        <b>Engine:</b> LangGraph Orchestrator<br>
        <b>Vector Store:</b> FAISS IndexFlatIP<br>
        <b>Embeddings:</b> MiniLM-L6-v2
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# TOP HEADER & DASHBOARD SUMMARY CARDS
# ==========================================
st.markdown(f"""
<div class="command-header-container">
    <div>
        <div class="command-title">Facility Incident Command Center</div>
        <div class="command-subtitle">AI-assisted diagnosis and maintenance decision support for campus infrastructure.</div>
    </div>
    <div class="header-status-strip">
        <span class="status-pill"><span class="status-dot-green"></span> System: ONLINE</span>
        <span class="status-pill">Knowledge Base: <b>{vector_store.index.ntotal}</b> Cases</span>
        <span class="status-pill">Technician Feedback: <b>{metrics['confirmed_accurate']}</b> Confirmed</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 4 Compact Summary KPI Cards
active_incidents_count = 1 if "current_result" in st.session_state else 0
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Active Incidents</div>
        <div class="kpi-value">{active_incidents_count}</div>
        <div class="kpi-meta" style="color: #64748B;">Ready for dispatch</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Cases in Knowledge Base</div>
        <div class="kpi-value">{vector_store.index.ntotal}</div>
        <div class="kpi-meta">● Active FAISS Index</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Technician Confirmations</div>
        <div class="kpi-value">{metrics['confirmed_accurate']}</div>
        <div class="kpi-meta">● Total Validated: {metrics['total_feedback']}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Diagnostic Accuracy</div>
        <div class="kpi-value">{metrics['accuracy_rate']}%</div>
        <div class="kpi-meta">● Continuous validation</div>
    </div>
    """, unsafe_allow_html=True)


# Quick Incident Scenarios Dataset
SCENARIOS = {
    "ac_fail": {
        "title": "AC FAILURE",
        "desc": "AC blowing warm air + abnormal noise",
        "equipment": "AC Unit / HVAC",
        "location": "Engineering Block - Room 204",
        "id": "HVAC-CHILLER-02",
        "issue": "Chiller unit cycling rapidly on high head pressure and blowing warm air into lecture classrooms",
        "symptoms": "Low suction pressure gauge, frost on thermal expansion valve, compressor short cycles every 4 minutes"
    },
    "gen_fail": {
        "title": "GENERATOR FAILURE",
        "desc": "Generator overheating + unstable output",
        "equipment": "Diesel Generator",
        "location": "Powerhouse Substation Yard",
        "id": "GEN-CUMMINS-01",
        "issue": "Generator engine shuts down after 15 minutes of running on high coolant temp",
        "symptoms": "Coolant temp gauge reaches 104°C, radiator upper hose cold, lower hose boiling hot, thermostat suspected"
    },
    "elev_fault": {
        "title": "ELEVATOR FAULT",
        "desc": "Elevator stopping between floors",
        "equipment": "Elevator",
        "location": "Science Tower - Bank A",
        "id": "ELEV-OTIS-04",
        "issue": "Elevator car stopped abruptly between 3rd and 4th floors with 3 students trapped inside",
        "symptoms": "Safety circuit tripped open, controller shows error F021 (Landing door lock circuit disrupted)"
    },
    "elec_fault": {
        "title": "ELECTRICAL FAULT",
        "desc": "Breaker trips repeatedly",
        "equipment": "Electrical Switchgear",
        "location": "Main 11kV Substation - Room 1",
        "id": "SWG-MAIN-415V-02",
        "issue": "Main 800A air circuit breaker (ACB) tripped repeatedly, cutting power to Academic Block B",
        "symptoms": "Microprocessor trip unit shows Ground Fault (GF) flag, no visible arc flash or burning odor"
    },
    "pump_fault": {
        "title": "PUMP CAVITATION",
        "desc": "Pressure loss + rattling in pump casing",
        "equipment": "Pump / Water Supply",
        "location": "Central Water Works - Pump Room",
        "id": "PUMP-BOOSTER-03",
        "issue": "Upper floors 4-7 experiencing zero water pressure during morning peak hours",
        "symptoms": "Booster pump #2 trips on drive fault VFD-OC (Overcurrent), loud gravel/rattling noise in bronze casing"
    }
}


# ==========================================
# VIEW 1: INCIDENT TRIAGE
# ==========================================
if nav_choice == "Incident Triage":
    # Multi-Agent Workflow Process Ribbon
    st.markdown("""
    <div class="process-ribbon">
        <div class="process-step"><span class="step-num-active">01</span> Report Incident</div>
        <span class="process-arrow">→</span>
        <div class="process-step"><span class="step-num">02</span> Retrieve Similar Cases (FAISS)</div>
        <span class="process-arrow">→</span>
        <div class="process-step"><span class="step-num">03</span> Diagnose Root Cause (LangGraph)</div>
        <span class="process-arrow">→</span>
        <div class="process-step"><span class="step-num">04</span> Recommend Action & Urgency</div>
        <span class="process-arrow">→</span>
        <div class="process-step"><span class="step-num">05</span> Validate (Technician Loop)</div>
    </div>
    """, unsafe_allow_html=True)

    # QUICK INCIDENT SCENARIOS (Clean Scenario Buttons)
    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #17212B; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.5rem;'>Quick Incident Scenarios (One-Click Auto-Fill)</div>", unsafe_allow_html=True)

    sc_cols = st.columns(len(SCENARIOS))
    for col, (sc_key, sc_data) in zip(sc_cols, SCENARIOS.items()):
        with col:
            if st.button(f"**{sc_data['title']}**\n\n{sc_data['desc']}", key=f"btn_{sc_key}", use_container_width=True):
                st.session_state["form_equipment"] = sc_data["equipment"]
                st.session_state["form_location"] = sc_data["location"]
                st.session_state["form_id"] = sc_data["id"]
                st.session_state["form_issue"] = sc_data["issue"]
                st.session_state["form_symptoms"] = sc_data["symptoms"]

    # NEW FACILITY INCIDENT FORM
    st.markdown("""
    <div style="margin-top: 1.2rem; margin-bottom: 0.4rem;">
        <span style="font-size: 1.15rem; font-weight: 700; color: #17212B;">NEW FACILITY INCIDENT</span>
        <span style="font-size: 0.85rem; color: #64748B; margin-left: 0.5rem;">Enter fault parameters for semantic retrieval and multi-agent diagnosis.</span>
    </div>
    """, unsafe_allow_html=True)

    # Populate form values from state or defaults
    cur_equipment = st.session_state.get("form_equipment", "AC Unit / HVAC")
    cur_location = st.session_state.get("form_location", "Engineering Block - Room 204")
    cur_id = st.session_state.get("form_id", "HVAC-CHILLER-02")
    cur_issue = st.session_state.get("form_issue", "Chiller unit cycling rapidly on high head pressure and blowing warm air into classrooms")
    cur_symptoms = st.session_state.get("form_symptoms", "Low suction pressure gauge, frost on thermal expansion valve, compressor short cycles every 4 minutes")

    with st.form("triage_form"):
        f_col1, f_col2, f_col3 = st.columns([1, 1, 1])

        with f_col1:
            eq_index = DISPLAY_LIST.index(cur_equipment) if cur_equipment in DISPLAY_LIST else 1
            equipment_type_display = st.selectbox("Equipment Type", DISPLAY_LIST, index=eq_index)
            equipment_type = EQUIPMENT_DISPLAY_MAP[equipment_type_display]

        with f_col2:
            equipment_id = st.text_input("Asset / Equipment ID", value=cur_id)

        with f_col3:
            location = st.text_input("Facility / Location", value=cur_location)

        f_issue = st.text_area(
            "Incident / Complaint",
            value=cur_issue,
            height=80,
            placeholder="Describe the failure, operational complaint, or malfunction..."
        )

        f_symptoms = st.text_area(
            "Observed Symptoms / Sensor Readings / Error Codes",
            value=cur_symptoms,
            height=70,
            placeholder="List specific telemetry, alarm codes (e.g. E-04), pressure/temperature readings..."
        )

        top_k = st.slider("Historical Cases to Compare", min_value=2, max_value=6, value=4)

        submit_analyze = st.form_submit_button("🔍 Analyze Incident", type="primary", use_container_width=True)
        st.caption("Retrieves similar maintenance cases and generates an explainable diagnosis and recommended action.")

    # Execution State
    if submit_analyze:
        if not f_issue.strip():
            st.error("Please enter a reported incident/complaint.")
        else:
            with st.status("Analyzing incident with Multi-Agent Orchestration...", expanded=True) as status_box:
                st.write("✓ Incident classified & parameters normalized")
                time.sleep(0.15)
                st.write("✓ Querying FAISS index for semantically similar maintenance records")
                start_t = time.time()
                result = orchestrator.process_incident(
                    reported_issue=f_issue,
                    symptoms=f_symptoms,
                    equipment_type=equipment_type,
                    location=location,
                    top_k=top_k
                )
                dur = round(time.time() - start_t, 2)
                st.write("✓ Diagnosis Agent analyzed symptom correlations & historical outcomes")
                st.write("✓ Recommendation Agent generated corrective procedure & safety plan")
                status_box.update(label="Analysis complete", state="complete", expanded=False)

            st.session_state["current_result"] = result
            st.session_state["incident_input"] = {
                "reported_issue": f_issue,
                "symptoms": f_symptoms,
                "equipment_type": equipment_type if equipment_type != "All" else (result["retrieved_cases"][0]["equipment_type"] if result["retrieved_cases"] else "HVAC"),
                "location": location,
                "equipment_id": equipment_id
            }
            st.session_state["pipeline_duration"] = dur

    # ==========================================
    # DIAGNOSIS & DECISION SUPPORT RESULTS
    # ==========================================
    if "current_result" in st.session_state:
        res = st.session_state["current_result"]
        inp = st.session_state["incident_input"]
        diag = res.get("diagnosis", {})
        recom = res.get("recommendation", {})
        retrieved = res.get("retrieved_cases", [])
        urgency = recom.get("urgency", "Medium")

        st.markdown("<hr style='margin: 1.5rem 0 1rem 0; border: none; border-top: 1px solid #CBD5E1;'>", unsafe_allow_html=True)

        # 1. INCIDENT SUMMARY STRIP
        st.markdown("### INCIDENT SUMMARY")
        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        with sum_col1:
            st.markdown(f"**Equipment:**<br>`{inp['equipment_type']}` ({inp['equipment_id']})", unsafe_allow_html=True)
        with sum_col2:
            st.markdown(f"**Location:**<br>{inp['location']}", unsafe_allow_html=True)
        with sum_col3:
            urgency_badge_map = {
                "Critical": "<span class='badge-critical'>● CRITICAL</span>",
                "High": "<span class='badge-high'>● HIGH</span>",
                "Medium": "<span class='badge-medium'>● MEDIUM</span>",
                "Low": "<span class='badge-low'>● LOW</span>"
            }
            badge_html = urgency_badge_map.get(urgency, "<span class='badge-medium'>● MEDIUM</span>")
            st.markdown(f"**Severity / Urgency:**<br>{badge_html}", unsafe_allow_html=True)
        with sum_col4:
            st.markdown(f"**Analysis Latency:**<br>{st.session_state.get('pipeline_duration', 0.5)}s", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. AI DIAGNOSIS & REASONING
        col_diag_card, col_action_card = st.columns([1, 1])

        with col_diag_card:
            st.markdown("""
            <div class="section-card">
                <div class="section-title">AI DIAGNOSIS</div>
                <div class="section-desc">Synthesized from historical failure patterns and telemetry correlation.</div>
            """, unsafe_allow_html=True)

            st.markdown(f"**LIKELY CAUSE:**\n### {diag.get('primary_root_cause')}")

            conf = float(diag.get("confidence_score", 85.0))
            st.markdown(f"**CONFIDENCE:** `{conf:.1f}%`")
            st.progress(min(1.0, conf / 100.0))

            st.markdown("<br><b>EVIDENCE-BASED REASONING:</b>", unsafe_allow_html=True)
            for i, step in enumerate(diag.get("reasoning_chain", []), 1):
                clean_step = step.replace("Step 1 (Case Corroboration):", "1. Corroboration:") \
                                 .replace("Step 2 (Symptom Triangulation):", "2. Telemetry:") \
                                 .replace("Step 3 (Recurrence Pattern Analysis):", "3. Pattern:") \
                                 .replace("Step 4 (Root Cause Synthesis):", "4. Synthesis:")
                st.markdown(f"<div class='reasoning-row'>{clean_step}</div>", unsafe_allow_html=True)

            if diag.get("differential_causes"):
                st.markdown("<div style='font-size:0.8rem; color:#64748B; margin-top:0.6rem;'><b>Differential Hypotheses:</b> " + ", ".join(diag.get("differential_causes")) + "</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        with col_action_card:
            st.markdown("""
            <div class="section-card">
                <div class="section-title">RECOMMENDED MAINTENANCE ACTION</div>
                <div class="section-desc">Prioritized corrective steps, safety protocols, and resource estimates.</div>
            """, unsafe_allow_html=True)

            st.markdown(f"**MAINTENANCE PRIORITY:** {badge_html}", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.82rem; color:#475569; margin-bottom:0.75rem;'><b>Operational Reason:</b> {recom.get('urgency_rationale', 'Assigned based on historical maintenance impact.')}</div>", unsafe_allow_html=True)

            res_c1, res_c2 = st.columns(2)
            with res_c1:
                st.metric("Estimated Time", f"{recom.get('estimated_time_hrs', 3.0)} Hours")
            with res_c2:
                cost_val = recom.get('estimated_cost_usd', 350)
                st.metric("Estimated Cost", f"${cost_val}")

            st.markdown("<br><b>Corrective Action Checklist:</b>", unsafe_allow_html=True)
            for idx, action_step in enumerate(recom.get("fix_steps", []), 1):
                st.markdown(f"<div style='font-size:0.85rem; padding: 4px 0;'><b>{idx}.</b> {action_step}</div>", unsafe_allow_html=True)

            st.markdown("<br><b>Safety Precautions & LOTO:</b>", unsafe_allow_html=True)
            for safe_item in recom.get("safety_precautions", []):
                st.markdown(f"<div style='font-size:0.82rem; color:#991B1B; background:#FEF2F2; padding:5px 8px; border-radius:4px; margin-bottom:4px; border:1px solid #FECACA;'>⚠️ {safe_item}</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # 3. SIMILAR HISTORICAL CASES (Semantic Retrieval Layer)
        st.markdown("### SIMILAR HISTORICAL CASES")
        st.caption(f"Semantic similarity search executed across {vector_store.index.ntotal} historical records via FAISS.")

        if retrieved:
            # Display clean tabular comparison
            table_rows = []
            for c in retrieved:
                table_rows.append({
                    "Case ID": c.get("id"),
                    "Equipment": f"{c.get('equipment_type')} ({c.get('equipment_id', '')})",
                    "Reported Symptom": c.get("symptoms", "")[:60] + "...",
                    "Previous Cause": c.get("root_cause", "")[:60] + "...",
                    "Resolution": c.get("fix_action", "")[:60] + "...",
                    "Similarity": f"{c.get('similarity_score', 0)}%",
                    "Outcome": "Resolved"
                })
            
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        # 4. TECHNICIAN VALIDATION (Compulsory Add-on / Feature 7)
        st.markdown("""
        <div class="validation-box">
            <div style="font-size:1.05rem; font-weight:700; color:#17212B; margin-bottom:0.2rem;">TECHNICIAN VALIDATION</div>
            <div style="font-size:0.85rem; color:#475569; margin-bottom:0.8rem;">
                <b>Compulsory Feedback Loop:</b> Confirmed cases are added to the maintenance knowledge base to improve future retrieval during this demo.
            </div>
        """, unsafe_allow_html=True)

        st.write("**Was this diagnosis correct?**")
        v_col1, v_col2 = st.columns([1, 1])

        with v_col1:
            tech_user = st.text_input("Technician Name / Tag", value="Lead Tech #12 (A. Vance)", key="val_tech_name")
            val_notes = st.text_input("Technician Field Confirmation Notes", value="Field physical inspection verified root cause.", key="val_notes")
            
            if st.button("✓ Diagnosis Correct — Confirm & Add to Knowledge Base", type="primary", use_container_width=True):
                success, msg, details = feedback_handler.record_feedback(
                    equipment_type=inp["equipment_type"],
                    reported_issue=inp["reported_issue"],
                    symptoms=inp["symptoms"],
                    root_cause=diag.get("primary_root_cause"),
                    fix_action=" ".join(recom.get("fix_steps", [])),
                    resolution_time_hrs=recom.get("estimated_time_hrs", 3.0),
                    cost_estimate_usd=recom.get("estimated_cost_usd", 350),
                    urgency=urgency,
                    is_accurate=True,
                    technician_name=tech_user,
                    notes=val_notes,
                    location=inp["location"],
                    equipment_id=inp["equipment_id"]
                )
                if success:
                    st.success(f"✓ {msg}")
                    st.info(f"Knowledge base updated! Total records: **{details['total_knowledge_base_size']}**. Subsequent query match confidence increased to **{details['new_confidence_score']}%**.")
                else:
                    st.error(msg)

        with v_col2:
            with st.expander("✕ Diagnosis Incorrect — Override Root Cause & Resolution"):
                override_cause = st.text_input("Actual Cause Identified in Field", value=diag.get("primary_root_cause"))
                override_fix = st.text_area("Actual Resolution Performed", value=" ".join(recom.get("fix_steps", [])))
                oc1, oc2 = st.columns(2)
                with oc1:
                    override_hours = st.number_input("Actual Time (Hours)", min_value=0.5, max_value=40.0, value=float(recom.get("estimated_time_hrs", 3.0)))
                with oc2:
                    override_cost = st.number_input("Actual Cost ($)", min_value=10, max_value=25000, value=int(recom.get("estimated_cost_usd", 350)))
                override_notes = st.text_input("Discrepancy Notes", value="Found damaged capacitor instead of low pressure.")

                if st.button("Save Technician Correction to Knowledge Base", use_container_width=True):
                    success, msg, details = feedback_handler.record_feedback(
                        equipment_type=inp["equipment_type"],
                        reported_issue=inp["reported_issue"],
                        symptoms=inp["symptoms"],
                        root_cause=override_cause,
                        fix_action=override_fix,
                        resolution_time_hrs=override_hours,
                        cost_estimate_usd=override_cost,
                        urgency=urgency,
                        is_accurate=False,
                        technician_name=tech_user,
                        notes=override_notes,
                        location=inp["location"],
                        equipment_id=inp["equipment_id"]
                    )
                    if success:
                        st.success(f"✓ {msg}")
                        st.info(f"Knowledge base updated with human-in-the-loop correction! Total records: **{details['total_knowledge_base_size']}**.")
                    else:
                        st.error(msg)

        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# VIEW 2: MULTI-INCIDENT ANALYSIS
# ==========================================
elif nav_choice == "Multi-Incident Analysis":
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <span style="font-size: 1.35rem; font-weight: 700; color: #17212B;">MULTI-INCIDENT ANALYSIS</span>
        <div style="font-size: 0.88rem; color: #64748B;">Simultaneous incident prioritization and dispatch queue during campus emergencies and power surges.</div>
    </div>
    """, unsafe_allow_html=True)

    BATCH_DATA = {
        "Campus Storm & Power Grid Transient (4 Simultaneous Outages)": [
            {
                "id": "INCIDENT 01",
                "equipment_type": "Elevator",
                "location": "Science Tower - Bank A",
                "reported_issue": "Elevator stopped between floors 4 and 5; trapped students calling intercom",
                "symptoms": "Door lock circuit open, hoistway safety relay tripped, emergency telephone ringing"
            },
            {
                "id": "INCIDENT 02",
                "equipment_type": "HVAC",
                "location": "Server Room - Data Center",
                "reported_issue": "Server room CRAC unit stopped cooling, room temperature climbing to 31°C",
                "symptoms": "Condenser fan not spinning, high head pressure trip E-04"
            },
            {
                "id": "INCIDENT 03",
                "equipment_type": "Electrical Switchgear",
                "location": "Main Substation - Room 1",
                "reported_issue": "Phase L2 busbar hotspot emitting ozone smell after lightning surge",
                "symptoms": "Infrared temp 112°C on connector splice, breaker humming loudly"
            },
            {
                "id": "INCIDENT 04",
                "equipment_type": "Water Supply & Pumps",
                "location": "Basement Mechanical Room",
                "reported_issue": "Water pooling on floor around domestic booster pump",
                "symptoms": "Shaft gland drip 10 drops/min, domestic header pressure steady at 55 PSI"
            }
        ]
    }

    b_scenario = list(BATCH_DATA.keys())[0]
    incidents_list = BATCH_DATA[b_scenario]

    # Show separate incident cards
    st.markdown("##### REPORTED INCIDENTS IN QUEUE:")
    q_cols = st.columns(len(incidents_list))
    for i, inc in enumerate(incidents_list):
        with q_cols[i]:
            st.markdown(f"""
            <div class="section-card" style="padding:1rem;">
                <div style="font-size:0.75rem; font-weight:700; color:#2563EB;">{inc['id']}</div>
                <div style="font-size:0.9rem; font-weight:700; color:#17212B; margin-top:3px;">{inc['equipment_type']}</div>
                <div style="font-size:0.78rem; color:#64748B;">{inc['location']}</div>
                <hr style="margin:6px 0; border:none; border-top:1px solid #E2E8F0;">
                <div style="font-size:0.78rem; color:#334155;">{inc['reported_issue'][:70]}...</div>
            </div>
            """, unsafe_allow_html=True)

    if st.button("🚀 Analyze All Incidents", type="primary", use_container_width=True):
        with st.spinner("Processing multi-agent pipeline for all simultaneous incidents..."):
            b_results = []
            for i, inc in enumerate(incidents_list):
                out = orchestrator.process_incident(
                    reported_issue=inc["reported_issue"],
                    symptoms=inc["symptoms"],
                    equipment_type=inc["equipment_type"],
                    location=inc["location"],
                    top_k=3
                )
                b_results.append({
                    "incident_id": inc["id"],
                    "equipment": inc["equipment_type"],
                    "location": inc["location"],
                    "reported_issue": inc["reported_issue"],
                    "diagnosed_cause": out["diagnosis"].get("primary_root_cause"),
                    "confidence": out["diagnosis"].get("confidence_score", 80.0),
                    "urgency": out["recommendation"].get("urgency", "Medium"),
                    "est_time": out["recommendation"].get("estimated_time_hrs", 3.0),
                    "est_cost": out["recommendation"].get("estimated_cost_usd", 350),
                    "primary_action": out["recommendation"].get("fix_steps", ["Inspect and repair"])[0]
                })

            urgency_priority = {"Critical": 1, "High": 2, "Medium": 3, "Low": 4}
            b_results.sort(key=lambda x: urgency_priority.get(x["urgency"], 5))
            st.session_state["multi_incident_results"] = b_results

    if "multi_incident_results" in st.session_state:
        b_res = st.session_state["multi_incident_results"]
        st.markdown("---")
        st.markdown("### PRIORITIZED DISPATCH QUEUE")

        sum_time = sum(r["est_time"] for r in b_res)
        sum_cost = sum(r["est_cost"] for r in b_res)
        crit_count = sum(1 for r in b_res if r["urgency"] == "Critical")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Critical Life/Safety Incidents", f"{crit_count}", delta="Requires Immediate Dispatch", delta_color="inverse")
        with m2:
            st.metric("Total Technician Duration", f"{sum_time:.1f} Hours")
        with m3:
            st.metric("Aggregate Repair Budget", f"${sum_cost:,}")

        for rank, item in enumerate(b_res, 1):
            badge_type = item["urgency"].lower()
            st.markdown(f"""
            <div class="section-card" style="margin-bottom:0.8rem; border-left: 4px solid {'#DC2626' if badge_type=='critical' else '#EA580C' if badge_type=='high' else '#CA8A04'};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-weight:700; color:#17212B; font-size:1rem;">PRIORITY #{rank}: {item['equipment']}</span>
                        <span style="font-size:0.85rem; color:#64748B; margin-left:8px;">{item['location']}</span>
                    </div>
                    <span class="badge-{badge_type}">● {item['urgency'].upper()}</span>
                </div>
                <div style="font-size:0.85rem; color:#1E293B; margin-top:8px;"><b>Complaint:</b> {item['reported_issue']}</div>
                <div style="font-size:0.85rem; color:#2563EB; margin-top:4px;"><b>Likely Cause:</b> {item['diagnosed_cause']} (Confidence: {item['confidence']}%)</div>
                <div style="font-size:0.82rem; color:#334155; margin-top:4px;"><b>Immediate Action:</b> {item['primary_action']}</div>
                <div style="font-size:0.8rem; color:#64748B; margin-top:6px;">Est. Time: {item['est_time']} hrs | Est. Cost: ${item['est_cost']}</div>
            </div>
            """, unsafe_allow_html=True)


# ==========================================
# VIEW 3: KNOWLEDGE BASE EXPLORER
# ==========================================
elif nav_choice == "Maintenance Knowledge Base":
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <span style="font-size: 1.35rem; font-weight: 700; color: #17212B;">MAINTENANCE KNOWLEDGE BASE</span>
        <div style="font-size: 0.88rem; color: #64748B;">Historical Maintenance Records across Campus Infrastructure.</div>
    </div>
    """, unsafe_allow_html=True)

    records = vector_store.get_all_records()
    df = pd.DataFrame(records)

    st.markdown(f"**{len(df)}** Historical Maintenance Cases Indexed in FAISS Vector Store.")

    # Clean filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        eq_filter = st.selectbox("Equipment Type", ["All"] + vector_store.get_equipment_types())
    with fc2:
        urg_filter = st.selectbox("Severity / Urgency", ["All", "Critical", "High", "Medium", "Low"])
    with fc3:
        search_kw = st.text_input("Search maintenance cases...", placeholder="e.g. refrigerant, bearing, contactor, water hammer")

    filtered_df = df.copy()
    if eq_filter != "All":
        filtered_df = filtered_df[filtered_df["equipment_type"] == eq_filter]
    if urg_filter != "All":
        filtered_df = filtered_df[filtered_df["urgency"] == urg_filter]
    if search_kw.strip():
        kw = search_kw.lower()
        filtered_df = filtered_df[
            filtered_df["reported_issue"].str.lower().str.contains(kw) |
            filtered_df["root_cause"].str.lower().str.contains(kw) |
            filtered_df["fix_action"].str.lower().str.contains(kw) |
            filtered_df["symptoms"].str.lower().str.contains(kw)
        ]

    st.dataframe(
        filtered_df[["id", "equipment_type", "equipment_id", "location", "reported_issue", "root_cause", "urgency", "resolution_time_hrs", "cost_estimate_usd", "date_logged"]],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.subheader("Inspect Case Dossier")
    sel_id = st.selectbox("Select Case ID to view full engineering record:", filtered_df["id"].tolist() if not filtered_df.empty else [])
    if sel_id:
        case_record = next((r for r in records if r["id"] == sel_id), None)
        if case_record:
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.markdown(f"""
                <div class="section-card">
                    <div style="font-weight:700; color:#2563EB;">#{case_record['id']}</div>
                    <div style="font-weight:700; font-size:1.1rem; color:#17212B; margin-top:4px;">{case_record['equipment_type']} ({case_record.get('equipment_id', '')})</div>
                    <div style="font-size:0.85rem; color:#64748B; margin-bottom:0.75rem;">{case_record.get('location', '')}</div>
                    <div style="font-size:0.85rem;"><b>Reported Issue:</b><br>{case_record['reported_issue']}</div>
                    <div style="font-size:0.85rem; margin-top:8px;"><b>Observed Symptoms:</b><br>{case_record['symptoms']}</div>
                </div>
                """, unsafe_allow_html=True)
            with d_col2:
                st.markdown(f"""
                <div class="section-card">
                    <div style="font-size:0.85rem;"><b>Diagnosed Root Cause:</b><br><span style="color:#B45309; font-weight:600;">{case_record['root_cause']}</span></div>
                    <div style="font-size:0.85rem; margin-top:8px;"><b>Action Taken:</b><br>{case_record['fix_action']}</div>
                    <hr style="margin:10px 0; border:none; border-top:1px solid #E2E8F0;">
                    <div style="font-size:0.8rem; color:#64748B;">
                        <b>Urgency:</b> {case_record['urgency']} | <b>Cost:</b> ${case_record['cost_estimate_usd']} | <b>Duration:</b> {case_record['resolution_time_hrs']} hrs<br>
                        <b>Technician:</b> {case_record.get('technician', 'N/A')}<br>
                        <b>Notes:</b> {case_record.get('technician_notes', 'N/A')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Filtered Cases to CSV",
        data=csv_data,
        file_name="maintenance_cases_export.csv",
        mime="text/csv"
    )


# ==========================================
# VIEW 4: SYSTEM ANALYTICS
# ==========================================
elif nav_choice == "System Analytics":
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <span style="font-size: 1.35rem; font-weight: 700; color: #17212B;">MAINTENANCE ANALYTICS & INSIGHTS</span>
        <div style="font-size: 0.88rem; color: #64748B;">Historical operational telemetry, mean time to repair, and diagnostic reliability.</div>
    </div>
    """, unsafe_allow_html=True)

    records = vector_store.get_all_records()
    df = pd.DataFrame(records)

    c_col1, c_col2 = st.columns(2)
    with c_col1:
        st.markdown("##### Most Common Equipment Failures")
        st.bar_chart(df["equipment_type"].value_counts())

    with c_col2:
        st.markdown("##### Maintenance Priority Distribution")
        st.bar_chart(df["urgency"].value_counts())

    c_col3, c_col4 = st.columns(2)
    with c_col3:
        st.markdown("##### Average Repair Cost by Equipment ($)")
        avg_cost = df.groupby("equipment_type")["cost_estimate_usd"].mean().round(0)
        st.bar_chart(avg_cost)

    with c_col4:
        st.markdown("##### Mean Time to Repair (MTTR in Hours)")
        avg_time = df.groupby("equipment_type")["resolution_time_hrs"].mean().round(1)
        st.bar_chart(avg_time)

    st.markdown("---")
    st.markdown("##### Technician Confirmation Telemetry")
    t_c1, t_c2, t_c3 = st.columns(3)
    with t_c1:
        st.metric("Total Confirmations Logged", f"{metrics['total_feedback']}")
    with t_c2:
        st.metric("Direct Accurate Matches", f"{metrics['confirmed_accurate']}")
    with t_c3:
        st.metric("Technician Corrections", f"{metrics['corrected']}")


# ==========================================
# VIEW 5: ARCHITECTURE & WORKFLOW
# ==========================================
elif nav_choice == "Architecture & Workflow":
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <span style="font-size: 1.35rem; font-weight: 700; color: #17212B;">MULTI-AGENT ARCHITECTURE & WORKFLOW</span>
        <div style="font-size: 0.88rem; color: #64748B;">High-level orchestrator architecture for Track B - B3.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-title">System Workflow Representation</div>
        <div style="font-size:0.9rem; color:#334155; line-height:2; margin-top:1rem; font-family:monospace; background:#F8FAFC; padding:1.25rem; border-radius:6px; border:1px solid #E2E8F0;">
            USER REPORT<br>
            &nbsp;&nbsp;↓<br>
            INCIDENT ANALYSIS AGENT<br>
            &nbsp;&nbsp;↓<br>
            HISTORICAL CASE RETRIEVAL (FAISS Vector Store)<br>
            &nbsp;&nbsp;↓<br>
            DIAGNOSIS AGENT (LangGraph Symptom Corroboration)<br>
            &nbsp;&nbsp;↓<br>
            RECOMMENDATION AGENT (Corrective Steps, Safety, Urgency, Cost/Time)<br>
            &nbsp;&nbsp;↓<br>
            TECHNICIAN VALIDATION (Human-in-the-Loop Feedback)<br>
            &nbsp;&nbsp;↓<br>
            KNOWLEDGE BASE UPDATE (Real-time Incremental FAISS Indexing)
        </div>
    </div>
    """, unsafe_allow_html=True)
