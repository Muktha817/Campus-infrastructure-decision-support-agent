"""
Campus/Facility Infrastructure Decision-Support Agent Dashboard
Track B - B3: Multi-Agent Orchestration & Decision Support
Built with Streamlit, LangGraph, FAISS & SentenceTransformers
"""

import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path

# Set Streamlit page config
st.set_page_config(
    page_title="Campus Infrastructure Decision Support Agent",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished modern enterprise UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .urgency-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .urgency-high {
        background-color: #FFEDD5;
        color: #C2410C;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .urgency-medium {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .urgency-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .reasoning-step {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 0 6px 6px 0;
        font-size: 0.93rem;
        color: #1E293B;
    }
    .feedback-box {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Imports of backend components
from src.retrieval.vector_store import MaintenanceVectorStore
from src.agents.orchestrator import InfrastructureDecisionOrchestrator
from src.feedback.feedback_handler import FeedbackHandler

# Initialize cached instances
@st.cache_resource(show_spinner="Initializing FAISS Semantic Knowledge Base & Models...")
def get_system():
    vector_store = MaintenanceVectorStore()
    orchestrator = InfrastructureDecisionOrchestrator(vector_store=vector_store)
    feedback_handler = FeedbackHandler(vector_store=vector_store)
    return vector_store, orchestrator, feedback_handler

vector_store, orchestrator, feedback_handler = get_system()

# Sidebar Navigation and System Status
st.sidebar.image("https://img.icons8.com/color/96/000000/maintenance.png", width=64)
st.sidebar.title("Facility Support AI")
st.sidebar.caption("Track B - B3: Multi-Agent Orchestration")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "🚨 New Incident Triage",
        "⚡ Multi-Complaint Batch Triage",
        "📚 Knowledge Base Explorer",
        "📊 System Analytics & Accuracy"
    ]
)

metrics = feedback_handler.get_metrics()
st.sidebar.markdown("---")
st.sidebar.subheader("System Telemetry")
st.sidebar.metric("Indexed Knowledge Base", f"{vector_store.index.ntotal} Records")
st.sidebar.metric("Technician Confirmations", f"{metrics['confirmed_accurate']} / {metrics['total_feedback']}")
st.sidebar.metric("Diagnostic Accuracy", f"{metrics['accuracy_rate']}%")
st.sidebar.caption(f"Backend Engine: LangGraph + FAISS ({orchestrator.llm_provider.mode.upper()})")

# Sample incident presets for rapid demo testing
SAMPLE_INCIDENTS = {
    "Select a pre-built scenario...": None,
    "HVAC: Chiller Cycling & Blowing Warm Air": {
        "equipment": "HVAC",
        "location": "Science Block - Roof",
        "issue": "Chiller unit cycling rapidly on high head pressure and blowing warm air into classrooms",
        "symptoms": "Low suction pressure gauge, frost/ice on thermal expansion valve, compressor short cycles every 4 minutes"
    },
    "Generator: Failed Automatic Load Test": {
        "equipment": "Diesel Generator",
        "location": "Powerhouse Substation Yard",
        "issue": "Cummins 500kVA emergency generator cranked very slowly and failed weekly auto-test",
        "symptoms": "Battery bus voltage plunges to 8.9V upon crank command, overcrank warning LED active, starter solenoid clicking"
    },
    "Elevator: Passenger Entrapment Between Floors": {
        "equipment": "Elevator",
        "location": "Science Tower - Bank A",
        "issue": "Elevator car stopped abruptly between 3rd and 4th floors with 3 students trapped inside",
        "symptoms": "Safety circuit tripped open, controller shows error F021 (Landing door lock circuit disrupted)"
    },
    "Water Pump: Pressure Loss & Impeller Cavitation": {
        "equipment": "Water Supply & Pumps",
        "location": "Central Water Works - Pump Room",
        "issue": "Floors 4 through 7 experiencing zero water pressure; booster pump running red hot",
        "symptoms": "VFD tripping on Overcurrent (OC-1), severe rattling/gravel noise in bronze pump casing, inlet screen intact"
    },
    "Switchgear: Substation Busbar Hot Spot Detected": {
        "equipment": "Electrical Switchgear",
        "location": "Main 11kV Substation - Room 1",
        "issue": "Thermographic infrared inspection identified severe thermal runaway on Phase L2 busbar connector",
        "symptoms": "Busbar joint running at 114°C under 60% rated load (adjacent phases at 42°C), slight ozone odor in switchroom"
    }
}

# ==========================================
# VIEW 1: NEW INCIDENT TRIAGE
# ==========================================
if nav_choice == "🚨 New Incident Triage":
    st.markdown('<div class="main-header">Campus Infrastructure Incident Triage</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-agent decision support for diagnosis, historical retrieval, fix planning, and technician validation.</div>', unsafe_allow_html=True)

    # Preset selector for easy demo
    col_preset, col_clear = st.columns([4, 1])
    with col_preset:
        chosen_sample = st.selectbox("🚀 Quick Demo Scenarios (One-Click Auto-Fill):", list(SAMPLE_INCIDENTS.keys()))
    with col_clear:
        st.write("")
        st.write("")
        if st.button("Reset Form"):
            chosen_sample = "Select a pre-built scenario..."

    preset_data = SAMPLE_INCIDENTS.get(chosen_sample)

    with st.form("incident_form"):
        col1, col2 = st.columns(2)
        equipment_types = ["All"] + vector_store.get_equipment_types()
        
        with col1:
            default_eq_idx = 0
            if preset_data and preset_data["equipment"] in equipment_types:
                default_eq_idx = equipment_types.index(preset_data["equipment"])
            equipment_type = st.selectbox("Equipment Type", equipment_types, index=default_eq_idx)
            
            locations = ["Campus Facility (General)"] + vector_store.get_locations()
            default_loc = preset_data["location"] if preset_data else locations[0]
            location = st.text_input("Facility Location", value=default_loc)

        with col2:
            equipment_id = st.text_input("Asset / Equipment Tag (Optional)", value="ASSET-" + equipment_type[:3].upper() if equipment_type != "All" else "ASSET-01")
            top_k = st.slider("Historical Cases to Correlate", min_value=2, max_value=6, value=4)

        reported_issue = st.text_area(
            "Reported Incident / Complaint",
            value=preset_data["issue"] if preset_data else "",
            placeholder="e.g. AC unit blowing hot air and making rattling noise in lecture hall...",
            height=85
        )

        symptoms = st.text_area(
            "Observed Symptoms / Sensor Readings / Error Codes",
            value=preset_data["symptoms"] if preset_data else "",
            placeholder="e.g. High head pressure gauge, suction line frosted, error code E-04 on digital display...",
            height=70
        )

        submit_btn = st.form_submit_button("🔍 Run Multi-Agent Diagnosis & Decision Support", type="primary", use_container_width=True)

    if submit_btn:
        if not reported_issue.strip():
            st.error("Please enter a reported issue/complaint to diagnose.")
        else:
            with st.spinner("Executing LangGraph multi-agent pipeline: Retrieving -> Diagnosing -> Recommending..."):
                start_time = time.time()
                result = orchestrator.process_incident(
                    reported_issue=reported_issue,
                    symptoms=symptoms,
                    equipment_type=equipment_type,
                    location=location,
                    top_k=top_k
                )
                duration = round(time.time() - start_time, 2)
                st.session_state["current_result"] = result
                st.session_state["incident_input"] = {
                    "reported_issue": reported_issue,
                    "symptoms": symptoms,
                    "equipment_type": equipment_type if equipment_type != "All" else (result["retrieved_cases"][0]["equipment_type"] if result["retrieved_cases"] else "General"),
                    "location": location,
                    "equipment_id": equipment_id
                }
                st.session_state["pipeline_duration"] = duration

    # Display Results if Available
    if "current_result" in st.session_state:
        res = st.session_state["current_result"]
        inp = st.session_state["incident_input"]
        diag = res.get("diagnosis", {})
        recom = res.get("recommendation", {})
        retrieved = res.get("retrieved_cases", [])
        urgency = recom.get("urgency", "Medium")

        st.markdown("---")
        
        # Urgency & Overview Banner
        urgency_classes = {
            "Critical": "urgency-critical",
            "High": "urgency-high",
            "Medium": "urgency-medium",
            "Low": "urgency-low"
        }
        badge_class = urgency_classes.get(urgency, "urgency-medium")

        header_col1, header_col2, header_col3 = st.columns([3, 1, 1])
        with header_col1:
            st.markdown(f"### Diagnosis Decision Summary <span class='{badge_class}'>URGENCY: {urgency.upper()}</span>", unsafe_allow_html=True)
            st.caption(f"Processed in {st.session_state.get('pipeline_duration', 0.5)}s via LangGraph pipeline")
        with header_col2:
            st.metric("Est. Downtime / Repair", f"{recom.get('estimated_time_hrs', 3.0)} hrs")
        with header_col3:
            st.metric("Est. Repair Cost", f"${recom.get('estimated_cost_usd', 350)}")

        # Multi-Agent Reasoning Chain & Root Cause
        col_diag, col_fix = st.columns([1, 1])

        with col_diag:
            st.markdown("#### 🧠 1. Diagnosis Reasoning & Root Cause")
            st.info(f"**Primary Probable Cause:**\n### {diag.get('primary_root_cause')}")
            
            conf = diag.get("confidence_score", 85.0)
            st.write(f"**Confidence Score:** {conf:.1f}%")
            st.progress(min(1.0, conf / 100.0))

            st.markdown("##### 📜 Plain-Language Reasoning Chain:")
            for step in diag.get("reasoning_chain", []):
                st.markdown(f"<div class='reasoning-step'>{step}</div>", unsafe_allow_html=True)

            if diag.get("differential_causes"):
                st.markdown("**Differential / Secondary Hypotheses:**")
                for diff in diag.get("differential_causes"):
                    st.write(f"• {diff}")

        with col_fix:
            st.markdown("#### 🛠️ 2. Fix Recommendation & Safety Protocol")
            st.markdown(f"**Urgency Rationale:** *{recom.get('urgency_rationale', '')}*")

            st.markdown("##### 📋 Recommended Fix Action Plan:")
            for i, step in enumerate(recom.get("fix_steps", []), 1):
                st.markdown(f"**{i}.** {step}")

            st.markdown("##### 🛡️ Mandatory Safety Precautions:")
            for safe in recom.get("safety_precautions", []):
                st.warning(f"⚠️ {safe}")

        # Retrieved Historical Cases (Feature 2)
        st.markdown("---")
        st.markdown("#### 🔍 3. Similar Historical Cases Retrieved from Knowledge Base")
        st.caption(f"Top {len(retrieved)} cases matched via FAISS semantic vector search over historical maintenance logs.")

        if retrieved:
            case_cols = st.columns(len(retrieved))
            for i, case in enumerate(retrieved):
                with case_cols[i]:
                    score = case.get("similarity_score", 0)
                    st.markdown(f"""
                    <div class="card">
                        <div style="font-weight:700; color:#1E3A8A; font-size:1.1rem;">#{case.get('id')}</div>
                        <div style="font-size:0.85rem; color:#6B7280;">{case.get('equipment_type')} ({case.get('equipment_id')})</div>
                        <div style="font-size:0.85rem; color:#4B5563; margin-top:4px;"><b>Match:</b> <span style="color:#059669; font-weight:700;">{score}%</span></div>
                        <hr style="margin:8px 0;">
                        <div style="font-size:0.85rem;"><b>Past Issue:</b> {case.get('reported_issue')[:75]}...</div>
                        <div style="font-size:0.85rem; margin-top:4px;"><b>Past Cause:</b> <span style="color:#B45309;">{case.get('root_cause')[:70]}...</span></div>
                        <div style="font-size:0.85rem; margin-top:4px;"><b>Resolution:</b> {case.get('fix_action')[:70]}...</div>
                        <div style="font-size:0.8rem; color:#6B7280; margin-top:6px;">Cost: ${case.get('cost_estimate_usd')} | {case.get('resolution_time_hrs')} hrs</div>
                    </div>
                    """, unsafe_allow_html=True)

        # =========================================================================
        # KEY FEATURE 7 | COMPULSORY ADD-ON: Technician Feedback & Learning Loop
        # =========================================================================
        st.markdown("---")
        st.markdown("### 👨‍🔧 4. Technician Feedback & Learning Loop (Compulsory Add-on)")
        st.markdown("""
        Confirm this diagnosis or enter technician field corrections.
        **Confirmed cases are appended to the knowledge base with their resolution and instantly re-indexed in FAISS**, improving future retrieval during this demo!
        """)

        feed_col1, feed_col2 = st.columns([1, 1])

        with feed_col1:
            st.markdown("#### Option A: One-Click Confirmation")
            st.write("If the agent's diagnosis matches the observed field problem, confirm it to update the model immediately:")
            tech_name_a = st.text_input("Technician Name / ID", value="Lead Tech Alex", key="tech_a")
            quick_notes = st.text_input("Resolution Field Notes (Optional)", value="Verified on site. Root cause confirmed.", key="notes_a")

            if st.button("✅ Confirm Diagnosis (Accurate) & Append to Knowledge Base", type="primary", use_container_width=True):
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
                    technician_name=tech_name_a,
                    notes=quick_notes,
                    location=inp["location"],
                    equipment_id=inp["equipment_id"]
                )
                if success:
                    st.success(f"🎉 **{msg}**")
                    st.balloons()
                    st.info(f"📈 **Real-Time Model Improvement**: Knowledge base expanded to **{details['total_knowledge_base_size']} records**! Querying this issue will now prioritize this verified incident with **{details['new_confidence_score']}% confidence**.")
                else:
                    st.error(msg)

        with feed_col2:
            st.markdown("#### Option B: Adjust / Correct Resolution")
            with st.expander("✏️ Technician Disagrees? Override Root Cause & Solution", expanded=False):
                correct_cause = st.text_input("Actual Root Cause Found", value=diag.get("primary_root_cause"))
                correct_fix = st.text_area("Actual Fix Action Applied", value=" ".join(recom.get("fix_steps", [])))
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    actual_time = st.number_input("Actual Hours Spent", min_value=0.5, max_value=40.0, value=float(recom.get("estimated_time_hrs", 3.0)))
                with col_c2:
                    actual_cost = st.number_input("Actual Cost ($)", min_value=10, max_value=20000, value=int(recom.get("estimated_cost_usd", 350)))
                tech_name_b = st.text_input("Technician Name", value="Field Tech Jordan", key="tech_b")
                adjust_notes = st.text_input("Technician Correction Comments", value="Component found burnt instead of loose contact.", key="notes_b")

                if st.button("💾 Save Technician Correction to Knowledge Base", use_container_width=True):
                    success, msg, details = feedback_handler.record_feedback(
                        equipment_type=inp["equipment_type"],
                        reported_issue=inp["reported_issue"],
                        symptoms=inp["symptoms"],
                        root_cause=correct_cause,
                        fix_action=correct_fix,
                        resolution_time_hrs=actual_time,
                        cost_estimate_usd=actual_cost,
                        urgency=urgency,
                        is_accurate=False,
                        technician_name=tech_name_b,
                        notes=adjust_notes,
                        location=inp["location"],
                        equipment_id=inp["equipment_id"]
                    )
                    if success:
                        st.success(f"🎉 **{msg}**")
                        st.info(f"📈 Knowledge base updated with human-in-the-loop correction! Total records: **{details['total_knowledge_base_size']}**.")
                    else:
                        st.error(msg)

# ==========================================
# VIEW 2: MULTI-COMPLAINT BATCH TRIAGE
# ==========================================
elif nav_choice == "⚡ Multi-Complaint Batch Triage":
    st.markdown('<div class="main-header">Multi-Complaint Batch Triage & Priority Dispatch</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation Parameter: Handling multiple simultaneous complaints during storm surges, power outages, or campus-wide events.</div>', unsafe_allow_html=True)

    st.write("During campus emergencies or power fluctuations, multiple failures arrive at the same time. The agent diagnoses each incident and produces an auto-prioritized dispatch queue based on urgency and risk.")

    BATCH_SCENARIOS = {
        "Campus Thunderstorm & Power Surge (4 Simultaneous Failures)": [
            {
                "equipment_type": "Elevator",
                "location": "Science Tower",
                "reported_issue": "Elevator stopped suddenly between floors 4 and 5; trapped students calling intercom",
                "symptoms": "Door lock circuit open, hoistway safety relay tripped, emergency telephone ringing"
            },
            {
                "equipment_type": "HVAC",
                "location": "Server Room - Data Center",
                "reported_issue": "Server room CRAC unit stopped cooling, room temperature climbing to 31°C",
                "symptoms": "Condenser fan not spinning, high head pressure trip E-04"
            },
            {
                "equipment_type": "Electrical Switchgear",
                "location": "Main Substation",
                "reported_issue": "Phase L2 busbar hotspot emitting ozone smell after lightning surge",
                "symptoms": "Infrared temp 112°C on connector splice, breaker humming loudly"
            },
            {
                "equipment_type": "Water Supply & Pumps",
                "location": "Basement Mechanical Room",
                "reported_issue": "Water pooling on floor around booster pump",
                "symptoms": "Minor shaft gland drip 10 drops/min, domestic pressure steady at 55 PSI"
            }
        ],
        "End-of-Semester High Load Breakdown (3 Failures)": [
            {
                "equipment_type": "Diesel Generator",
                "location": "Hospital Wing Bunker",
                "reported_issue": "Emergency diesel generator failed auto-start test for emergency surgery ward",
                "symptoms": "Battery bus voltage 9.1V, overcrank failure light active"
            },
            {
                "equipment_type": "HVAC",
                "location": "Main Auditorium",
                "reported_issue": "Auditorium AHU screeching deafeningly during graduation rehearsal",
                "symptoms": "Serpentine belt smoking and shredded rubber in air plenum"
            },
            {
                "equipment_type": "Water Supply & Pumps",
                "location": "Hostel Block A Sump",
                "reported_issue": "Sump pump failed to start during heavy rain; water rising near boiler base",
                "symptoms": "Mercury float switch tilted upside down and tangled in discharge line"
            }
        ]
    }

    selected_batch_name = st.selectbox("Select Batch Disaster Scenario:", list(BATCH_SCENARIOS.keys()))
    batch_items = BATCH_SCENARIOS[selected_batch_name]

    st.write(f"**Selected {len(batch_items)} simultaneous complaints to process:**")
    for idx, item in enumerate(batch_items, 1):
        st.markdown(f"**Ticket #{idx}:** `{item['equipment_type']}` at *{item['location']}* — {item['reported_issue']}")

    if st.button("🚀 Run Batch Multi-Agent Triage & Prioritize Queue", type="primary", use_container_width=True):
        batch_results = []
        progress_bar = st.progress(0.0)

        for i, item in enumerate(batch_items):
            res = orchestrator.process_incident(
                reported_issue=item["reported_issue"],
                symptoms=item["symptoms"],
                equipment_type=item["equipment_type"],
                location=item["location"],
                top_k=3
            )
            item_out = {
                "ticket_id": f"TKT-BATCH-{i+1:02d}",
                "equipment_type": item["equipment_type"],
                "location": item["location"],
                "reported_issue": item["reported_issue"],
                "diagnosed_cause": res["diagnosis"].get("primary_root_cause"),
                "urgency": res["recommendation"].get("urgency", "Medium"),
                "est_time": res["recommendation"].get("estimated_time_hrs", 3.0),
                "est_cost": res["recommendation"].get("estimated_cost_usd", 350),
                "fix_action": res["recommendation"].get("fix_steps", ["Inspect and repair"])[0],
                "confidence": res["diagnosis"].get("confidence_score", 80.0)
            }
            batch_results.append(item_out)
            progress_bar.progress((i + 1) / len(batch_items))

        # Sort by urgency priority: Critical (1) -> High (2) -> Medium (3) -> Low (4)
        urgency_priority = {"Critical": 1, "High": 2, "Medium": 3, "Low": 4}
        batch_results.sort(key=lambda x: urgency_priority.get(x["urgency"], 5))

        st.session_state["batch_results"] = batch_results

    if "batch_results" in st.session_state:
        b_res = st.session_state["batch_results"]
        st.markdown("---")
        st.markdown("### 📋 Prioritized Dispatch Queue (Highest Urgency First)")

        total_downtime = sum(r["est_time"] for r in b_res)
        total_budget = sum(r["est_cost"] for r in b_res)
        critical_count = sum(1 for r in b_res if r["urgency"] == "Critical")

        stat1, stat2, stat3 = st.columns(3)
        with stat1:
            st.metric("Critical Hazards Identified", f"{critical_count} Immediate Action Required", delta="High Risk", delta_color="inverse")
        with stat2:
            st.metric("Total Est. Technician Hours", f"{total_downtime:.1f} hrs")
        with stat3:
            st.metric("Total Est. Repair Budget", f"${total_budget:,}")

        for rank, r in enumerate(b_res, 1):
            urgency_tag = r["urgency"]
            tag_color = {
                "Critical": "#DC2626",
                "High": "#EA580C",
                "Medium": "#CA8A04",
                "Low": "#16A34A"
            }.get(urgency_tag, "#4B5563")

            with st.expander(f"**Priority #{rank} [{urgency_tag.upper()}]** {r['equipment_type']} at {r['location']}", expanded=(rank <= 2)):
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.markdown(f"**Complaint:** {r['reported_issue']}")
                    st.markdown(f"**Diagnosed Root Cause:** `{r['diagnosed_cause']}` (Confidence: {r['confidence']}%)")
                    st.markdown(f"**Primary Action:** {r['fix_action']}")
                with col_b:
                    st.markdown(f"<span style='background-color:{tag_color}; color:white; padding:4px 10px; border-radius:4px; font-weight:700;'>{urgency_tag.upper()}</span>", unsafe_allow_html=True)
                    st.write(f"⏱️ **Time:** {r['est_time']} hrs")
                    st.write(f"💰 **Cost:** ${r['est_cost']}")

# ==========================================
# VIEW 3: KNOWLEDGE BASE EXPLORER
# ==========================================
elif nav_choice == "📚 Knowledge Base Explorer":
    st.markdown('<div class="main-header">Historical Maintenance Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Key Feature 1: Searchable knowledge base of 265+ past maintenance records across campus facilities.</div>', unsafe_allow_html=True)

    records = vector_store.get_all_records()
    df = pd.DataFrame(records)

    # Filter Controls
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        eq_filter = st.selectbox("Filter by Equipment", ["All"] + vector_store.get_equipment_types())
    with fcol2:
        urg_filter = st.selectbox("Filter by Urgency", ["All", "Critical", "High", "Medium", "Low"])
    with fcol3:
        search_kw = st.text_input("Keyword Search (Issue / Cause / Fix)", placeholder="e.g. capacitor, refrigerant, float switch...")

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

    st.write(f"Displaying **{len(filtered_df)}** of **{len(df)}** historical maintenance records:")

    st.dataframe(
        filtered_df[["id", "equipment_type", "equipment_id", "location", "reported_issue", "root_cause", "urgency", "resolution_time_hrs", "cost_estimate_usd", "date_logged"]],
        use_container_width=True,
        hide_index=True
    )

    # Detailed Record Viewer
    st.markdown("---")
    st.subheader("🔍 Inspect Single Record Details")
    selected_id = st.selectbox("Select Record ID to view complete maintenance dossier:", filtered_df["id"].tolist())
    
    if selected_id:
        rec = next((r for r in records if r["id"] == selected_id), None)
        if rec:
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Record ID:** {rec['id']}")
                st.write(f"**Equipment:** {rec['equipment_type']} ({rec.get('equipment_id', '')})")
                st.write(f"**Location:** {rec.get('location', '')}")
                st.write(f"**Reported Issue:** {rec['reported_issue']}")
                st.write(f"**Symptoms:** {rec['symptoms']}")
            with c2:
                st.write(f"**Root Cause:** {rec['root_cause']}")
                st.write(f"**Fix Action:** {rec['fix_action']}")
                st.write(f"**Urgency:** {rec['urgency']} | **Cost:** ${rec['cost_estimate_usd']} | **Time:** {rec['resolution_time_hrs']} hrs")
                st.write(f"**Technician:** {rec.get('technician', '')}")
                st.write(f"**Technician Notes:** {rec.get('technician_notes', '')}")

    # CSV Export
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Filtered Records to CSV",
        data=csv_data,
        file_name="campus_maintenance_knowledge_base.csv",
        mime="text/csv"
    )

# ==========================================
# VIEW 4: SYSTEM ANALYTICS & ACCURACY
# ==========================================
elif nav_choice == "📊 System Analytics & Accuracy":
    st.markdown('<div class="main-header">Campus Infrastructure Reliability & Decision Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Recurring failure patterns, cost distributions, and decision-support accuracy metrics.</div>', unsafe_allow_html=True)

    records = vector_store.get_all_records()
    df = pd.DataFrame(records)

    # Top KPI Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Total Historical Records</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    with kpi2:
        avg_cost = int(df["cost_estimate_usd"].mean())
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Avg Repair Cost</div>
            <div class="metric-value">${avg_cost}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        avg_time = round(df["resolution_time_hrs"].mean(), 1)
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Mean Time to Repair (MTTR)</div>
            <div class="metric-value">{avg_time} hrs</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Technician Validation Rate</div>
            <div class="metric-value">{metrics['accuracy_rate']}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualizations
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Equipment Failure Frequency")
        counts = df["equipment_type"].value_counts()
        st.bar_chart(counts)

    with chart_col2:
        st.subheader("Urgency Level Breakdown")
        urg_counts = df["urgency"].value_counts()
        st.bar_chart(urg_counts)

    cost_col1, cost_col2 = st.columns(2)
    with cost_col1:
        st.subheader("Average Repair Cost by Equipment Type ($)")
        avg_cost_eq = df.groupby("equipment_type")["cost_estimate_usd"].mean().round(0)
        st.bar_chart(avg_cost_eq)

    with cost_col2:
        st.subheader("Average Repair Duration (Hours)")
        avg_time_eq = df.groupby("equipment_type")["resolution_time_hrs"].mean().round(1)
        st.bar_chart(avg_time_eq)

    st.markdown("---")
    st.subheader("Feedback Loop Learning Telemetry")
    st.write(f"- **Total Human-in-the-Loop Feedback Submissions:** {metrics['total_feedback']}")
    st.write(f"- **Direct Confirmations (Accurate Diagnoses):** {metrics['confirmed_accurate']}")
    st.write(f"- **Technician Override/Refinements:** {metrics['corrected']}")
    st.write(f"- **Live Knowledge Base Size:** {vector_store.index.ntotal} vectors indexed in FAISS")
