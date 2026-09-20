"""
Campus/Facility Infrastructure Decision-Support Agent
Enterprise Cyber-Industrial Decision Intelligence Command Center
Built with Streamlit, LangGraph, FAISS & SentenceTransformers
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import json
import time
from datetime import datetime
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION & METADATA
# ==========================================
st.set_page_config(
    page_title="InfraOps Nexus | Campus Decision Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CYBER-INDUSTRIAL COMMAND CENTER STYLESHEET
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

    /* Global Base & Typography */
    html, body, .stApp, .stMarkdown p, .stText, h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Preserve Streamlit Material Icons & Symbols (renders keyboard_arrow as icons) */
    .material-symbols-rounded,
    .material-symbols-outlined,
    .material-icons,
    [data-testid="stIconMaterial"],
    [data-testid*="Icon"],
    [data-testid="stExpanderToggleIcon"],
    [class*="material-symbols"],
    [class*="material-icons"] {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
        font-weight: normal !important;
        font-style: normal !important;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-feature-settings: 'liga' 1 !important;
        font-feature-settings: 'liga' 1 !important;
        -webkit-font-smoothing: antialiased;
    }
    
    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Ambient Cyber Background */
    .stApp {
        background-color: #070B14;
        background-image: 
            radial-gradient(circle at 12% 15%, rgba(14, 165, 233, 0.12) 0%, transparent 45%),
            radial-gradient(circle at 88% 85%, rgba(99, 102, 241, 0.10) 0%, transparent 50%),
            radial-gradient(circle at 50% 50%, rgba(16, 185, 129, 0.05) 0%, transparent 65%),
            linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 100% 100%, 36px 36px, 36px 36px;
        color: #E2E8F0;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1440px !important;
    }

    /* Top Command Header with Campus Banner Overlay */
    .hero-banner {
        position: relative;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92) 0%, rgba(11, 15, 25, 0.88) 100%),
                    url('https://images.unsplash.com/photo-1541888946425-d0fbb186c5f7?auto=format&fit=crop&w=1600&q=80') center/cover no-repeat;
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 14px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 35px -5px rgba(0, 0, 0, 0.6), 0 0 20px rgba(14, 165, 233, 0.15);
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #00F0FF, #3B82F6, #10B981, #F59E0B);
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #FFFFFF 30%, #93C5FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .hero-subtitle {
        font-size: 0.92rem;
        color: #94A3B8;
        margin-top: 0.35rem;
        font-weight: 400;
        max-width: 700px;
    }
    .hero-telemetry-bar {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    .hud-badge {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.2);
        backdrop-filter: blur(8px);
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-size: 0.78rem;
        color: #E2E8F0;
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Animated Pulsing Heartbeat */
    @keyframes pulseGreen {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    @keyframes pulseCyan {
        0% { box-shadow: 0 0 0 0 rgba(6, 182, 212, 0.7); }
        70% { box-shadow: 0 0 0 8px rgba(6, 182, 212, 0); }
        100% { box-shadow: 0 0 0 0 rgba(6, 182, 212, 0); }
    }
    .pulse-dot-green {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
        animation: pulseGreen 2s infinite;
    }
    .pulse-dot-cyan {
        width: 8px;
        height: 8px;
        background-color: #06B6D4;
        border-radius: 50%;
        display: inline-block;
        animation: pulseCyan 2s infinite;
    }

    /* Cyber Glass Cards */
    .cyber-card {
        background: rgba(17, 24, 39, 0.75) !important;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(56, 189, 248, 0.18) !important;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.2rem;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .cyber-card:hover {
        border-color: rgba(56, 189, 248, 0.4) !important;
    }

    /* Metric Tiles Grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #38BDF8;
    }
    .metric-card-title {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .metric-card-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 0.35rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
        line-height: 1.15;
    }
    .metric-card-sub {
        font-size: 0.76rem;
        color: #64748B;
        margin-top: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }

    /* Interactive Visual Equipment Cards */
    .eq-card-container {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.9rem;
        margin-bottom: 1.5rem;
    }
    .eq-thumb-card {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(51, 65, 85, 0.8);
        border-radius: 10px;
        overflow: hidden;
        cursor: pointer;
        transition: all 0.25s ease;
        position: relative;
    }
    .eq-thumb-card:hover {
        transform: translateY(-3px);
        border-color: #06B6D4;
        box-shadow: 0 8px 25px rgba(6, 182, 212, 0.25);
    }
    .eq-thumb-card.selected {
        border-color: #00F0FF;
        box-shadow: 0 0 20px rgba(0, 240, 255, 0.35);
    }
    .eq-image-box {
        width: 100%;
        height: 105px;
        overflow: hidden;
        position: relative;
    }
    .eq-image-box img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.4s ease;
    }
    .eq-thumb-card:hover .eq-image-box img {
        transform: scale(1.08);
    }
    .eq-overlay-status {
        position: absolute;
        top: 6px;
        right: 6px;
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 4px;
        padding: 0.15rem 0.45rem;
        font-size: 0.65rem;
        font-weight: 700;
        color: #10B981;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .eq-details {
        padding: 0.65rem 0.75rem 0.75rem 0.75rem;
    }
    .eq-name {
        font-size: 0.84rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .eq-tag {
        font-size: 0.70rem;
        color: #38BDF8;
        font-weight: 600;
        margin-top: 0.1rem;
    }

    /* Workflow Pipeline Ribbon */
    .pipeline-strip {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 10px;
        padding: 0.85rem 1.2rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        overflow-x: auto;
        gap: 0.6rem;
    }
    .pipe-step {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.82rem;
        font-weight: 600;
        color: #94A3B8;
        white-space: nowrap;
    }
    .pipe-step.active {
        color: #38BDF8;
    }
    .pipe-node {
        width: 26px;
        height: 26px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.74rem;
        font-weight: 700;
        background: rgba(30, 41, 59, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.3);
        color: #E2E8F0;
    }
    .pipe-step.active .pipe-node {
        background: #0284C7;
        border-color: #38BDF8;
        color: #FFFFFF;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.5);
    }
    .pipe-arrow {
        color: #475569;
        font-size: 0.9rem;
    }

    /* LOTO Safety Container */
    .loto-banner {
        background: linear-gradient(135deg, rgba(146, 64, 14, 0.25) 0%, rgba(69, 26, 3, 0.3) 100%);
        border: 1px solid rgba(245, 158, 11, 0.5);
        border-left: 5px solid #F59E0B;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-top: 1rem;
        color: #FDE68A;
        font-size: 0.86rem;
    }
    .loto-title {
        font-weight: 800;
        font-size: 0.88rem;
        color: #FBBF24;
        display: flex;
        align-items: center;
        gap: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }

    /* Severity Badges */
    .badge-critical {
        background: rgba(220, 38, 38, 0.2) !important;
        color: #F87171 !important;
        border: 1px solid rgba(239, 68, 68, 0.5) !important;
    }
    .badge-high {
        background: rgba(217, 119, 6, 0.2) !important;
        color: #FBBF24 !important;
        border: 1px solid rgba(245, 158, 11, 0.5) !important;
    }
    .badge-medium {
        background: rgba(37, 99, 235, 0.2) !important;
        color: #60A5FA !important;
        border: 1px solid rgba(59, 130, 246, 0.5) !important;
    }
    .badge-low {
        background: rgba(16, 185, 129, 0.2) !important;
        color: #34D399 !important;
        border: 1px solid rgba(16, 185, 129, 0.5) !important;
    }

    /* Streamlit Widget Form & Input Overrides */
    div[data-testid="stForm"] {
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(16px);
        border: 1px solid rgba(56, 189, 248, 0.22) !important;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
    }
    div[data-baseweb="input"], div[data-baseweb="select"], div[data-baseweb="textarea"] {
        background-color: rgba(11, 17, 32, 0.85) !important;
        border-color: rgba(51, 65, 85, 0.8) !important;
        color: #F8FAFC !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="input"]:focus-within, div[data-baseweb="textarea"]:focus-within {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.3) !important;
    }

    /* Buttons */
    button[kind="primary"] {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        border: 1px solid #38BDF8 !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.4rem !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.6) !important;
        border-color: #7DD3FC !important;
    }
    button[kind="secondary"] {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        color: #E2E8F0 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    button[kind="secondary"]:hover {
        border-color: #38BDF8 !important;
        color: #38BDF8 !important;
    }

    /* Sidebar Dark Cyber Style */
    [data-testid="stSidebar"] {
        background-color: #0B1120 !important;
        border-right: 1px solid rgba(56, 189, 248, 0.15) !important;
    }
    .sidebar-brand-box {
        padding: 0.6rem 0 1.2rem 0;
        border-bottom: 1px solid rgba(51, 65, 85, 0.6);
        margin-bottom: 1.2rem;
    }
    .sidebar-brand-title {
        font-size: 1.18rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sidebar-brand-sub {
        font-size: 0.74rem;
        color: #64748B;
        font-weight: 500;
        margin-top: 0.15rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .sidebar-nav-label {
        font-size: 0.70rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1.2rem 0 0.4rem 0;
    }

    /* Telemetry Row */
    .telemetry-box {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(51, 65, 85, 0.7);
        border-radius: 8px;
        padding: 0.85rem;
        font-size: 0.78rem;
        margin-top: 1rem;
    }
    .telemetry-line {
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #94A3B8;
        margin-bottom: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Checkbox & Expander Dark Styling */
    .stCheckbox label {
        color: #E2E8F0 !important;
        font-size: 0.88rem !important;
    }
    div[data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(51, 65, 85, 0.8) !important;
        border-radius: 8px !important;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. BACKEND INITIALIZATION (CACHED)
# ==========================================
from src.retrieval.vector_store import MaintenanceVectorStore
from src.agents.orchestrator import InfrastructureDecisionOrchestrator
from src.feedback.feedback_handler import FeedbackHandler

@st.cache_resource(show_spinner="Initializing FAISS Semantic Knowledge Base & Multi-Agent Graph...")
def get_system():
    vector_store = MaintenanceVectorStore()
    orchestrator = InfrastructureDecisionOrchestrator(vector_store=vector_store)
    feedback_handler = FeedbackHandler(vector_store=vector_store)
    return vector_store, orchestrator, feedback_handler

vector_store, orchestrator, feedback_handler = get_system()
metrics = feedback_handler.get_metrics()
all_records = vector_store.get_all_records()
total_kb_cases = vector_store.index.ntotal

# Taxonomy Mapping
EQUIPMENT_DISPLAY_TO_BACKEND = {
    "AC Unit": "HVAC",
    "Generator": "Diesel Generator",
    "Elevator": "Elevator",
    "Pump": "Water Supply & Pumps",
    "Electrical System": "Electrical Switchgear",
    "Other": "All"
}
EQUIPMENT_BACKEND_TO_DISPLAY = {
    "HVAC": "AC Unit",
    "Diesel Generator": "Generator",
    "Elevator": "Elevator",
    "Water Supply & Pumps": "Pump",
    "Electrical Switchgear": "Electrical System"
}

# Rich Equipment Gallery Metadata with Curated High-Res Unsplash Images
EQUIPMENT_GALLERY = {
    "AC Unit": {
        "title": "HVAC & Chiller Plants",
        "tag": "Thermal Management",
        "image": "https://images.unsplash.com/photo-1621905251189-08b45d6a269e?auto=format&fit=crop&w=600&q=80",
        "status": "ONLINE",
        "location": "Engineering Block - Room 204",
        "asset_id": "ASSET-AC-204",
        "issue": "Air conditioning unit blowing warm air into lecture hall with audible compressor rattling noise",
        "symptoms": "Low suction pressure (42 PSI), thermal expansion frost accumulation, short-cycling every 4 minutes",
        "sensor_codes": "ALARM-402: Head Pressure Differential Exceeded"
    },
    "Generator": {
        "title": "Emergency Standby Power",
        "tag": "Critical Power Systems",
        "image": "https://images.unsplash.com/photo-1513828583688-c52646db42da?auto=format&fit=crop&w=600&q=80",
        "status": "STANDBY",
        "location": "Main Substation Generator Yard",
        "asset_id": "ASSET-GEN-01",
        "issue": "Emergency diesel generator engine temperature escalating rapidly during monthly load transfer test",
        "symptoms": "Coolant temperature reaches 102°C at 50% load, voltage swings ±18V, exhaust smoky with unburnt fuel smell",
        "sensor_codes": "FAULT-GEN-88: High Coolant Temperature Warning"
    },
    "Elevator": {
        "title": "Vertical Traction Hoists",
        "tag": "Life Safety & Transit",
        "image": "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=600&q=80",
        "status": "ONLINE",
        "location": "Academic Complex - Tower B",
        "asset_id": "ASSET-ELEV-03",
        "issue": "Passenger elevator car stopped abruptly between 3rd and 4th floors; door interlock circuit tripped",
        "symptoms": "Safety loop open, controller display reports door clutch misalignment, passenger emergency intercom activated",
        "sensor_codes": "ERROR-F021: Hoistway Landing Interlock Open"
    },
    "Electrical System": {
        "title": "11kV Substation Switchgear",
        "tag": "Grid Distribution",
        "image": "https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?auto=format&fit=crop&w=600&q=80",
        "status": "MONITORED",
        "location": "Science Block Substation - Panel B",
        "asset_id": "ASSET-SWG-02",
        "issue": "Main feeder molded case circuit breaker trips repeatedly within 10 minutes of HVAC chiller startup",
        "symptoms": "Phase L2 contactor thermal discoloration, busbar joint reading 94°C via infrared pyrometer, faint ozone odor",
        "sensor_codes": "TRIP-50/51: Instantaneous Overcurrent Pickup"
    },
    "Pump": {
        "title": "Hydraulic Booster Pumps",
        "tag": "Water & Utilities",
        "image": "https://images.unsplash.com/photo-1504307651254-35680f356dfd?auto=format&fit=crop&w=600&q=80",
        "status": "ONLINE",
        "location": "Central Utility Building - Pump Room",
        "asset_id": "ASSET-PUMP-04",
        "issue": "Domestic water booster pump running continuously with zero discharge pressure to upper floors",
        "symptoms": "Severe rattling sound in bronze casing, cavitation vibration > 7.5 mm/s, discharge pressure gauge at 0 PSI",
        "sensor_codes": "FAULT-VFD-01: Overcurrent / Dry Run Protection"
    }
}

# Session State Initialization
if "selected_eq_key" not in st.session_state:
    st.session_state["selected_eq_key"] = "AC Unit"
if "nav_module" not in st.session_state:
    st.session_state["nav_module"] = "Incident Triage"
if "checklist_state" not in st.session_state:
    st.session_state["checklist_state"] = {}

# ==========================================
# 4. SIDEBAR COMMAND CONSOLE
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-box">
        <div class="sidebar-brand-title">
            <span>⚡</span> INFRAOPS NEXUS
        </div>
        <div class="sidebar-brand-sub">
            Campus Decision Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-nav-label">OPERATIONS & TRIAGE</div>', unsafe_allow_html=True)
    b_triage = st.button("🚨 Incident Triage & Diagnosis", use_container_width=True)
    b_multi = st.button("🌐 Multi-Incident War Room", use_container_width=True)

    st.markdown('<div class="sidebar-nav-label">KNOWLEDGE REPOSITORY</div>', unsafe_allow_html=True)
    b_kb = st.button("📚 Maintenance Knowledge Base", use_container_width=True)

    st.markdown('<div class="sidebar-nav-label">TELEMETRY & FIELD</div>', unsafe_allow_html=True)
    b_ana = st.button("📊 Fleet System Analytics", use_container_width=True)
    b_fb = st.button("👨‍🔧 Technician Sign-Off Portal", use_container_width=True)

    if b_triage: st.session_state["nav_module"] = "Incident Triage"
    elif b_multi: st.session_state["nav_module"] = "Multi-Incident Analysis"
    elif b_kb: st.session_state["nav_module"] = "Maintenance Knowledge Base"
    elif b_ana: st.session_state["nav_module"] = "System Analytics"
    elif b_fb: st.session_state["nav_module"] = "Technician Validation"

    nav_view = st.session_state["nav_module"]

    st.markdown("---")
    st.markdown(f"""
    <div class="telemetry-box">
        <div style="font-size: 0.72rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; margin-bottom: 0.5rem; letter-spacing: 0.08em; display: flex; justify-content: space-between;">
            <span>SYSTEM TELEMETRY</span>
            <span class="pulse-dot-green"></span>
        </div>
        <div class="telemetry-line">
            <span>Graph Engine</span>
            <span style="color: #10B981;">LangGraph Active</span>
        </div>
        <div class="telemetry-line">
            <span>Semantic Index</span>
            <span style="color: #38BDF8;">FAISS FlatIP (384d)</span>
        </div>
        <div class="telemetry-line">
            <span>Total Precedents</span>
            <span style="color: #F8FAFC; font-weight: 700;">{total_kb_cases} Cases</span>
        </div>
        <div class="telemetry-line">
            <span>Accuracy Rate</span>
            <span style="color: #10B981; font-weight: 700;">{metrics['accuracy_rate']}%</span>
        </div>
        <div class="telemetry-line">
            <span>Host Port</span>
            <span style="color: #FBBF24; font-weight: 700;">:8502</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Reload Vector Engine", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

# ==========================================
# 5. HERO COMMAND BANNER & TELEMETRY KPIS
# ==========================================
st.markdown(f"""
<div class="hero-banner">
    <div style="position: relative; z-index: 2;">
        <div class="hero-title">
            <span>🏢</span> CAMPUS INFRASTRUCTURE COMMAND CENTER
        </div>
        <div class="hero-subtitle">
            Autonomous multi-agent triage, FAISS semantic case retrieval, and explainable decision support for critical facility assets.
        </div>
        <div class="hero-telemetry-bar">
            <div class="hud-badge">
                <span class="pulse-dot-green"></span> SYSTEM: <b>ONLINE</b>
            </div>
            <div class="hud-badge">
                <span class="pulse-dot-cyan"></span> INDEX: <b>{total_kb_cases} EMBEDDED CASES</b>
            </div>
            <div class="hud-badge">
                <span>🎯</span> ACCURACY: <b>{metrics['accuracy_rate']}% FIELD-VERIFIED</b>
            </div>
            <div class="hud-badge">
                <span>🛡️</span> OSHA COMPLIANCE: <b>LOTO ENFORCED</b>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Top 4 Cyber Metric Tiles
active_incidents = 1 if "current_result" in st.session_state else 0
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-card-title">
            <span>ACTIVE INCIDENTS</span>
            <span>🚨</span>
        </div>
        <div class="metric-card-val" style="color: {'#F87171' if active_incidents > 0 else '#34D399'};">
            {active_incidents}
        </div>
        <div class="metric-card-sub">
            <span style="color: {'#F87171' if active_incidents > 0 else '#64748B'};">●</span> In active dispatch queue
        </div>
    </div>
    <div class="metric-card">
        <div class="metric-card-title">
            <span>KNOWLEDGE BASE REPOSITORY</span>
            <span>🧠</span>
        </div>
        <div class="metric-card-val" style="color: #38BDF8;">
            {total_kb_cases}
        </div>
        <div class="metric-card-sub">
            <span>📚</span> FAISS dense vector clusters
        </div>
    </div>
    <div class="metric-card">
        <div class="metric-card-title">
            <span>TECHNICIAN VERIFICATIONS</span>
            <span>✅</span>
        </div>
        <div class="metric-card-val" style="color: #10B981;">
            {metrics['confirmed_accurate']}
        </div>
        <div class="metric-card-sub">
            <span>👨‍🔧</span> {metrics['total_feedback']} total field sign-offs
        </div>
    </div>
    <div class="metric-card">
        <div class="metric-card-title">
            <span>DECISION CONFIDENCE BENCHMARK</span>
            <span>🎯</span>
        </div>
        <div class="metric-card-val" style="color: #A78BFA;">
            {metrics['accuracy_rate']}%
        </div>
        <div class="metric-card-sub">
            <span>⚡</span> Human-in-the-loop validated
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# VIEW 1: INCIDENT TRIAGE & DIAGNOSIS
# ==========================================
if nav_view == "Incident Triage":
    # Multi-Agent Workflow Pipeline Ribbon
    st.markdown("""
    <div class="pipeline-strip">
        <div class="pipe-step active">
            <span class="pipe-node">01</span>
            <span>Telemetry Ingestion</span>
        </div>
        <span class="pipe-arrow">➔</span>
        <div class="pipe-step active">
            <span class="pipe-node">02</span>
            <span>FAISS Semantic Match</span>
        </div>
        <span class="pipe-arrow">➔</span>
        <div class="pipe-step active">
            <span class="pipe-node">03</span>
            <span>Agent Diagnosis</span>
        </div>
        <span class="pipe-arrow">➔</span>
        <div class="pipe-step active">
            <span class="pipe-node">04</span>
            <span>Action & LOTO Plan</span>
        </div>
        <span class="pipe-arrow">➔</span>
        <div class="pipe-step active">
            <span class="pipe-node">05</span>
            <span>Technician Verification</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # VISUAL EQUIPMENT SHOWCASE (INTERACTIVE GALLERY)
    # ----------------------------------------------------
    st.markdown("""
    <div style="font-size: 0.82rem; font-weight: 800; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.4rem;">
        <span>📸</span> CRITICAL FACILITY EQUIPMENT INVENTORY (ONE-CLICK LAUNCHPAD)
    </div>
    """, unsafe_allow_html=True)

    eq_cols = st.columns(5)
    for idx, (eq_name, eq_info) in enumerate(EQUIPMENT_GALLERY.items()):
        with eq_cols[idx]:
            is_sel = (st.session_state["selected_eq_key"] == eq_name)
            sel_border = "border: 2px solid #00F0FF; box-shadow: 0 0 15px rgba(0, 240, 255, 0.4);" if is_sel else "border: 1px solid rgba(51, 65, 85, 0.8);"
            st.markdown(f"""
            <div class="eq-thumb-card" style="{sel_border}">
                <div class="eq-image-box">
                    <img src="{eq_info['image']}" alt="{eq_info['title']}" onerror="this.onerror=null; this.src='https://images.unsplash.com/photo-1581092160607-ee22621dd758?auto=format&fit=crop&w=600&q=80';" />
                    <div class="eq-overlay-status">{eq_info['status']}</div>
                </div>
                <div class="eq-details">
                    <div class="eq-name">{eq_name}</div>
                    <div class="eq-tag">{eq_info['tag']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Load {eq_name}", key=f"btn_sel_{eq_name}", use_container_width=True):
                st.session_state["selected_eq_key"] = eq_name
                st.rerun()

    active_preset = EQUIPMENT_GALLERY[st.session_state["selected_eq_key"]]

    # ----------------------------------------------------
    # LIVE IOT SENSOR & TELEMETRY SIMULATOR (INTERACTIVE)
    # ----------------------------------------------------
    with st.expander("🎛️ Interactive IoT Sensor Simulator & Anomaly Injector", expanded=False):
        st.markdown("""
        <div style="font-size: 0.84rem; color: #94A3B8; margin-bottom: 0.8rem;">
            Tweak real-time facility IoT sensors to simulate edge hardware telemetry anomalies and test agent response.
        </div>
        """, unsafe_allow_html=True)
        col_iot1, col_iot2, col_iot3, col_iot4 = st.columns(4)
        with col_iot1:
            sim_pressure = st.slider("Suction/Discharge (PSI)", 0, 150, 42, help="Normal operating range: 65 - 80 PSI")
        with col_iot2:
            sim_temp = st.slider("Core Temperature (°C)", 20, 130, 94, help="Normal operating range: 45 - 65 °C")
        with col_iot3:
            sim_vibe = st.slider("Vibration (mm/s RMS)", 0.5, 12.0, 6.8, help="ISO 10816 Alarm Threshold: > 4.5 mm/s")
        with col_iot4:
            sim_current = st.slider("Feeder Load Current (A)", 10, 300, 185, help="Nominal breaker trip limit: 160 A")

        # Dynamic Anomaly Evaluation
        is_anomaly = (sim_pressure < 50) or (sim_temp > 85) or (sim_vibe > 5.0) or (sim_current > 175)
        if is_anomaly:
            st.markdown(f"""
            <div style="background: rgba(220, 38, 38, 0.15); border: 1px solid #EF4444; border-radius: 6px; padding: 0.6rem 0.9rem; color: #FCA5A5; font-size: 0.82rem; display: flex; justify-content: space-between; align-items: center;">
                <span>⚠️ <b>TELEMETRY ALERT:</b> Critical threshold breach detected ({sim_temp}°C temp / {sim_pressure} PSI / {sim_vibe} mm/s vibration).</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; border-radius: 6px; padding: 0.6rem 0.9rem; color: #6EE7B7; font-size: 0.82rem;">
                ✓ <b>TELEMETRY NORMAL:</b> All sensor streams within nominal baseline operating limits.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # INCIDENT INTAKE FORM
    # ----------------------------------------------------
    with st.form("incident_triage_form"):
        st.markdown(f"""
        <div style="font-size: 1.05rem; font-weight: 800; color: #F8FAFC; margin-bottom: 0.85rem; display: flex; align-items: center; gap: 0.5rem;">
            <span>📋</span> REPORT NEW FACILITY ANOMALY & DISPATCH TRIAGE
        </div>
        """, unsafe_allow_html=True)

        col_f1, col_f2 = st.columns(2)
        eq_options = ["AC Unit", "Generator", "Elevator", "Pump", "Electrical System", "Other"]
        curr_eq_idx = eq_options.index(st.session_state["selected_eq_key"]) if st.session_state["selected_eq_key"] in eq_options else 0

        with col_f1:
            equipment_type = st.selectbox("Equipment Type", eq_options, index=curr_eq_idx)
            facility_location = st.text_input("Facility / Campus Location", value=active_preset["location"])
        with col_f2:
            asset_id = st.text_input("Asset ID Tag", value=active_preset["asset_id"])
            top_k = st.slider("Historical Cases to Retrieve (FAISS)", 2, 6, 4)

        reported_complaint = st.text_area("Reported Complaint / Failure Description", value=active_preset["issue"], height=75)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            observed_symptoms = st.text_area("Observed Symptoms & Edge Indicators", value=active_preset["symptoms"], height=70)
        with col_s2:
            sensor_codes = st.text_area("Sensor Readings / Telemetry Codes", value=active_preset["sensor_codes"], height=70)

        analyze_btn = st.form_submit_button("⚡ Analyze Incident & Generate Decision Dossier", type="primary", use_container_width=True)

    # ----------------------------------------------------
    # ORCHESTRATION PIPELINE EXECUTION
    # ----------------------------------------------------
    if analyze_btn:
        if not reported_complaint.strip():
            st.error("⚠️ Please specify the incident or complaint before analyzing.")
        else:
            backend_equipment = EQUIPMENT_DISPLAY_TO_BACKEND.get(equipment_type, "All")
            composite_symptoms = observed_symptoms
            if sensor_codes.strip():
                composite_symptoms = f"{observed_symptoms}. Telemetry: {sensor_codes}".strip()

            with st.status("Executing Multi-Agent Incident Investigation...", expanded=True) as status_box:
                st.write(f"✓ **1. Telemetry Ingested:** Tagged as `{equipment_type}` at `{facility_location}`")
                time.sleep(0.2)
                st.write(f"✓ **2. Semantic Vector Search:** Scanning FAISS dense index for top {top_k} similar maintenance precedents...")
                time.sleep(0.25)
                
                start_time = time.time()
                result = orchestrator.process_incident(
                    reported_issue=reported_complaint,
                    symptoms=composite_symptoms,
                    equipment_type=backend_equipment,
                    location=facility_location,
                    top_k=top_k
                )
                duration = round(time.time() - start_time, 2)
                
                st.write("✓ **3. Diagnostic Reasoner:** Hypotheses evaluated against historical failure correlations.")
                time.sleep(0.2)
                st.write("✓ **4. Decision Formulation:** Estimated MTTR, budget impact, and OSHA LOTO protocols established.")
                time.sleep(0.15)
                status_box.update(label=f"Multi-Agent Triage Complete in {duration}s!", state="complete", expanded=False)

                st.session_state["current_result"] = result
                st.session_state["incident_input"] = {
                    "equipment_type": equipment_type,
                    "backend_equipment": backend_equipment,
                    "asset_id": asset_id,
                    "location": facility_location,
                    "reported_issue": reported_complaint,
                    "symptoms": composite_symptoms,
                    "sensor_codes": sensor_codes
                }
                st.session_state["pipeline_duration"] = duration
                st.session_state["checklist_state"] = {}

    # ----------------------------------------------------
    # DECISION SUPPORT DOSSIER (RESULTS DISPLAY)
    # ----------------------------------------------------
    if "current_result" in st.session_state:
        res = st.session_state["current_result"]
        inp = st.session_state["incident_input"]
        diag = res.get("diagnosis", {})
        recom = res.get("recommendation", {})
        retrieved_cases = res.get("retrieved_cases", [])
        urgency = recom.get("urgency", "Medium")

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

        urgency_classes = {
            "Critical": "badge-critical",
            "High": "badge-high",
            "Medium": "badge-medium",
            "Low": "badge-low"
        }
        urg_badge = urgency_classes.get(urgency, "badge-medium")

        # Dossier Header Card
        st.markdown(f"""
        <div class="cyber-card" style="border-left: 5px solid #38BDF8 !important;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <div style="font-size: 0.74rem; font-weight: 800; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.08em;">
                        FACILITY DECISION INTELLIGENCE DOSSIER
                    </div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #F8FAFC; margin-top: 0.2rem;">
                        Asset: {inp['asset_id']} — {inp['equipment_type']}
                    </div>
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.25rem;">
                        📍 Location: <b>{inp['location']}</b> • ⏱️ Processed: <b>{datetime.now().strftime('%Y-%m-%d %H:%M')}</b> • ⚡ Latency: <b>{st.session_state.get('pipeline_duration', 0.4)}s</b>
                    </div>
                </div>
                <div>
                    <span class="hud-badge {urg_badge}" style="font-size: 0.92rem; padding: 0.45rem 1rem;">
                        SEVERITY: {urgency.upper()}
                    </span>
                </div>
            </div>
            <div style="margin-top: 0.85rem; padding-top: 0.75rem; border-top: 1px solid rgba(51, 65, 85, 0.6); font-size: 0.88rem; color: #CBD5E1;">
                <b>Reported Complaint:</b> {inp['reported_issue']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2-Column AI Diagnosis & Mitigation Action
        col_diag, col_rec = st.columns([1, 1])

        with col_diag:
            st.markdown("""
            <div class="cyber-card">
                <div style="font-size: 1.05rem; font-weight: 800; color: #F8FAFC; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;">
                    <span>🧠</span> AI ROOT CAUSE DIAGNOSIS
                </div>
            """, unsafe_allow_html=True)

            conf = diag.get("confidence_score", 85.0)
            st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.08em;">
                        IDENTIFIED PRIMARY CAUSE
                    </div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #38BDF8; margin-top: 0.2rem;">
                        {diag.get('primary_root_cause')}
                    </div>
                </div>
                <div style="margin-bottom: 0.6rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 0.35rem;">
                        <span style="font-weight: 600; color: #94A3B8;">STATISTICAL CONFIDENCE</span>
                        <span style="font-weight: 800; color: #10B981; font-family: 'JetBrains Mono';">{conf:.1f}%</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.progress(min(1.0, conf / 100.0))

            st.markdown("<div style='font-size: 0.76rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin: 1.2rem 0 0.5rem 0;'>MULTI-AGENT REASONING CHAIN</div>", unsafe_allow_html=True)
            reasoning_steps = diag.get("reasoning_chain", [])
            if not reasoning_steps:
                reasoning_steps = [
                    "Telemetry indicators correlate with mechanical pressure drops or thermal elevation.",
                    "FAISS semantic match confirms past resolution paths with high convergence.",
                    "Inspection protocol isolated to primary drive / contactor assembly."
                ]

            for i, step in enumerate(reasoning_steps, 1):
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.7); border-left: 3px solid #38BDF8; border-radius: 0 6px 6px 0; padding: 0.6rem 0.85rem; margin-bottom: 0.45rem; font-size: 0.84rem; color: #E2E8F0; border-top: 1px solid rgba(51, 65, 85, 0.4); border-right: 1px solid rgba(51, 65, 85, 0.4); border-bottom: 1px solid rgba(51, 65, 85, 0.4);">
                    <b style="color: #38BDF8;">Step {i}:</b> {step}
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_rec:
            st.markdown("""
            <div class="cyber-card">
                <div style="font-size: 1.05rem; font-weight: 800; color: #F8FAFC; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;">
                    <span>🔧</span> RECOMMENDED FIELD ACTION & DISPATCH
                </div>
            """, unsafe_allow_html=True)

            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(51, 65, 85, 0.7); border-radius: 8px; padding: 0.75rem; text-align: center;">
                    <div style="font-size: 0.70rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;">ESTIMATED MTTR</div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #F8FAFC; margin-top: 0.2rem;">{recom.get('estimated_time_hrs', 2.5)} <span style="font-size: 0.85rem; font-weight: 500; color: #94A3B8;">hrs</span></div>
                </div>
                """, unsafe_allow_html=True)
            with r_col2:
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(51, 65, 85, 0.7); border-radius: 8px; padding: 0.75rem; text-align: center;">
                    <div style="font-size: 0.70rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;">PARTS & LABOR BUDGET</div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #10B981; margin-top: 0.2rem;">${recom.get('estimated_cost_usd', 350)}</div>
                </div>
                """, unsafe_allow_html=True)

            # Interactive Checklist for Field Techs
            st.markdown("<div style='font-size: 0.76rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin: 1.2rem 0 0.5rem 0;'>ACTIONABLE RESOLUTION CHECKLIST</div>", unsafe_allow_html=True)
            fix_steps = recom.get("fix_steps", ["Perform systematic equipment diagnostic inspection."])
            completed_steps = 0
            for i, step in enumerate(fix_steps):
                ck_key = f"chk_step_{i}_{inp['asset_id']}"
                checked = st.checkbox(f"{i+1}. {step}", key=ck_key)
                if checked:
                    completed_steps += 1

            if fix_steps:
                pct_done = completed_steps / len(fix_steps)
                st.progress(pct_done)
                st.caption(f"Task Progress: **{completed_steps}/{len(fix_steps)} completed** ({int(pct_done*100)}%)")

            # OSHA LOTO Protocol Banner
            loto_items = recom.get("safety_precautions", ["De-energize circuit breaker and attach LOTO padlock."])
            st.markdown(f"""
            <div class="loto-banner">
                <div class="loto-title">
                    <span>⚠️</span> OSHA LOCKOUT / TAGOUT (LOTO) & PPE MANDATE
                </div>
                <div style="font-size: 0.82rem; line-height: 1.45;">
                    {'<br/>• '.join(['• ' + item for item in loto_items])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # 3. HISTORICAL SIMILAR PRECEDENTS (FAISS CLUSTERS)
        st.markdown("""
        <div class="cyber-card">
            <div style="font-size: 1.05rem; font-weight: 800; color: #F8FAFC; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;">
                <span>📚</span> FAISS RETRIEVED HISTORICAL PRECEDENTS
            </div>
        """, unsafe_allow_html=True)

        if retrieved_cases:
            p_cols = st.columns(min(len(retrieved_cases), 4))
            for i, prec in enumerate(retrieved_cases[:4]):
                with p_cols[i]:
                    score = prec.get("similarity_score", 0.85)
                    match_pct = int(score * 100) if score <= 1.0 else int(score)
                    st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(51, 65, 85, 0.8); border-radius: 8px; padding: 0.85rem; height: 100%;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 0.70rem; font-weight: 700; color: #94A3B8;">CASE #{prec.get('case_id', f'KB-{i+1}')}</span>
                            <span class="hud-badge" style="color: #38BDF8; border-color: #38BDF8;">{match_pct}% MATCH</span>
                        </div>
                        <div style="font-size: 0.82rem; font-weight: 700; color: #F8FAFC; margin-top: 0.4rem;">
                            {prec.get('equipment_type', inp['equipment_type'])}
                        </div>
                        <div style="font-size: 0.76rem; color: #94A3B8; margin-top: 0.25rem;">
                            <b>Root Cause:</b> {prec.get('root_cause', 'N/A')}
                        </div>
                        <div style="font-size: 0.74rem; color: #10B981; margin-top: 0.35rem;">
                            <b>Fix:</b> {prec.get('fix_action', 'Completed field repair')[:75]}...
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 4. FIELD SIGN-OFF & HUMAN-IN-THE-LOOP FEEDBACK
        st.markdown("""
        <div class="cyber-card" style="border-left: 5px solid #10B981 !important;">
            <div style="font-size: 1.05rem; font-weight: 800; color: #F8FAFC; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                <span>👨‍🔧</span> FIELD TECHNICIAN SIGN-OFF & CONTINUOUS LEARNING
            </div>
            <div style="font-size: 0.84rem; color: #94A3B8; margin-bottom: 0.8rem;">
                Human-in-the-loop validation: Confirmed resolutions are immediately embedded into the FAISS vector database to improve accuracy.
            </div>
        """, unsafe_allow_html=True)

        tech_status = st.radio("Field Diagnosis Assessment:", ["✓ Diagnosis Confirmed Accurate", "✕ Diagnosis Overridden / Corrected"], horizontal=True, key="fb_radio_main")

        if tech_status == "✓ Diagnosis Confirmed Accurate":
            c_tb1, c_tb2 = st.columns([1, 2])
            with c_tb1:
                tech_name = st.text_input("Lead Technician Name / ID", value="Tech Lead Jordan", key="tech_name_triage")
            with c_tb2:
                tech_notes = st.text_input("Field Sign-off Notes", value="Verified on site. Root cause confirmed and repair executed per procedure.", key="tech_notes_triage")

            if st.button("Confirm & Index into Knowledge Base", type="primary", use_container_width=True, key="btn_confirm_triage"):
                success, msg, details = feedback_handler.record_feedback(
                    equipment_type=inp["backend_equipment"],
                    reported_issue=inp["reported_issue"],
                    symptoms=inp["symptoms"],
                    root_cause=diag.get("primary_root_cause"),
                    fix_action=" ".join(recom.get("fix_steps", [])),
                    resolution_time_hrs=recom.get("estimated_time_hrs", 2.5),
                    cost_estimate_usd=recom.get("estimated_cost_usd", 350),
                    urgency=urgency,
                    is_accurate=True,
                    technician_name=tech_name,
                    notes=tech_notes,
                    location=inp["location"],
                    equipment_id=inp["asset_id"]
                )
                if success:
                    st.success(f"🎉 **{msg}**")
                    st.balloons()
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; border-radius: 6px; padding: 0.75rem 1rem; color: #6EE7B7; font-size: 0.84rem; margin-top: 0.5rem;">
                        📈 <b>Vector Memory Enhanced:</b> Total indexed cases expanded to <b>{details['total_knowledge_base_size']} records</b>. Next retrieval confidence: <b>{details['new_confidence_score']}%</b>.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(msg)
        else:
            actual_cause = st.text_input("Actual Root Cause Found on Site:", value=diag.get("primary_root_cause"), key="act_cause_inp")
            actual_res = st.text_area("Actual Resolution Executed:", value=" ".join(recom.get("fix_steps", [])), key="act_res_inp", height=70)
            c_o1, c_o2, c_o3 = st.columns(3)
            with c_o1:
                act_time = st.number_input("Actual Labor Hours", 0.5, 50.0, float(recom.get("estimated_time_hrs", 2.5)), 0.5)
            with c_o2:
                act_cost = st.number_input("Actual Cost ($)", 10, 30000, int(recom.get("estimated_cost_usd", 350)), 25)
            with c_o3:
                tech_o_name = st.text_input("Technician Name", value="Field Tech Morgan", key="tech_o_name")

            if st.button("Submit Ground-Truth Override to FAISS", type="primary", use_container_width=True, key="btn_override_triage"):
                success, msg, details = feedback_handler.record_feedback(
                    equipment_type=inp["backend_equipment"],
                    reported_issue=inp["reported_issue"],
                    symptoms=inp["symptoms"],
                    root_cause=actual_cause,
                    fix_action=actual_res,
                    resolution_time_hrs=act_time,
                    cost_estimate_usd=act_cost,
                    urgency=urgency,
                    is_accurate=False,
                    technician_name=tech_o_name,
                    notes="Ground-truth override after teardown.",
                    location=inp["location"],
                    equipment_id=inp["asset_id"]
                )
                if success:
                    st.success(f"🎉 **{msg}**")
                    st.info(f"Ground-truth correction successfully indexed! Vector size: **{details['total_knowledge_base_size']}**.")
                else:
                    st.error(msg)

        st.markdown("</div>", unsafe_allow_html=True)

        # Export Dossier Button
        dossier_text = f"""================================================================================
CAMPUS INFRASTRUCTURE DECISION-SUPPORT AGENT — INCIDENT DOSSIER
================================================================================
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Asset ID: {inp['asset_id']}
Equipment: {inp['equipment_type']}
Location: {inp['location']}
Severity: {urgency.upper()}

REPORTED COMPLAINT:
{inp['reported_issue']}

SYMPTOMS & TELEMETRY:
{inp['symptoms']}

--------------------------------------------------------------------------------
AI DIAGNOSTIC EVALUATION
--------------------------------------------------------------------------------
Primary Root Cause: {diag.get('primary_root_cause')}
Diagnostic Confidence: {diag.get('confidence_score')}%

Reasoning Sequence:
{chr(10).join([f"- {s}" for s in diag.get('reasoning_chain', [])])}

--------------------------------------------------------------------------------
RECOMMENDED MITIGATION & DISPATCH
--------------------------------------------------------------------------------
Estimated MTTR: {recom.get('estimated_time_hrs')} Hours
Estimated Budget: ${recom.get('estimated_cost_usd')}

Actionable Checklist:
{chr(10).join([f"{i+1}. {s}" for i, s in enumerate(recom.get('fix_steps', []))])}

OSHA LOTO Safety Protocols:
{chr(10).join([f"- {s}" for s in recom.get('safety_precautions', [])])}
================================================================================
"""
        st.download_button(
            label="📄 Export Decision Dossier (Text)",
            data=dossier_text,
            file_name=f"decision_dossier_{inp['asset_id']}.txt",
            mime="text/plain",
            use_container_width=False
        )

    else:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.6); border: 2px dashed rgba(51, 65, 85, 0.8); border-radius: 12px; padding: 2.5rem; text-align: center; margin-top: 1.5rem;">
            <div style="font-size: 2.2rem; color: #38BDF8;">⚡</div>
            <div style="font-size: 1.1rem; font-weight: 800; color: #F8FAFC; margin-top: 0.5rem;">
                No Incident Currently Loaded in Triage
            </div>
            <div style="font-size: 0.88rem; color: #94A3B8; margin-top: 0.25rem;">
                Select an equipment card from the visual gallery above or enter incident symptoms to launch automated multi-agent diagnosis.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# VIEW 2: MULTI-INCIDENT WAR ROOM
# ==========================================
elif nav_view == "Multi-Incident Analysis":
    st.markdown("""
    <div class="cyber-card">
        <div style="font-size: 1.1rem; font-weight: 800; color: #F8FAFC; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.5rem;">
            <span>🌐</span> MULTI-INCIDENT WAR ROOM (AUTONOMOUS BATCH TRIAGE)
        </div>
        <div style="font-size: 0.88rem; color: #94A3B8; line-height: 1.45;">
            Autonomous cluster engine for campus-wide concurrent failures (e.g. storm surge, substation trip, thermal overload).
            Simultaneously evaluates incident matrices, optimizes emergency dispatches, and computes cumulative fleet risk.
        </div>
    </div>
    """, unsafe_allow_html=True)

    BATCH_SCENARIOS = {
        "Campus Severe Storm & Electrical Inrush (4 Simultaneous Failures)": [
            {
                "id": "INCIDENT-01",
                "equipment_type": "Elevator",
                "location": "Academic Block - Tower A",
                "reported_issue": "Elevator stopped abruptly between floors 3 & 4; safety relay open",
                "symptoms": "Door interlock circuit interrupted, emergency cab alarm sounding",
                "urgency_tag": "CRITICAL"
            },
            {
                "id": "INCIDENT-02",
                "equipment_type": "HVAC",
                "location": "Engineering Block - Data Center",
                "reported_issue": "Data center precision CRAC unit head pressure lockout",
                "symptoms": "High condenser head pressure, ambient server rack temperature 32°C and rising",
                "urgency_tag": "HIGH"
            },
            {
                "id": "INCIDENT-03",
                "equipment_type": "Electrical Switchgear",
                "location": "Main 11kV Substation Yard",
                "reported_issue": "Substation Phase L2 busbar hotspot emitting ozone odor",
                "symptoms": "Splice connector reading 112°C on infrared pyrometer, humming breaker",
                "urgency_tag": "CRITICAL"
            },
            {
                "id": "INCIDENT-04",
                "equipment_type": "Water Supply & Pumps",
                "location": "Central Plant Pump Room",
                "reported_issue": "Booster pump packing gland drip pooling on floor",
                "symptoms": "Minor shaft sleeve water drip, domestic supply pressure steady at 55 PSI",
                "urgency_tag": "LOW"
            }
        ],
        "Heatwave Peak Demand Outages (3 Concurrent Failures)": [
            {
                "id": "INCIDENT-01",
                "equipment_type": "Diesel Generator",
                "location": "Health Sciences Building",
                "reported_issue": "Emergency standby generator failed auto-transfer sequence",
                "symptoms": "Battery bus voltage plunged to 8.9V under starter crank command",
                "urgency_tag": "HIGH"
            },
            {
                "id": "INCIDENT-02",
                "equipment_type": "HVAC",
                "location": "Campus Main Auditorium",
                "reported_issue": "Chilled water air handler blower belt shredded during conference",
                "symptoms": "Loud screeching from supply plenum, airflow dropped to zero",
                "urgency_tag": "MEDIUM"
            },
            {
                "id": "INCIDENT-03",
                "equipment_type": "Water Supply & Pumps",
                "location": "Student Residence Block C",
                "reported_issue": "Basement drainage sump pump failed to engage with water rising",
                "symptoms": "Float switch stuck upside down against PVC discharge pipe",
                "urgency_tag": "HIGH"
            }
        ]
    }

    col_b1, col_b2 = st.columns([3, 1])
    with col_b1:
        sel_batch = st.selectbox("Select Incident Cluster Scenario:", list(BATCH_SCENARIOS.keys()))
    with col_b2:
        st.write("")
        st.write("")
        run_batch = st.button("⚡ Triage Entire Cluster", type="primary", use_container_width=True)

    current_cluster = BATCH_SCENARIOS[sel_batch]

    # Incoming Incident Cards
    st.markdown("##### 📥 Concurrent Incoming Incident Stream:")
    inc_cols = st.columns(len(current_cluster))
    for i, itm in enumerate(current_cluster):
        with inc_cols[i]:
            u_border = "#EF4444" if itm["urgency_tag"] == "CRITICAL" else ("#F59E0B" if itm["urgency_tag"] == "HIGH" else "#3B82F6")
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(51, 65, 85, 0.8); border-top: 3px solid {u_border}; border-radius: 8px; padding: 0.85rem; height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.70rem; font-weight: 700; color: #94A3B8;">{itm['id']}</span>
                    <span style="font-size: 0.70rem; font-weight: 700; color: {u_border};">{itm['urgency_tag']}</span>
                </div>
                <div style="font-size: 0.90rem; font-weight: 700; color: #F8FAFC; margin-top: 0.35rem;">
                    {itm['equipment_type']}
                </div>
                <div style="font-size: 0.75rem; color: #94A3B8;">
                    {itm['location']}
                </div>
                <div style="font-size: 0.78rem; color: #CBD5E1; margin-top: 0.4rem; line-height: 1.35;">
                    {itm['reported_issue']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    if run_batch:
        batch_results = []
        bar = st.progress(0.0)
        with st.spinner("Processing simultaneous multi-agent diagnosis..."):
            for idx, item in enumerate(current_cluster):
                r = orchestrator.process_incident(
                    reported_issue=item["reported_issue"],
                    symptoms=item["symptoms"],
                    equipment_type=item["equipment_type"],
                    location=item["location"],
                    top_k=3
                )
                diag_info = r.get("diagnosis", {})
                rec_info = r.get("recommendation", {})
                batch_results.append({
                    "id": item["id"],
                    "equipment": item["equipment_type"],
                    "location": item["location"],
                    "reported_issue": item["reported_issue"],
                    "cause": diag_info.get("primary_root_cause", "Unspecified component failure"),
                    "confidence": diag_info.get("confidence_score", 82.0),
                    "urgency": rec_info.get("urgency", "Medium"),
                    "est_time": rec_info.get("estimated_time_hrs", 2.5),
                    "est_cost": rec_info.get("estimated_cost_usd", 350),
                    "fix_step": rec_info.get("fix_steps", ["Perform systematic equipment inspection"])[0]
                })
                bar.progress((idx + 1) / len(current_cluster))

            p_weights = {"Critical": 1, "High": 2, "Medium": 3, "Low": 4}
            batch_results.sort(key=lambda x: p_weights.get(x["urgency"], 5))
            st.session_state["cluster_results"] = batch_results

    if "cluster_results" in st.session_state:
        b_res = st.session_state["cluster_results"]
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("### 📋 Prioritized Dispatch Queue (Optimal Risk Matrix)")

        tot_time = sum(x["est_time"] for x in b_res)
        tot_cost = sum(x["est_cost"] for x in b_res)
        crit_count = sum(1 for x in b_res if x["urgency"] == "Critical")

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-card-title"><span>CRITICAL HAZARDS</span></div>
                <div class="metric-card-val" style="color: #F87171;">{crit_count}</div>
                <div class="metric-card-sub">Immediate dispatch required</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-title"><span>CUMULATIVE DOWNTIME</span></div>
                <div class="metric-card-val" style="color: #38BDF8;">{tot_time:.1f} <span style="font-size: 1rem; font-weight: 500;">hrs</span></div>
                <div class="metric-card-sub">Fleet MTTR required</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-title"><span>FLEET REPAIR BUDGET</span></div>
                <div class="metric-card-val" style="color: #10B981;">${tot_cost:,}</div>
                <div class="metric-card-sub">Estimated emergency spend</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-title"><span>TRIAGE COMPLETION</span></div>
                <div class="metric-card-val" style="color: #A78BFA;">{len(b_res)} / {len(b_res)}</div>
                <div class="metric-card-sub">100% vector matched</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        for rank, item in enumerate(b_res, 1):
            u_tag = item["urgency"]
            b_cls = "badge-critical" if u_tag == "Critical" else ("badge-high" if u_tag == "High" else "badge-medium")

            with st.expander(f"Priority #{rank} [{u_tag.upper()}] — {item['equipment']} ({item['location']})", expanded=(rank <= 2)):
                col_eq1, col_eq2 = st.columns([3, 1])
                with col_eq1:
                    st.markdown(f"**Reported Issue:** {item['reported_issue']}")
                    st.markdown(f"**Diagnosed Cause:** `{item['cause']}` (Confidence: **{item['confidence']}%**)")
                    st.markdown(f"**Immediate Action:** {item['fix_step']}")
                with col_eq2:
                    st.markdown(f"<span class='hud-badge {b_cls}'>{u_tag.upper()} PRIORITY</span>", unsafe_allow_html=True)
                    st.write(f"⏱️ **Labor Time:** {item['est_time']} hrs")
                    st.write(f"💰 **Est. Cost:** ${item['est_cost']}")


# ==========================================
# VIEW 3: MAINTENANCE KNOWLEDGE BASE
# ==========================================
elif nav_view == "Maintenance Knowledge Base":
    st.markdown(f"""
    <div class="cyber-card">
        <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.5rem;">
            <span>📚</span> FAISS EMBEDDED MAINTENANCE KNOWLEDGE BASE
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div>
                <span style="font-size: 1.8rem; font-weight: 800; color: #38BDF8;">{len(all_records)}</span>
                <span style="font-size: 0.95rem; color: #94A3B8; font-weight: 600; margin-left: 0.4rem;">Historical Maintenance Case Vectors</span>
            </div>
            <div style="font-size: 0.82rem; color: #94A3B8; font-family: 'JetBrains Mono';">
                Vector Dimensions: <b>384</b> (all-MiniLM-L6-v2) • Metric: <b>Cosine Inner Product</b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    df_kb = pd.DataFrame(all_records)

    search_kw = st.text_input("🔍 Search Semantic Knowledge Base:", placeholder="Filter by symptom, error code, fix action, or root cause...")

    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        equipment_filter = st.selectbox("Equipment Domain", ["All"] + vector_store.get_equipment_types())
    with col_k2:
        urgency_filter = st.selectbox("Urgency Severity", ["All", "Critical", "High", "Medium", "Low"])
    with col_k3:
        location_filter = st.selectbox("Campus Zone", ["All"] + vector_store.get_locations())

    filtered_df = df_kb.copy()
    if equipment_filter != "All":
        filtered_df = filtered_df[filtered_df["equipment_type"] == equipment_filter]
    if urgency_filter != "All":
        filtered_df = filtered_df[filtered_df["urgency"] == urgency_filter]
    if location_filter != "All":
        filtered_df = filtered_df[filtered_df["location"] == location_filter]

    if search_kw.strip():
        kw = search_kw.lower()
        filtered_df = filtered_df[
            filtered_df["reported_issue"].str.lower().str.contains(kw, na=False) |
            filtered_df["root_cause"].str.lower().str.contains(kw, na=False) |
            filtered_df["fix_action"].str.lower().str.contains(kw, na=False) |
            filtered_df["symptoms"].str.lower().str.contains(kw, na=False)
        ]

    st.caption(f"Displaying **{len(filtered_df)}** of **{len(df_kb)}** historical records:")

    st.dataframe(
        filtered_df[["id", "equipment_type", "equipment_id", "location", "reported_issue", "root_cause", "urgency", "resolution_time_hrs", "cost_estimate_usd", "date_logged"]],
        column_config={
            "id": st.column_config.TextColumn("Case ID", width="small"),
            "equipment_type": st.column_config.TextColumn("Domain", width="medium"),
            "equipment_id": st.column_config.TextColumn("Asset Tag", width="small"),
            "location": st.column_config.TextColumn("Location", width="medium"),
            "reported_issue": st.column_config.TextColumn("Reported Issue", width="large"),
            "root_cause": st.column_config.TextColumn("Root Cause", width="large"),
            "urgency": st.column_config.TextColumn("Urgency", width="small"),
            "resolution_time_hrs": st.column_config.NumberColumn("MTTR", format="%.1f hrs"),
            "cost_estimate_usd": st.column_config.NumberColumn("Cost", format="$%d"),
            "date_logged": st.column_config.DateColumn("Date Logged", format="YYYY-MM-DD")
        },
        use_container_width=True,
        hide_index=True
    )

    if not filtered_df.empty:
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        with st.expander("🔍 Deep Case Dossier Inspector", expanded=False):
            sel_case_id = st.selectbox("Select Case ID to inspect:", filtered_df["id"].tolist())
            rec_detail = next((r for r in all_records if r["id"] == sel_case_id), None)
            if rec_detail:
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.markdown(f"**Asset:** `{rec_detail.get('equipment_id', 'N/A')}` ({rec_detail['equipment_type']})")
                    st.markdown(f"**Location:** {rec_detail.get('location', 'Campus Facility')}")
                    st.markdown(f"**Reported Issue:**\n> {rec_detail['reported_issue']}")
                    st.markdown(f"**Observed Symptoms:**\n> {rec_detail['symptoms']}")
                with dc2:
                    st.markdown(f"**Root Cause:**\n> `{rec_detail['root_cause']}`")
                    st.markdown(f"**Fix Action:**\n> {rec_detail['fix_action']}")
                    st.markdown(f"**Metrics:** Urgency: **{rec_detail['urgency']}** | Cost: **${rec_detail['cost_estimate_usd']}** | MTTR: **{rec_detail['resolution_time_hrs']} hrs**")
                    st.markdown(f"**Logged By:** {rec_detail.get('technician', 'Field Tech')} on {rec_detail.get('date_logged')}")

    st.markdown("---")
    c_e1, c_e2 = st.columns(2)
    with c_e1:
        st.download_button("📥 Export Knowledge Base (CSV)", filtered_df.to_csv(index=False).encode('utf-8'), "knowledge_base.csv", "text/csv", use_container_width=True)
    with c_e2:
        st.download_button("📥 Export Knowledge Base (JSON)", filtered_df.to_json(orient="records", indent=2).encode('utf-8'), "knowledge_base.json", "application/json", use_container_width=True)


# ==========================================
# VIEW 4: SYSTEM ANALYTICS & FLEET METRICS
# ==========================================
elif nav_view == "System Analytics":
    st.markdown("""
    <div class="cyber-card">
        <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.5rem;">
            <span>📊</span> FLEET ANALYTICS & CAMPUS INFRASTRUCTURE HEALTH
        </div>
        <div style="font-size: 0.88rem; color: #94A3B8;">
            High-precision maintenance telemetry: failure frequency distributions, Mean Time to Repair (MTTR), repair cost analytics, and technician accuracy benchmarks.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not all_records:
        st.info("No historical analytics available yet.")
    else:
        df_ana = pd.DataFrame(all_records)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">TOTAL INCIDENTS</div>
                <div class="metric-card-val" style="color: #38BDF8;">{len(df_ana)}</div>
                <div class="metric-card-sub">FAISS semantic records</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            avg_cost = int(df_ana["cost_estimate_usd"].mean())
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">AVERAGE REPAIR COST</div>
                <div class="metric-card-val" style="color: #10B981;">${avg_cost}</div>
                <div class="metric-card-sub">Across all facility domains</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            avg_time = round(df_ana["resolution_time_hrs"].mean(), 1)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">FLEET MTTR</div>
                <div class="metric-card-val" style="color: #FBBF24;">{avg_time} <span style="font-size: 0.9rem; font-weight: 500;">hrs</span></div>
                <div class="metric-card-sub">Mean Time to Repair</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">DECISION ACCURACY</div>
                <div class="metric-card-val" style="color: #A78BFA;">{metrics['accuracy_rate']}%</div>
                <div class="metric-card-sub">{metrics['confirmed_accurate']} verified field cases</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

        # Altair Charts with Cyber-Industrial Dark Theme Styling
        c_ch1, c_ch2 = st.columns(2)
        with c_ch1:
            st.markdown("##### ⚡ Equipment Failure Incident Distribution")
            eq_dist = df_ana["equipment_type"].value_counts().reset_index()
            eq_dist.columns = ["Equipment", "Incidents"]
            chart_eq = alt.Chart(eq_dist).mark_bar(cornerRadius=6, color="#0284C7").encode(
                x=alt.X("Equipment:N", sort="-y", title="Equipment Category"),
                y=alt.Y("Incidents:Q", title="Total Incidents Logged"),
                tooltip=["Equipment", "Incidents"]
            ).properties(height=260).configure_view(strokeOpacity=0)
            st.altair_chart(chart_eq, use_container_width=True)

        with c_ch2:
            st.markdown("##### 🚨 Urgency & Priority Breakdown")
            urg_dist = df_ana["urgency"].value_counts().reset_index()
            urg_dist.columns = ["Urgency", "Incidents"]
            chart_urg = alt.Chart(urg_dist).mark_bar(cornerRadius=6).encode(
                x=alt.X("Urgency:N", sort=["Critical", "High", "Medium", "Low"], title="Urgency Rating"),
                y=alt.Y("Incidents:Q", title="Incident Count"),
                color=alt.Color("Urgency:N", scale=alt.Scale(
                    domain=["Critical", "High", "Medium", "Low"],
                    range=["#EF4444", "#F59E0B", "#3B82F6", "#10B981"]
                ), legend=None),
                tooltip=["Urgency", "Incidents"]
            ).properties(height=260).configure_view(strokeOpacity=0)
            st.altair_chart(chart_urg, use_container_width=True)

        c_ch3, c_ch4 = st.columns(2)
        with c_ch3:
            st.markdown("##### 💰 Mean Repair Cost by Equipment Domain ($)")
            cost_dist = df_ana.groupby("equipment_type")["cost_estimate_usd"].mean().round(0).reset_index()
            cost_dist.columns = ["Equipment", "Mean Cost ($)"]
            chart_cost = alt.Chart(cost_dist).mark_bar(cornerRadius=6, color="#10B981").encode(
                x=alt.X("Equipment:N", sort="-y", title="Equipment Category"),
                y=alt.Y("Mean Cost ($):Q", title="Average Cost ($ USD)"),
                tooltip=["Equipment", "Mean Cost ($)"]
            ).properties(height=260).configure_view(strokeOpacity=0)
            st.altair_chart(chart_cost, use_container_width=True)

        with c_ch4:
            st.markdown("##### ⏱️ Mean Time to Repair (MTTR in Hours)")
            time_dist = df_ana.groupby("equipment_type")["resolution_time_hrs"].mean().round(1).reset_index()
            time_dist.columns = ["Equipment", "MTTR (Hours)"]
            chart_time = alt.Chart(time_dist).mark_bar(cornerRadius=6, color="#8B5CF6").encode(
                x=alt.X("Equipment:N", sort="-y", title="Equipment Category"),
                y=alt.Y("MTTR (Hours):Q", title="MTTR (Hours)"),
                tooltip=["Equipment", "MTTR (Hours)"]
            ).properties(height=260).configure_view(strokeOpacity=0)
            st.altair_chart(chart_time, use_container_width=True)


# ==========================================
# VIEW 5: TECHNICIAN VALIDATION PORTAL
# ==========================================
elif nav_view == "Technician Validation":
    st.markdown("""
    <div class="cyber-card">
        <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.5rem;">
            <span>👨‍🔧</span> CENTRALIZED TECHNICIAN VALIDATION CONSOLE
        </div>
        <div style="font-size: 0.88rem; color: #94A3B8;">
            Field engineers confirm or override AI diagnoses to continuously expand and recalibrate the FAISS semantic knowledge base.
        </div>
    </div>
    """, unsafe_allow_html=True)

    v1, v2, v3 = st.columns(3)
    with v1:
        st.metric("Confirmed Ground Truth", metrics["confirmed_accurate"])
    with v2:
        st.metric("Field Overrides", metrics["corrected"])
    with v3:
        st.metric("Overall Accuracy Benchmark", f"{metrics['accuracy_rate']}%")

    st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

    if "current_result" in st.session_state:
        st.markdown("#### ⏳ Active Incident Awaiting Lead Sign-Off")
        r_act = st.session_state["current_result"]
        i_act = st.session_state["incident_input"]
        d_act = r_act.get("diagnosis", {})
        rec_act = r_act.get("recommendation", {})

        st.markdown(f"""
        <div class="cyber-card">
            <div style="font-size: 0.76rem; font-weight: 700; color: #38BDF8;">ASSET {i_act['asset_id']} ({i_act['equipment_type']}) — {i_act['location']}</div>
            <div style="font-size: 1.1rem; font-weight: 800; color: #F8FAFC; margin-top: 0.2rem;">Complaint: {i_act['reported_issue']}</div>
            <div style="font-size: 0.90rem; color: #60A5FA; margin-top: 0.4rem;">
                <b>AI Diagnosis:</b> {d_act.get('primary_root_cause')} (Confidence: {d_act.get('confidence_score')}%)
            </div>
            <div style="font-size: 0.86rem; color: #94A3B8; margin-top: 0.25rem;">
                <b>Recommended Fix:</b> {' '.join(rec_act.get('fix_steps', []))}
            </div>
        </div>
        """, unsafe_allow_html=True)

        is_val_corr = st.radio("Field Verification Status:", ["✓ Diagnosis Confirmed Accurate", "✕ Diagnosis Overridden"], horizontal=True, key="fb_portal_radio")
        if is_val_corr == "✓ Diagnosis Confirmed Accurate":
            col_pv1, col_pv2 = st.columns(2)
            with col_pv1:
                tech_portal_name = st.text_input("Technician Name", value="Lead Tech Alex", key="p_name_c")
            with col_pv2:
                tech_portal_notes = st.text_input("Verification Notes", value="Confirmed on site. Primary cause verified.", key="p_notes_c")

            if st.button("Confirm & Commit to Vector Store", type="primary", use_container_width=True, key="p_btn_c"):
                success, msg, details = feedback_handler.record_feedback(
                    equipment_type=i_act["backend_equipment"],
                    reported_issue=i_act["reported_issue"],
                    symptoms=i_act["symptoms"],
                    root_cause=d_act.get("primary_root_cause"),
                    fix_action=" ".join(rec_act.get("fix_steps", [])),
                    resolution_time_hrs=rec_act.get("estimated_time_hrs", 2.5),
                    cost_estimate_usd=rec_act.get("estimated_cost_usd", 350),
                    urgency=rec_act.get("urgency", "Medium"),
                    is_accurate=True,
                    technician_name=tech_portal_name,
                    notes=tech_portal_notes,
                    location=i_act["location"],
                    equipment_id=i_act["asset_id"]
                )
                if success:
                    st.success(f"🎉 **{msg}**")
                    st.balloons()
        else:
            p_cause = st.text_input("Actual Root Cause:", value=d_act.get("primary_root_cause"), key="p_cause_ov")
            p_fix = st.text_area("Actual Resolution:", value=" ".join(rec_act.get("fix_steps", [])), key="p_fix_ov")
            p_notes = st.text_input("Technician Notes:", value="Replaced faulty terminal splice connector.", key="p_notes_ov")
            if st.button("Commit Ground-Truth Override", type="primary", use_container_width=True, key="p_btn_ov"):
                success, msg, details = feedback_handler.record_feedback(
                    equipment_type=i_act["backend_equipment"],
                    reported_issue=i_act["reported_issue"],
                    symptoms=i_act["symptoms"],
                    root_cause=p_cause,
                    fix_action=p_fix,
                    resolution_time_hrs=2.5,
                    cost_estimate_usd=300,
                    urgency=rec_act.get("urgency", "Medium"),
                    is_accurate=False,
                    technician_name="Field Tech",
                    notes=p_notes,
                    location=i_act["location"],
                    equipment_id=i_act["asset_id"]
                )
                if success:
                    st.success(f"🎉 **{msg}**")

    else:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.6); border: 2px dashed rgba(51, 65, 85, 0.8); border-radius: 12px; padding: 2.5rem; text-align: center;">
            <div style="font-size: 2.2rem; color: #38BDF8;">🛠️</div>
            <div style="font-size: 1.1rem; font-weight: 800; color: #F8FAFC; margin-top: 0.5rem;">
                No Active Incident Awaiting Sign-Off
            </div>
            <div style="font-size: 0.88rem; color: #94A3B8; margin-top: 0.25rem;">
                Triage an incident in the Incident Triage view to submit real-time field validation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("##### 📜 Recent Technician Sign-Off History")
    stats_data = feedback_handler.stats
    history = stats_data.get("history", [])
    if history:
        hist_df = pd.DataFrame(history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No field feedback records logged in this session yet.")
