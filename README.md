# Campus/Facility Infrastructure Decision-Support Agent 🏢⚡
> **Track B - B3: Multi-Agent Orchestration & Decision Support**

An AI-powered decision support system that helps facility and campus operations teams diagnose equipment failures, retrieve relevant historical maintenance records, formulate actionable repair steps with cost/time/urgency estimates, explain the reasoning chain in plain language, and continuously learn through a live technician feedback loop.

---

## 🌟 Key Features

1. **Searchable Historical Knowledge Base**:
   - 265+ realistic maintenance records covering 5 critical facility systems: HVAC & Chillers, Backup Diesel Generators, Elevators & Escalators, Water Supply & Booster Pumps, and Electrical Switchgears & Transformers.
   - Rich metadata: symptoms, root causes, repair procedures, duration (hrs), cost ($), urgency, and technician notes.

2. **FAISS Semantic Similarity Search**:
   - Vector search powered by `sentence-transformers` (`all-MiniLM-L6-v2`) and `faiss-cpu`.
   - Real-time percentage matching, keyword filtering, and equipment scoping.

3. **LangGraph Multi-Agent Reasoning Pipeline**:
   - **Retrieval Agent**: Formulates queries and pulls top-k historical cases.
   - **Diagnosis Agent**: Cross-references symptoms against historical failure modes and synthesizes the root cause with confidence scoring.
   - **Recommendation Agent**: Formulates step-by-step repair actions, safety precautions (LOTO, PPE), estimated repair time, cost, and urgency level.
   - **Plain-Language Reasoning Chain**: Explains the exact reasoning steps leading to the diagnosis.

4. **Technician Feedback Loop (Compulsory Add-on - Key Feature 7)**:
   - "Confirm Diagnosis (Accurate)" and "Adjust & Confirm" buttons on every incident.
   - Confirmed cases are immediately appended to `data/maintenance_records.json` and dynamically re-indexed in the active FAISS vector store in real-time without restarting the app.
   - Live telemetry tracks human-in-the-loop accuracy and demonstrates real-time learning.

5. **Multi-Complaint Batch Triage**:
   - Handles multi-incident emergencies (e.g. storm power surges, campus-wide blackouts).
   - Automatically prioritizes tickets into a dispatch queue (Critical -> High -> Medium -> Low), calculating cumulative technician hours and budget requirements.

6. **Interactive Dashboard**:
   - Clean, modern Streamlit UI with 4 dedicated views:
     - 🚨 *New Incident Triage* (with 1-click demo scenario buttons)
     - ⚡ *Multi-Complaint Batch Triage*
     - 📚 *Knowledge Base Explorer* (with filtering and CSV export)
     - 📊 *System Analytics & Accuracy*

---

## 🏗️ Architecture

```mermaid
graph TD
    A[New Incident Complaint / Batch Complaints] --> B[Incident Ingestion & Preprocessing]
    B --> C[Retrieval Agent (FAISS + SentenceTransformers)]
    C -->|Top-K Historical Cases + Similarity Scores| D[Diagnosis Agent (LangGraph)]
    D -->|Root Cause Hypotheses + Reasoning Chain| E[Recommendation & Urgency Agent (LangGraph)]
    E -->|Repair Steps, Cost, Time & Urgency| F[Synthesizer / Decision Report]
    F --> G[Interactive Streamlit Dashboard]
    G --> H[Technician Feedback Loop (Feature 7)]
    H -->|Confirmed Resolution| C
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Installed packages: `streamlit`, `langgraph`, `sentence-transformers`, `faiss-cpu`, `pandas`, `numpy`, `pytest`

### 2. Generate / Inspect Synthetic Data
```bash
python data/generate_synthetic_data.py
```

### 3. Run Automated Tests
```bash
python -m pytest tests/test_system.py -v
```

### 4. Launch the Web Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Testing the Technician Feedback Loop (Live Demo)
1. Navigate to **🚨 New Incident Triage**.
2. Select **"Elevator: Passenger Entrapment Between Floors"** from the Quick Demo Scenarios dropdown.
3. Click **"Run Multi-Agent Diagnosis"**.
4. Observe the diagnosis, retrieved similar past cases, and reasoning chain.
5. Click **"✅ Confirm Diagnosis (Accurate) & Append to Knowledge Base"**.
6. Notice the balloons and real-time confirmation message: the case is saved to disk and indexed in FAISS immediately.
7. Subsequent searches for similar elevator entrapment issues will now retrieve this newly verified record with higher similarity confidence!
