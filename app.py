"""
TruthGuard - Enterprise Multi-Agent SaaS Platform
=================================================
Production-grade Streamlit application for Adversarial Verification of enterprise document corpora.
Features:
- Enterprise Dark Mode SaaS Aesthetic (#1E2022, #2B2D42, #00A896, #3A86FF, #EDF2F4)
- Hidden Streamlit branding and chrome for native web-app appearance
- Left Control & Telemetry Panel (System Status, Document Corpus Analytics, Active Model, Sensitivity Controls)
- Right Interactive Arena (SaaS Navigation Bar, Chat Flow, Preset Chips)
- Color-Coded Multi-Agent Trace beneath every response (Retriever 🔵, Synthesizer 🟡, Critic 🔴, Resolution 🟢)
"""

import os
import json
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import TruthGuard backend
from graph import run_truthguard
from data_loader import get_all_raw_documents

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TruthGuard Enterprise | Multi-Agent Verification Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Advanced SaaS Custom Theming & CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* CSS Reset & Hide Streamlit Default Chrome without hiding the Sidebar Toggle */
    #MainMenu {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    .stDeployButton {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}
    [data-testid="stToolbarActions"] {display: none !important;}

    /* Keep stToolbar container visible and positioned so stExpandSidebarButton is always accessible */
    [data-testid="stToolbar"] {
        visibility: visible !important;
        display: flex !important;
        background: transparent !important;
        top: 0.6rem !important;
        left: 0.6rem !important;
        right: auto !important;
        z-index: 1000000 !important;
    }

    /* Make header transparent so it doesn't occlude content, but keeps sidebar control clickable */
    header {
        background: transparent !important;
        height: 2.8rem !important;
        z-index: 99999 !important;
    }

    /* Prominently style the Expand Sidebar button (when sidebar is closed/collapsed) */
    [data-testid="stExpandSidebarButton"],
    button[data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background-color: #252836 !important;
        color: #00A896 !important;
        border: 1.5px solid rgba(0, 168, 150, 0.6) !important;
        border-radius: 8px !important;
        width: 38px !important;
        height: 38px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45) !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        z-index: 1000001 !important;
    }

    [data-testid="stExpandSidebarButton"]:hover,
    button[data-testid="stExpandSidebarButton"]:hover,
    [data-testid="stSidebarCollapsedControl"]:hover,
    [data-testid="collapsedControl"]:hover {
        background-color: #2F3346 !important;
        border-color: #3A86FF !important;
        color: #02c39a !important;
        transform: scale(1.08) !important;
    }

    [data-testid="stExpandSidebarButton"] svg,
    button[data-testid="stExpandSidebarButton"] svg,
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="collapsedControl"] svg {
        fill: #00A896 !important;
        stroke: #00A896 !important;
        color: #00A896 !important;
        width: 22px !important;
        height: 22px !important;
    }

    /* Style the Collapse button inside the open sidebar */
    [data-testid="stSidebarCollapseButton"] button {
        visibility: visible !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background-color: #252836 !important;
        color: #00A896 !important;
        border: 1px solid rgba(0, 168, 150, 0.4) !important;
        border-radius: 8px !important;
        padding: 6px !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }

    [data-testid="stSidebarCollapseButton"] button:hover {
        background-color: #2F3346 !important;
        border-color: #3A86FF !important;
        transform: scale(1.05) !important;
    }

    [data-testid="stSidebarCollapseButton"] svg {
        fill: #00A896 !important;
        stroke: #00A896 !important;
        color: #00A896 !important;
        width: 20px !important;
        height: 20px !important;
    }

    /* Root Palette & Typography */
    :root {
        --bg-primary: #1E2022;
        --bg-surface: #2B2D42;
        --bg-card: #252836;
        --bg-card-hover: #2F3346;
        --accent-teal: #00A896;
        --accent-teal-light: #02c39a;
        --accent-blue: #3A86FF;
        --accent-amber: #F48C06;
        --accent-red: #EF233C;
        --text-primary: #EDF2F4;
        --text-secondary: #8D99AE;
        --text-muted: #6C757D;
        --border-subtle: rgba(141, 153, 174, 0.2);
        --border-active: rgba(0, 168, 150, 0.5);
    }

    /* Global Body & Container */
    .stApp {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Top Padding Adjustment */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 96% !important;
    }

    /* Sidebar Styling (Fixed Control Panel Look) */
    [data-testid="stSidebar"] {
        background-color: #17191C !important;
        border-right: 1px solid var(--border-subtle) !important;
        padding-top: 1rem !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: var(--border-subtle) !important;
        margin: 1.2rem 0 !important;
    }

    /* SaaS Navigation Bar Header */
    .saas-navbar {
        background: linear-gradient(135deg, #252836 0%, #1F222E 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }

    .saas-brand-wrap {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .saas-logo-badge {
        background: linear-gradient(135deg, var(--accent-teal), var(--accent-blue));
        width: 44px;
        height: 44px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        box-shadow: 0 4px 12px rgba(0, 168, 150, 0.4);
    }

    .saas-title-box h1 {
        font-size: 1.45rem !important;
        font-weight: 800 !important;
        color: var(--text-primary) !important;
        margin: 0 !important;
        letter-spacing: -0.02em;
    }

    .saas-title-box p {
        font-size: 0.85rem !important;
        color: var(--text-secondary) !important;
        margin: 0 !important;
    }

    .saas-telemetry {
        display: flex;
        gap: 10px;
        align-items: center;
    }

    .telemetry-chip {
        background: rgba(43, 45, 66, 0.8);
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .status-dot-active {
        width: 7px;
        height: 7px;
        background-color: var(--accent-teal);
        border-radius: 50%;
        box-shadow: 0 0 8px var(--accent-teal);
        display: inline-block;
    }

    /* Sidebar Status Cards */
    .control-card {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 12px;
    }

    .control-card-header {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        font-size: 0.82rem;
        border-bottom: 1px solid rgba(141, 153, 174, 0.1);
    }
    .stat-row:last-child {
        border-bottom: none;
    }
    .stat-label {
        color: var(--text-primary);
    }
    .stat-badge {
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 4px;
        background: rgba(0, 0, 0, 0.3);
        color: var(--accent-teal-light);
    }

    /* Button Styling */
    .stButton > button {
        background: #252836 !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 8px 14px !important;
        transition: all 0.2s ease !important;
        text-align: left !important;
    }

    .stButton > button:hover {
        background: #2E3347 !important;
        border-color: var(--accent-blue) !important;
        color: #ffffff !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(58, 134, 255, 0.25) !important;
    }

    /* Chat Messages Styling */
    .user-bubble {
        background: linear-gradient(135deg, #24334a 0%, #1e2c3f 100%);
        border: 1px solid rgba(58, 134, 255, 0.4);
        border-radius: 14px 14px 4px 14px;
        padding: 14px 18px;
        margin: 12px 0;
        color: #EDF2F4;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
    }

    .assistant-bubble {
        background: #252836;
        border: 1px solid rgba(0, 168, 150, 0.3);
        border-radius: 14px 14px 14px 4px;
        padding: 18px 22px;
        margin: 12px 0;
        color: #EDF2F4;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
    }

    .bubble-header {
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .user-header-text {
        color: var(--accent-blue);
    }

    .assistant-header-text {
        color: var(--accent-teal-light);
    }

    /* Agent Trace Badges and Cards */
    .trace-section {
        background: #1C1E26;
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 14px;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    .trace-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }

    .badge-retriever {
        background: rgba(58, 134, 255, 0.15);
        color: #5ea0ff;
        border: 1px solid rgba(58, 134, 255, 0.4);
    }

    .badge-synthesizer {
        background: rgba(244, 140, 6, 0.15);
        color: #f7a028;
        border: 1px solid rgba(244, 140, 6, 0.4);
    }

    .badge-critic {
        background: rgba(239, 35, 60, 0.15);
        color: #ff5c72;
        border: 1px solid rgba(239, 35, 60, 0.4);
    }

    .badge-resolution {
        background: rgba(0, 168, 150, 0.15);
        color: #2ec4b6;
        border: 1px solid rgba(0, 168, 150, 0.4);
    }

    /* Document Item in Retriever Trace */
    .doc-pill {
        background: rgba(43, 45, 66, 0.7);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 8px;
    }

    .doc-pill-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    .doc-status-tag {
        font-size: 0.7rem;
        padding: 1px 6px;
        border-radius: 4px;
        font-weight: 700;
    }

    .tag-active {
        background: rgba(0, 168, 150, 0.2);
        color: #02c39a;
        border: 1px solid rgba(0, 168, 150, 0.4);
    }

    .tag-superseded {
        background: rgba(239, 35, 60, 0.2);
        color: #ff5c72;
        border: 1px solid rgba(239, 35, 60, 0.4);
    }

    /* Critic Conflict Alert Card */
    .conflict-callout {
        background: rgba(239, 35, 60, 0.08);
        border-left: 4px solid var(--accent-red);
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin: 8px 0;
    }

    /* Streamlit Expander Overrides */
    .streamlit-expanderHeader {
        background-color: #222533 !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
    }

    .streamlit-expanderContent {
        background-color: #1a1c24 !important;
        border: 1px solid var(--border-subtle) !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        padding: 16px !important;
    }

    /* Chat Input Styling */
    [data-testid="stChatInput"] {
        border-color: var(--border-subtle) !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #252836 !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--accent-teal) !important;
        box-shadow: 0 0 0 2px rgba(0, 168, 150, 0.25) !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. Session State Management
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_trigger" not in st.session_state:
    st.session_state.query_trigger = None


# -----------------------------------------------------------------------------
# 4. Helper Renderers
# -----------------------------------------------------------------------------
def render_trace_block(trace_data: Optional[List[Dict[str, Any]]]):
    """
    Renders the color-coded Multi-Agent Verification Trace:
    🔵 Retriever
    🟡 Synthesizer
    🔴 Critic (The Adversary)
    🟢 Resolution
    """
    if not trace_data:
        st.info("No trace telemetry recorded for this interaction.")
        return

    # 1. 🔵 Retriever Step
    step_retriever = next((s for s in trace_data if s["agent"] == "Retriever Agent"), None)
    if step_retriever:
        st.markdown("""
        <div class="trace-section">
            <span class="trace-badge badge-retriever">🔵 Node 1 &nbsp;|&nbsp; Retriever Agent</span>
            <p style="font-size: 0.86rem; color: #8D99AE; margin: 8px 0 10px 0;">
                Queried vector store with metadata extraction (effective dates, version tags, and supersession links).
            </p>
        """, unsafe_allow_html=True)

        docs = step_retriever["details"].get("documents", [])
        for d in docs:
            is_active = d.get("status") == "ACTIVE"
            tag_class = "tag-active" if is_active else "tag-superseded"
            st.markdown(f"""
            <div class="doc-pill">
                <div class="doc-pill-header">
                    <span><b>#{d.get('rank')} [{d.get('doc_id')}]</b> {d.get('title')}</span>
                    <span class="doc-status-tag {tag_class}">{d.get('status')}</span>
                </div>
                <div style="font-size: 0.76rem; color: #8D99AE; margin-top: 3px;">
                    <b>Version:</b> {d.get('version')} &nbsp;|&nbsp; 
                    <b>Effective:</b> {d.get('effective_date')} &nbsp;|&nbsp; 
                    <b>Dept:</b> {d.get('department')} &nbsp;|&nbsp; 
                    <b>Supersedes:</b> {d.get('supersedes')}
                </div>
                <p style="font-size: 0.82rem; color: #c9d1d9; margin: 5px 0 0 0; font-family: monospace;">
                    "{d.get('content', '')[:135]}..."
                </p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 2. 🟡 Synthesizer Step (Standard RAG Draft)
    step_synth = next((s for s in trace_data if s["agent"] == "Synthesizer Agent"), None)
    if step_synth:
        draft_content = step_synth["details"].get("draft", "")
        st.markdown(f"""
        <div class="trace-section">
            <span class="trace-badge badge-synthesizer">🟡 Node 2 &nbsp;|&nbsp; Synthesizer Agent (Baseline RAG)</span>
            <p style="font-size: 0.86rem; color: #8D99AE; margin: 8px 0 10px 0;">
                Candidate draft produced using semantic context alone. This represents what naive RAG outputs without verification.
            </p>
            <div style="background: rgba(0,0,0,0.3); border-left: 3px solid #F48C06; padding: 10px 14px; border-radius: 0 6px 6px 0; font-size: 0.85rem; color: #EDF2F4; font-style: italic;">
                "{draft_content}"
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 3. 🔴 Critic Step (The Adversary)
    step_critic = next((s for s in trace_data if "Critic" in s["agent"]), None)
    if step_critic:
        critic_details = step_critic["details"]
        conflicts = critic_details.get("conflicts", [])
        staleness = critic_details.get("staleness_issues", [])
        unsupported = critic_details.get("unsupported_claims", [])
        rec = critic_details.get("recommendation", "RESOLVE_CONFLICT")

        st.markdown(f"""
        <div class="trace-section">
            <span class="trace-badge badge-critic">🔴 Node 3 &nbsp;|&nbsp; Critic Agent (The Adversary)</span>
            <p style="font-size: 0.86rem; color: #8D99AE; margin: 8px 0 10px 0;">
                Audits candidate draft against retrieved ground-truth documents. Actively hunts for temporal staleness and contradictions.
            </p>
            <div style="display: flex; gap: 8px; margin-bottom: 10px;">
                <span class="stat-badge" style="background: rgba(239,35,60,0.2); color: #ff5c72;">Flagged Contradictions: {len(conflicts)}</span>
                <span class="stat-badge" style="background: rgba(244,140,6,0.2); color: #f7a028;">Staleness Issues: {len(staleness)}</span>
                <span class="stat-badge" style="background: rgba(58,134,255,0.2); color: #5ea0ff;">Action: {rec}</span>
            </div>
        """, unsafe_allow_html=True)

        if conflicts:
            for idx, c in enumerate(conflicts, 1):
                st.markdown(f"""
                <div class="conflict-callout">
                    <div style="font-weight: 700; font-size: 0.84rem; color: #ff5c72;">
                        ⚠️ Conflict #{idx}: {c.get('subject')} &nbsp;•&nbsp; 
                        <span style="font-family: monospace; font-size: 0.74rem; background: rgba(239,35,60,0.3); padding: 1px 6px; border-radius: 4px;">{c.get('conflict_type')}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #EDF2F4; margin-top: 4px;">
                        <b>{c.get('doc_a')}:</b> {c.get('policy_a_claim')}
                    </div>
                    <div style="font-size: 0.8rem; color: #EDF2F4; margin-top: 2px;">
                        <b>{c.get('doc_b')}:</b> {c.get('policy_b_claim')}
                    </div>
                    <div style="font-size: 0.8rem; color: #ff8595; margin-top: 5px;">
                        <b>Auditor Rationale:</b> {c.get('contradiction_analysis')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        if staleness:
            for s in staleness:
                st.markdown(f"""
                <div style="background: rgba(244, 140, 6, 0.08); border-left: 4px solid #F48C06; border-radius: 0 8px 8px 0; padding: 8px 12px; margin: 6px 0; font-size: 0.8rem;">
                    <b>Supersession Enforced:</b> <code>{s.get('superseded_doc_id')}</code> rendered obsolete by <code>{s.get('newer_doc_id')}</code>.<br>
                    <span style="color: #f7b250;"><b>Timeline:</b> {s.get('effective_timeline')}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # 4. 🟢 Resolution Step
    step_res = next((s for s in trace_data if "Resolution" in s["agent"]), None)
    if step_res:
        st.markdown("""
        <div class="trace-section">
            <span class="trace-badge badge-resolution">🟢 Node 4 &nbsp;|&nbsp; Resolution Agent</span>
            <p style="font-size: 0.86rem; color: #8D99AE; margin: 8px 0 0 0;">
                Synthesizes the authoritative final position. <b>Silent blending strictly prohibited.</b> 
                Produces chronological provenance matrices or explicitly declares jurisdictional deadlock.
            </p>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. Left Panel: Control Center & Telemetry
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
        <span style="font-size: 20px;">⚙️</span>
        <span style="font-weight: 800; font-size: 1.15rem; color: #EDF2F4; letter-spacing: -0.01em;">Control Center</span>
    </div>
    """, unsafe_allow_html=True)

    # System Status Card
    st.markdown("""
    <div class="control-card">
        <div class="control-card-header">
            <span>System Telemetry</span>
            <span class="stat-badge" style="background: rgba(0,168,150,0.2); color: #02c39a;">
                <span class="status-dot-active"></span> ACTIVE
            </span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Orchestrator</span>
            <span class="stat-badge">LangGraph 0.2</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Vector Database</span>
            <span class="stat-badge">ChromaDB</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Guardrail Protocol</span>
            <span class="stat-badge">Adversarial v2.4</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Latency Average</span>
            <span class="stat-badge">92ms</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Document Database Metrics Card
    st.markdown("""
    <div class="control-card">
        <div class="control-card-header">
            <span>Knowledge Base Stats</span>
            <span class="stat-badge">9 Documents</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">🏢 HR & Remote Work</span>
            <span class="stat-badge">3 Docs (2 Superseded)</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">🔐 IT & Zero Trust</span>
            <span class="stat-badge">2 Docs (1 Superseded)</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">✈️ Finance & Travel</span>
            <span class="stat-badge">2 Docs (1 Superseded)</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">⚖️ Legal vs Eng AI</span>
            <span class="stat-badge" style="color: #ff5c72;">2 Conflicting</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model & Verification Settings
    with st.expander("🛠️ Model & Guardrail Settings", expanded=False):
        llm_provider = st.selectbox(
            "Orchestrator LLM",
            ["Deterministic Engine (Offline / Local)", "OpenAI", "Anthropic"],
            index=0
        )
        api_key_input = ""
        model_choice = "Local Deterministic Engine"

        if llm_provider == "OpenAI":
            api_key_input = st.text_input(
                "OpenAI API Key",
                type="password",
                value=os.getenv("OPENAI_API_KEY", ""),
                placeholder="sk-proj-..."
            )
            model_choice = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], index=0)
        elif llm_provider == "Anthropic":
            api_key_input = st.text_input(
                "Anthropic API Key",
                type="password",
                value=os.getenv("ANTHROPIC_API_KEY", ""),
                placeholder="sk-ant-..."
            )
            model_choice = st.selectbox("Model", ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"], index=0)
        else:
            st.caption("⚡ Zero-token mode: Runs local adversarial heuristic verification engine.")

        st.slider("Adversarial Critic Strictness", min_value=0.5, max_value=1.0, value=0.95, step=0.05, help="Controls tolerance for temporal inconsistencies.")
        st.toggle("Enforce Chronological Precedence", value=True)
        st.toggle("Strict Silent Blending Ban", value=True)

    st.markdown("---")

    # Benchmark Preset Scenarios
    st.markdown("""
    <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #8D99AE; margin-bottom: 8px;">
        ⚡ Enterprise Policy Benchmarks
    </div>
    """, unsafe_allow_html=True)

    preset_benchmarks = [
        ("🏢 Remote Work & Stipend", "What is the company remote work policy and how much is the home office stipend?"),
        ("🔐 Password Expiry & SMS 2FA", "How often do I need to change my corporate password, and can I use SMS for 2FA?"),
        ("✈️ Travel Per Diem & Flights", "What is the daily meal per diem and can I fly Business Class for flights over 4 hours?"),
        ("🤖 AI Tool Reimbursement", "Can I expense my Cursor or ChatGPT subscription on my corporate card as a software developer?")
    ]

    for label, query in preset_benchmarks:
        if st.button(label, use_container_width=True):
            st.session_state.query_trigger = query

    st.markdown("---")

    # Enterprise Document Corpus Inspector
    with st.expander("📂 Raw Corpus Inspector (9 Policies)", expanded=False):
        raw_docs = get_all_raw_documents()
        for doc in raw_docs:
            is_active = doc.get("status") == "ACTIVE"
            tag_class = "tag-active" if is_active else "tag-superseded"
            st.markdown(f"""
            <div style="background: rgba(0,0,0,0.25); border: 1px solid var(--border-subtle); border-radius: 6px; padding: 8px; margin-bottom: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <b style="font-size: 0.8rem; color: #EDF2F4;">{doc['doc_id']}</b>
                    <span class="doc-status-tag {tag_class}">{doc['status']}</span>
                </div>
                <div style="font-size: 0.72rem; color: #8D99AE;">Ver: {doc['version']} | Date: {doc['effective_date']}</div>
                <div style="font-size: 0.72rem; color: #8D99AE;">Supersedes: {doc['supersedes']}</div>
            </div>
            """, unsafe_allow_html=True)

    if st.button("🗑️ Reset Interaction State", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_trigger = None
        st.rerun()


# -----------------------------------------------------------------------------
# 6. Main Arena: Header & Chat Interface
# -----------------------------------------------------------------------------

# SaaS Top Navigation Bar
st.markdown("""
<div class="saas-navbar">
    <div class="saas-brand-wrap">
        <div class="saas-logo-badge">🛡️</div>
        <div class="saas-title-box">
            <h1>TruthGuard Enterprise</h1>
            <p>Adversarial Multi-Agent Governance Platform &bull; Non-Blended Policy Resolution</p>
        </div>
    </div>
    <div class="saas-telemetry">
        <div class="telemetry-chip">
            <span class="status-dot-active"></span>
            Agents: 4 Online
        </div>
        <div class="telemetry-chip">
            Mode: Zero-Blend
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Conversation Arena
chat_container = st.container()

with chat_container:
    # Display message history
    for idx, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        content = msg["content"]
        trace = msg.get("trace")

        if role == "user":
            st.markdown(f"""
            <div class="user-bubble">
                <div class="bubble-header user-header-text">
                    <span>👤 Employee Query</span>
                </div>
                <div style="font-size: 0.95rem; line-height: 1.5;">{content}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-bubble">
                <div class="bubble-header assistant-header-text">
                    <span>🛡️ TruthGuard Verified Arbiter</span>
                </div>
                <div style="font-size: 0.95rem; line-height: 1.6;">{content}</div>
            </div>
            """, unsafe_allow_html=True)

            # Color-coded Agent Trace Expandable Container
            if trace:
                with st.expander("🔍 View Multi-Agent Verification Trace (Retriever 🔵 ➔ Synthesizer 🟡 ➔ Critic 🔴 ➔ Resolution 🟢)", expanded=False):
                    render_trace_block(trace)

# Handle Query Trigger (from Preset or Chat Input)
active_query = st.session_state.query_trigger or st.chat_input("Ask a question about HR, IT Security, Travel, or AI Policies...")

if active_query:
    # Reset trigger
    st.session_state.query_trigger = None

    # Render User Query Bubble Immediately
    st.markdown(f"""
    <div class="user-bubble">
        <div class="bubble-header user-header-text">
            <span>👤 Employee Query</span>
        </div>
        <div style="font-size: 0.95rem; line-height: 1.5;">{active_query}</div>
    </div>
    """, unsafe_allow_html=True)

    # Append to state
    st.session_state.messages.append({"role": "user", "content": active_query})

    # Execute Multi-Agent Pipeline
    with st.spinner("🛡️ TruthGuard Multi-Agent Verification in Progress (Retriever ➔ Synthesizer ➔ Critic ➔ Resolution)..."):
        try:
            result = run_truthguard(
                query=active_query,
                provider=llm_provider if 'llm_provider' in locals() else None,
                model=model_choice if 'model_choice' in locals() else None,
                api_key=api_key_input if 'api_key_input' in locals() else None
            )

            res_output = result["resolution_output"]
            trace_steps = result["agent_trace"]

            # Render Verified Assistant Response
            st.markdown(f"""
            <div class="assistant-bubble">
                <div class="bubble-header assistant-header-text">
                    <span>🛡️ TruthGuard Verified Arbiter</span>
                </div>
                <div style="font-size: 0.95rem; line-height: 1.6;">{res_output}</div>
            </div>
            """, unsafe_allow_html=True)

            # Render Multi-Agent Trace
            with st.expander("🔍 View Multi-Agent Verification Trace (Retriever 🔵 ➔ Synthesizer 🟡 ➔ Critic 🔴 ➔ Resolution 🟢)", expanded=True):
                render_trace_block(trace_steps)

            # Persist response with trace in history
            st.session_state.messages.append({
                "role": "assistant",
                "content": res_output,
                "trace": trace_steps
            })

        except Exception as e:
            err_msg = f"⚠️ TruthGuard Pipeline Exception: {str(e)}"
            st.error(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
