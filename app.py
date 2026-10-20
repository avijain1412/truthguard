"""
TruthGuard - Streamlit Interactive Frontend
===========================================
Interactive Demo & Evaluation Console for the TruthGuard Adversarial Verification Multi-Agent System.
Features:
- Chat interface with real-time adversarial verification
- Comprehensive 'Agent Trace' viewer detailing Retriever, Synthesizer, Critic, and Resolution steps
- Preset scenarios showcasing temporal supersessions and departmental deadlocks
- Corpus explorer inspecting raw enterprise documents and version metadata
"""

import os
import json
import streamlit as st
from datetime import datetime

# Import TruthGuard backend
from graph import run_truthguard
from data_loader import get_all_raw_documents

# Streamlit Page Config
st.set_page_config(
    page_title="TruthGuard | Adversarial Verification Multi-Agent System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern enterprise look and feel
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0e1117 0%, #161b22 90%);
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Card */
    .tg-header {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.95), rgba(33, 38, 45, 0.85));
        border: 1px solid rgba(56, 139, 253, 0.3);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    
    .tg-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff, #bc8cff, #f778ba);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .tg-subtitle {
        color: #8b949e;
        font-size: 1.05rem;
        line-height: 1.5;
    }

    /* Badges */
    .badge-active {
        background-color: rgba(46, 160, 67, 0.2);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.4);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .badge-superseded {
        background-color: rgba(248, 81, 73, 0.2);
        color: #f85149;
        border: 1px solid rgba(248, 81, 73, 0.4);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .badge-amend {
        background-color: rgba(210, 153, 34, 0.2);
        color: #d29922;
        border: 1px solid rgba(210, 153, 34, 0.4);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Trace Step Containers */
    .trace-card {
        background: rgba(22, 27, 34, 0.8);
        border-left: 4px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-bottom: 14px;
        border-top: 1px solid rgba(48, 54, 61, 0.5);
        border-right: 1px solid rgba(48, 54, 61, 0.5);
        border-bottom: 1px solid rgba(48, 54, 61, 0.5);
    }

    .trace-card-adversary {
        border-left-color: #f85149;
        background: rgba(248, 81, 73, 0.05);
    }

    .trace-card-resolution {
        border-left-color: #bc8cff;
        background: rgba(188, 140, 255, 0.05);
    }

    .trace-card-retriever {
        border-left-color: #58a6ff;
    }

    .trace-card-synthesizer {
        border-left-color: #d29922;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_trace" not in st.session_state:
    st.session_state.latest_trace = None
if "latest_query" not in st.session_state:
    st.session_state.latest_query = ""


# =====================================================================
# Sidebar: Configuration, Presets & Corpus Browser
# =====================================================================
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    llm_provider = st.selectbox(
        "LLM Orchestrator",
        ["OpenAI", "Anthropic", "Deterministic Fallback Engine (Offline)"],
        index=0
    )

    api_key_input = ""
    if llm_provider == "OpenAI":
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            placeholder="sk-proj-...",
            help="Leave blank to use OPENAI_API_KEY from environment or switch to Fallback Engine."
        )
        model_choice = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], index=0)
    elif llm_provider == "Anthropic":
        api_key_input = st.text_input(
            "Anthropic API Key",
            type="password",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            placeholder="sk-ant-...",
            help="Leave blank to use ANTHROPIC_API_KEY from environment."
        )
        model_choice = st.selectbox("Model", ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"], index=0)
    else:
        st.info("💡 Running in 100% local deterministic mode. Ideal for testing and demonstrations without API consumption.")
        model_choice = "Local Deterministic Engine"

    st.markdown("---")
    st.markdown("### 🎯 Benchmark Scenarios")
    st.caption("Click any preset to test conflicting policies:")

    preset_scenarios = [
        ("🏢 Remote Work Policy & Stipend", "What is the company remote work policy and how much is the home office stipend?"),
        ("🔐 Password Expiry & SMS 2FA", "How often do I need to change my corporate password, and can I use SMS for 2FA?"),
        ("✈️ Travel Per Diem & Domestic Flights", "What is the daily meal per diem and can I fly Business Class for flights over 4 hours?"),
        ("🤖 AI Coding Assistant Expenses", "Can I expense my Cursor or ChatGPT subscription on my corporate card as a software developer?")
    ]

    selected_query_preset = None
    for label, query_text in preset_scenarios:
        if st.button(label, use_container_width=True):
            selected_query_preset = query_text

    st.markdown("---")
    with st.expander("📂 Enterprise Corpus Browser (9 Docs)", expanded=False):
        raw_docs = get_all_raw_documents()
        for doc in raw_docs:
            status_badge = (
                f"<span class='badge-active'>{doc['status']}</span>"
                if doc['status'] == "ACTIVE"
                else f"<span class='badge-superseded'>{doc['status']}</span>"
            )
            st.markdown(
                f"**`{doc['doc_id']}`** {status_badge}<br>"
                f"<small><b>{doc['title']}</b> | Version: {doc['version']} | Date: {doc['effective_date']}</small>",
                unsafe_allow_html=True
            )
            st.caption(f"Supersedes: {doc['supersedes']}")
            st.code(doc['content'][:150] + "...", language="text")
            st.markdown("<hr style='margin: 8px 0; border-color: #30363d;'>", unsafe_allow_html=True)

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.latest_trace = None
        st.session_state.latest_query = ""
        st.rerun()


# =====================================================================
# Main Interface: Header & Overview
# =====================================================================
st.markdown("""
<div class="tg-header">
    <div class="tg-title">🛡️ TruthGuard</div>
    <div class="tg-subtitle">
        <b>Adversarial Verification Multi-Agent System</b> for Enterprise Knowledge Bases.<br>
        Traditional RAG fails silently when enterprise corpora contain conflicting policies, obsolete standards, 
        and superseded amendments. TruthGuard uses an adversarial Critic Agent to actively break candidate drafts, 
        uncover contradictions, and enforce non-blended provenance resolution.
    </div>
</div>
""", unsafe_allow_html=True)

# Create layout with two primary columns: Left = Chat & Final Resolution, Right = Live Agent Trace
col_chat, col_trace = st.columns([1.15, 0.85], gap="large")


# =====================================================================
# Left Column: Chat & Verified Output
# =====================================================================
with col_chat:
    st.subheader("💬 Verified Enterprise Assistant")

    # Display past chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Determine input query (from preset button or chat input)
    user_query = selected_query_preset or st.chat_input("Ask a question about HR, IT Security, Travel, or AI Policies...")

    if user_query:
        # Add user query to conversation
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.latest_query = user_query
        with st.chat_message("user"):
            st.markdown(user_query)

        # Run TruthGuard Pipeline
        with st.chat_message("assistant"):
            with st.spinner("🛡️ Orchestrating Multi-Agent Pipeline (Retriever ➔ Synthesizer ➔ Critic ➔ Resolution)..."):
                try:
                    result = run_truthguard(
                        query=user_query,
                        provider=llm_provider,
                        model=model_choice,
                        api_key=api_key_input
                    )
                    final_ans = result["resolution_output"]
                    st.markdown(final_ans)

                    # Save to state
                    st.session_state.messages.append({"role": "assistant", "content": final_ans})
                    st.session_state.latest_trace = result["agent_trace"]

                except Exception as e:
                    error_msg = f"⚠️ Pipeline Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


# =====================================================================
# Right Column: Visual Agent Trace (The "Thoughts" of TruthGuard)
# =====================================================================
with col_trace:
    st.subheader("🧠 Multi-Agent Thought & Trace")
    st.caption("Inspect what each specialized agent observed, drafted, audited, and resolved.")

    trace_data = st.session_state.latest_trace

    if not trace_data:
        st.info("👈 Submit a query or select a preset scenario from the sidebar to visualize the multi-agent reasoning trace in real time.")
    else:
        # Step 1: Retriever Agent
        step_retriever = next((s for s in trace_data if s["agent"] == "Retriever Agent"), None)
        if step_retriever:
            with st.expander(f"{step_retriever['icon']} Step 1: {step_retriever['agent']}", expanded=True):
                st.markdown(f"**Status:** `{step_retriever['status']}` | <small>{step_retriever['timestamp']}</small>", unsafe_allow_html=True)
                st.markdown(f"*{step_retriever['summary']}*")
                
                docs = step_retriever["details"].get("documents", [])
                st.markdown(f"**Retrieved Passages ({len(docs)}):**")
                for d in docs:
                    badge_cls = "badge-active" if d['status'] == "ACTIVE" else "badge-superseded"
                    st.markdown(
                        f"<div style='background: rgba(33, 38, 45, 0.6); padding: 10px; border-radius: 6px; margin-bottom: 8px; border: 1px solid rgba(48, 54, 61, 0.6);'>"
                        f"<b>#{d['rank']} [{d['doc_id']}]</b> <span class='{badge_cls}'>{d['status']}</span><br>"
                        f"<small><b>Title:</b> {d['title']} | <b>Ver:</b> {d['version']} | <b>Effective:</b> {d['effective_date']}</small><br>"
                        f"<small style='color: #8b949e;'>Supersedes: {d['supersedes']}</small><br>"
                        f"<p style='font-size: 0.85rem; margin-top: 6px; color: #c9d1d9;'>\"{d['content'][:140]}...\"</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

        # Step 2: Synthesizer Agent (Standard RAG)
        step_synth = next((s for s in trace_data if s["agent"] == "Synthesizer Agent"), None)
        if step_synth:
            with st.expander(f"{step_synth['icon']} Step 2: {step_synth['agent']} (Standard RAG)", expanded=False):
                st.markdown(f"**Status:** `{step_synth['status']}` | <small>{step_synth['timestamp']}</small>", unsafe_allow_html=True)
                st.markdown(f"*{step_synth['summary']}*")
                st.warning("⚠️ Notice: A standard naive RAG model stops here and outputs this initial unverified draft directly to the user:")
                st.markdown(f"> *{step_synth['details'].get('draft')}*")

        # Step 3: Critic Agent (The Adversary)
        step_critic = next((s for s in trace_data if "Critic" in s["agent"]), None)
        if step_critic:
            with st.expander(f"{step_critic['icon']} Step 3: {step_critic['agent']} (Adversarial Audit)", expanded=True):
                st.markdown(f"**Audit Result:** `{step_critic['status']}` | <small>{step_critic['timestamp']}</small>", unsafe_allow_html=True)
                st.markdown(f"**Adversary Verdict:** *{step_critic['summary']}*")

                critic_details = step_critic["details"]
                conflicts = critic_details.get("conflicts", [])
                staleness = critic_details.get("staleness_issues", [])
                unsupported = critic_details.get("unsupported_claims", [])

                if conflicts:
                    st.markdown("#### 🚨 Flagged Contradictions:")
                    for idx, c in enumerate(conflicts, 1):
                        st.markdown(
                            f"<div style='background: rgba(248, 81, 73, 0.1); border: 1px solid rgba(248, 81, 73, 0.4); border-radius: 6px; padding: 10px; margin-bottom: 8px;'>"
                            f"<b>Conflict #{idx}: {c.get('subject')}</b> (Severity: <code>{c.get('severity')}</code>)<br>"
                            f"<b>Doc A ({c.get('doc_a')}):</b> {c.get('policy_a_claim')}<br>"
                            f"<b>Doc B ({c.get('doc_b')}):</b> {c.get('policy_b_claim')}<br>"
                            f"<p style='color: #ff7b72; font-size: 0.86rem; margin-top: 4px;'><b>Adversary Analysis:</b> {c.get('contradiction_analysis')}</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                if staleness:
                    st.markdown("#### ⏳ Staleness & Supersession Violations:")
                    for s in staleness:
                        st.markdown(
                            f"<div style='background: rgba(210, 153, 34, 0.1); border: 1px solid rgba(210, 153, 34, 0.4); border-radius: 6px; padding: 10px; margin-bottom: 8px;'>"
                            f"<b>Obsolete Doc:</b> <code>{s.get('superseded_doc_id')}</code> ➔ <b>Superseded By:</b> <code>{s.get('newer_doc_id')}</code><br>"
                            f"<small><b>Invalidated Rules:</b> {s.get('superseded_rules')}</small><br>"
                            f"<small style='color: #e3b341;'><b>Timeline:</b> {s.get('effective_timeline')}</small>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                if unsupported:
                    st.markdown("#### ⚠️ Unsupported Assertions / Silent Blends:")
                    for u in unsupported:
                        st.markdown(f"- **Claim:** *{u.get('claim')}*\n  **Reason:** {u.get('reason')}")

        # Step 4: Resolution Agent
        step_res = next((s for s in trace_data if "Resolution" in s["agent"]), None)
        if step_res:
            with st.expander(f"{step_res['icon']} Step 4: {step_res['agent']}", expanded=True):
                st.markdown(f"**Status:** `{step_res['status']}` | <small>{step_res['timestamp']}</small>", unsafe_allow_html=True)
                st.markdown(f"*{step_res['summary']}*")
                st.success("✅ Output verified against enterprise temporal governance rules. Silent blending prohibited.")
