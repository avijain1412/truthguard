# 🛡️ TruthGuard: Adversarial Verification Multi-Agent System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An Adversarial Verification Multi-Agent Architecture designed to answer questions from deliberately inconsistent enterprise document corpora (conflicting policies, obsolete baselines, and superseded amendments) without hallucinating a silent blend.**

---

## 📌 Executive Summary & The Problem

Enterprise knowledge repositories (SharePoint, Notion, Confluence, internal legal wikis) are rarely clean. Over years of organizational turnover, they become plagued with:
- **Superseded Amendments:** A 2021 policy permits 100% remote work, while a 2025 executive directive mandates 4 days in-office and discontinues stipends.
- **Revoked Security Standards:** An older 2022 IT standard enforces 90-day password expiration and SMS 2FA, while a 2024 Zero Trust directive strictly prohibits periodic rotation and mandates FIDO2 passkeys.
- **Inter-Departmental Deadlocks:** Engineering authorizes $100/mo personal AI copilot expense reimbursement, while Legal and InfoSec simultaneously enact a zero-tolerance ban on decentralized AI tooling.

### Why Standard RAG Fails Catastrophically
Traditional Retrieval-Augmented Generation (RAG) approaches treat all retrieved context with equal chronological validity. When presented with conflicting documents, traditional RAG models will:
1. **Hallucinate a "Silent Blend":** Artificially merge conflicting rules (e.g., *"Employees may work remote 1 day a month or 3 days a week and receive between $0 and $1,200"*).
2. **Surface Superseded, Non-Compliant Information:** Present revoked compliance rules as current fact, exposing organizations to legal, operational, and cybersecurity risk.
3. **Hide the Discrepancy:** Fail to inform the user that a policy conflict exists.

---

## ⚡ The TruthGuard Solution

**TruthGuard** replaces naive RAG with an **adversarial multi-agent pipeline** built on **LangGraph**. A dedicated **Critic Agent ("The Adversary")** actively audits candidate answers, cross-references temporal metadata, and detects contradictions. If contradictions or staleness are found, the system **NEVER silently blends the answer**; instead, the **Resolution Agent** surfaces both positions with provenance, outlines chronological supersessions, or flags unresolvable governance deadlocks.

```
                          ┌───────────────────────────┐
                          │        User Query         │
                          └─────────────┬─────────────┘
                                        ▼
                          ┌───────────────────────────┐
                          │      Retriever Agent      │
                          │   (Chroma / Metadata)     │
                          └─────────────┬─────────────┘
                                        ▼
                          ┌───────────────────────────┐
                          │     Synthesizer Agent     │
                          │   (Initial Candidate)     │
                          └─────────────┬─────────────┘
                                        ▼
                          ┌───────────────────────────┐
                          │   Critic Agent (Adversary)│
                          │  (Pydantic Structured)    │
                          └─────────────┬─────────────┘
                                        │
                       Has Conflicts / Staleness Detected?
                                       / \
                                     YES  NO
                                     /     \
                                    ▼       ▼
    ┌───────────────────────────────────┐  ┌──────────────────────────────────┐
    │          Resolution Agent         │  │     Finalize Approved Agent      │
    │  - Non-blended Provenance Table   │  │   - Stamped Compliance Badges    │
    │  - Supersession Hierarchy         │  │   - Grounded Answer Citations    │
    │  - Inter-Dept Deadlock Escalation │  └─────────────────┬────────────────┘
    └─────────────────┬─────────────────┘                    │
                      │                                      │
                      └──────────────────┬───────────────────┘
                                         ▼
                          ┌───────────────────────────┐
                          │   Streamlit Chat & Trace  │
                          └───────────────────────────┘
```

---

## 🏛️ Multi-Agent Architecture

| Agent | Icon | Role & Responsibilities |
| :--- | :---: | :--- |
| **Retriever Agent** | 🔍 | Performs vector similarity search across the enterprise corpus. Enriches each passage with structured metadata (`doc_id`, `version`, `effective_date`, `status`, `supersedes`, `department`, `authority_level`). |
| **Synthesizer Agent** | ✍️ | Acts as a baseline RAG agent. Drafts an initial candidate answer using only retrieved context to demonstrate what standard RAG would generate. |
| **Critic Agent**<br>*(The Adversary)* | ⚔️ | **The Core Auditor.** Actively attacks the candidate draft using Pydantic structured output (`CriticVerdict`). Audits for:<br>• **Cross-document contradictions**<br>• **Temporal staleness & supersessions**<br>• **Unsupported claims / silent blending**<br>• **Categorization:** Resolvable vs. Unresolvable Deadlock. |
| **Resolution Agent** | ⚖️ | Invoked when conflicts exist. **Guaranteed zero silent blending.** Formats output with Status Banners, Chronological Provenance Matrices, Adversarial Deconstruction, and actionable employee guidance. |
| **Finalize Approved Agent** | ✅ | Invoked when zero conflicts are detected. Stamps verified provenance badges onto the draft. |

---

## 📊 Pre-Loaded Inconsistent Corpus Scenarios

TruthGuard comes pre-loaded with realistic, deliberately conflicting enterprise documents:

### 1. Remote Work & Stipend Conflict
* **`HR-POL-2021-04` (v1.0, 2021):** 100% remote work allowed; 1 day/mo in-office; $1,200 annual stipend.
* **`HR-POL-2023-09` (v2.0, 2023):** 3 days/week in-office (within 50 miles); stipend reduced to $500; remote grandfathered.
* **`EXEC-MEMO-2024-11` (v3.0, Effective Jan 1, 2025):** 4 days/week mandatory in-office (Mon-Thu); **stipend discontinued ($0)**; **supersedes and revokes all prior policies and grandfathering**.

### 2. IT Security: Password Rotation & 2FA
* **`SEC-STD-2022-01` (v2.1, 2022):** 12-char passwords; mandatory 90-day rotation; SMS 2FA required for VPN.
* **`SEC-BULLETIN-2024-03` (v3.0, 2024):** Zero Trust (NIST SP 800-63B); **periodic rotation strictly prohibited**; 16-char minimum; **SMS 2FA forbidden** due to SIM-swapping; FIDO2 passkeys mandatory.

### 3. Travel Expense & Domestic Business Class
* **`FIN-POL-2022-05` (v4.0, 2022):** Flat $75/day meal per diem; receipts optional under $25; Business Class permitted for flights > 4 hours.
* **`FIN-AMEND-2024-08` (v4.2, 2024):** Tiered per diem ($60 std / $85 Tier-1); receipts mandatory for all expenses; **domestic Business Class strictly abolished**.

### 4. AI Developer Tooling (Inter-Departmental Deadlock)
* **`ENG-PROC-2024-01` (Engineering, Jan 2024):** Engineers authorized to expense up to $100/mo for GitHub Copilot, Cursor, or Claude Pro on corporate cards without prior approval.
* **`LEGAL-SEC-2024-02` (Legal/InfoSec, Feb 2024):** Zero individual expense reimbursements approved for AI tools; expensing personal AI licenses is a compliance violation; all AI must be centrally licensed through Central IT.
* *Adversarial Result:* Flagged as an **Unresolvable Inter-Departmental Deadlock** requiring human executive arbitration.

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/truthguard.git
cd truthguard
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables (Optional)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. 

> **💡 Zero-Token Demo Mode:** TruthGuard includes an intelligent **Deterministic Fallback Engine**. If no API keys are configured, the app runs completely offline with full multi-agent trace visualization, Pydantic schemas, and provenance matrices!

---

## 🖥️ Running the Application

### Launch the Streamlit Interactive UI
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Standalone CLI Execution
You can also run and inspect each module directly from the command line:

```bash
# Test Corpus & Vector Search
python data_loader.py

# Test Full Multi-Agent Pipeline
python graph.py
```

---

## 🔬 Live Agent Trace Feature

The Streamlit UI features a dedicated **Multi-Agent Thought & Trace Inspector** alongside the chat view:

1. **Retriever Trace:** View exact passages fetched, version history, effective dates, and similarity rankings.
2. **Synthesizer Trace:** Inspect the unverified initial draft to see where traditional RAG models hallucinate or blend data.
3. **Critic Trace (The Adversary):**
   - Severity level (`CRITICAL`, `HIGH`, `MEDIUM`).
   - Pairwise contradiction analysis (`Doc A Claim` vs `Doc B Claim`).
   - Supersession timelines and invalidated clauses.
   - Flagged silent blends or unsupported claims.
4. **Resolution Trace:** View the non-blended provenance matrix and policy arbitration rationale.

---

## 📂 Project Structure

```
truthguard/
├── app.py               # Streamlit frontend with interactive Agent Trace viewer
├── graph.py             # LangGraph StateGraph, multi-agent nodes & Pydantic schemas
├── data_loader.py       # Vector store initializer with conflicting enterprise corpus
├── requirements.txt     # Locked production dependencies
├── .env.example         # Template for OpenAI/Anthropic API keys
└── README.md            # Comprehensive project documentation & architecture guide
```

---

## 🛡️ Traditional RAG vs. TruthGuard Comparison

| Feature | Traditional RAG | TruthGuard Multi-Agent |
| :--- | :---: | :---: |
| **Handling Inconsistent Documents** | ❌ Hallucinates blend | ✅ Detects pairwise contradictions |
| **Temporal / Version Awareness** | ❌ Blind to dates | ✅ Enforces chronological supersession |
| **Departmental Deadlocks** | ❌ Picks random passage | ✅ Flags unresolvable governance deadlock |
| **Provenance Visibility** | ❌ Opaque or unverified | ✅ Interactive Agent Trace & Provenance Matrix |
| **Silent Blending Prevention** | ❌ None | ✅ Strictly prohibited by architectural invariant |
| **Output Grounding** | ⚠️ Probabilistic | ✅ Adversarially verified by Pydantic Critic |

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
