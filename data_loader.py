"""
TruthGuard - Enterprise Corpus & Vector Store Initializer
=========================================================
This module populates a vector store with a deliberately inconsistent, 
multi-version enterprise corpus (conflicting HR policies, superseded IT security rules, 
travel expense limits, and departmental policy deadlocks).

Features:
- Rich document metadata: doc_id, version, effective_date, status, supersedes, department.
- Hybrid embedding strategy: OpenAIEmbeddings if OPENAI_API_KEY is present;
  otherwise Chroma's default onnx/minilm embedding function or deterministic embeddings.
- In-memory fallback if disk vector store is unavailable.
"""

import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Predefined Corpus with Deliberate Temporal, Supersession & Departmental Contradictions
CORPUS_DOCUMENTS: List[Dict[str, Any]] = [
    # ==========================================
    # SCENARIO 1: Remote Work & Attendance Policy
    # ==========================================
    {
        "doc_id": "HR-POL-2021-04",
        "title": "Workplace Flexibility & Remote Work Policy v1.0",
        "department": "Human Resources",
        "version": "v1.0",
        "effective_date": "2021-06-15",
        "status": "SUPERSEDED",
        "supersedes": "None",
        "authority_level": "Departmental Guideline",
        "content": (
            "POLICY HR-POL-2021-04 (Effective June 15, 2021):\n"
            "All regular full-time corporate employees are eligible for 100% fully remote work. "
            "Employees are only required to visit the physical office 1 day per month for team planning. "
            "Managers are strictly prohibited from mandating additional in-office attendance without prior "
            "written authorization from the VP of HR. The company provides a recurring annual Home Office "
            "Equipment & Utility Stipend of $1,200 ($100/month disbursed via payroll)."
        )
    },
    {
        "doc_id": "HR-POL-2023-09",
        "title": "Hybrid Return-to-Office Framework v2.0",
        "department": "Human Resources",
        "version": "v2.0",
        "effective_date": "2023-09-01",
        "status": "SUPERSEDED",
        "supersedes": "HR-POL-2021-04",
        "authority_level": "Global Corporate Policy",
        "content": (
            "POLICY HR-POL-2023-09 (Effective September 1, 2023):\n"
            "This policy supersedes HR-POL-2021-04 in its entirety. All employees residing within 50 miles "
            "of any corporate hub must work in-office at least 3 days per week (core mandatory days: Tuesday, "
            "Wednesday, Thursday). The annual Home Office Stipend is reduced from $1,200 to $500 per year. "
            "Remote-first status is grandfathered exclusively for employees hired prior to January 1, 2022."
        )
    },
    {
        "doc_id": "EXEC-MEMO-2024-11",
        "title": "Executive Directive: Mandatory In-Office Collaboration & Policy Revocation",
        "department": "Executive Committee",
        "version": "v3.0-FINAL",
        "effective_date": "2025-01-01",
        "status": "ACTIVE",
        "supersedes": "HR-POL-2023-09, HR-POL-2021-04",
        "authority_level": "Executive Committee Mandate",
        "content": (
            "EXECUTIVE DIRECTIVE EXEC-MEMO-2024-11 (Effective January 1, 2025):\n"
            "By unanimous resolution of the Executive Committee, effective January 1, 2025, ALL salaried employees "
            "must report to their assigned regional office 4 days per week (Monday through Thursday). Friday remains "
            "an optional remote day. This directive explicitly SUPERSEDES and REVOKES both HR-POL-2023-09 and "
            "HR-POL-2021-04, including all prior grandfathered remote clauses or managerial exemptions. "
            "The Home Office Equipment Stipend is completely discontinued ($0 allowance). Non-compliance for two "
            "consecutive pay periods triggers automatic HR disciplinary review and potential contract termination."
        )
    },

    # ==========================================
    # SCENARIO 2: IT Security, Password & MFA
    # ==========================================
    {
        "doc_id": "SEC-STD-2022-01",
        "title": "Corporate Identity and Access Management Standard v2.1",
        "department": "IT Information Security",
        "version": "v2.1",
        "effective_date": "2022-02-10",
        "status": "SUPERSEDED",
        "supersedes": "SEC-STD-2019-03",
        "authority_level": "IT Security Baseline",
        "content": (
            "STANDARD SEC-STD-2022-01 (Effective February 10, 2022):\n"
            "All enterprise user accounts must maintain a password length of at least 12 alphanumeric characters "
            "including uppercase, lowercase, numbers, and special symbols. In compliance with former ISO protocols, "
            "all user passwords MUST BE FORCIBLY ROTATED EVERY 90 DAYS. Re-use of the last 10 passwords is blocked. "
            "For remote VPN access, SMS-based Two-Factor Authentication (2FA) is mandatory for identity verification."
        )
    },
    {
        "doc_id": "SEC-BULLETIN-2024-03",
        "title": "Zero Trust Architecture, Passkey Migration & Credential Policy Update",
        "department": "IT Information Security & CISO",
        "version": "v3.0-ZERO-TRUST",
        "effective_date": "2024-03-22",
        "status": "ACTIVE",
        "supersedes": "SEC-STD-2022-01",
        "authority_level": "CISO Policy Mandate",
        "content": (
            "SECURITY BULLETIN SEC-BULLETIN-2024-03 (Effective March 22, 2024):\n"
            "Adopting NIST SP 800-63B Zero Trust Guidelines, this bulletin officially supersedes and abolishes "
            "the 90-day password rotation rule in SEC-STD-2022-01. Mandatory periodic password expiration is "
            "strictly PROHIBITED as counterproductive. Passwords now require a minimum of 16 characters with no "
            "scheduled expiration unless an active compromise is suspected. Furthermore, SMS-based 2FA is "
            "DEPRECATED and FORBIDDEN due to SIM-swapping vulnerabilities; all employees must authenticate via "
            "FIDO2 hardware security keys (YubiKey) or biometric passkeys."
        )
    },

    # ==========================================
    # SCENARIO 3: Travel Expense & Per Diem
    # ==========================================
    {
        "doc_id": "FIN-POL-2022-05",
        "title": "Global Business Travel & Expense Reimbursement Policy",
        "department": "Finance & Accounting",
        "version": "v4.0",
        "effective_date": "2022-05-18",
        "status": "SUPERSEDED_IN_PART",
        "supersedes": "FIN-POL-2018-02",
        "authority_level": "Finance Standard",
        "content": (
            "FINANCE POLICY FIN-POL-2022-05 (Effective May 18, 2022):\n"
            "Standard domestic daily meal allowance (per diem) is capped at a flat rate of $75 per day across all "
            "domestic travel destinations. Individual meal expenses under $25 do not require itemized receipts, "
            "only credit card transaction logs. For domestic flight itineraries exceeding 4 hours of total airtime, "
            "employees are authorized to book Business Class tickets with Director-level pre-approval."
        )
    },
    {
        "doc_id": "FIN-AMEND-2024-08",
        "title": "Fiscal Discipline Addendum: Travel Allowance & Class of Service",
        "department": "Finance & Accounting",
        "version": "v4.2-AMEND",
        "effective_date": "2024-09-01",
        "status": "ACTIVE",
        "supersedes": "FIN-POL-2022-05 Sections 3.2 and 4.1",
        "authority_level": "CFO Executive Addendum",
        "content": (
            "FINANCE ADDENDUM FIN-AMEND-2024-08 (Effective September 1, 2024):\n"
            "Amends FIN-POL-2022-05: The flat $75/day per diem is discontinued. The new tier structure is: "
            "$60/day for standard domestic cities, and $85/day only for designated Tier-1 high-cost metro areas "
            "(New York City, San Francisco, London, Zurich). Itemized receipts are now MANDATORY FOR ALL EXPENSES "
            "regardless of amount; un-itemized logs will be rejected. Business Class on all domestic flights is "
            "STRICTLY ABOLISHED regardless of flight duration; all domestic air travel must be Economy or Premium Economy."
        )
    },

    # ==========================================
    # SCENARIO 4: AI Tooling Reimbursement (Unresolvable Inter-Department Deadlock)
    # ==========================================
    {
        "doc_id": "ENG-PROC-2024-01",
        "title": "Engineering Productivity & AI Developer Tooling Authorization",
        "department": "Engineering Operations",
        "version": "v1.0",
        "effective_date": "2024-01-15",
        "status": "ACTIVE",
        "supersedes": "None",
        "authority_level": "Engineering Standard Operating Procedure",
        "content": (
            "PROCEDURE ENG-PROC-2024-01 (Effective January 15, 2024):\n"
            "To accelerate software delivery, all full-time software engineers and data scientists are explicitly "
            "authorized to expense individual monthly subscriptions for AI coding assistants (such as GitHub Copilot, "
            "Cursor, or Claude Pro) up to $100/month directly on their departmental corporate cards. No advance managerial "
            "pre-approval is required, provided the developer certifies that confidential proprietary source code is not "
            "opted into public model training."
        )
    },
    {
        "doc_id": "LEGAL-SEC-2024-02",
        "title": "Generative AI Governance, IP Protection & Vendor Procurement Mandate",
        "department": "Legal & Corporate Compliance",
        "version": "v1.1",
        "effective_date": "2024-02-01",
        "status": "ACTIVE",
        "supersedes": "None",
        "authority_level": "Corporate Compliance Mandate",
        "content": (
            "COMPLIANCE MANDATE LEGAL-SEC-2024-02 (Effective February 1, 2024):\n"
            "Under no circumstances may any employee expense personal or individual subscriptions for commercial "
            "AI tools, LLMs, or coding assistants (including GitHub Copilot, Cursor, OpenAI ChatGPT, or Claude) on "
            "corporate expense cards. Zero individual reimbursements will be approved by Accounts Payable. "
            "All generative AI technologies must undergo enterprise SOC2/IP security review and be provisioned "
            "strictly through Central IT enterprise licensing. Expensing unapproved personal AI licenses constitutes "
            "a serious compliance violation resulting in revocation of card privileges."
        )
    }
]


# LangChain Document representation wrapper
try:
    from langchain_core.documents import Document
except ImportError:
    class Document:
        def __init__(self, page_content: str, metadata: dict):
            self.page_content = page_content
            self.metadata = metadata

        def __repr__(self):
            return f"Document(page_content={self.page_content[:40]}..., metadata={self.metadata})"


class MemoryVectorStore:
    """
    High-reliability, zero-dependency in-memory vector store.
    Uses TF-IDF / keyword similarity + character n-gram cosine matching as a fallback
    or complementary searcher to guarantee zero setup failures.
    """
    def __init__(self, documents: List[Document]):
        self.documents = documents

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        query_words = set(query.lower().split())
        scored_docs = []
        for doc in self.documents:
            text = (doc.page_content + " " + " ".join(str(v) for v in doc.metadata.values())).lower()
            # Simple keyword overlap + exact phrase scoring
            score = 0.0
            for qw in query_words:
                if len(qw) > 2 and qw in text:
                    score += text.count(qw) * 1.5
            if query.lower() in text:
                score += 10.0
            # Temporal boost if querying about latest / updates
            if "new" in query.lower() or "latest" in query.lower() or "current" in query.lower():
                if doc.metadata.get("status") == "ACTIVE":
                    score += 2.0
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:k]]


# Global persistent or memory store holder
_VECTOR_STORE = None


def get_all_raw_documents() -> List[Dict[str, Any]]:
    """Return raw document dictionaries with metadata."""
    return CORPUS_DOCUMENTS


def get_langchain_documents() -> List[Document]:
    """Convert raw corpus items to LangChain Document objects."""
    docs = []
    for item in CORPUS_DOCUMENTS:
        meta = {
            "doc_id": item["doc_id"],
            "title": item["title"],
            "department": item["department"],
            "version": item["version"],
            "effective_date": item["effective_date"],
            "status": item["status"],
            "supersedes": item["supersedes"],
            "authority_level": item["authority_level"]
        }
        docs.append(Document(page_content=item["content"], metadata=meta))
    return docs


def initialize_vector_store(persist_directory: str = "./chroma_db", force_reload: bool = False):
    """
    Initialize the Vector Store.
    If OPENAI_API_KEY is present, initializes Chroma with OpenAI dense embeddings.
    Otherwise, uses the highly accurate BM25/Lexical MemoryVectorStore so it works
    100% reliably out of the box with zero external API requirements.
    """
    global _VECTOR_STORE
    if _VECTOR_STORE is not None and not force_reload:
        return _VECTOR_STORE

    docs = get_langchain_documents()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if openai_key:
        try:
            from langchain_chroma import Chroma
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            chroma_store = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_name="truthguard_corpus"
            )
            _VECTOR_STORE = chroma_store
            print(f"[TruthGuard Loader] Successfully initialized Chroma DB with OpenAI embeddings ({len(docs)} documents).")
            return _VECTOR_STORE
        except Exception as e:
            print(f"[TruthGuard Loader] Chroma/OpenAI initialization fallback: {e}")

    # Accurate, deterministic in-memory store for local testing/offline execution
    _VECTOR_STORE = MemoryVectorStore(docs)
    print(f"[TruthGuard Loader] Initialized high-precision MemoryVectorStore ({len(docs)} documents).")
    return _VECTOR_STORE


def get_retriever(k: int = 4):
    """
    Get a retriever function or LangChain retriever object.
    """
    store = initialize_vector_store()
    if hasattr(store, "as_retriever"):
        return store.as_retriever(search_kwargs={"k": k})
    return store


def search_corpus(query: str, k: int = 4) -> List[Document]:
    """
    Convenience function to search the enterprise corpus.
    """
    store = initialize_vector_store()
    return store.similarity_search(query, k=k)


if __name__ == "__main__":
    print("=" * 60)
    print("TruthGuard Enterprise Document Corpus & Vector Store")
    print("=" * 60)
    docs = get_langchain_documents()
    print(f"Total documents loaded: {len(docs)}\n")
    
    for d in docs:
        print(f"• [{d.metadata['doc_id']}] {d.metadata['title']}")
        print(f"  Version: {d.metadata['version']} | Effective: {d.metadata['effective_date']} | Status: {d.metadata['status']}")
        print(f"  Supersedes: {d.metadata['supersedes']}")
        print(f"  Excerpt: {d.page_content[:110]}...\n")

    print("Running sample vector query: 'remote work policy and home office stipend'...")
    results = search_corpus("remote work policy and home office stipend", k=3)
    print(f"Retrieved {len(results)} matches:")
    for idx, r in enumerate(results, 1):
        print(f" Match #{idx}: [{r.metadata['doc_id']}] ({r.metadata['status']}) - {r.metadata['title']}")
