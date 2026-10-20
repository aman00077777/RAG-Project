import os
import sys
import tempfile
import time
from typing import List, Tuple, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# LangChain and vector store imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader

# -----------------------------------------------------------------------------
# 1. Environment & API Key Configuration
# -----------------------------------------------------------------------------
load_dotenv()

# Resolve Gemini API Key (checking both GOOGLE_API_KEY and GEMINI_API_KEY)
env_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if env_key and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = env_key

# -----------------------------------------------------------------------------
# 2. Streamlit Page Configuration & Design System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ContextAI — Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End SaaS CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1050px;
    }

    /* Header styling */
    .app-header {
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }
    .brand-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 0;
        color: #f8fafc;
    }
    .brand-badge {
        font-size: 0.72rem;
        font-weight: 600;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: #ffffff;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .brand-tagline {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 0.35rem;
        font-weight: 400;
    }

    /* Metric & Status Cards */
    .stat-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
    }
    .stat-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .stat-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 0.2rem;
    }

    /* Welcome / Hero State */
    .hero-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 2.2rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    .hero-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    .feature-item {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 10px;
        padding: 1rem;
    }
    .feature-icon {
        font-size: 1.3rem;
        margin-bottom: 0.4rem;
    }
    .feature-title {
        font-weight: 600;
        font-size: 0.9rem;
        color: #e2e8f0;
        margin-bottom: 0.25rem;
    }
    .feature-desc {
        font-size: 0.8rem;
        color: #94a3b8;
        line-height: 1.4;
    }

    /* Active Knowledge Bar */
    .knowledge-bar {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 10px;
        padding: 0.65rem 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.2rem;
    }
    .knowledge-info {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 0.86rem;
        color: #cbd5e1;
    }
    .status-dot {
        height: 8px;
        width: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #10b981;
    }

    /* Source Citation Accordion */
    .source-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #6366f1;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.84rem;
        color: #cbd5e1;
    }
    .source-header {
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.35rem;
        font-size: 0.82rem;
        display: flex;
        justify-content: space-between;
    }
    .source-text {
        color: #94a3b8;
        line-height: 1.5;
        white-space: pre-wrap;
    }

    /* File List Chips */
    .file-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: #cbd5e1;
        font-size: 0.78rem;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }

    /* Suggested Question Buttons */
    div[data-testid="stButton"] > button {
        border-radius: 8px;
        font-size: 0.84rem;
        transition: all 0.2s ease;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 3. Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

if "active_source" not in st.session_state:
    st.session_state.active_source = "None"

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# -----------------------------------------------------------------------------
# 4. Resource Caching (Hugging Face Embedding Model)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load and cache the Hugging Face embedding model (384-dim)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# -----------------------------------------------------------------------------
# 5. Document Processing & Ingestion Helpers
# -----------------------------------------------------------------------------
def load_file_content(uploaded_file) -> List[Document]:
    """Parse an uploaded PDF or TXT file into LangChain Documents."""
    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()

    if not file_bytes or len(file_bytes.strip()) == 0:
        raise ValueError(f"The uploaded file '{filename}' is empty.")

    documents: List[Document] = []

    if filename.lower().endswith(".pdf"):
        # Write to temporary file for PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path)
            loaded_docs = loader.load()

            for doc in loaded_docs:
                page_text = doc.page_content.strip()
                if page_text:
                    # Clean metadata for display
                    doc.metadata["source"] = filename
                    documents.append(doc)

            if not documents:
                raise ValueError(
                    f"No extractable text found in '{filename}'. "
                    "The PDF might be scanned, image-only, or password protected."
                )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    elif filename.lower().endswith(".txt"):
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1", errors="replace")

        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError(f"The text file '{filename}' contains no readable text.")

        documents.append(
            Document(
                page_content=cleaned_text,
                metadata={"source": filename, "page": 1},
            )
        )
    else:
        raise ValueError(
            f"Unsupported file format: '{filename}'. Please upload PDF or TXT files."
        )

    return documents


def generate_suggested_questions(files_metadata: List[Dict[str, Any]]) -> List[str]:
    """Generate intelligent sample questions based on the ingested documents."""
    names = [f["name"].lower() for f in files_metadata]

    # Heuristic suggestions tailored to common document topics
    if any("dl" in n or "deep" in n for n in names):
        return [
            "What is deep learning and how does it differ from traditional ML?",
            "What is the role of backpropagation and loss functions in neural networks?",
            "How do recurrent neural networks (RNNs) and Transformers handle sequential data?",
            "Can you provide a summary of the main conclusions from the document?",
        ]
    elif any("gru" in n or "rnn" in n or "lstm" in n for n in names):
        return [
            "What is a Gated Recurrent Unit (GRU) and how does it work?",
            "What are the differences between GRU and LSTM architectures?",
            "What problem do gating mechanisms solve in sequence modeling?",
            "Summarize the key mathematical equations or updates in the text.",
        ]
    else:
        first_doc = files_metadata[0]["name"] if files_metadata else "the document"
        return [
            f"What are the main topics discussed in {first_doc}?",
            "Can you summarize the key findings or takeaways?",
            "What specific methodologies, techniques, or definitions are introduced?",
            "Are there any limitations or future directions mentioned?",
        ]


def process_uploaded_documents(uploaded_files, embedding_model) -> Tuple[Chroma, List[Dict[str, Any]], int]:
    """Split, embed, and index documents into a session-isolated Chroma vectorstore."""
    all_docs: List[Document] = []
    files_info: List[Dict[str, Any]] = []

    for uf in uploaded_files:
        docs = load_file_content(uf)
        all_docs.extend(docs)
        files_info.append({
            "name": uf.name,
            "size": f"{len(uf.getvalue()) / 1024:.1f} KB",
            "pages": len(docs),
        })

    # Chunking: chunk_size=1000, chunk_overlap=200
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(all_docs)

    if not chunks:
        raise ValueError("Document splitting resulted in 0 text chunks.")

    # Create temporary isolated directory for the session
    session_temp_dir = tempfile.mkdtemp(prefix="contextai_chroma_session_")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=session_temp_dir,
    )

    return vectorstore, files_info, len(chunks)


# -----------------------------------------------------------------------------
# 6. RAG Generation Pipeline
# -----------------------------------------------------------------------------
RAG_SYSTEM_PROMPT = """You are ContextAI, an intelligent, objective, and precise document research assistant.
Your goal is to answer the user's question using ONLY the provided document context excerpts.

Strict Rules:
1. Ground every statement in the provided context.
2. If the context does not contain enough information to answer the question, clearly respond:
   "I cannot find the answer in the provided documents."
3. Never invent facts, make assumptions, or introduce outside knowledge not present in the context.
4. Structure your response clearly using markdown headings, bullet points, and concise paragraphs where appropriate.
5. Refer to specific document sources or section titles if they appear in the excerpts.
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", "Context from documents:\n{context}\n\nQuestion: {question}"),
])


def execute_rag_query(query: str, vectorstore: Chroma, api_key: str) -> Tuple[str, List[Document]]:
    """Execute MMR retrieval and generate an answer with Gemini 2.5 Flash."""
    # Retrieval configuration: MMR
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.7,
        },
    )

    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return (
            "I could not retrieve any relevant context from your documents to answer this question.",
            [],
        )

    # Format context for the LLM
    context_chunks = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", "N/A")
        context_chunks.append(
            f"--- [Excerpt {i} | Source: {source} (Page {page})] ---\n{doc.page_content}"
        )
    context_str = "\n\n".join(context_chunks)

    # Initialize Gemini 2.5 Flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key,
    )

    prompt_messages = PROMPT_TEMPLATE.format_messages(
        context=context_str,
        question=query,
    )

    response = llm.invoke(prompt_messages)
    return response.content, retrieved_docs


# -----------------------------------------------------------------------------
# 7. Sidebar: Brand, Document Upload & Knowledge Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
            <div style="font-size: 1.45rem; font-weight: 700; color: #f8fafc; display: flex; align-items: center; gap: 0.5rem;">
                <span>🧠</span> ContextAI
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.2rem;">
                Your documents. Your questions. Clear answers.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # API Key Verification Indicator
    effective_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if effective_api_key:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:0.4rem; padding:0.4rem 0.7rem; background:rgba(16, 185, 129, 0.1); border:1px solid rgba(16, 185, 129, 0.25); border-radius:6px; margin-bottom:1rem;">
                <span class="status-dot"></span>
                <span style="font-size:0.78rem; color:#10b981; font-weight:500;">Google Gemini API Connected</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="padding:0.6rem 0.8rem; background:rgba(239, 68, 68, 0.1); border:1px solid rgba(239, 68, 68, 0.25); border-radius:6px; margin-bottom:1rem;">
                <span style="font-size:0.8rem; color:#ef4444; font-weight:600;">⚠️ API Key Missing</span>
                <div style="font-size:0.75rem; color:#cbd5e1; margin-top:0.2rem;">
                    Add <code>GOOGLE_API_KEY</code> to your <code>.env</code> file.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Knowledge Base Selection
    st.markdown("### 📚 Knowledge Source")
    source_mode = st.radio(
        "Select Knowledge Source",
        options=["Upload Custom Documents", "Use Existing Database (chroma-db-local)"],
        index=0 if st.session_state.active_source != "Existing Database" else 1,
        label_visibility="collapsed",
    )

    # Option A: Upload Custom Documents
    if source_mode == "Upload Custom Documents":
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
            help="Select one or multiple PDF or TXT documents to index.",
        )

        if uploaded_files:
            st.markdown(f"**Selected Files ({len(uploaded_files)}):**")
            chips_html = "".join([
                f"<div class='file-chip'>📄 {f.name} <span style='color:#94a3b8;'>({len(f.getvalue())/1024:.1f} KB)</span></div>"
                for f in uploaded_files
            ])
            st.markdown(f"<div>{chips_html}</div>", unsafe_allow_html=True)

        col_proc, col_clear = st.columns([1.6, 1])
        with col_proc:
            process_btn = st.button(
                "⚡ Process Documents",
                use_container_width=True,
                type="primary",
                disabled=not bool(uploaded_files),
            )
        with col_clear:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.uploader_key += 1
                st.session_state.vectorstore = None
                st.session_state.processed_files = []
                st.session_state.total_chunks = 0
                st.session_state.suggested_questions = []
                st.session_state.active_source = "None"
                st.rerun()

        if process_btn and uploaded_files:
            try:
                with st.status("Ingesting and indexing documents...", expanded=True) as status:
                    st.write("1. Reading and validating documents...")
                    embedding_model = load_embedding_model()

                    st.write("2. Splitting into semantic chunks (size=1000, overlap=200)...")
                    st.write("3. Generating embeddings with all-MiniLM-L6-v2 (384-dim)...")

                    start_time = time.time()
                    vs, file_stats, num_chunks = process_uploaded_documents(
                        uploaded_files, embedding_model
                    )
                    elapsed = time.time() - start_time

                    status.update(
                        label=f"Indexed {len(uploaded_files)} files ({num_chunks} chunks) in {elapsed:.1f}s!",
                        state="complete",
                        expanded=False,
                    )

                st.session_state.vectorstore = vs
                st.session_state.processed_files = file_stats
                st.session_state.total_chunks = num_chunks
                st.session_state.suggested_questions = generate_suggested_questions(file_stats)
                st.session_state.active_source = f"{len(file_stats)} Uploaded File(s)"
                st.success(f"Knowledge Base ready with {num_chunks} chunks!")
                st.rerun()

            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")

    # Option B: Use Existing Database (chroma-db-local)
    else:
        st.info("Existing database contains previously indexed documents (Deep Learning knowledge base).")
        load_existing_btn = st.button(
            "Load Existing Database",
            use_container_width=True,
            type="primary" if st.session_state.active_source != "Existing Database" else "secondary",
        )

        if load_existing_btn:
            try:
                with st.spinner("Connecting to chroma-db-local..."):
                    embedding_model = load_embedding_model()
                    if not os.path.exists("chroma-db-local"):
                        raise FileNotFoundError("Directory 'chroma-db-local' was not found.")

                    vs = Chroma(
                        persist_directory="chroma-db-local",
                        embedding_function=embedding_model,
                    )
                    count = vs._collection.count()

                st.session_state.vectorstore = vs
                st.session_state.processed_files = [{"name": "chroma-db-local (Deep Learning)", "size": "Local DB", "pages": count}]
                st.session_state.total_chunks = count
                st.session_state.suggested_questions = [
                    "What is deep learning and how is it used in AI?",
                    "What are the main components of a neural network?",
                    "How does reinforcement learning work (Agent, State, Action, Reward)?",
                    "Explain the difference between RNNs and feed-forward networks.",
                ]
                st.session_state.active_source = "Existing Database (chroma-db-local)"
                st.success(f"Connected to existing database ({count} chunks)!")
                st.rerun()

            except Exception as e:
                st.error(f"Failed to load existing database: {str(e)}")

    # Knowledge Base Status Panel
    st.markdown("---")
    st.markdown("### 📊 System Status")

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Active Source</div>
            <div class="stat-value" style="font-size: 0.95rem; word-break: break-word;">{st.session_state.active_source}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Total Chunks</div>
                <div class="stat-value">{st.session_state.total_chunks}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_s2:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-label">Embedding</div>
                <div class="stat-value" style="font-size: 0.85rem;">MiniLM (384d)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Retrieval Configuration Details
    with st.expander("⚙️ RAG Configuration", expanded=False):
        st.caption("**LLM**: Gemini 2.5 Flash (temperature=0)")
        st.caption("**Retriever**: Maximal Marginal Relevance (MMR)")
        st.caption("**k**: 4 | **fetch_k**: 10 | **lambda_mult**: 0.7")
        st.caption("**Chunk Size**: 1000 chars | **Overlap**: 200 chars")

    # Clear Chat Option
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -----------------------------------------------------------------------------
# 8. Main Application Area
# -----------------------------------------------------------------------------

# Main Brand Header
st.markdown(
    """
    <div class="app-header">
        <div class="brand-title">
            <span>🧠 ContextAI</span>
            <span class="brand-badge">RAG Assistant</span>
        </div>
        <div class="brand-tagline">
            Your documents. Your questions. Clear answers.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# API Key Missing Warning Banner
if not effective_api_key:
    st.warning(
        "**API Key Required**: No `GOOGLE_API_KEY` was detected in your environment. "
        "Please ensure `GOOGLE_API_KEY` is set in your `.env` file to chat with your documents.",
        icon="🔑",
    )

# Active Knowledge Indicator Bar
if st.session_state.vectorstore is not None:
    st.markdown(
        f"""
        <div class="knowledge-bar">
            <div class="knowledge-info">
                <span class="status-dot"></span>
                <span>Active Knowledge: <strong>{st.session_state.active_source}</strong> ({st.session_state.total_chunks} chunks ready)</span>
            </div>
            <div style="font-size: 0.78rem; color: #94a3b8;">
                MMR Retrieval • Gemini 2.5 Flash
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Empty / Welcome State (when no knowledge base is active)
if st.session_state.vectorstore is None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Welcome to ContextAI</div>
            <div class="hero-subtitle">
                An enterprise-grade document question-answering assistant powered by LangChain,
                Chroma vector search with Maximal Marginal Relevance (MMR), and Google Gemini.
            </div>
            <div class="feature-grid">
                <div class="feature-item">
                    <div class="feature-icon">📁</div>
                    <div class="feature-title">1. Upload Documents</div>
                    <div class="feature-desc">Upload one or multiple PDF or TXT files in the sidebar, or connect to your local ChromaDB.</div>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">⚡</div>
                    <div class="feature-title">2. Semantic Indexing</div>
                    <div class="feature-desc">Documents are split into 1000-char chunks and embedded using 384-dimensional MiniLM embeddings.</div>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">🎯</div>
                    <div class="feature-title">3. Grounded Answers</div>
                    <div class="feature-desc">Ask questions and receive precise answers grounded strictly in retrieved context, with source citations.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Suggested Questions Section
if st.session_state.vectorstore is not None and st.session_state.suggested_questions:
    st.markdown("<p style='font-size: 0.82rem; color: #94a3b8; font-weight: 600; margin-bottom: 0.4rem; text-transform: uppercase; letter-spacing: 0.05em;'>💡 Suggested Questions</p>", unsafe_allow_html=True)
    cols = st.columns(len(st.session_state.suggested_questions))
    for idx, (col, q_text) in enumerate(zip(cols, st.session_state.suggested_questions)):
        with col:
            if st.button(q_text, key=f"sug_{idx}", use_container_width=True):
                st.session_state.pending_query = q_text
                st.rerun()

# Render Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display expandable source citations if available
        if msg.get("sources"):
            with st.expander(f"📄 View Sources ({len(msg['sources'])} excerpts retrieved)", expanded=False):
                for idx, src in enumerate(msg["sources"], 1):
                    source_name = src.get("source", "Unknown Document")
                    page_num = src.get("page", "N/A")
                    content_snippet = src.get("content", "")

                    st.markdown(
                        f"""
                        <div class="source-box">
                            <div class="source-header">
                                <span>Excerpt #{idx} — <strong>{source_name}</strong></span>
                                <span style="color:#94a3b8;">Page {page_num}</span>
                            </div>
                            <div class="source-text">{content_snippet}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Handle Chat Input & Pending Questions
prompt_input = st.chat_input(
    "Ask a question about your documents..."
    if st.session_state.vectorstore is not None
    else "Upload documents in the sidebar first..."
)

# Check if user submitted via chat input OR clicked a suggested question
active_query = prompt_input or st.session_state.pop("pending_query", None)

if active_query:
    # Validation checks
    if st.session_state.vectorstore is None:
        st.error("Please upload and process documents or select the existing database in the sidebar before asking questions.")
    elif not effective_api_key:
        st.error("Google Gemini API Key is missing. Please configure `GOOGLE_API_KEY` in your `.env` file.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": active_query, "sources": []})
        with st.chat_message("user"):
            st.markdown(active_query)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("ContextAI is retrieving context and generating an answer..."):
                try:
                    answer, docs = execute_rag_query(
                        query=active_query,
                        vectorstore=st.session_state.vectorstore,
                        api_key=effective_api_key,
                    )

                    # Extract source metadata for storage
                    source_metadata = [
                        {
                            "source": d.metadata.get("source", "Unknown"),
                            "page": d.metadata.get("page", "N/A"),
                            "content": d.page_content,
                        }
                        for d in docs
                    ]

                    # Render response
                    st.markdown(answer)

                    # Render source expander
                    if source_metadata:
                        with st.expander(f"📄 View Sources ({len(source_metadata)} excerpts retrieved)", expanded=False):
                            for idx, src in enumerate(source_metadata, 1):
                                st.markdown(
                                    f"""
                                    <div class="source-box">
                                        <div class="source-header">
                                            <span>Excerpt #{idx} — <strong>{src['source']}</strong></span>
                                            <span style="color:#94a3b8;">Page {src['page']}</span>
                                        </div>
                                        <div class="source-text">{src['content']}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # Append to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": source_metadata,
                    })

                except Exception as e:
                    error_msg = f"An error occurred while generating the response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"⚠️ **Error:** {error_msg}",
                        "sources": [],
                    })
