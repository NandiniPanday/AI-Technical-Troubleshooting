import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TechFix AI",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# COLOR PALETTE
# ============================================================

DARK_RED = "#9D0B0B"
RED = "#DA2D2D"
ORANGE = "#EB8242"
YELLOW = "#F6DA63"

CREAM = "#FFFDF8"
LIGHT_CREAM = "#FFF8E8"
TEXT = "#2F2520"
MUTED = "#6F625B"
BORDER = "#E8DCD2"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    f"""
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {{
        background-color: {CREAM};
        color: {TEXT};
        font-family: "Trebuchet MS", Arial, sans-serif;
    }}

    .main .block-container {{
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }}


    /* =====================================================
       GENERAL TEXT
       ===================================================== */

    p {{
        font-family: "Trebuchet MS", Arial, sans-serif;
        font-size: 18px;
        line-height: 1.65;
        color: {TEXT};
    }}

    label {{
        font-family: "Trebuchet MS", Arial, sans-serif !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        color: {TEXT} !important;
    }}


    /* =====================================================
       SIDEBAR
       ===================================================== */

    [data-testid="stSidebar"] {{
        background-color: {LIGHT_CREAM};
        border-right: 2px solid #F0DED0;
    }}

    [data-testid="stSidebar"] * {{
        color: {TEXT};
    }}

    .sidebar-brand {{
        font-family: Georgia, "Times New Roman", serif;
        font-size: 30px;
        font-weight: bold;
        color: {DARK_RED} !important;
        margin-bottom: 8px;
    }}

    .sidebar-description {{
        font-size: 16px;
        line-height: 1.65;
        color: {MUTED} !important;
    }}

    .sidebar-heading {{
        font-family: Georgia, "Times New Roman", serif;
        font-size: 21px;
        font-weight: bold;
        color: {DARK_RED} !important;
        margin-top: 20px;
        margin-bottom: 10px;
    }}

    .sidebar-divider {{
        border: none;
        border-top: 1px solid #E6D7CC;
        margin: 25px 0;
    }}


    /* =====================================================
       HERO
       ===================================================== */

    .hero-title {{
        font-family: Georgia, "Times New Roman", serif;
        font-size: 46px;
        font-weight: bold;
        color: {DARK_RED};
        margin-bottom: 8px;
    }}

    .hero-subtitle {{
        font-family: "Trebuchet MS", Arial, sans-serif;
        font-size: 20px;
        line-height: 1.6;
        color: {MUTED};
        max-width: 850px;
    }}


    /* =====================================================
       SECTION HEADINGS
       ===================================================== */

    .section-title {{
        font-family: Georgia, "Times New Roman", serif;
        font-size: 27px;
        font-weight: bold;
        color: {DARK_RED};
        margin-top: 28px;
        margin-bottom: 12px;
    }}


    /* =====================================================
       TEXT AREA
       ===================================================== */

    textarea {{
        background-color: white !important;
        color: {TEXT} !important;

        border: 2px solid #E5D6CA !important;
        border-radius: 12px !important;

        font-family: "Trebuchet MS", Arial, sans-serif !important;
        font-size: 18px !important;

        padding: 14px !important;
    }}

    textarea:focus {{
        border: 2px solid {ORANGE} !important;
        box-shadow: 0 0 0 2px rgba(235, 130, 66, 0.15) !important;
    }}


    /* =====================================================
       NORMAL STREAMLIT BUTTONS
       ===================================================== */

    div.stButton > button {{
        background-color: white;
        color: {DARK_RED};

        border: 2px solid {ORANGE};

        border-radius: 10px;

        min-height: 52px;

        font-family: "Trebuchet MS", Arial, sans-serif;
        font-size: 17px;
        font-weight: 700;
    }}

    div.stButton > button:hover {{
        background-color: {YELLOW};
        color: {DARK_RED};
        border-color: {RED};
    }}


    /* =====================================================
       MAIN TROUBLESHOOT BUTTON
       ===================================================== */

    div.stButton > button[kind="primary"] {{
        background-color: {DARK_RED};
        color: white;
        border: 2px solid {DARK_RED};

        min-height: 56px;

        font-size: 19px;
        font-weight: bold;
    }}

    div.stButton > button[kind="primary"]:hover {{
        background-color: {RED};
        color: white;
        border-color: {RED};
    }}


    /* =====================================================
       RESULT
       ===================================================== */

    .answer-heading {{
        font-family: Georgia, "Times New Roman", serif;
        font-size: 29px;
        font-weight: bold;
        color: {DARK_RED};
        margin-top: 32px;
        margin-bottom: 12px;
    }}


    /* =====================================================
       SOURCES
       ===================================================== */

    .sources-heading {{
        font-family: Georgia, "Times New Roman", serif;
        font-size: 24px;
        font-weight: bold;
        color: {DARK_RED};
        margin-top: 30px;
        margin-bottom: 10px;
    }}


    /* =====================================================
       TECH BADGES
       ===================================================== */

    .tech-badge {{
        display: inline-block;

        background-color: white;

        border: 1px solid #E7D4C7;
        border-radius: 20px;

        padding: 7px 12px;
        margin: 3px 3px;

        font-family: "Trebuchet MS", Arial, sans-serif;
        font-size: 14px;
        font-weight: 600;

        color: {DARK_RED} !important;
    }}


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {{
        text-align: center;

        color: {MUTED};

        font-family: "Trebuchet MS", Arial, sans-serif;
        font-size: 15px;

        padding-top: 25px;
        margin-top: 45px;

        border-top: 1px solid {BORDER};
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD API KEY
# ============================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_api_key = None

if not groq_api_key:

    st.error(
        "GROQ_API_KEY is missing. "
        "Add it to your .env file locally or Streamlit Secrets."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">🔧 TechFix AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-description">
        Your AI-powered technical troubleshooting assistant.
        Describe a problem and get step-by-step guidance
        using Retrieval-Augmented Generation.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<hr class="sidebar-divider">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-heading">⚙️ How it works</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
**1. Ask**

Describe your technical problem.

**2. Retrieve**

Relevant information is retrieved from the technical knowledge base.

**3. Generate**

The AI generates a troubleshooting solution using the retrieved information.
""")

    st.markdown(
        '<hr class="sidebar-divider">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-heading">🧠 Technologies</div>',
        unsafe_allow_html=True
    )

    technologies = [
        "Python",
        "Streamlit",
        "LangChain",
        "Groq",
        "RAG",
        "ChromaDB",
        "HuggingFace"
    ]

    for tech in technologies:

        st.markdown(
            f'<span class="tech-badge">{tech}</span>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<hr class="sidebar-divider">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-heading">📚 Knowledge Base</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
The assistant uses technical documents covering:

• Hardware diagnostics

• Laptop troubleshooting

• Network troubleshooting
""")


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="hero-title">🔧 TechFix AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-subtitle">
    AI-powered technical troubleshooting assistant
    that helps diagnose hardware, laptop and network problems.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# LOAD / CREATE CHROMA DATABASE
# ============================================================

@st.cache_resource
def load_vector_store():

    db_path = Path("chroma_db")

    embeddings = load_embeddings()

    # If Chroma database already exists, load it
    if db_path.exists():

        return Chroma(
            persist_directory=str(db_path),
            embedding_function=embeddings
        )

    # Otherwise create the knowledge base from PDFs
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    documents = []

    documents_path = Path("documents")

    for pdf_file in documents_path.glob("*.pdf"):

        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    if not documents:
        return None

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    # Create Chroma database
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(db_path)
    )

    return vector_store


vector_store = load_vector_store()


# ============================================================
# CHECK KNOWLEDGE BASE
# ============================================================

if vector_store is None:

    st.error("Knowledge base not found.")

    st.info(
        "Run `python ingest.py` once in your project folder "
        "to create the ChromaDB knowledge base."
    )

    st.stop()


# ============================================================
# RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 4
    }
)


# ============================================================
# LOAD GROQ MODEL
# ============================================================

@st.cache_resource
def load_llm(api_key):

    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        groq_api_key=api_key
    )


llm = load_llm(groq_api_key)


# ============================================================
# PROBLEM INPUT
# ============================================================

st.markdown(
    '<div class="section-title">📝 What problem are you facing?</div>',
    unsafe_allow_html=True
)


# Initialize question in session state
if "question" not in st.session_state:
    st.session_state.question = ""


question = st.text_area(
    "Describe your technical problem",
    value=st.session_state.question,
    placeholder=(
        "Example: My laptop is connected to Wi-Fi "
        "but websites are not opening..."
    ),
    height=150,
    label_visibility="collapsed"
)


st.session_state.question = question


# ============================================================
# QUICK PROBLEMS
# ============================================================

st.markdown(
    '<div class="section-title">💡 Try a common problem</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    if st.button(
        "💻  Laptop won't turn on",
        use_container_width=True
    ):

        st.session_state.question = (
            "My laptop won't turn on."
        )

        st.rerun()


with col2:

    if st.button(
        "🌐  Wi-Fi has no internet",
        use_container_width=True
    ):

        st.session_state.question = (
            "My laptop is connected to Wi-Fi "
            "but there is no internet."
        )

        st.rerun()


with col3:

    if st.button(
        "🔌  Device is overheating",
        use_container_width=True
    ):

        st.session_state.question = (
            "My laptop is overheating."
        )

        st.rerun()


# ============================================================
# TROUBLESHOOT BUTTON
# ============================================================

st.write("")

if st.button(
    "🔍  Troubleshoot My Problem",
    type="primary",
    use_container_width=True
):

    question = st.session_state.question

    if not question.strip():

        st.warning(
            "Please describe your technical problem first."
        )

    else:

        # -----------------------------------------------
        # RETRIEVE DOCUMENTS
        # -----------------------------------------------

        with st.spinner(
            "🔎 Searching the technical knowledge base..."
        ):

            relevant_docs = retriever.invoke(
                question
            )


        # -----------------------------------------------
        # BUILD CONTEXT
        # -----------------------------------------------

        context = "\n\n".join(
            doc.page_content
            for doc in relevant_docs
        )


        # -----------------------------------------------
        # PROMPT
        # -----------------------------------------------

        prompt = f"""
You are an AI Technical Troubleshooting Assistant.

Use ONLY the information provided in the context.

Answer the user's technical problem clearly.

Structure your answer as:

Possible Cause:
Briefly explain what may be causing the issue.

Troubleshooting Steps:
Give clear numbered steps the user can follow.

When to Seek Professional Help:
Mention when professional assistance may be required.

If the context does not contain enough information,
say so clearly instead of inventing information.

Context:
{context}

User Problem:
{question}
"""


        # -----------------------------------------------
        # GENERATE RESPONSE
        # -----------------------------------------------

        with st.spinner(
            "🤖 Generating your troubleshooting solution..."
        ):

            response = llm.invoke(
                prompt
            )


        # =================================================
        # ANSWER
        # =================================================

        st.markdown(
            '<div class="answer-heading">💡 Suggested Solution</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            response.content
        )


        # =================================================
        # SOURCES
        # =================================================

        st.markdown(
            '<div class="sources-heading">📚 Sources Used</div>',
            unsafe_allow_html=True
        )

        shown_sources = set()

        for doc in relevant_docs:

            source = os.path.basename(
                doc.metadata.get(
                    "source",
                    "Unknown"
                )
            )

            if source not in shown_sources:

                st.markdown(
                    f"📄 **{source}**"
                )

                shown_sources.add(source)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
    TechFix AI&nbsp;&nbsp;•&nbsp;&nbsp;
    Built with Python, LangChain, ChromaDB, Groq & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
