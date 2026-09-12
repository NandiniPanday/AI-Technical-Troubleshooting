import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="AI Technical Troubleshooting Assistant",
    page_icon="🔧"
)

st.title("🔧 AI Technical Troubleshooting Assistant")

st.write(
    "Describe your technical problem and I will provide "
    "step-by-step troubleshooting guidance."
)


# -----------------------------
# Load API key
# -----------------------------
load_dotenv()

groq_api_key = st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY is missing. Add it in Streamlit Secrets.")
    st.stop()


# -----------------------------
# Connect to Groq LLM
# -----------------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    groq_api_key=groq_api_key
)


# -----------------------------
# Create knowledge base
# -----------------------------
@st.cache_resource
def create_vector_store():

    documents = []

    # Search both the repository root and documents folder
    pdf_paths = list(Path(".").glob("*.pdf"))
    pdf_paths += list(Path("documents").glob("*.pdf"))

    # Remove duplicate paths
    pdf_paths = list(dict.fromkeys(pdf_paths))

    if not pdf_paths:
        raise FileNotFoundError(
            "No PDF documents found in the repository."
        )

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create ChromaDB in memory
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store


# -----------------------------
# Load vector store
# -----------------------------
try:
    vector_store = create_vector_store()
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

except Exception as e:
    st.error(f"Could not create the knowledge base: {e}")
    st.stop()


# -----------------------------
# User question
# -----------------------------
question = st.text_area(
    "Describe your problem:",
    placeholder=(
        "Example: My laptop is connected to Wi-Fi "
        "but websites are not opening."
    )
)


# -----------------------------
# Process question
# -----------------------------
if st.button("Find Solution"):

    if not question.strip():
        st.warning("Please describe your problem.")

    else:

        # Retrieve relevant documents
        relevant_docs = retriever.invoke(question)

        # Combine retrieved information
        context = "\n\n".join(
            [doc.page_content for doc in relevant_docs]
        )

        # Prompt for the AI
        prompt = f"""
You are an AI Technical Troubleshooting Assistant.

Use ONLY the information provided in the context below.

Give the user:

1. A short explanation of the possible problem.
2. Clear step-by-step troubleshooting instructions.
3. A short note if the problem may require professional support.

If the context does not contain enough information to answer,
say that the available documents do not provide enough information.

Context:
{context}

User's Problem:
{question}
"""

        # Ask Groq
        response = llm.invoke(prompt)

        # Display answer
        st.subheader("💡 Suggested Solution")
        st.write(response.content)

        # Display sources
        st.subheader("📚 Sources")

        shown_sources = set()

        for doc in relevant_docs:

            source = os.path.basename(
                doc.metadata.get("source", "Unknown")
            )

            if source not in shown_sources:
                st.write(f"📄 {source}")
                shown_sources.add(source)
