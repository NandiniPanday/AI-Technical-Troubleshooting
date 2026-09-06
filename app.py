import os
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load API key from .env
load_dotenv()

# -----------------------------
# 1. Connect to Groq LLM
# -----------------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# -----------------------------
# 2. Load the same embedding model
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# 3. Connect to ChromaDB
# -----------------------------
vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# -----------------------------
# 4. Create Retriever
# -----------------------------
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)

# -----------------------------
# 5. Streamlit UI
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

# User question
question = st.text_area(
    "Describe your problem:",
    placeholder="Example: My laptop is connected to Wi-Fi but websites are not opening."
)

# -----------------------------
# 6. Process question
# -----------------------------
if st.button("Find Solution"):

    if not question.strip():
        st.warning("Please describe your problem.")
    else:

        # Search ChromaDB
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
