from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os


# 1. Load all PDF files
documents = []

for file in os.listdir("documents"):
    if file.endswith(".pdf"):
        pdf_path = os.path.join("documents", file)

        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

print(f"Loaded {len(documents)} pages from PDFs.")


# 2. Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} text chunks.")


# 3. Convert text chunks into embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating embeddings...")


# 4. Store embeddings in ChromaDB
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("RAG knowledge base created successfully!")