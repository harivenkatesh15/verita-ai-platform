import os
from typing import List
from dotenv import load_dotenv

# Modern LangChain Imports
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables (API Keys)
load_dotenv()

# --- 1. DATA LOADING ---
def load_pdfs(data_path: str):
    """Extracts text from all PDF files in a directory."""
    loader = DirectoryLoader(
        data_path, 
        glob="*.pdf", 
        show_progress=True, 
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# --- 2. DATA CLEANING ---
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Removes heavy metadata to keep the vector store lightweight."""
    minimal_docs = []
    for doc in docs:
        minimal_doc = Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source", "")}
        )
        minimal_docs.append(minimal_doc)
    return minimal_docs

# --- 3. DATA SPLITTING ---
def text_split(minimal_docs: List[Document]):
    """Splits documents into smaller chunks for better AI context."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = splitter.split_documents(minimal_docs)
    return text_chunks

# --- 4. EMBEDDINGS ---
def download_embeddings():
    """Downloads the HuggingFace model for vector conversion."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

