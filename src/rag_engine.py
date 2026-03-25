from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, DOCS_PATH, EMBEDDING_MODEL
import os


def load_documents(path: str = DOCS_PATH) -> list:
    """Load all PDF and TXT files from the docs folder."""
    documents = []
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
        else:
            continue
        documents.extend(loader.load())
    return documents


def build_vector_store(documents: list) -> FAISS:
    """Split documents into chunks and create a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def get_retriever(vector_store: FAISS):
    """Return a retriever that fetches the 3 most relevant chunks."""
    return vector_store.as_retriever(search_kwargs={"k": 3})