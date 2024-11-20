from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from chromadb import PersistentClient
import os

CHROMA_PATH = "/tmp/chroma"

def populate_database(documents):
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    # Initialize Chroma client
    client = PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection("documents")
    
    # Prepare documents for insertion
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"doc_{i}" for i in range(len(chunks))]
    
    # Add documents to collection
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )