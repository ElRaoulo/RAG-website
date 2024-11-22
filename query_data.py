import sys
import importlib

# Attempt to import pysqlite3 and patch sqlite3 if needed
try:
    import pysqlite3 as sqlite3
    sys.modules['sqlite3'] = sqlite3
except ImportError:
    pass

import argparse
from chromadb import PersistentClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
from langchain_core.messages import HumanMessage
import os

CHROMA_PATH = "/tmp/chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str, k: int = 2):
    # Initialize Chroma client
    client = PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection("documents")
    
    # Search for similar documents
    results = collection.query(
        query_texts=[query_text],
        n_results=k
    )
    
    # Prepare context
    context_text = "\n\n---\n\n".join(results['documents'][0]) if results['documents'] else ""
    
    # Format prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Initialize Gemini Pro
    model = GoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    response_text = model.invoke(prompt)

    sources = results['metadatas'][0] if results['metadatas'] else []
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return prompt, response_text