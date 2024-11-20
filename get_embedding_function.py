from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

def get_embedding_function():
    """Returns the Google Generative AI embedding function."""
    # Make sure GOOGLE_API_KEY is set in your environment variables
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings