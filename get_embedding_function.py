from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embedding_function():
    """Returns the Google Generative AI embedding function."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings