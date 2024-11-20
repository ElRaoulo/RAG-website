import streamlit as st
import os
import logging
from query_data import query_rag
from populate_database import populate_database, split_documents
from langchain_core.documents import Document
import PyPDF2
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine Chroma path based on environment
CHROMA_PATH = os.getenv('CHROMA_PATH', tempfile.gettempdir() + '/chroma')

# Ensure the directory exists
os.makedirs(CHROMA_PATH, exist_ok=True)

def validate_pdf(file):
    """Validate PDF file before processing."""
    try:
        PyPDF2.PdfReader(file)
        return True
    except Exception as e:
        st.error(f"Invalid PDF file: {e}")
        return False

# Function to read PDF content
def read_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        documents = []
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": file.name,
                    "page": page_num + 1
                }
            ))
        return documents
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return []

def main():
    # Add a title
    st.title("RAG Document Query Application")

    # Check for API key
    if 'GOOGLE_API_KEY' not in st.secrets:
        st.error("Google API Key is not configured. Please set up in Streamlit secrets.")
        st.stop()

    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    # Add a sidebar title
    st.sidebar.title("Document Upload")

    # Add file uploader to sidebar
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file:", type=["pdf"])

    if uploaded_file:
        if validate_pdf(uploaded_file):
            if st.sidebar.button("Process Document"):
                with st.spinner("Processing document..."):
                    try:
                        documents = read_pdf(uploaded_file)
                        if documents:
                            populate_database(documents)
                            st.sidebar.success("Document processed successfully!")
                        else:
                            st.sidebar.error("Failed to process the document.")
                    except Exception as e:
                        st.sidebar.error(f"Error processing document: {e}")

    # Main query interface
    st.subheader("Ask Questions")
    query_text = st.text_input("Enter your question about the document:")

    k = st.slider("Number of context chunks", 1, 5, 2)
    show_context = st.checkbox("Show context")

    if query_text:
        with st.spinner("Generating answer..."):
            try:
                prompt, response = query_rag(query_text=query_text, k=k)
                if show_context:
                    st.write("Context:")
                    st.write(prompt)
                st.write("Response:")
                st.write(response)
            except Exception as e:
                st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()