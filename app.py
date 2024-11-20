import streamlit as st
import os
import logging
from query_data import query_rag
from populate_database import populate_database, split_documents
from langchain_core.documents import Document
import PyPDF2

# Initialize session state for database
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key from Streamlit secrets
if 'GOOGLE_API_KEY' not in st.secrets:
    st.error("GOOGLE_API_KEY not found in secrets.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

CHROMA_PATH = "/tmp/chroma"

# Function to read PDF content
def read_pdf(file):
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

# Add a title
st.title("RAG Application")

# Add a sidebar title
st.sidebar.title("Document Upload")

# Add file uploader to sidebar
uploaded_file = st.sidebar.file_uploader("Upload a PDF file:", type=["pdf"])

if uploaded_file:
    if st.sidebar.button("Process Document"):
        with st.spinner("Processing document..."):
            documents = read_pdf(uploaded_file)
            populate_database(documents)
            st.success("Document processed successfully!")

# Main query interface
st.subheader("Ask Questions")
query_text = st.text_input("Enter your question about the document:")

k = st.slider("Number of context chunks", 1, 5, 2)
show_context = st.checkbox("Show context")

if query_text:
    with st.spinner("Generating answer..."):
        prompt, response = query_rag(query_text=query_text, k=k)
        if show_context:
            st.write("Context:")
            st.write(prompt)
        st.write("Response:")
        st.write(response)