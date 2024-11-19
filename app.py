import streamlit as st
from query_data import query_rag
from populate_database import populate_database, split_documents
from langchain.schema.document import Document
import PyPDF2
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
import logging

CHROMA_PATH = "chroma"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to clear the database
def clear_database():
    try:
        # Initialize the database
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Get all document IDs
        all_ids = db.get()["ids"]
        
        if all_ids:
            # Delete all documents
            db.delete(all_ids)
            logger.info(f"Deleted {len(all_ids)} documents from the database.")
        else:
            logger.info("Database is already empty.")
        
        return True
    except Exception as e:
        logger.error(f"An error occurred while clearing the database: {str(e)}")
        return False

# Add a title
st.title("RAG Application")

# Add a sidebar title
st.sidebar.title("Sidebar Menu")

# Add widgets to the sidebar
uploaded_file = st.sidebar.file_uploader("Upload a PDF file:", type=["pdf"])

# Function to read PDF content
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Add buttons to submit and clear database in the sidebar
col1, col2 = st.sidebar.columns(2)

if col1.button("Submit"):
    if uploaded_file is not None:
        # Read the PDF content
        pdf_content = read_pdf(uploaded_file)
        
        # Create a Document object
        doc = Document(page_content=pdf_content, metadata={"source": uploaded_file.name})
        
        # Split the document
        chunks = split_documents([doc])
        
        # Populate the database
        populate_database(chunks)
        st.sidebar.success("Database populated successfully!")
    else:
        st.sidebar.error("Please upload a PDF file first.")

if col2.button("Clear Database"):
    if clear_database():
        st.sidebar.success("Database cleared successfully!")
    else:
        st.sidebar.error("Failed to clear the database.")

# Main Page
query_text = st.text_input("Enter your question about the PDF:")

# Create two columns for the slider and checkbox
col1, col2 = st.columns(2)

# Add the slider to the first column
with col1:
    k = st.slider("Choose a number for k", 1, 5, 2, help="k determines how many chunks the RAG retrieves as context.")

# Add the checkbox to the second column, centered
with col2:
    # Create three sub-columns within col2 to center the checkbox
    left_spacer, checkbox_col, right_spacer = st.columns([1,2,1])
    with checkbox_col:
        print_context = st.checkbox("Print context", help="Activate to print out the retrieved context.")

if query_text:
    prompt, response = query_rag(query_text=query_text, k=k)
    if print_context:
        st.write("Context:")
        st.write(prompt)
    st.write("Response:")
    st.write(response)