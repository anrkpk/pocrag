import os
import streamlit as st
from dotenv import load_dotenv

from utils.embeddings.chunking import load_and_split_sources
from utils.embeddings.retriever import create_retriever, create_hybrid_retriever
from utils.embeddings.generator import create_generator
from utils.embeddings.vector_store import create_vector_store

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
VECTOR_STORE_PATH = "vectorstore/my_index"

def render():
    """Render the upload page."""
    st.title("Upload PDF Files")
    st.write("Upload your PDF files or enter URLs to create or update the Knowledge Base")

    # Session state initialization
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.pdf_names = []
        st.session_state.uploaded_files = []
        st.session_state.chunks = None
        st.session_state.vector_store = None
        st.session_state.retriever = None
        st.session_state.generator = None
        st.session_state.pdf_name = None

    uploaded_files = st.file_uploader("Choose PDF file(s)", type="pdf", accept_multiple_files=True)
    url_input = st.text_area("🔗 Enter URL(s) (comma or newline separated)")

    url_list = []
    if url_input.strip():
        url_lines = url_input.replace(',', '\n').splitlines()
        url_list = [u.strip() for u in url_lines if u.strip()]
        st.session_state.url_list = url_list

    # Check for new files
    new_files = False
    current_file_names = [file.name for file in uploaded_files] if uploaded_files else []
    if set(current_file_names) != set(st.session_state.pdf_names):
        new_files = True
        st.session_state.uploaded_files = uploaded_files

    # Display uploaded files and URLs
    if uploaded_files:
        st.write(f"📚 {len(uploaded_files)} file(s) selected")
        for file in uploaded_files:
            st.write(f"- {file.name}")
    if url_list:
        st.write(f"🌐 {len(url_list)} URL(s) provided")
        for url in url_list:
            st.write(f"- {url}")

    if st.button("Process Documents") or new_files:
        if not api_key:
            st.error("Please set your GROQ API key in the .env file.")
            return

        with st.spinner("Processing documents..."):
            st.session_state.pdf_names = current_file_names

            # Step 1: Load and chunk documents
            chunks = load_and_split_sources(
                pdf_files=uploaded_files,
                urls=url_list,
                custom_metadata={"uploaded_by": "admin", "project": "customer-care-ai"}
            )
            st.session_state.chunks = chunks
            total_sources = len(uploaded_files or []) + len(url_list or [])

            st.success(f"✅ Processed {len(chunks)} chunks from {total_sources} source(s)!")

            # Step 2: Create or update vector store
            st.write("🔄 Creating or updating vector store...")
            vector_store = create_vector_store(
                chunks=chunks if chunks else None,
                save_path=VECTOR_STORE_PATH,
                load_existing=True,
                allow_dangerous_deserialization=True
            )
            st.session_state.vector_store = vector_store
            st.write("✅ Vector store ready")

            # Step 3: Set up retriever
            st.write("🔄 Setting up retriever...")
            retriever = create_hybrid_retriever(vector_store=vector_store)
            st.session_state.retriever = retriever
            st.write("✅ Retriever ready")

            # Step 4: Set up generator
            st.write("🔄 Setting up generator...")
            generator = create_generator(retriever=retriever)
            st.session_state.generator = generator
            st.write("✅ Generator ready")

            st.session_state.initialized = True
            st.success("Knowledge base ready! You can now chat in the Chatbot tab.")

    # Fallback: Load existing vector store if not already loaded
    if not st.session_state.vector_store and os.path.exists(VECTOR_STORE_PATH):
        st.write("🔄 Loading existing vector store...")
        try:
            vector_store = create_vector_store(
                chunks=None,
                save_path=VECTOR_STORE_PATH,
                load_existing=True,
                allow_dangerous_deserialization=True
            )
            st.session_state.vector_store = vector_store
            st.session_state.retriever = create_hybrid_retriever(vector_store=vector_store)
            st.session_state.generator = create_generator(retriever=st.session_state.retriever)
            st.session_state.initialized = True
            st.success("Loaded existing knowledge base.")
        except Exception as e:
            st.warning(f"⚠️ Failed to load vector store: {e}")

    # Clear state button
    if st.session_state.vector_store and st.button("Clear Document"):
        for key in ['initialized', 'chunks', 'vector_store', 'retriever', 'generator', 'pdf_name', 'pdf_names']:
            st.session_state.pop(key, None)
        st.success("Document state cleared.")
        st.experimental_rerun()
