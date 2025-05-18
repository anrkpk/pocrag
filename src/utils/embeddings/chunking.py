from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import os
import tempfile



def load_and_split_sources(pdf_files=None, urls=None, chunk_size=1000, chunk_overlap=200, custom_metadata=None):
    """Load and chunk PDF files and URLs, and attach metadata."""
    documents = []
    # Load PDFs
    if pdf_files:
        for file in pdf_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            try:
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata = {
                        "source": file.name,
                        "type": "pdf",
                        "uploaded_at": datetime.now().isoformat()
                    }
                    if custom_metadata:
                        doc.metadata.update(custom_metadata)
                documents.extend(docs)
            finally:
                os.remove(temp_file_path)

    # Load URLs
    if urls:
        loader = WebBaseLoader(urls)
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata = {
                "source": urls[i] if i < len(urls) else "unknown",
                "type": "url",
                "fetched_at": datetime.now().isoformat()
            }
            if custom_metadata:
                doc.metadata.update(custom_metadata)
        documents.extend(docs)

    # Chunk all documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    # Add chunk ID metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks
    
