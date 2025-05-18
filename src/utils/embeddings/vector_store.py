from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.documents import Document  # or from langchain.schema

import os

def create_vector_store(chunks=None, embedding_model="all-MiniLM-L6-v2", save_path="vectorstore/my_index", load_existing=False, allow_dangerous_deserialization=False):
    """
    Create or load a FAISS vector store, optionally updating it with new chunks.

    Args:
        chunks (list or None): Optional list of Document chunks to add.
        embedding_model (str): SentenceTransformer model to use.
        save_path (str): Path to store or load the vector store.
        load_existing (bool): Whether to load from disk if it exists.
        allow_dangerous_deserialization (bool): Allow loading potentially unsafe data.
        
    Returns:
        FAISS: Loaded or updated vector store.
    """
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
        vector_store = None

        # Load existing store if required
        if load_existing and os.path.exists(save_path):
            vector_store = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=allow_dangerous_deserialization)
            if chunks:
                if not all(isinstance(chunk, Document) for chunk in chunks):
                    raise ValueError("Chunks must be LangChain Document instances.")
                vector_store.add_documents(chunks)
                vector_store.save_local(save_path)
                
        else:
            if chunks:
                if not all(isinstance(chunk, Document) for chunk in chunks):
                    raise ValueError("Chunks must be LangChain Document instances.")
                vector_store = FAISS.from_documents(chunks, embeddings)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                vector_store.save_local(save_path)
            else:
                raise FileNotFoundError(f"No vector store found at {save_path} and no chunks provided to create one.")

        return vector_store

    except Exception as e:
        raise RuntimeError(f"Error in create_vector_store: {e}")



def dupli_create_vector_store(chunks, embedding_model="all-MiniLM-L6-v2", save_path="vectorstore/my_index"):
    """
    Create a vector store from the given chunks using SentenceTransformer embeddings.
    
    Args:
        chunks (list): List of document chunks to be embedded.
        embedding_model (str): Name of the SentenceTransformer model to use.
        save_path (str): Optional path to save the vector store.
        
    Returns:
        FAISS: The created vector store.
    """
    try:
        
        
        # Initialize the embedding model
        embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
        # Validate chunks
         # Validate chunks
        if not isinstance(chunks, list) or not chunks or not all(isinstance(chunk, Document) for chunk in chunks):
            raise ValueError("Chunks must be a non-empty list of LangChain Document objects.")
        
        # Create the vector store
        vector_store = FAISS.from_documents(chunks, embeddings)

        # Check if the save_path is valid and writable
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if os.access(save_dir, os.W_OK):
            vector_store.save_local(save_path)
        
    except Exception as e:
        raise RuntimeError(f"Failed to create vector store: {e}")
        
    return vector_store


