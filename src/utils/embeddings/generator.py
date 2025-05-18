
import os
from dotenv import load_dotenv

load_dotenv()

import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

api_key = os.getenv("GROQ_API_KEY")


def create_generator(retriever, model_name="llama3-8b-8192", api_key=api_key, temperature=0.7, max_tokens=150, custom_prompt=None):
    """
    Create a generator from the given retriever.
    
    Args:
        retriever (FAISS): The FAISS vector store to create a retriever from.
        model_name (str): The name of the language model to use.
        temperature (float): The temperature for the language model.
        max_tokens (int): The maximum number of tokens for the language model.
        
    Returns:
        LLMChain: The created generator.
    """
    try:
        # Initialize the language model
        llm = ChatGroq(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=api_key
        )
        #llm = OpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        
        # Create prompt template
        if custom_prompt:
           prompt = ChatPromptTemplate.from_template(custom_prompt)
        else:
            prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant. Use the following context to answer the question.
            If you don't know the answer based on the context, say "I don't have enough information to answer this question."
            
            Context:
            {context}
            
            Question:
            {input}
            """
        )
    
        # Create document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
    
        # Create retrieval chain
        rag_chain = create_retrieval_chain(retriever, document_chain)

    except Exception as e:
        raise RuntimeError(f"Failed to create generator: {e}")
    
    return rag_chain