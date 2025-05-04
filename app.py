import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import os
import torch
import textwrap

import os
from dotenv import load_dotenv

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline

from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain 

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


#from constants import CHROMA_SETTINGS
"""
checkpoint = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.float32)

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.float32,
        max_length=512,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm
    """
    
    

def llm_groq():
    return ChatGroq(
    groq_api_key=groq_api_key, 
    model_name="llama3-8b-8192",
    temperature=0.7,
    top_p=0.95,
    max_tokens=512
    )
    


def qa_llm():
    #llm = llm_pipeline()
    llm = llm_groq()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa.invoke(instruction)
    answer = generated_text['result']
    return answer, generated_text

def main():
    st.title("Tesla chatbot for User Q n A")
    question = st.text_input("Enter your question:")
    if st.button("Submit"):
        if question:
            with st.spinner("Generating answer..."):
                answer, generated_text = process_answer(question)
                st.success("Answer generated!")
                st.write("Answer:", answer)
                st.write("Source documents:")
                for doc in generated_text['source_documents']:
                    st.write(doc.metadata['source'])
        else:
            st.warning("Please enter a question.")


if __name__ == "__main__":
    main()