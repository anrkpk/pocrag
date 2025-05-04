
import streamlit as st
import torch

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline

from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# ========== GROQ LLM ==========
@st.cache_resource
def llm_groq():
    return ChatGroq(
        groq_api_key="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",  # Replace with environment variable in production
        model_name="llama3-8b-8192",
        temperature=0.7,
        top_p=0.95,
        max_tokens=512
    )

# ========== QA Chain Setup ==========
@st.cache_resource
def qa_llm():
    llm = llm_groq()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa

# ========== Process Answer ==========
def process_answer(instruction):
    qa = qa_llm()
    generated_text = qa.invoke({"query": instruction})
    answer = generated_text['result']
    return answer, generated_text

# ========== Streamlit UI ==========
def main():
    st.title("🚗 Tesla Chatbot - User Q&A")
    question = st.text_input("Enter your question:")
    
    if st.button("Submit"):
        if question:
            with st.spinner("Generating answer..."):
                answer, generated_text = process_answer(question)
                st.success("Answer generated!")
                st.write("### 💬 Answer:")
                st.write(answer)

                st.write("### 📄 Source Documents:")
                for doc in generated_text['source_documents']:
                    st.write(doc.metadata.get('source', 'No source metadata found'))
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
