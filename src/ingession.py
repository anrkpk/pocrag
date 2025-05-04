import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from constants import CHROMA_SETTINGS

persist_directory = "db"

def main():
    print("Scanning for PDF files...")
    documents = []

    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(root, file)
                print(f"Loading: {file_path}")
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())

    if not documents:
        print("No PDF documents found.")
        return

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    print("Loading HuggingFace embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Creating Chroma vector store...")
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory   
        )
    db.persist()
    db = None

    print("Ingestion complete!")
    print(f"Your documents are stored in the '{persist_directory}' directory.")
    print("You can now run privateGPT.py to query your documents.")

if __name__ == "__main__":
    main()